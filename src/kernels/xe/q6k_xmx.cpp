// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// b70-local: the decode's Q6_K products for 2-8 tokens on Xe2's XMX.  native_mmvq (dp4a) reads a Q6_K row once for
// every column but does every column's arithmetic again: at the MTP window's 3 tokens it ran 1.6x its one-token time
// (B70, 4096 x 8192: 39 -> 64 us), the arithmetic, not the bytes.  Here a sub-group decodes 16 rows' blocks to FP16 in
// local memory (each lane one row: 256 values a block, the row's own scale and d folded in) and multiplies them with
// the tokens' FP16 rows on the matrix engine (8 x 16 x 16 tiles), so 1 to 8 tokens cost about the same.  The tokens
// are FP16 (finer than native_mmvq's int8 per 32), the weights d * scale * q in FP16: not bitwise native_mmvq's.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <mutex>
#include <utility>
#include <vector>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
constexpr int SG = 16, NSG = 4, TM = 8, TN = 16, TK = 16, BLK = 256, HB = 128;

sycl::queue& Q(void* stream) {
    return stream ? *static_cast<sycl::queue*>(stream) : core::Runtime::get().compute();
}

template<typename E>
inline auto lptr(E* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template<typename E>
inline auto gptr(const E* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
        const_cast<E*>(p));
}


// mode 3: no local memory - lane j decodes its own row j's 16 values of a K step straight into its part of the B
// tile (joint_matrix_apply visits a lane's elements in K order for a column-major B on Xe2: checked by the parity test)
void q6k_rows_xmx_reg(const uint8_t* w, const sycl::half* x, float* y, int n_in, int n_out, int T, sycl::queue& q) {
    const int bpr = n_in / BLK;
    const size_t groups = (size_t) (n_out / TN);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lc(sycl::range<1>((size_t) NSG * TM * TN), h);
        h.parallel_for(sycl::nd_range<1>(groups * NSG * SG, NSG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int n0 = (int) it.get_group(0) * TN;
            const size_t rstride = (size_t) bpr * 210;
            const uint8_t* row = w + (size_t) (n0 + lane) * rstride;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c;
            mx::joint_matrix_fill(sg, c, 0.0f);
            for (int kb = sgid; kb < bpr; kb += NSG) {
                const uint8_t* ql = row + (size_t) kb * 128;
                const uint8_t* qh = row + (size_t) bpr * 128 + (size_t) kb * 64;
                const int8_t* sc = reinterpret_cast<const int8_t*>(row + (size_t) bpr * 192 + (size_t) kb * 16);
                const float d = (float) *reinterpret_cast<const sycl::half*>(row + (size_t) bpr * 208 + (size_t) kb * 2);
                #pragma unroll 2
                for (int nh = 0; nh < 2; ++nh) {
                    // this half's 128 values: ql bytes 64 nh .. +63 and qh bytes 32 nh .. +31, four 16-byte loads
                    const sycl::vec<uint32_t, 4> l0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(ql + 64 * nh);
                    const sycl::vec<uint32_t, 4> l1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(ql + 64 * nh + 16);
                    const sycl::vec<uint32_t, 4> l2 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(ql + 64 * nh + 32);
                    const sycl::vec<uint32_t, 4> l3 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(ql + 64 * nh + 48);
                    const sycl::vec<uint32_t, 4> h0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(qh + 32 * nh);
                    const sycl::vec<uint32_t, 4> h1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(qh + 32 * nh + 16);
                    auto qlb = [&](int b) -> uint32_t {   // ql byte 64 nh + b
                        const sycl::vec<uint32_t, 4>& v = b < 16 ? l0 : b < 32 ? l1 : b < 48 ? l2 : l3;
                        return (v[(b % 16) / 4] >> (8 * (b % 4))) & 0xffu;
                    };
                    auto qhb = [&](int b) -> uint32_t {   // qh byte 32 nh + b
                        const sycl::vec<uint32_t, 4>& v = b < 16 ? h0 : h1;
                        return (v[(b % 16) / 4] >> (8 * (b % 4))) & 0xffu;
                    };
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {          // values 128 nh + 32 m + l
                        #pragma unroll
                        for (int g = 0; g < 2; ++g) {      // two K steps of 16: l = 16 g .. 16 g + 15
                            const float s = d * (float) sc[8 * nh + 2 * m + g];
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::col_major> b;
                            int i = 0;
                            mx::joint_matrix_apply(sg, b, [&](sycl::half& v) {
                                const int l = 16 * g + i;
                                const uint32_t lb = qlb((m & 1) ? 32 + l : l), hb = qhb(l);
                                const uint32_t lo4 = (m & 2) ? (lb >> 4) : (lb & 0xFu);
                                v = (sycl::half) (s * (float) ((int) (lo4 | (((hb >> (2 * m)) & 3u) << 4)) - 32));
                                ++i;
                            });
                            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> a;
                            mx::joint_matrix_load(sg, a, gptr(x + (size_t) kb * BLK + 128 * nh + 32 * m + 16 * g), (size_t) n_in);
                            mx::joint_matrix_mad(sg, c, a, b, c);
                        }
                    }
                }
            }
            int i = 0;
            mx::joint_matrix_apply(sg, c, [&](float& v) {
                lc[((size_t) sgid * TM + i) * TN + lane] = v;
                ++i;
            });
            sycl::group_barrier(it.get_group());
            if (sgid != 0) return;
            for (int t = 0; t < T; ++t) {
                float v = 0.0f;
                for (int qq = 0; qq < NSG; ++qq) v += lc[((size_t) qq * TM + t) * TN + lane];
                y[(size_t) t * n_out + n0 + lane] = v;
            }
        });
    });
}

// b70-local: the same for kNativeQ8Rows (a row: its blocks' 32 int8 values each, then their FP16 scales): the decode's
// Q8_0 projections for 2-8 tokens (the MTP window) on XMX, the tokens in FP16.  native_mmvq's dp4a redoes every
// column's arithmetic, so the window's projections cost about T times one token's.
void q8_rows_xmx(const uint8_t* w, const sycl::half* x, float* y, int n_in, int n_out, int T, sycl::queue& q) {
    const int bpr = n_in / 32;
    const size_t groups = (size_t) (n_out / TN);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lc(sycl::range<1>((size_t) NSG * TM * TN), h);
        h.parallel_for(sycl::nd_range<1>(groups * NSG * SG, NSG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int n0 = (int) it.get_group(0) * TN;
            const uint8_t* row = w + (size_t) (n0 + lane) * (size_t) bpr * 34;
            const sycl::half* sc = reinterpret_cast<const sycl::half*>(row + (size_t) bpr * 32);
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c;
            mx::joint_matrix_fill(sg, c, 0.0f);
            for (int kb = sgid; kb < bpr; kb += NSG) {
                const sycl::vec<uint32_t, 4> q0 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(row + (size_t) kb * 32);
                const sycl::vec<uint32_t, 4> q1 = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(row + (size_t) kb * 32 + 16);
                const float d = (float) sc[kb];
                #pragma unroll
                for (int g = 0; g < 2; ++g) {   // two K steps of 16
                    const sycl::vec<uint32_t, 4>& v = g == 0 ? q0 : q1;
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::col_major> b;
                    int i = 0;
                    mx::joint_matrix_apply(sg, b, [&](sycl::half& e) {
                        const int8_t q8 = (int8_t) ((v[i / 4] >> (8 * (i % 4))) & 0xffu);
                        e = (sycl::half) (d * (float) q8);
                        ++i;
                    });
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> a;
                    mx::joint_matrix_load(sg, a, gptr(x + (size_t) kb * 32 + 16 * g), (size_t) n_in);
                    mx::joint_matrix_mad(sg, c, a, b, c);
                }
            }
            int i = 0;
            mx::joint_matrix_apply(sg, c, [&](float& v) {
                lc[((size_t) sgid * TM + i) * TN + lane] = v;
                ++i;
            });
            sycl::group_barrier(it.get_group());
            if (sgid != 0) return;
            for (int t = 0; t < T; ++t) {
                float v = 0.0f;
                for (int qq = 0; qq < NSG; ++qq) v += lc[((size_t) qq * TM + t) * TN + lane];
                y[(size_t) t * n_out + n0 + lane] = v;
            }
        });
    });
}

}  // namespace

bool native_q8_xmx_enabled() {
    static const bool on = [] { const char* e = std::getenv("STRATA_Q8_XMX"); return e != nullptr && std::atoi(e) != 0; }();
    return on;
}
bool native_q8_rows_xmx_ok(int n_in, int n_out, int T) noexcept {
    return n_out % TN == 0 && n_in % (NSG * 32) == 0 && T >= 2 && T <= TM;
}
// x: T rows of n_in floats -> FP16 in a per-queue scratch of 8 rows (rows past T zero), then the XMX product into y
void native_q8_rows_xmx(const void* weights, const float* x, float* y, int n_in, int n_out, int T, void* stream) {
    if (!native_q8_rows_xmx_ok(n_in, n_out, T)) throw std::invalid_argument("native_q8_rows_xmx: n_out % 16, n_in % 128, 2 <= T <= 8");
    sycl::queue& q = Q(stream);
    static std::mutex mu;
    static std::vector<std::pair<sycl::queue*, std::pair<sycl::half*, size_t>>> bufs;
    sycl::half* xh = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& b : bufs)
            if (b.first == &q && b.second.second >= (size_t) n_in * TM) xh = b.second.first;
        if (xh == nullptr) {
            xh = sycl::malloc_device<sycl::half>((size_t) 16384 * TM, q);
            q.memset(xh, 0, sizeof(sycl::half) * (size_t) 16384 * TM);
            bufs.push_back({&q, {xh, (size_t) 16384 * TM}});
        }
    }
    const size_t total = (size_t) n_in * (size_t) TM, live = (size_t) n_in * (size_t) T;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) { xh[i] = i < live ? (sycl::half) x[i] : (sycl::half) 0.0f; });
    q8_rows_xmx(static_cast<const uint8_t*>(weights), xh, y, n_in, n_out, T, q);
}

bool native_q6k_xmx_enabled() {
    static const bool on = [] { const char* e = std::getenv("STRATA_Q6K_XMX"); return e != nullptr && std::atoi(e) != 0; }();
    return on;
}

void native_f32_to_f16_rows(const float* x, void* xh, int n, int T, void* stream) {
    sycl::half* h = static_cast<sycl::half*>(xh);
    const size_t total = (size_t) n * (size_t) T;
    Q(stream).parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) { h[i] = (sycl::half) x[i]; });
}

// weights: kNativeQ6KRows (a row: its blocks' ql, then qh, then the scales, then d).  xh: 8 rows of n_in FP16 (rows
// past T finite, e.g. zeros).  y: T rows of n_out floats.  n_out a multiple of 16, n_in of 512.
void native_q6k_rows_xmx(const void* weights, const void* xh, float* y, int n_in, int n_out, int T, void* stream) {
    if (n_out % TN != 0 || n_in % (NSG * BLK) != 0 || T < 1 || T > TM)
        throw std::invalid_argument("native_q6k_rows_xmx: n_out % 16, n_in % 1024, 1 <= T <= 8");
    const uint8_t* w = static_cast<const uint8_t*>(weights);
    const sycl::half* x = static_cast<const sycl::half*>(xh);
    const int bpr = n_in / BLK;
    const size_t groups = (size_t) (n_out / TN);   // a work-group: 16 rows; its NSG sub-groups split the blocks
    const int mode = [] { const char* e = std::getenv("STRATA_Q6K_XMX_MODE"); return e ? std::atoi(e) : 0; }();
    if (mode == 3) return q6k_rows_xmx_reg(w, x, y, n_in, n_out, T, Q(stream));
    Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> lw(sycl::range<1>((size_t) NSG * TN * HB), h);
        sycl::local_accessor<float, 1> lc(sycl::range<1>((size_t) NSG * TM * TN), h);
        h.parallel_for(sycl::nd_range<1>(groups * NSG * SG, NSG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int n0 = (int) it.get_group(0) * TN;
            sycl::half* my = &lw[(size_t) sgid * TN * HB];
            const size_t rstride = (size_t) bpr * 210;
            const uint8_t* rows = w + (size_t) n0 * rstride;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c;
            mx::joint_matrix_fill(sg, c, 0.0f);
            for (int kb = sgid; kb < bpr; kb += NSG) {
                for (int nh = 0; nh < 2; ++nh) {   // the block's two halves of 128 values in turn (4 KiB a sub-group)
                if (mode != 2) {
                // lane j reads 4 bytes of a row's ql and the 4 qh bytes they pair with (the lanes side by side over
                // the half): ql byte 64 nh + 32 hi + l gives values 128 nh + 32 hi + l (low nibble) and
                // 128 nh + 64 + 32 hi + l (high nibble), with qh byte 32 nh + l's bits 2 hi and 4 + 2 hi
                const int hi = lane / 8, l4 = 4 * (lane % 8);
                const int is = l4 / 16;
                #pragma unroll 4
                for (int r = 0; r < TN; ++r) {
                    const uint8_t* row = rows + (size_t) r * rstride;
                    const uint32_t lv = *reinterpret_cast<const uint32_t*>(row + (size_t) kb * 128 + 64 * nh + 32 * hi + l4);
                    const uint32_t hv = *reinterpret_cast<const uint32_t*>(row + (size_t) bpr * 128 + (size_t) kb * 64 + 32 * nh + l4);
                    const int8_t* sc = reinterpret_cast<const int8_t*>(row + (size_t) bpr * 192 + (size_t) kb * 16);
                    const float d = (float) *reinterpret_cast<const sycl::half*>(row + (size_t) bpr * 208 + (size_t) kb * 2);
                    const float s_lo = d * (float) sc[8 * nh + is + 2 * hi], s_hi = d * (float) sc[8 * nh + is + 4 + 2 * hi];
                    sycl::vec<sycl::half, 4> olo, ohi;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const uint32_t qlb = (lv >> (8 * u)) & 0xffu, qhb = (hv >> (8 * u)) & 0xffu;
                        olo[u] = (sycl::half) (s_lo * (float) ((int) ((qlb & 0xFu) | (((qhb >> (2 * hi)) & 3u) << 4)) - 32));
                        ohi[u] = (sycl::half) (s_hi * (float) ((int) ((qlb >> 4) | (((qhb >> (4 + 2 * hi)) & 3u) << 4)) - 32));
                    }
                    sycl::half* dst = my + (size_t) r * HB + 32 * hi + l4;
                    *reinterpret_cast<sycl::vec<sycl::half, 4>*>(dst) = olo;
                    *reinterpret_cast<sycl::vec<sycl::half, 4>*>(dst + 64) = ohi;
                }
                }
                sycl::group_barrier(sg);
                if (mode != 1)
                #pragma unroll 4
                for (int k = 0; k < HB; k += TK) {
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> a;
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::col_major> b;
                    mx::joint_matrix_load(sg, a, gptr(x + (size_t) kb * BLK + 128 * nh + k), (size_t) n_in);
                    mx::joint_matrix_load(sg, b, lptr(my + k), (size_t) HB);
                    mx::joint_matrix_mad(sg, c, a, b, c);
                }
                sycl::group_barrier(sg);
                }
            }
            // the sub-groups' partial sums through local memory, summed by sub-group 0.  Xe's accumulator layout: lane
            // j holds column j, its element i row i (as iq_mmq's intel_body reads it)
            int i = 0;
            mx::joint_matrix_apply(sg, c, [&](float& v) {
                lc[((size_t) sgid * TM + i) * TN + lane] = v;
                ++i;
            });
            sycl::group_barrier(it.get_group());
            if (sgid != 0) return;
            for (int t = 0; t < T; ++t) {
                float v = 0.0f;
                for (int q = 0; q < NSG; ++q) v += lc[((size_t) q * TM + t) * TN + lane];
                y[(size_t) t * n_out + n0 + lane] = v;
            }
        });
    });
}

}  // namespace strata::kernels
