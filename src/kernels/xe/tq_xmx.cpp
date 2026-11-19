// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/tq_xmx.cpp - b70-local (11-19): Inkling 2's trellis-coded (tq2) experts on the XMX engines, for windows
// with many entries an expert (the prompt).  The blob and the codes' order are tq_kernels.cpp's.  First the entries'
// inputs are rotated into a scratch row each (x' = Hb (sign * x): a work-item a 16 inputs, H16 in it and H4 over 4
// neighbouring lanes).  Then a work-group of 4 sub-groups takes 64 rows (4 bands) of one expert's projection and up to
// 8 * MT of its entries: per 256 inputs each lane decodes one tile (its band, its 16 inputs: the decode kernel's
// coalesced loads) into local memory as FP16, the entries' 256 inputs are gathered as FP16, and each sub-group multiplies
// its band on the XMX engines (8 x 16 x 16 tiles, FP32 sums); the row gains go on the FP32 sums.  B70 (64 experts x 5
// entries, a prompt window's share): gate / up 5.8 ms against 6.5 for tq_kernels.cpp's two-entry kernel; the decode
// into local memory is still ~4x slower a tile than that kernel's (11 instructions a weight).  The unrolled MMA loop
// spills 4.7 KB; unrolled by 1, 4 or 8 it does not, and runs slower (8.0 / 7.9 / 8.0 ms); writing each weight to local
// memory alone (no row of 16 floats) 8.4 ms; 16 entries a pass in the large register file 9.1 ms.
#include "strata/kernels/tq_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <unordered_map>


namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using half = sycl::half;

constexpr int TM = 8, TN = 16, TK = 16;   // one XMX tile
constexpr int NSG = 4, RT = NSG * 16;     // sub-groups (bands) a work-group, its rows
constexpr int KC = 256;                   // inputs a step: 16 tiles of a band


template<typename P>
inline auto lptr(P& acc) { return acc.template get_multi_ptr<sycl::access::decorated::no>(); }

struct Buf { float* p = nullptr; size_t n = 0; };
std::mutex buf_mu;
std::unordered_map<size_t, Buf>& bufs() { static std::unordered_map<size_t, Buf> m; return m; }
// the rotated inputs of this queue's device (slot 0: x' for gate / up, slot 1: h' for down); an older, smaller buffer
// stays allocated for the graphs recorded with it
float* scratch(sycl::queue& q, size_t n, int slot) {
    std::lock_guard<std::mutex> lk(buf_mu);
    Buf& b = bufs()[std::hash<sycl::device>{}(q.get_device()) * 4 + (size_t) slot];
    if (b.n < n) {
        b.p = sycl::malloc_device<float>(n, q);
        b.n = b.p ? n : 0;
    }
    if (b.p == nullptr) throw std::runtime_error("tq experts (XMX): no scratch for the rotated inputs");
    return b.p;
}

struct View {
    int64_t ff;
    const uint8_t* gate;
    const uint8_t* up;
    const uint8_t* down;
    const half* gains;
    const uint8_t* gu_sign;
    const uint8_t* d_sign;
};
inline View view(const uint8_t* blob, int64_t N) {
    View v;
    v.ff = *(const int32_t*) blob;
    const uint8_t* b = blob + 256;
    const size_t gu = (size_t) (v.ff / 16) * (size_t) (N / 16) * kTqTileBytes;
    v.gate = b;
    v.up = b + gu;
    v.down = b + 2 * gu;
    v.gains = (const half*) (b + 3 * gu);
    v.gu_sign = (const uint8_t*) (v.gains + 2 * v.ff + N);
    v.d_sign = v.gu_sign + N / 8;
    return v;
}

// out[e][0, K) = Hb (sign * in[row(e)]) for every entry: K = N (gate / up: row = ent_tok[e]) or the expert's ff (down:
// row = e); a work-item 16 inputs; lanes past K still take part in the exchanges (K is a multiple of 64)
sycl::event rotate(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_tok, const float* in, int64_t ld_in, float* out,
                   int64_t ld_out, int64_t N, int64_t KMAX, bool down) {
    const size_t items = (size_t) ((KMAX / 16 + 15) / 16 * 16);
    return q.parallel_for(sycl::nd_range<2>({cg, items}, {1, 16}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t c = (int64_t) it.get_global_id(1);
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const View v = view((const uint8_t*) grp_ptr[g], N);
            const int64_t K = down ? v.ff : N;
            const bool live = c * 16 < K;
            const int64_t cc = live ? c : 0;
            const uint8_t* sgn = down ? v.d_sign : v.gu_sign;
            const uint32_t sb = (uint32_t) sgn[2 * cc] | ((uint32_t) sgn[2 * cc + 1] << 8);
            for (int e = grp_start[g]; e < grp_start[g + 1]; ++e) {
                const float* xp = in + (size_t) (down ? e : ent_tok[e]) * (size_t) ld_in + cc * 16;
                float xv[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) xv[i] = ((sb >> i) & 1) ? -xp[i] : xp[i];
#pragma unroll
                for (int hh = 1; hh < 16; hh <<= 1)
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        if (!(i & hh)) { const float a0 = xv[i], a1 = xv[i | hh]; xv[i] = a0 + a1; xv[i | hh] = a0 - a1; }
#pragma unroll
                for (int m = 1; m < 4; m <<= 1)
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float o = sycl::permute_group_by_xor(sg, xv[i], m);
                        xv[i] = (lane & m) ? o - xv[i] : xv[i] + o;
                    }
                if (live) {
                    float* op = out + (size_t) e * (size_t) ld_out + c * 16;
#pragma unroll
                    for (int i = 0; i < 16; ++i) op[i] = xv[i] * 0.125f;
                }
            }
        }
    });
}

// DOWN false: gate and up (xr: x' [entry][N]; out gate / up [entry][FF]); DOWN true: down (xr: h' [entry][FF], K = ff;
// out [ent_dst][N])
template<int MT, bool DOWN, int GRF>
struct TqXmx {
    const unsigned long long* grp_ptr;
    const int32_t* grp_start;
    const int32_t* n_groups;
    const int32_t* ent_dst;
    const float* xr;
    int64_t ldx;
    float* o0;
    float* o1;
    int64_t N, FF;
    int n_rt;   // row tiles of one projection (gate / up: FF / 64, both in the grid; down: N / 64)
    sycl::local_accessor<half, 1> wt;    // [RT][KC]: a row's KC inputs contiguous (B column-major)
    sycl::local_accessor<half, 1> xa;    // [8 * MT][KC]
    sycl::local_accessor<float, 1> ot;   // [8 * MT][RT]

    void operator()(sycl::nd_item<2> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int tid = (int) it.get_local_linear_id();
        const int rt = (int) it.get_group(1);
        const bool up = !DOWN && rt >= n_rt;
        const int64_t r0 = (int64_t) (up ? rt - n_rt : rt) * RT;
        constexpr int EP = TM * MT, WG = NSG * 16;
        half* const wtp = lptr(wt).get();
        half* const xap = lptr(xa).get();
        float* const otp = lptr(ot).get();
        // the inputs' rows past an entry pass's n are never written: their products land in output rows that are not
        // stored; zero once so that no stale NaN meets a zero weight
        for (int idx = tid; idx < EP * KC; idx += WG) xap[idx] = (half) 0.0f;
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const View v = view((const uint8_t*) grp_ptr[g], N);
            const int64_t ff = v.ff;
            if (!DOWN && r0 >= ff) continue;   // the same for the whole work-group
            const int64_t K = DOWN ? ff : N, NC = K / 16;
            const uint8_t* M = DOWN ? v.down : (up ? v.up : v.gate);
            const int64_t band = r0 / 16 + sgid;   // this sub-group's band
            const uint8_t* bandp = M + (size_t) band * (size_t) NC * kTqTileBytes;
            const half* gains = v.gains + (DOWN ? 2 * ff : (up ? ff : 0));
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += EP) {
                const int n = sycl::min(EP, e1 - e);
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c[MT];
                for (int i = 0; i < MT; ++i) mx::joint_matrix_fill(sg, c[i], 0.0f);
                for (int64_t k0 = 0; k0 < K; k0 += KC) {
                    // the weights: lane = tile (k0 / 16 + lane) of the sub-group's band, as FP16 (value, no gain)
                    const int64_t c0 = k0 / 16;
                    const int nt = (int) sycl::min<int64_t>(16, NC - c0);   // a multiple of 4
                    half* w = wtp + (size_t) sgid * 16 * KC + lane * 16;
                    if (lane < nt) {
                        const uint32_t* p = (const uint32_t*) (bandp + (size_t) c0 * kTqTileBytes) + lane;
                        uint32_t d[16];
#pragma unroll
                        for (int i = 0; i < 16; ++i) { d[i] = *p; p += nt; }   // steps by nt (AGENTS.md: IGC's GEP simplification)
                        const uint32_t d16 = ((const uint16_t*) (p - lane))[lane];
                        auto tw = [&](int k) -> uint32_t { return k < 32 ? ((d[k >> 1] >> (16 * (k & 1))) & 0xFFFFu) : d16; };
                        auto val = [](uint32_t z) { return (float) tq_bsum(z * 34038481u + 76625530u); };
                        // a row's 16 values as floats, then one conversion and one 32-byte store (inserting FP16 lanes
                        // one by one into a vector was 10x slower)
                        auto put = [&](const float* f, int r) {
                            sycl::vec<float, 16> v;
#pragma unroll
                            for (int i = 0; i < 16; ++i) v[i] = (f[i] - 510.0f) * (1.0f / 147.8f);
                            *reinterpret_cast<sycl::vec<half, 16>*>(w + r * KC) = v.convert<half>();
                        };
                        uint32_t s = tw(0);
                        float row[16];
                        row[0] = val(s);
#pragma unroll
                        for (int wd = 0; wd < 32; ++wd) {
                            const uint32_t U = s | (tw(1 + wd) << 14);
#pragma unroll
                            for (int u = 1; u <= 8; ++u) {
                                const int t = wd * 8 + u;
                                if (t > 255) break;
                                row[t & 15] = val((U >> (2 * u)) & 0x3FFFu);
                                if ((t & 15) == 15) put(row, t >> 4);
                            }
                            s = (U >> 16) & 0x3FFFu;
                        }
                    } else {
#pragma unroll
                        for (int r = 0; r < 16; ++r) *reinterpret_cast<sycl::vec<half, 16>*>(w + r * KC) = sycl::vec<half, 16>((half) 0.0f);
                    }
                    // the n entries' 256 rotated inputs, 4 a work-item (zero past K)
                    for (int idx = tid; idx < n * (KC / 4); idx += WG) {
                        const int er = idx / (KC / 4), kk = (idx % (KC / 4)) * 4;
                        sycl::vec<float, 4> x4(0.0f);
                        if (k0 + kk < K) x4 = *reinterpret_cast<const sycl::vec<float, 4>*>(xr + (size_t) (e + er) * (size_t) ldx + k0 + kk);
                        *reinterpret_cast<sycl::vec<half, 4>*>(xap + er * KC + kk) = x4.convert<half>();
                    }
                    sycl::group_barrier(it.get_group());
                    for (int kk = 0; kk < KC; kk += TK) {
                        mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> bm;
                        mx::joint_matrix_load(sg, bm, lptr(wt) + sgid * TN * KC + kk, KC);
                        for (int i = 0; i < MT; ++i) {
                            mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> a;
                            mx::joint_matrix_load(sg, a, lptr(xa) + i * TM * KC + kk, KC);
                            mx::joint_matrix_mad(sg, c[i], a, bm, c[i]);
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
                for (int i = 0; i < MT; ++i)
                    mx::joint_matrix_store(sg, c[i], lptr(ot) + i * TM * RT + sgid * TN, RT, mx::layout::row_major);
                sycl::group_barrier(it.get_group());
                for (int idx = tid; idx < n * RT; idx += WG) {
                    const int er = idx / RT, rr = idx % RT;
                    const int64_t r = r0 + rr;
                    const float y = otp[er * RT + rr] * (float) gains[r];
                    if (DOWN) o0[(size_t) ent_dst[e + er] * (size_t) N + r] = y;
                    else (up ? o1 : o0)[(size_t) (e + er) * (size_t) FF + r] = y;
                }
                sycl::group_barrier(it.get_group());
            }
        }
    }
    // GRF: the register file (128 for MT 1; 256 would hold MT 2's accumulators, at half the threads)
    auto get(syclex::properties_tag) const {
        return syclex::properties{sycl::ext::intel::experimental::grf_size<GRF>, syclex::sub_group_size<16>};
    }
};

template<int MT, bool DOWN, int GRF>
sycl::event launch(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_dst, const float* xr, int64_t ldx, float* o0, float* o1,
                   int64_t N, int64_t FF) {
    const int n_rt = (int) ((DOWN ? N : FF) / RT);
    const size_t tiles = (size_t) (DOWN ? n_rt : 2 * n_rt);
    return q.submit([&](sycl::handler& h) {
        TqXmx<MT, DOWN, GRF> k{grp_ptr, grp_start, n_groups, ent_dst, xr, ldx, o0, o1, N, FF, n_rt,
                          sycl::local_accessor<half, 1>(sycl::range<1>(RT * KC), h),
                          sycl::local_accessor<half, 1>(sycl::range<1>(TM * MT * KC), h),
                          sycl::local_accessor<float, 1>(sycl::range<1>(TM * MT * RT), h)};
        h.parallel_for(sycl::nd_range<2>({cg, tiles * NSG * 16}, {1, (size_t) NSG * 16}), k);
    });
}

}  // namespace

bool tq_xmx_on(sycl::queue& q, int64_t cap_entries, int64_t N, int64_t FF) {
    // STRATA_TQ_XMX: 0 off, else the entries a window needs (default 64: a window of 8 tokens' top-8; the decode and the
    // speculative windows take tq_kernels.cpp's kernels)
    static const long min_e = [] { const char* v = std::getenv("STRATA_TQ_XMX"); return v ? std::strtol(v, nullptr, 10) : 64L; }();
    if (min_e <= 0 || cap_entries < min_e || N % RT != 0 || FF % RT != 0 || !xmx_available(XmxType::f16)) return false;
    const size_t need = (size_t) (RT * KC + TM * KC) * sizeof(half) + (size_t) (TM * RT) * sizeof(float);
    return q.get_device().get_info<sycl::info::device::local_mem_size>() >= need;
}

sycl::event tq_xmx_gu(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                      const int32_t* n_groups, const int32_t* ent_tok, int64_t cap_entries, const float* x, float* gate,
                      float* up, int64_t N, int64_t FF) {
    float* xr = scratch(q, (size_t) cap_entries * (size_t) N, 0);
    rotate(q, cg, grp_ptr, grp_start, n_groups, ent_tok, x, N, xr, N, N, N, false);
    // 8 entries a pass in the default register file: 2 passes' worth (16 entries) in the large one ran 1.6x slower (half
    // the threads; B70, 64 experts x 5 entries)
    return launch<1, false, 128>(q, cg, grp_ptr, grp_start, n_groups, nullptr, xr, N, gate, up, N, FF);
}

sycl::event tq_xmx_down(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                        const int32_t* n_groups, const int32_t* ent_dst, int64_t cap_entries, const float* h, float* out,
                        int64_t N, int64_t FF) {
    float* hr = scratch(q, (size_t) cap_entries * (size_t) FF, 1);
    rotate(q, cg, grp_ptr, grp_start, n_groups, nullptr, h, FF, hr, FF, N, FF, true);
    return launch<1, true, 128>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hr, FF, out, nullptr, N, FF);
}

}  // namespace strata::kernels
