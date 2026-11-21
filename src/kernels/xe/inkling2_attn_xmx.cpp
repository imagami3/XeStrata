// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/inkling2_attn_xmx.cpp - Inkling 2's attention for a prompt window on Xe's matrix engines (b70-local,
// 2026-11-19).
//
// inkling2_attention gives every (token, head) a work-group whose work-items each take a whole key row: on the B70 a
// 512-token window's attention was 36M clock ticks a sliding layer and 121M a full one at ~4.7K cells (a third and two
// thirds of the layer).  This kernel follows glm5_mla_xmx.cpp's scheme:
//   * a work-group takes the heads that share a KV head for 16 tokens (96 rows; 16 tokens of one head where the heads
//     do not fit) and walks the cells from the first token's window start to the last token's position, 16 at a time,
//     with an online softmax per row; sub-group s owns dims [64 (s % 2), + 64) of rows [8 (s / 2), + 8);
//   * the chunk's K and V rows (FP32 in the cache) go into local memory as FP16 hi + lo, q (scaled by a power of two
//     that puts its largest value near 2^14) and p likewise: three products each (hi.hi + lo.hi + hi.lo), so they keep
//     about 22 bits;
//   * the relative bias r . table[pos - j] (pos - j < extent), the causal and sliding-window masks and the sink (a
//     logit with a zero value, in the denominator) are applied with the scores in FP32.
// Against the plain kernel the outputs differ by at most ~3e-6 of their largest value (tests/core/
// inkling2_attn_xmx_test.cpp).
#include "strata/kernels/inkling2.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_inkling2_attn_xmx
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace ix = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using sycl::half;

constexpr int D = 128;             // a head's width
constexpr int NG = 2;              // 64-dim groups
constexpr int SGS = 16;            // sub-group size
constexpr int CH = 16;             // cells a chunk: lane `lane` owns cell `lane` in the softmax
constexpr int RS = D + 8;          // row strides in halves (16-byte aligned rows)
constexpr int RELMAX = 32;
[[maybe_unused]] constexpr int TM = 8, TN = 16, TK = 16;
[[maybe_unused]] constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
[[maybe_unused]] constexpr float TAU = 8.0f;                     // lazy rescaling, as qsa_prompt_attn.cpp
[[maybe_unused]] constexpr float PSCALE = 16384.0f / 256.0f;     // 2^(14 - TAU): p' = p * PSCALE stays within FP16
[[maybe_unused]] constexpr float LOG2E = 1.4426950408889634f;

template<typename A>
inline auto at(const A& a, size_t off) {
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

// RT tiles of 8 rows a work-group (R rows: `hpw` heads that share a KV head x R / hpw tokens, row = token * hpw +
// head); sub-groups 2 x RT
template<int RT>
struct AttXmx {
    static constexpr int R = 8 * RT, NSG = NG * RT, WG = NSG * SGS;
    const float* q;       // [T][H][D]
    const float* r;       // [T][H][rel]
    const float* sinks;   // [T][H]
    const float* kc;      // [cells][nkv][D]
    const float* vc;
    const float* table;   // [extent][rel]
    const int32_t* steps;
    int T, H, G, hpw, rel, extent, window;
    float* out;
    sycl::local_accessor<half, 1> kv;      // kh, kl, vh, vl [CH][RS]; at the start qh, ql [R][RS] (the larger)
    sycl::local_accessor<float, 3> part;   // [NG][R][CH]
    sycl::local_accessor<half, 2> pp;      // [2][R * CH]: p' hi, lo
    sycl::local_accessor<float, 1> mrow, lsum, alpha;
    sycl::local_accessor<float, 2> rv;     // [R][RELMAX]
    sycl::local_accessor<int, 1> rpos;     // [R]: the row's position (-1: no token)

    static void split(float x, half& hi, half& lo) {
        hi = half(x);
        lo = half(x - f32_from_f16(sycl::bit_cast<uint16_t>(hi)));
    }

    void operator()(sycl::nd_item<1> it) const {
#if STRATA_DEVICE_NOT_INTEL
        (void) it;
#else
        const auto grp = it.get_group();
        const auto sg = it.get_sub_group();
        const int bt = R / hpw, nblk = (T + bt - 1) / bt;
        const int wgi = (int) it.get_group(0), h0 = (wgi / nblk) * hpw, t0 = (wgi % nblk) * bt;
        const int kvh = h0 / G;
        const int t = (int) it.get_local_id(0), s = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int g = s % NG, mt = s / NG;
        // row rr: token t0 + rr / hpw, head h0 + rr % hpw
        auto tok = [&](int rr) { return t0 + rr / hpw; };
        auto qrow = [&](int rr) { return (int64_t) tok(rr) * H + h0 + rr % hpw; };
        const int64_t width = (int64_t) (H / G) * D;
        const size_t QH = 0, QL = (size_t) R * RS;
        const size_t KH = 0, KL = (size_t) CH * RS, VH = 2 * (size_t) CH * RS, VL = 3 * (size_t) CH * RS;

        // q: the rows, times 2^(14 - e) for its largest |value| < 2^e, as FP16 hi + lo; the rows' r and positions
        float qm = 0.0f;
        for (int i = t; i < R * D; i += WG) {
            const int rr = i / D, d = i % D;
            if (tok(rr) < T) qm = sycl::fmax(qm, sycl::fabs(q[qrow(rr) * D + d]));
        }
        qm = sycl::reduce_over_group(grp, qm, sycl::maximum<float>());
        const int qe = qm > 0.0f ? (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFF) - 126 : 0;
        const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(LOG2E / sycl::sqrt((float) D), qe - 14);
        for (int i = t; i < R * D; i += WG) {
            const int rr = i / D, d = i % D;
            half hi(0.0f), lo(0.0f);
            if (tok(rr) < T) split(q[qrow(rr) * D + d] * qup, hi, lo);
            kv[QH + (size_t) rr * RS + d] = hi;
            kv[QL + (size_t) rr * RS + d] = lo;
        }
        for (int i = t; i < R * RELMAX; i += WG) {
            const int rr = i / RELMAX, k = i % RELMAX;
            rv[rr][k] = tok(rr) < T && k < rel ? r[qrow(rr) * rel + k] : 0.0f;
        }
        for (int rr = t; rr < R; rr += WG) {
            const bool live = tok(rr) < T;
            rpos[rr] = live ? steps[(int64_t) tok(rr) * kStepCount + kStepPos] : -1;
            mrow[rr] = live ? sinks[qrow(rr)] * LOG2E : 0.0f;   // the sink's logit
            lsum[rr] = 1.0f;                                     // its term at m = itself
        }
        sycl::group_barrier(grp);
        mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> qa[4], qb[4];
        for (int kk = 0; kk < 4; ++kk) {
            const size_t off = (size_t) mt * TM * RS + (size_t) g * 64 + (size_t) kk * TK;
            mx::joint_matrix_load(sg, qa[kk], at(kv, QH + off), RS);
            mx::joint_matrix_load(sg, qb[kk], at(kv, QL + off), RS);
        }
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[4];
        for (int nt = 0; nt < 4; ++nt) mx::joint_matrix_fill(sg, acc[nt], 0.0f);

        // the cells: from the first row's window start to the last live row's position
        int last = 0;
        for (int i = 0; i < R; ++i) last = sycl::max(last, rpos[i]);
        const int first = rpos[0], lo0 = window > 0 ? sycl::max(0, first - window + 1) : 0;
        for (int c0 = lo0; c0 <= last; c0 += CH) {
            const int nh = sycl::min(CH, last + 1 - c0);
            sycl::group_barrier(grp);   // the previous chunk (and q's staging) is done with kv, part and pp
            for (int i = t; i < CH * (D / 4); i += WG) {   // K and V rows as FP16 hi + lo, 4 values a piece
                const int c = i / (D / 4), pc = (i % (D / 4)) * 4;
                sycl::vec<float, 4> k4(0.0f), v4(0.0f);
                if (c < nh) {
                    const int64_t o = (int64_t) (c0 + c) * width + (int64_t) kvh * D + pc;
                    k4 = *reinterpret_cast<const sycl::vec<float, 4>*>(kc + o);
                    v4 = *reinterpret_cast<const sycl::vec<float, 4>*>(vc + o);
                }
                for (int e = 0; e < 4; ++e) {
                    half hi, lo;
                    split(k4[e], hi, lo);
                    kv[KH + (size_t) c * RS + pc + e] = hi;
                    kv[KL + (size_t) c * RS + pc + e] = lo;
                    split(v4[e], hi, lo);
                    kv[VH + (size_t) c * RS + pc + e] = hi;
                    kv[VL + (size_t) c * RS + pc + e] = lo;
                }
            }
            sycl::group_barrier(grp);

            {   // q.k of this sub-group's 8 rows and 64 dims over the chunk's 16 cells
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c;
                mx::joint_matrix_fill(sg, c, 0.0f);
                for (int kk = 0; kk < 4; ++kk) {
                    mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> bh, bl;
                    mx::joint_matrix_load(sg, bh, at(kv, KH + (size_t) g * 64 + (size_t) kk * TK), RS);
                    mx::joint_matrix_load(sg, bl, at(kv, KL + (size_t) g * 64 + (size_t) kk * TK), RS);
                    mx::joint_matrix_mad(sg, c, qa[kk], bh, c);
                    mx::joint_matrix_mad(sg, c, qb[kk], bh, c);
                    mx::joint_matrix_mad(sg, c, qa[kk], bl, c);
                }
                mx::joint_matrix_store(sg, c, at(part, (size_t) (g * R + mt * TM) * CH), CH, mx::layout::row_major);
            }
            sycl::group_barrier(grp);

            // online softmax: sub-group s takes rows 4s .. 4s + 3, lane = cell; the groups' partials in order
            for (int i = 0; i < R / NSG; ++i) {
                const int rr = s * (R / NSG) + i, c = lane, j = c0 + c;
                const int pos = rpos[rr];
                const int jlo = window > 0 ? sycl::max(0, pos - window + 1) : 0;
                float x = NEG_INF;
                if (pos >= 0 && c < nh && j >= jlo && j <= pos) {
                    float b = 0.0f;
                    const int dist = pos - j;
                    if (dist < extent) {
                        const float* tb = table + (int64_t) dist * rel;
                        for (int k = 0; k < rel; ++k) b += rv[rr][k] * tb[k];
                    }
                    x = (part[0][rr][c] + part[1][rr][c]) * qdown + b * LOG2E;
                }
                float mxv = x;
                for (int o = 1; o < CH; o <<= 1) mxv = sycl::fmax(mxv, sycl::permute_group_by_xor(sg, mxv, o));
                const float m_old = mrow[rr];
                const float m_new = mxv > m_old + TAU ? mxv : m_old;
                const float e = x == NEG_INF ? 0.0f : sycl::exp2(x - m_new);
                float sum = e;
                for (int o = 1; o < CH; o <<= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
                sycl::group_barrier(sg);   // every lane has read mrow[rr] before lane 0 moves it
                half hi, lo;
                split(e * PSCALE, hi, lo);
                pp[0][(size_t) rr * CH + c] = hi;
                pp[1][(size_t) rr * CH + c] = lo;
                if (c == 0) {
                    const float a = m_old == m_new ? 1.0f : sycl::exp2(m_old - m_new);
                    alpha[rr] = a;
                    lsum[rr] = sycl::fma(lsum[rr], a, sum);
                    mrow[rr] = m_new;
                }
            }
            sycl::group_barrier(grp);

            {   // p.v of this sub-group's 8 rows and 64 dims
                float f[TM];
                bool same = true;
                for (int i = 0; i < TM; ++i) {
                    f[i] = alpha[mt * TM + i];
                    same = same && f[i] == 1.0f;
                }
                if (!same)
                    for (int nt = 0; nt < 4; ++nt) {
                        int i = 0;
                        mx::joint_matrix_apply(sg, acc[nt], [&](float& x) { x *= f[i++]; });
                    }
                mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> ph, pl;
                mx::joint_matrix_load(sg, ph, at(pp, (size_t) mt * TM * CH), CH);
                mx::joint_matrix_load(sg, pl, at(pp, (size_t) R * CH + (size_t) mt * TM * CH), CH);
                for (int nt = 0; nt < 4; ++nt) {
                    mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::row_major> bh, bl;
                    mx::joint_matrix_load(sg, bh, at(kv, VH + (size_t) g * 64 + (size_t) nt * TN), RS);
                    mx::joint_matrix_load(sg, bl, at(kv, VL + (size_t) g * 64 + (size_t) nt * TN), RS);
                    mx::joint_matrix_mad(sg, acc[nt], ph, bh, acc[nt]);
                    mx::joint_matrix_mad(sg, acc[nt], pl, bh, acc[nt]);
                    mx::joint_matrix_mad(sg, acc[nt], ph, bl, acc[nt]);
                }
            }
        }
        sycl::group_barrier(grp);
        for (int nt = 0; nt < 4; ++nt)
            ix::joint_matrix_apply(sg, acc[nt], [&](float& x, size_t row, size_t col) {
                const int rr = mt * TM + (int) row;
                if (tok(rr) < T) out[qrow(rr) * D + g * 64 + nt * TN + (int) col] = x / (PSCALE * lsum[rr]);
            });
#endif
    }
    auto get(syclex::properties_tag) const {
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<SGS>};
    }
};

template<int RT>
bool launch(sycl::event& ev, sycl::queue& queue, const float* q, const float* r, const float* sinks, const float* kc,
                   const float* vc, const float* table, const int32_t* steps, int T, int H, int G, int hpw, int rel,
                   int extent, int window, float* out) {
    using K = AttXmx<RT>;
    const size_t kvh = (size_t) std::max(2 * K::R * RS, 4 * CH * RS);
    const size_t need = kvh * 2 + (size_t) NG * K::R * CH * 4 + (size_t) 2 * K::R * CH * 2 + (size_t) 3 * K::R * 4 +
                        (size_t) K::R * RELMAX * 4 + (size_t) K::R * 4;
    const auto dev = queue.get_device();
    if (dev.get_info<sycl::info::device::local_mem_size>() < need ||
        dev.get_info<sycl::info::device::max_work_group_size>() < (size_t) K::WG)
        return false;
    const int bt = K::R / hpw, nblk = (T + bt - 1) / bt;
    ev = queue.submit([&](sycl::handler& h) {
        K k{q, r, sinks, kc, vc, table, steps, T, H, G, hpw, rel, extent, window, out,
            sycl::local_accessor<half, 1>(sycl::range<1>(kvh), h),
            sycl::local_accessor<float, 3>(sycl::range<3>(NG, K::R, CH), h),
            sycl::local_accessor<half, 2>(sycl::range<2>(2, (size_t) K::R * CH), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(K::R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(K::R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(K::R), h),
            sycl::local_accessor<float, 2>(sycl::range<2>(K::R, RELMAX), h),
            sycl::local_accessor<int, 1>(sycl::range<1>(K::R), h)};
        h.parallel_for(sycl::nd_range<1>((size_t) nblk * (H / hpw) * K::WG, K::WG), k);
    });
    return true;
}

}  // namespace

#if defined(STRATA_GB10_SG32)
bool inkling2_attention_gb10(const float* q, const float* r, const float* sinks, const float* kc, const float* vc,
                             const float* table, const int32_t* steps, int n_head, int n_head_kv, int dim, int rel,
                             int extent, int window, float* out, int n_tokens, void* stream);   // a64_ink_attn.cpp
#endif
bool inkling2_attention_xmx(const float* q, const float* r, const float* sinks, const float* kc, const float* vc,
                            const float* table, const int32_t* steps, int n_head, int n_head_kv, int dim, int rel,
                            int extent, int window, float* out, int n_tokens, void* stream) {
#if defined(STRATA_GB10_SG32)   // GB10 port: NVIDIA's prompt-window kernel (a64_ink_attn.cpp)
    if (inkling2_attention_gb10(q, r, sinks, kc, vc, table, steps, n_head, n_head_kv, dim, rel, extent, window, out,
                                n_tokens, stream))
        return true;
#endif
    static const bool on = [] {
        const char* v = std::getenv("STRATA_INK_ATT_XMX");
        return v == nullptr || v[0] != '0';
    }();
    if (!on || n_tokens <= 0 || dim != D || rel > RELMAX || n_head_kv <= 0 || n_head % n_head_kv != 0 ||
        !xmx_available(XmxType::f16))
        return false;
    auto& queue = core::Runtime::get().stream(stream);
    const int G = n_head / n_head_kv;
    // the heads that share a KV head together (their K and V rows converted once a chunk for all of them) where 96 (or
    // 48) rows hold a whole number of tokens of them, else 16 tokens of one head; STRATA_INK_ATT_ROWS caps the rows.
    // B70, 512 tokens of 48 heads over 8 KV heads, 1,040 / 4,512 cells: 16 rows of one head 10.0 / 30.4 ms, 48 rows
    // (8 tokens x 6 heads) 7.4 / 18.2 ms, 96 rows 6.7 / 15.2 ms (the plain kernel 17.2 / 114.8 ms)
    static const int rows = [] { const char* v = std::getenv("STRATA_INK_ATT_ROWS"); return v ? std::atoi(v) : 96; }();
    sycl::event e;
    bool ok = false;
    if (rows >= 96 && 96 % G == 0)
        ok = launch<12>(e, queue, q, r, sinks, kc, vc, table, steps, n_tokens, n_head, G, G, rel, extent, window, out);
    if (!ok && rows >= 48 && 48 % G == 0)
        ok = launch<6>(e, queue, q, r, sinks, kc, vc, table, steps, n_tokens, n_head, G, G, rel, extent, window, out);
    if (!ok) ok = launch<2>(e, queue, q, r, sinks, kc, vc, table, steps, n_tokens, n_head, G, 1, rel, extent, window, out);
    if (!ok) return false;
    if (!stream) core::Runtime::get().wait(e, "inkling2_attention_xmx");
    return true;
}

}  // namespace strata::kernels
