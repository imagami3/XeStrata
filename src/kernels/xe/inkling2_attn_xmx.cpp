// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/inkling2_attn_xmx.cpp - Inkling 2's attention for a prompt window on Xe's matrix engines (b70-local,
// 2026-11-19).
//
// inkling2_attention gives every (token, head) a work-group whose work-items each take a whole key row: on the B70 a
// 512-token window's attention was 36M clock ticks a sliding layer and 121M a full one at ~4.7K cells (a third and two
// thirds of the layer).  This kernel follows glm5_mla_xmx.cpp's scheme:
//   * a work-group of 4 sub-groups takes 16 tokens of one head and walks the cells from the first token's window start
//     to the last token's position, 16 at a time, with an online softmax per token; sub-groups 2g and 2g + 1 own dims
//     [64g, 64g + 64), rows 0-7 and 8-15;
//   * the chunk's K and V rows (FP32 in the cache) go into local memory as FP16 hi + lo, q (scaled by a power of two
//     that puts its largest value near 2^14) and p likewise: three products each (hi.hi + lo.hi + hi.lo), so they keep
//     about 22 bits;
//   * the relative bias r . table[pos - j] (pos - j < extent), the causal and sliding-window masks and the sink (a
//     logit with a zero value, in the denominator) are applied with the scores in FP32.
// The attention part of a layer then took 22M ticks (sliding) and 48M (full); against the plain kernel the outputs
// differ by at most ~2e-6 of their largest value (tests/core/inkling2_attn_xmx_test.cpp).
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
constexpr int R = 16;              // tokens (rows of q, p and the output) a work-group
constexpr int NG = 2;              // 64-dim groups
constexpr int SGS = 16;            // sub-group size
constexpr int NSG = 2 * NG;        // sub-groups: a group's dims x a tile of 8 rows
constexpr int WG = NSG * SGS;      // work-items
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

struct AttXmx {
    const float* q;       // [T][H][D]
    const float* r;       // [T][H][rel]
    const float* sinks;   // [T][H]
    const float* kc;      // [cells][nkv][D]
    const float* vc;
    const float* table;   // [extent][rel]
    const int32_t* steps;
    int T, H, G, rel, extent, window;
    float* out;
    sycl::local_accessor<half, 1> kv;      // kh, kl, vh, vl [CH][RS]; at the start qh, ql [R][RS]
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
        const int nblk = (T + R - 1) / R;
        const int wgi = (int) it.get_group(0), hh = wgi / nblk, t0 = (wgi % nblk) * R;
        const int kvh = hh / G;
        const int t = (int) it.get_local_id(0), s = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int g = s / 2, mt = s % 2;
        const int64_t width = (int64_t) (H / G) * D;
        const size_t QH = 0, QL = (size_t) R * RS;
        const size_t KH = 0, KL = (size_t) CH * RS, VH = 2 * (size_t) CH * RS, VL = 3 * (size_t) CH * RS;

        // q: 16 tokens, times 2^(14 - e) for its largest |value| < 2^e, as FP16 hi + lo; the rows' r and positions
        float qm = 0.0f;
        for (int i = t; i < R * D; i += WG) {
            const int rr = i / D, d = i % D;
            if (t0 + rr < T) qm = sycl::fmax(qm, sycl::fabs(q[((int64_t) (t0 + rr) * H + hh) * D + d]));
        }
        qm = sycl::reduce_over_group(grp, qm, sycl::maximum<float>());
        const int qe = qm > 0.0f ? (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFF) - 126 : 0;
        const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(LOG2E / sycl::sqrt((float) D), qe - 14);
        for (int i = t; i < R * D; i += WG) {
            const int rr = i / D, d = i % D;
            half hi(0.0f), lo(0.0f);
            if (t0 + rr < T) split(q[((int64_t) (t0 + rr) * H + hh) * D + d] * qup, hi, lo);
            kv[QH + (size_t) rr * RS + d] = hi;
            kv[QL + (size_t) rr * RS + d] = lo;
        }
        for (int i = t; i < R * RELMAX; i += WG) {
            const int rr = i / RELMAX, k = i % RELMAX;
            rv[rr][k] = t0 + rr < T && k < rel ? r[((int64_t) (t0 + rr) * H + hh) * rel + k] : 0.0f;
        }
        if (t < R) {
            const bool live = t0 + t < T;
            rpos[t] = live ? steps[(int64_t) (t0 + t) * kStepCount + kStepPos] : -1;
            mrow[t] = live ? sinks[(int64_t) (t0 + t) * H + hh] * LOG2E : 0.0f;   // the sink's logit
            lsum[t] = 1.0f;                                                      // its term at m = itself
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
                if (t0 + rr < T)
                    out[((int64_t) (t0 + rr) * H + hh) * D + g * 64 + nt * TN + (int) col] = x / (PSCALE * lsum[rr]);
            });
#endif
    }
    auto get(syclex::properties_tag) const {
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<SGS>};
    }
};

}  // namespace

bool inkling2_attention_xmx(const float* q, const float* r, const float* sinks, const float* kc, const float* vc,
                            const float* table, const int32_t* steps, int n_head, int n_head_kv, int dim, int rel,
                            int extent, int window, float* out, int n_tokens, void* stream) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_INK_ATT_XMX");
        return v == nullptr || v[0] != '0';
    }();
    if (!on || n_tokens <= 0 || dim != D || rel > RELMAX || n_head_kv <= 0 || n_head % n_head_kv != 0 ||
        !xmx_available(XmxType::f16))
        return false;
    auto& queue = core::Runtime::get().stream(stream);
    const size_t need = (size_t) 4 * CH * RS * 2 + (size_t) NG * R * CH * 4 + (size_t) 2 * R * CH * 2 + (size_t) 3 * R * 4 +
                        (size_t) R * RELMAX * 4 + (size_t) R * 4;
    if (queue.get_device().get_info<sycl::info::device::local_mem_size>() < need || 2 * R * RS > 4 * CH * RS) return false;
    const int nblk = (n_tokens + R - 1) / R;
    const auto e = queue.submit([&](sycl::handler& h) {
        AttXmx k{q, r, sinks, kc, vc, table, steps, n_tokens, n_head, n_head / n_head_kv, rel, extent, window, out,
                 sycl::local_accessor<half, 1>(sycl::range<1>((size_t) 4 * CH * RS), h),
                 sycl::local_accessor<float, 3>(sycl::range<3>(NG, R, CH), h),
                 sycl::local_accessor<half, 2>(sycl::range<2>(2, (size_t) R * CH), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
                 sycl::local_accessor<float, 2>(sycl::range<2>(R, RELMAX), h),
                 sycl::local_accessor<int, 1>(sycl::range<1>(R), h)};
        h.parallel_for(sycl::nd_range<1>((size_t) nblk * n_head * WG, WG), k);
    });
    if (!stream) core::Runtime::get().wait(e, "inkling2_attention_xmx");
    return true;
}

}  // namespace strata::kernels
