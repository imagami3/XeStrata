// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/glm5_mla_xmx.cpp - GLM-5.3-Flash's MLA attention over the selection, for a prompt, on Xe's matrix
// engines (b70-local, 2026-10-27).
//
// glm5_mla_attn gives every (token, head) a work-group of scalar loops; on the B70 a 3,024-token prompt took ~850 ms a
// DSA block there (about a fifth of the whole prompt).  The heads of a token share its selected cells' 512-wide
// latent rows (the MLA's K and V are the same row), so this kernel follows qsa_prompt_attn.cpp's scheme with K = V:
//   * a work-group of 16 sub-groups takes 16 heads of one token and walks its cells 16 at a time with an online
//     softmax; sub-groups 2g and 2g + 1 own dims [64g, 64g + 64) (g < 8), rows 0-7 and 8-15;
//   * the chunk's latent rows are gathered once into local memory as FP16 (they are FP16 already) and serve both q.k
//     and p.v;
//   * q is scaled by a power of two that puts its largest value near 2^14 and split into FP16 hi + lo, and so is p:
//     two products each, so they keep about 22 bits;
//   * the eight groups' q.k partials are added in a fixed order;
//   * q and the output stay in registers as joint_matrix tiles.
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_glm5_mla_xmx
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

constexpr int HD = 512;            // the latent's width
constexpr int R = 16;              // heads (rows of q, p and the output) a work-group
constexpr int NG = 8;              // 64-dim groups
constexpr int SGS = 16;            // sub-group size
constexpr int NSG = 2 * NG;        // sub-groups: a group's dims x a tile of 8 rows
constexpr int WG = NSG * SGS;      // work-items
constexpr int CH = 16;             // cells a chunk: lane `lane` owns cell `lane` when p is split
constexpr int QS = HD + 8;         // row strides in halves (16-byte aligned rows)
constexpr int KR = HD + 8;
[[maybe_unused]] constexpr int TM = 8, TN = 16, TK = 16;
[[maybe_unused]] constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
[[maybe_unused]] constexpr float TAU = 8.0f;                     // lazy rescaling, as qsa_prompt_attn.cpp
[[maybe_unused]] constexpr float PSCALE = 16384.0f / 256.0f;     // 2^(14 - TAU): p' = p * PSCALE stays within FP16
constexpr int LAT_HALVES = 2 * R * QS > CH * KR ? 2 * R * QS : CH * KR;   // q's staging shares the rows' space

template<typename A>
inline auto at(const A& a, size_t off) {
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

struct MlaXmx {
    const float* q;          // [T][H][HD]
    const uint16_t* latent;  // [cells][HD] FP16
    const int32_t* ids;      // [T][cap]
    const int32_t* steps;
    int64_t cap;
    int H, KP;
    float scale_log2;
    float* out;
    sycl::local_accessor<half, 1> lat;     // rows [CH][KR]; at the start qh [R][QS], ql [R][QS]
    sycl::local_accessor<long long, 1> rows;
    sycl::local_accessor<float, 3> part;   // [NG][R][CH]; part[0] then holds p
    sycl::local_accessor<half, 2> pp;      // [2][R * CH]: p' hi, lo
    sycl::local_accessor<float, 1> mrow, lsum, alpha;

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
        const int64_t wgi = (int64_t) it.get_group(0), tok = wgi / (H / R);
        const int h0 = (int) (wgi % (H / R)) * R;
        const int t = (int) it.get_local_id(0), s = (int) sg.get_group_linear_id();
        const int g = s / 2, mt = s % 2;
        const float* qp = q + ((int64_t) tok * H + h0) * HD;
        float* op = out + ((int64_t) tok * H + h0) * HD;
        const int32_t* sel = ids + tok * cap;
        const int64_t chosen = sycl::min<int64_t>(steps[tok * kStepCount + kStepWidth], cap);
        const int32_t pos = steps[tok * kStepCount + kStepPos];
        const int64_t tail0 = KP > 0 && pos >= 0 ? (int64_t) ((pos + 1) / KP) * KP : 0;
        const int64_t width = chosen + (KP > 0 && pos >= 0 ? (int64_t) pos + 1 - tail0 : 0);
        if (width <= 0) {   // uniform over the work-group
            for (int i = t; i < R * HD; i += WG) op[i] = 0.0f;
            return;
        }
        const size_t QH = 0, QL = (size_t) R * QS;

        // q: 16 heads, times 2^(14 - e) for its largest |value| < 2^e, as FP16 hi + lo
        float qm = 0.0f;
        for (int i = t; i < R * HD; i += WG) qm = sycl::fmax(qm, sycl::fabs(qp[i]));
        qm = sycl::reduce_over_group(grp, qm, sycl::maximum<float>());
        const int qe = qm > 0.0f ? (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFF) - 126 : 0;
        const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
        for (int i = t; i < R * HD; i += WG) {
            const int r = i / HD, d = i % HD;
            half hi, lo;
            split(qp[i] * qup, hi, lo);
            lat[QH + (size_t) r * QS + d] = hi;
            lat[QL + (size_t) r * QS + d] = lo;
        }
        if (t < R) { mrow[t] = NEG_INF; lsum[t] = 0.0f; }
        sycl::group_barrier(grp);
        mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> qa[4], qb[4];
        for (int kk = 0; kk < 4; ++kk) {
            const size_t off = (size_t) mt * TM * QS + (size_t) g * 64 + (size_t) kk * TK;
            mx::joint_matrix_load(sg, qa[kk], at(lat, QH + off), QS);
            mx::joint_matrix_load(sg, qb[kk], at(lat, QL + off), QS);
        }
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[4];
        for (int nt = 0; nt < 4; ++nt) mx::joint_matrix_fill(sg, acc[nt], 0.0f);

        for (int64_t c0 = 0; c0 < width; c0 += CH) {
            const int nh = (int) sycl::min<int64_t>(CH, width - c0);
            if (t < CH) {
                long long r = -1;
                if (t < nh) {
                    const int64_t j = c0 + t;
                    r = j < chosen ? (long long) sel[j] : (long long) (tail0 + (j - chosen));
                }
                rows[t] = r;
            }
            sycl::group_barrier(grp);   // rows ready; the previous chunk (and q's staging) is done with lat, part, pp
            for (int i = t; i < CH * (HD / 8); i += WG) {   // the rows as FP16, 8 values a piece
                const int c = i / (HD / 8), pc = i % (HD / 8);
                const long long r = rows[c];
                sycl::vec<half, 8> x(half(0.0f));
                if (r >= 0) x = *reinterpret_cast<const sycl::vec<half, 8>*>(latent + r * HD + (int64_t) pc * 8);
                *reinterpret_cast<sycl::vec<half, 8>*>(&lat[(size_t) c * KR + (size_t) pc * 8]) = x;
            }
            sycl::group_barrier(grp);

            {   // q.k of this sub-group's 8 rows and 64 dims over the chunk's 16 cells
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c;
                mx::joint_matrix_fill(sg, c, 0.0f);
                for (int kk = 0; kk < 4; ++kk) {
                    mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> b;
                    mx::joint_matrix_load(sg, b, at(lat, (size_t) g * 64 + (size_t) kk * TK), KR);
                    mx::joint_matrix_mad(sg, c, qa[kk], b, c);
                    mx::joint_matrix_mad(sg, c, qb[kk], b, c);
                }
                mx::joint_matrix_store(sg, c, at(part, (size_t) (g * R + mt * TM) * CH), CH, mx::layout::row_major);
            }
            sycl::group_barrier(grp);

            // online softmax: row t / 16, one cell a work-item; the groups' partials in order
            {
                const int r = t / CH, c = t % CH;
                float sc = 0.0f;
                for (int gg = 0; gg < NG; ++gg) sc += part[gg][r][c];
                const float x = c < nh && rows[c] >= 0 ? sc * qdown : NEG_INF;
                float mxv = x;
                for (int o = 1; o < CH; o <<= 1) mxv = sycl::fmax(mxv, sycl::permute_group_by_xor(sg, mxv, o));
                const float m_old = mrow[r];
                const float m_new = m_old == NEG_INF || mxv > m_old + TAU ? mxv : m_old;
                const float e = x == NEG_INF ? 0.0f : sycl::exp2(x - m_new);
                float sum = e;
                for (int o = 1; o < CH; o <<= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
                sycl::group_barrier(sg);   // every lane has read mrow[r] before lane 0 moves it
                half hi, lo;
                split(e * PSCALE, hi, lo);
                pp[0][(size_t) r * CH + c] = hi;
                pp[1][(size_t) r * CH + c] = lo;
                if (c == 0) {
                    const float a = m_old == m_new ? 1.0f : m_old == NEG_INF ? 0.0f : sycl::exp2(m_old - m_new);
                    alpha[r] = a;
                    lsum[r] = sycl::fma(lsum[r], a, sum);
                    mrow[r] = m_new;
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
                    mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::row_major> b;
                    mx::joint_matrix_load(sg, b, at(lat, (size_t) g * 64 + (size_t) nt * TN), KR);
                    mx::joint_matrix_mad(sg, acc[nt], ph, b, acc[nt]);
                    mx::joint_matrix_mad(sg, acc[nt], pl, b, acc[nt]);
                }
            }
        }
        sycl::group_barrier(grp);
        for (int nt = 0; nt < 4; ++nt)
            ix::joint_matrix_apply(sg, acc[nt], [&](float& x, size_t row, size_t col) {
                const int r = mt * TM + (int) row;
                const float l = lsum[r];
                op[(int64_t) r * HD + g * 64 + nt * TN + (int) col] = l > 0.0f ? x / (PSCALE * l) : 0.0f;
            });
#endif
    }
    auto get(syclex::properties_tag) const {
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<SGS>};
    }
};

}  // namespace

bool glm5_mla_attn_xmx(const float* q, const uint16_t* latent, const int32_t* ids, int64_t cap, const int32_t* steps,
                       int n_heads, float scale, int64_t n_tokens, float* out, void* stream, int tail_kpool) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MLA_XMX");
        return v == nullptr || v[0] != '0';
    }();
    if (!on || n_tokens <= 0 || n_heads % R != 0 || !xmx_available(XmxType::f16)) return false;
    auto& queue = core::Runtime::get().stream(stream);
    const size_t need = (size_t) LAT_HALVES * 2 + (size_t) CH * 8 + (size_t) NG * R * CH * 4 + (size_t) 2 * R * CH * 2 +
                        (size_t) 3 * R * 4;
    if (queue.get_device().get_info<sycl::info::device::local_mem_size>() < need) return false;
    const float scale_log2 = 1.4426950408889634f * scale;
    const auto e = queue.submit([&](sycl::handler& h) {
        MlaXmx k{q, latent, ids, steps, cap, n_heads, tail_kpool, scale_log2, out,
                 sycl::local_accessor<half, 1>(sycl::range<1>((size_t) LAT_HALVES), h),
                 sycl::local_accessor<long long, 1>(sycl::range<1>(CH), h),
                 sycl::local_accessor<float, 3>(sycl::range<3>(NG, R, CH), h),
                 sycl::local_accessor<half, 2>(sycl::range<2>(2, (size_t) R * CH), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>(R), h)};
        h.parallel_for(sycl::nd_range<1>((size_t) (n_tokens * (n_heads / R)) * WG, WG), k);
    });
    if (!stream) core::Runtime::get().wait(e, "glm5_mla_attn_xmx");
    return true;
}

}  // namespace strata::kernels
