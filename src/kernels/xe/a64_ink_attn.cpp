// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// GB10 local port (aarch64 + CUDA, 11-21): Inkling 2's attention for a prompt window on NVIDIA, in place of the
// XMX kernel (inkling2_attn_xmx.cpp, Intel only).  inkling2_attention (inkling2.cpp) gives every (token, head) its own
// work-group that reads every key row from global memory: 58 ms a layer a 512-token window on GB10 (42% of the prompt).
// Here a work-group takes the G heads that share one KV head for TT tokens (TT * G rows, two work-items a row: dims
// [64 h, +64)) and walks the cells from the tile's lowest window start to its last position, CK at a time: the chunk's
// K and V rows go into local memory once for all the rows, a row's CK scores stay in registers (its two halves summed
// with one exchange), then one online-softmax step a chunk.  FP32 throughout; the sink starts the softmax (a logit with
// a zero value), as in inkling2_attention.  STRATA_INK_ATT_GB10=0: the plain kernel.
#include "strata/kernels/inkling2.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <limits>

namespace strata::kernels {
namespace {
constexpr int D = 128, HD = 64, CK = 16, TT = 16, SG = 32;
}  // namespace

bool inkling2_attention_gb10(const float* q, const float* r, const float* sinks, const float* kc, const float* vc,
                             const float* table, const int32_t* steps, int n_head, int n_head_kv, int dim, int rel,
                             int extent, int window, float* out, int n_tokens, void* stream) {
    static const bool on = [] { const char* v = std::getenv("STRATA_INK_ATT_GB10"); return v == nullptr || v[0] != '0'; }();
    if (!on || n_tokens <= 0 || dim != D || rel > 32 || rel % 2 != 0 || n_head_kv <= 0 || n_head % n_head_kv != 0)
        return false;
    const int G = n_head / n_head_kv;
    const int rows = TT * G;
    const int wg = ((2 * rows + SG - 1) / SG) * SG;
    if (wg > 1024) return false;
    const int ntile = (n_tokens + TT - 1) / TT;
    const float qs = 1.0f / sycl::sqrt((float) D);
    const int64_t width = (int64_t) n_head_kv * D;
    const int RH = rel / 2;
    auto& queue = core::Runtime::get().stream(stream);
    const sycl::event e = queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::float4, 1> sk(sycl::range<1>(CK * D / 4), h);
        sycl::local_accessor<sycl::float4, 1> sv(sycl::range<1>(CK * D / 4), h);
        sycl::local_accessor<int64_t, 1> srange(sycl::range<1>(2), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_head_kv * ntile * wg, (size_t) wg),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            constexpr float NEG = -std::numeric_limits<float>::infinity();
            const auto sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0);
            const int grp = (int) it.get_group(0), kvh = grp / ntile, t0 = (grp % ntile) * TT;
            const int row = tid >> 1, hf = tid & 1;
            const bool live = row < rows && t0 + row / G < n_tokens;
            const int t = live ? t0 + row / G : t0, hh = kvh * G + (live ? row % G : 0);
            const int64_t pos = live ? (int64_t) steps[(int64_t) t * kStepCount + kStepPos] : -1;
            const int64_t lo = (live && window > 0) ? sycl::max((int64_t) 0, pos - window + 1) : 0;
            if (tid == 0) {   // the tile's cells: the lowest window start to the last position
                int64_t a = std::numeric_limits<int64_t>::max(), b = -1;
                for (int i = 0; i < TT && t0 + i < n_tokens; ++i) {
                    const int64_t p = steps[(int64_t) (t0 + i) * kStepCount + kStepPos];
                    const int64_t l0 = window > 0 ? sycl::max((int64_t) 0, p - window + 1) : 0;
                    a = sycl::min(a, l0);
                    b = sycl::max(b, p);
                }
                srange[0] = a;
                srange[1] = b;
            }
            float qv[HD], acc[HD], rv[16];
            const float* qp = q + ((int64_t) t * n_head + hh) * D + hf * HD;
#pragma unroll
            for (int d = 0; d < HD; ++d) {
                qv[d] = live ? qp[d] * qs : 0.0f;
                acc[d] = 0.0f;
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) rv[i] = (live && i < RH) ? r[((int64_t) t * n_head + hh) * rel + hf * RH + i] : 0.0f;
            float m = live ? sinks[(int64_t) t * n_head + hh] : 0.0f, l = 1.0f;
            sycl::group_barrier(it.get_group());
            const int64_t c_lo = srange[0], c_hi = srange[1];
            for (int64_t c0 = c_lo; c0 <= c_hi; c0 += CK) {
                const int nc = (int) sycl::min<int64_t>(CK, c_hi + 1 - c0);
                for (int i = tid; i < CK * D / 4; i += wg) {
                    const int jj = i / (D / 4), dd = i % (D / 4);
                    if (jj < nc) {
                        const int64_t off = (c0 + jj) * width + (int64_t) kvh * D + dd * 4;
                        sk[i] = *reinterpret_cast<const sycl::float4*>(kc + off);
                        sv[i] = *reinterpret_cast<const sycl::float4*>(vc + off);
                    } else {
                        sk[i] = sycl::float4(0.0f);
                        sv[i] = sycl::float4(0.0f);
                    }
                }
                sycl::group_barrier(it.get_group());
                float s[CK];
                float cm = NEG;
#pragma unroll
                for (int jj = 0; jj < CK; ++jj) {
                    const sycl::float4* kr = &sk[jj * (D / 4) + hf * (HD / 4)];
                    float p = 0.0f;
#pragma unroll
                    for (int d4 = 0; d4 < HD / 4; ++d4) {
                        const sycl::float4 k4 = kr[d4];
                        p += qv[4 * d4] * k4.x() + qv[4 * d4 + 1] * k4.y() + qv[4 * d4 + 2] * k4.z() + qv[4 * d4 + 3] * k4.w();
                    }
                    const int64_t j = c0 + jj, dist = pos - j;
                    if (live && jj < nc && dist >= 0 && dist < extent) {
                        const float* tb = table + dist * rel + hf * RH;
                        for (int i = 0; i < RH; ++i) p += rv[i] * tb[i];
                    }
                    p += sycl::permute_group_by_xor(sg, p, 1);
                    const bool ok = live && jj < nc && j >= lo && j <= pos;
                    s[jj] = ok ? p : NEG;
                    cm = sycl::fmax(cm, s[jj]);
                }
                const float nm = sycl::fmax(m, cm), a = sycl::exp(m - nm);
                l *= a;
#pragma unroll
                for (int d = 0; d < HD; ++d) acc[d] *= a;
#pragma unroll
                for (int jj = 0; jj < CK; ++jj) {
                    const float pj = s[jj] == NEG ? 0.0f : sycl::exp(s[jj] - nm);
                    l += pj;
                    const sycl::float4* vr = &sv[jj * (D / 4) + hf * (HD / 4)];
#pragma unroll
                    for (int d4 = 0; d4 < HD / 4; ++d4) {
                        const sycl::float4 v4 = vr[d4];
                        acc[4 * d4] += pj * v4.x();
                        acc[4 * d4 + 1] += pj * v4.y();
                        acc[4 * d4 + 2] += pj * v4.z();
                        acc[4 * d4 + 3] += pj * v4.w();
                    }
                }
                m = nm;
                sycl::group_barrier(it.get_group());
            }
            if (live) {
                float* op = out + ((int64_t) t * n_head + hh) * D + hf * HD;
                const float il = 1.0f / l;
#pragma unroll
                for (int d = 0; d < HD; ++d) op[d] = acc[d] * il;
            }
        });
    });
    if (!stream) core::Runtime::get().wait(e, "inkling2_attention_gb10");
    return true;
}

}  // namespace strata::kernels
