// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/glm5.cpp - see include/strata/kernels/glm5.hpp.
//
// Transcribed from llama.cpp f9a95ae (src/models/glm5-next.cpp; ggml/src/ggml-cpu/ops.cpp's dsv4_hc_comb,
// gated_delta_net and swiglu_clamp); MIT License, Copyright (c) 2023-2026 The ggml authors, see
// third_party/main/ggml/LICENSE.  The KDA step keeps the layout and the sub-group shape of native_gdn.cpp.
#include <cstdlib>
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"
#include "q8_1_finite.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_glm5
#include "strata/kernels/device_arch_tag.hpp"

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels {
namespace {

constexpr int HC = 4;                  // the mHC streams (model_spec refuses another count)
constexpr int MIX = (2 + HC) * HC;     // pre, post and the HC x HC comb
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }
void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}

inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

}  // namespace

void glm5_hc_pre(const float* x, const float* mixes, const float* scale, const float* base, float rms_eps, float hc_eps,
                 int iters, float* y, float* post, float* comb, int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (n_embd <= 0 || iters < 1) throw core::DeviceError("glm5_hc_pre: n_embd " + std::to_string(n_embd) + ", iters " +
                                                          std::to_string(iters));
    // b70-local: 1024 work-items a token, not 256 (a token's 4N values read 64 apart was most of the time at T = 3:
    // three work-groups on the whole card)
    constexpr int WG = 1024;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> pre_l(sycl::range<1>(HC), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t t = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const float* xt = x + t * HC * N;
            // the RMS of the token's 4N stream values
            float ss = 0.0f;
            for (int64_t i = lid; i < HC * N; i += WG) ss += xt[i] * xt[i];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            // b70-local: the 4 x 4 Sinkhorn on 16 lanes of the first sub-group, lane L holding c[L] (dst = L % 4,
            // src = L / 4); each sum gathers its four values in the serial loop's order, so every value is bitwise the
            // one-lane loop's (21.6 us -> a few, B70, T = 3)
            const sycl::sub_group sg = it.get_sub_group();
            if (sg.get_group_linear_id() == 0) {
                const int L = (int) sg.get_local_linear_id();
                const float inv = 1.0f / sycl::sqrt(ss / (float) (HC * N) + rms_eps);
                if (L < HC) {
                    pre_l[L] = sigmoid(mixes[t * MIX + L] * inv * scale[0] + base[L]) + hc_eps;
                    post[t * HC + L] = 2.0f * sigmoid(mixes[t * MIX + HC + L] * inv * scale[1] + base[HC + L]);
                }
                const int Lc = L < HC * HC ? L : 0, dst = Lc % HC, src = Lc / HC;
                float cv = (mixes[t * MIX + 2 * HC + Lc] * inv) * scale[2] + base[2 * HC + Lc];
                auto at = [&](float v, int idx) { return sycl::select_from_group(sg, v, idx); };
                {   // softmax over dst for this src
                    float mx = -INFINITY;
                    for (int d = 0; d < HC; ++d) mx = sycl::fmax(mx, at(cv, d + HC * src));
                    const float e = sycl::exp(cv - mx);
                    float sum = 0.0f;
                    for (int d = 0; d < HC; ++d) sum += at(e, d + HC * src);
                    const float inv_sum = 1.0f / sum;
                    cv = e * inv_sum + hc_eps;
                }
                auto norm_cols = [&]() {
                    float sum = hc_eps;
                    for (int s2 = 0; s2 < HC; ++s2) sum += at(cv, dst + HC * s2);
                    const float inv_sum = 1.0f / sum;
                    cv *= inv_sum;
                };
                auto norm_rows = [&]() {
                    float sum = hc_eps;
                    for (int d = 0; d < HC; ++d) sum += at(cv, d + HC * src);
                    const float inv_sum = 1.0f / sum;
                    cv *= inv_sum;
                };
                norm_cols();
                for (int i = 1; i < iters; ++i) {
                    norm_rows();
                    norm_cols();
                }
                if (L < HC * HC) comb[t * HC * HC + L] = cv;
            }
            sycl::group_barrier(it.get_group());
            const float p0 = pre_l[0], p1 = pre_l[1], p2 = pre_l[2], p3 = pre_l[3];
            for (int64_t n = lid; n < N; n += WG)   // ggml: x0 p0, then + x1 p1, + x2 p2, + x3 p3
                y[t * N + n] = ((xt[n] * p0 + xt[N + n] * p1) + xt[2 * N + n] * p2) + xt[3 * N + n] * p3;
        });
    });
    sync_if_needed(stream, event, "glm5_hc_pre");
}


// b70-local: the mHC read of a block in one launch a token (hc_read's four: the streams' q8_1, the 24 mixes' Q8_0
// products, glm5_hc_pre, the weighted RMS norm with its q8_1): 35 us -> one kernel at T = 3 (B70).  The mixes are
// FP32 products of the streams with the Q8_0 rows (not of the streams' q8_1), so the values differ from the four
// launches' by those roundings; the Sinkhorn, y and the norm are theirs.  fn: kNativeQ8Rows (a row's qs, then its d),
// [24][4N].  hn [T][N] and its q8_1 blocks (xq), post [T][4], comb [T][16].
void glm5_hc_read_fused(const float* hs, const void* fn_rows, const float* scale, const float* base, float rms_eps,
                        float hc_eps, int iters, const float* gamma, float norm_eps, float* hn, void* xq, float* post,
                        float* comb, int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    constexpr int WG = 1024, NW = WG / WARP, NV = MIX + 1;   // the 24 mixes and the sum of squares
    const int64_t N = n_embd, HN = HC * N;
    if (HN % WG != 0 || N % WG != 0 || HN / WG > 32) throw core::DeviceError("glm5_hc_read_fused: n_embd");
    const int bpr = (int) (HN / 32);
    const uint8_t* W = static_cast<const uint8_t*>(fn_rows);
    struct Q81b { sycl::half2 ds; int8_t qs[32]; };
    Q81b* blocks = static_cast<Q81b*>(xq);
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) NW * NV), h);
        sycl::local_accessor<float, 1> mixl(sycl::range<1>(NV), h);
        sycl::local_accessor<float, 1> pre_l(sycl::range<1>(HC), h);
        sycl::local_accessor<float, 1> sq(sycl::range<1>(NW), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int64_t t = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int w = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const float* xt = hs + t * HN;
            float acc[NV];
            #pragma unroll
            for (int v = 0; v < NV; ++v) acc[v] = 0.0f;
            for (int64_t i = lid; i < HN; i += WG) {
                const float x = xt[i];
                acc[MIX] += x * x;
                const int kb = (int) (i / 32);
                #pragma unroll
                for (int r = 0; r < MIX; ++r) {
                    const uint8_t* row = W + (size_t) r * (size_t) bpr * 34;
                    const float d = (float) *reinterpret_cast<const sycl::half*>(row + (size_t) bpr * 32 + (size_t) kb * 2);
                    acc[r] += x * (d * (float) (int8_t) row[i]);
                }
            }
            #pragma unroll
            for (int v = 0; v < NV; ++v) {
                const float sv = sycl::reduce_over_group(sg, acc[v], sycl::plus<float>());
                if (lane == 0) red[w * NV + v] = sv;
            }
            sycl::group_barrier(it.get_group());
            if (w == 0) {
                for (int v = 0; v < NV; ++v) {
                    const float sv = sycl::reduce_over_group(sg, lane < NW ? red[lane * NV + v] : 0.0f, sycl::plus<float>());
                    if (lane == 0) mixl[v] = sv;
                }
                sycl::group_barrier(sg);
                // glm5_hc_pre's pre / post / comb from the mixes (the Sinkhorn on 16 lanes, as there)
                const int L = lane;
                const float inv = 1.0f / sycl::sqrt(mixl[MIX] / (float) HN + rms_eps);
                if (L < HC) {
                    pre_l[L] = sigmoid(mixl[L] * inv * scale[0] + base[L]) + hc_eps;
                    post[t * HC + L] = 2.0f * sigmoid(mixl[HC + L] * inv * scale[1] + base[HC + L]);
                }
                const int Lc = L < HC * HC ? L : 0, dst = Lc % HC, src = Lc / HC;
                float cv = (mixl[2 * HC + Lc] * inv) * scale[2] + base[2 * HC + Lc];
                auto at = [&](float v, int idx) { return sycl::select_from_group(sg, v, idx); };
                {
                    float mx = -INFINITY;
                    for (int d = 0; d < HC; ++d) mx = sycl::fmax(mx, at(cv, d + HC * src));
                    const float e = sycl::exp(cv - mx);
                    float sum = 0.0f;
                    for (int d = 0; d < HC; ++d) sum += at(e, d + HC * src);
                    cv = e * (1.0f / sum) + hc_eps;
                }
                auto norm_cols = [&]() {
                    float sum = hc_eps;
                    for (int s2 = 0; s2 < HC; ++s2) sum += at(cv, dst + HC * s2);
                    cv *= 1.0f / sum;
                };
                auto norm_rows = [&]() {
                    float sum = hc_eps;
                    for (int d = 0; d < HC; ++d) sum += at(cv, d + HC * src);
                    cv *= 1.0f / sum;
                };
                norm_cols();
                for (int i = 1; i < iters; ++i) {
                    norm_rows();
                    norm_cols();
                }
                if (L < HC * HC) comb[t * HC * HC + L] = cv;
            }
            sycl::group_barrier(it.get_group());
            const float p0 = pre_l[0], p1 = pre_l[1], p2 = pre_l[2], p3 = pre_l[3];
            constexpr int PER = 8;   // N / WG at most (N = 4096: 4)
            float y[PER];
            float ys = 0.0f;
            const int per = (int) (N / WG);
            for (int k = 0; k < per; ++k) {
                const int64_t n = lid + (int64_t) k * WG;
                y[k] = ((xt[n] * p0 + xt[N + n] * p1) + xt[2 * N + n] * p2) + xt[3 * N + n] * p3;
                ys += y[k] * y[k];
            }
            ys = sycl::reduce_over_group(sg, ys, sycl::plus<float>());
            if (lane == 0) sq[w] = ys;
            sycl::group_barrier(it.get_group());
            const float tot = sycl::reduce_over_group(sg, lane < NW ? sq[lane] : 0.0f, sycl::plus<float>());
            const float sc = sycl::rsqrt(tot / (float) N + norm_eps);
            for (int k = 0; k < per; ++k) {
                const int64_t n = lid + (int64_t) k * WG;
                const float v = sc * y[k] * gamma[n];
                hn[t * N + n] = v;
                const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
                const float sum = sycl::reduce_over_group(sg, v, sycl::plus<float>());
                const float d = xe::q8_1_finite(amax / 127.0f);
                const int64_t bi = (t * N + n) / 32;
                blocks[bi].qs[(t * N + n) % 32] = xe::q8_1_quant(v, d, amax);
                if (lane == 0) blocks[bi].ds = sycl::half2(sycl::half(d), sycl::half(xe::q8_1_finite(sum)));
            }
        });
    });
    sync_if_needed(stream, event, "glm5_hc_read_fused");
}


// b70-local: the softmax probability of row 0's most likely token (the greedy draft's), written as a float to
// out (a mapped host word): the adaptive draft length's confidence.  One work-group over the n logits.
void glm5_top_prob(const float* logits, int n, float* out, void* stream) {
    constexpr int WG = 1024;
    const auto event = queue_for(stream).parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const int lid = (int) it.get_local_id(0);
        float mx = -INFINITY;
        for (int i = lid; i < n; i += WG) mx = sycl::fmax(mx, logits[i]);
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float sum = 0.0f;
        for (int i = lid; i < n; i += WG) sum += sycl::exp(logits[i] - mx);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        if (lid == 0) *out = 1.0f / sum;
    });
    sync_if_needed(stream, event, "glm5_top_prob");
}

void glm5_hc_init(const float* x, float* hs, int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0 || n_embd <= 0) return;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * HC * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / (HC * N), n = i % N;
        hs[i] = x[t * N + n];
    });
    sync_if_needed(stream, event, "glm5_hc_init");
}

void glm5_transpose_abd(const float* src, float* dst, int64_t a, int64_t b, int64_t d, void* stream) {
    if (a <= 0 || b <= 0 || d <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (a * b * d)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], ia = i / (b * d), ib = (i / d) % b, k = i % d;
        dst[(ib * a + ia) * d + k] = src[i];
    });
    sync_if_needed(stream, event, "glm5_transpose_abd");
}

void glm5_hc_post(const float* f, const float* residual, const float* post, const float* comb, float* out,
                  int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0 || n_embd <= 0) return;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / N, n = i % N;
        const float fv = f[i];
        float r[HC];
        for (int s = 0; s < HC; ++s) r[s] = residual[(t * HC + s) * N + n];
        for (int d = 0; d < HC; ++d) {
            float v = fv * post[t * HC + d];
            for (int s = 0; s < HC; ++s) v += r[s] * comb[(t * HC + s) * HC + d];
            out[(t * HC + d) * N + n] = v;
        }
    });
    sync_if_needed(stream, event, "glm5_hc_post");
}

void glm5_hc_mean(const float* x, float* y, int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0 || n_embd <= 0) return;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / N, n = i % N;
        // each stream's pointer one step of N from the last: written as xt[N], xt[2 * N], xt[3 * N], IGC 2.27 to 2.30
        // read xt[N], xt[8 * N], xt[9 * N] on Xe2 GPUs, a fault past the buffer (AGENTS.md, "Intel Graphics Compiler";
        // the runtime turns the pass off, this form compiles right either way)
        const float* x0 = x + t * HC * N + n;
        const float* x1 = x0 + N;
        const float* x2 = x1 + N;
        const float* x3 = x2 + N;
        y[i] = (((*x0 + *x1) + *x2) + *x3) * (1.0f / HC);
    });
    sync_if_needed(stream, event, "glm5_hc_mean");
}

void glm5_router_sigmoid(const float* logits, const float* bias, int n_tokens, int n_expert, int k, bool norm,
                         float scale, int32_t* ids, float* weights, void* stream) {
    if (n_tokens <= 0) return;
    if (n_expert <= 0 || k <= 0 || k > WARP || k > n_expert || n_expert > 16 * WARP)
        throw core::DeviceError("glm5_router_sigmoid: " + std::to_string(k) + " of " + std::to_string(n_expert) +
                                " experts (at most 32 picks)");
    // b70-local: STRATA_MOE_KEEP=k1 (< k) keeps only the first k1 picks with the weights they have among all k
    // (normalised over the k picks, as trained; Chen and Yao's k2 = trained k): the others get weight 0 (quality runs;
    // they are still computed). Unset or >= k: unchanged.
    static const int keep_env = [] { const char* v = std::getenv("STRATA_MOE_KEEP"); return v ? std::atoi(v) : 0; }();
    const int keep = keep_env > 0 && keep_env < k ? keep_env : k;
    // b70-local: a work-item an expert, each counting the experts ahead of it in the same order (larger biased score,
    // then lower index): rank i < k is pass i's pick of the kernel below, so the ids, their order and the weights are
    // the same (the logits bitwise).  One sub-group making k passes took ~19 us on the B70 (384 experts, k 8), this
    // one ~7 (STRATA_DECODE_PROFILE).  STRATA_ROUTER_RANK=0: the passes.
    static const bool rank_on = [] { const char* v = std::getenv("STRATA_ROUTER_RANK"); return v == nullptr || v[0] != '0'; }();
    auto& q = queue_for(stream);
    const size_t wg_r = (size_t) ((n_expert + WARP - 1) / WARP) * WARP;
    if (rank_on && wg_r <= q.get_device().get_info<sycl::info::device::max_work_group_size>()) {
        const auto ev = q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> sv(sycl::range<1>(wg_r), h);
            sycl::local_accessor<float, 1> pw(sycl::range<1>(WARP), h);
            h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * wg_r, wg_r), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const int t = (int) it.get_group(0), e = (int) it.get_local_id(0);
                const float* l = logits + (size_t) t * n_expert;
                const float me = e < n_expert ? sigmoid(l[e]) : 0.0f;
                const float x = e < n_expert ? me + bias[e] : -INFINITY;
                sv[e] = x;
                sycl::group_barrier(it.get_group());
                // every score into the sub-group's registers (lane i holds scores i, i + 32, ...), then compared by
                // shuffles: no memory read in the loop (a loop over local memory took ~17 us: a read's latency each)
                const sycl::sub_group sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id();
                constexpr int kPer = 16;
                float v[kPer];
                #pragma unroll
                for (int j = 0; j < kPer; ++j) v[j] = j * WARP + lane < n_expert ? sv[j * WARP + lane] : -INFINITY;
                int r = 0;
                #pragma unroll
                for (int j = 0; j < kPer; ++j) {
                    if (j * WARP >= n_expert) break;
                    #pragma unroll
                    for (int sl = 0; sl < WARP; ++sl) {
                        const float y = sycl::select_from_group(sg, v[j], sl);
                        const int jj = j * WARP + sl;
                        r += (y > x || (y == x && jj < e)) ? 1 : 0;
                    }
                }
                if (e < n_expert && r < k) {
                    ids[(size_t) t * k + r] = e;
                    pw[r] = me;
                }
                sycl::group_barrier(it.get_group());
                if (e < WARP) {   // the first sub-group: lane i has pick i, summed as the pass kernel sums them
                    const float mine = e < k ? pw[e] : 0.0f;
                    float div = 1.0f;
                    if (norm) div = sycl::fmax(sycl::reduce_over_group(sg, mine, sycl::plus<float>()), 6.103515625e-05f);
                    if (e < k) weights[(size_t) t * k + e] = e < keep ? mine / div * scale : 0.0f;
                }
            });
        });
        sync_if_needed(stream, ev, "glm5_router_sigmoid");
        return;
    }
    const auto event = q.parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * WARP, WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_group(0), lane = (int) sg.get_local_linear_id();
            const float* l = logits + (size_t) t * n_expert;
            // b70-local: each lane's biased scores computed once (n_expert / WARP of them, up to kRouterPer), then k
            // passes over them in registers: pass i takes the best below pass i - 1's pick in the order (larger biased
            // score, then lower index), the same picks as before (26.7 -> a few us at 288 experts, B70)
            constexpr int kRouterPer = 16;
            float v[kRouterPer];
            #pragma unroll
            for (int j = 0; j < kRouterPer; ++j) {
                const int e = lane + j * WARP;
                v[j] = e < n_expert ? sigmoid(l[e]) + bias[e] : -INFINITY;
            }
            float pv = INFINITY, mine = 0.0f;
            int pi = -1;
            for (int i = 0; i < k; ++i) {
                float bv = -INFINITY;
                int bi = n_expert;
                #pragma unroll
                for (int j = 0; j < kRouterPer; ++j) {
                    const int e = lane + j * WARP;
                    if (e >= n_expert) break;
                    const float x = v[j];
                    if ((x < pv || (x == pv && e > pi)) && (x > bv || (x == bv && e < bi))) { bv = x; bi = e; }
                }
                for (int off = WARP / 2; off > 0; off >>= 1) {
                    const float ov = sycl::permute_group_by_xor(sg, bv, off);
                    const int oi = sycl::permute_group_by_xor(sg, bi, off);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                if (lane == i) {
                    mine = sigmoid(l[bi]);
                    ids[(size_t) t * k + i] = bi;
                }
                pv = bv;
                pi = bi;
            }
            float div = 1.0f;
            if (norm) {
                const float sum = sycl::reduce_over_group(sg, lane < k ? mine : 0.0f, sycl::plus<float>());
                div = sycl::fmax(sum, 6.103515625e-05f);   // 2**-14, as llama.cpp
            }
            if (lane < k) weights[(size_t) t * k + lane] = lane < keep ? mine / div * scale : 0.0f;
        });
    sync_if_needed(stream, event, "glm5_router_sigmoid");
}

void glm5_kda_gate(const float* f, const float* dt_bias, const float* ssm_a, float lb, float* g, int64_t n_heads,
                   int64_t head_dim, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (n_heads <= 0 || head_dim <= 0 || !(lb < 0.0f) || !std::isfinite(lb))
        throw core::DeviceError("glm5_kda_gate: a finite negative lower bound and positive head shapes");
    const int64_t C = n_heads * head_dim;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], c = i % C, h = c / head_dim;
        const float x = (f[i] + dt_bias[c]) * ssm_a[h];
        g[i] = lb * sigmoid(-x);
    });
    sync_if_needed(stream, event, "glm5_kda_gate");
}

void glm5_kda_conv(const float* x, float* state, const float* w, float* y, int64_t n_channels, int64_t n_tokens,
                   bool update, void* stream) {
    constexpr int K = 4;
    if (n_tokens <= 0 || n_channels <= 0) return;
    const int64_t C = n_channels, T = n_tokens;
    // a work-item a channel, the tokens in order (the window is short: one, or the drafts)
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = (int64_t) id[0];
        float win[K];
        for (int j = 0; j < K - 1; ++j) win[j] = state[c * (K - 1) + j];
        const float* wc = w + c * K;
        for (int64_t t = 0; t < T; ++t) {
            win[K - 1] = x[t * C + c];
            float sum = 0.0f;
            for (int j = 0; j < K; ++j) sum += win[j] * wc[j];
            y[t * C + c] = sum / (1.0f + sycl::exp(-sum));
            for (int j = 0; j < K - 1; ++j) win[j] = win[j + 1];
        }
        if (update)
            for (int j = 0; j < K - 1; ++j) state[c * (K - 1) + j] = win[j];
    });
    sync_if_needed(stream, event, "glm5_kda_conv");
}

void glm5_l2_heads(float* x, int64_t n_heads, float eps, int64_t n_tokens, void* stream) {
    constexpr int S = 128;
    if (n_tokens <= 0 || n_heads <= 0) return;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) (n_tokens * n_heads) * S, S), [=](sycl::nd_item<1> it) {
            float* v = x + (int64_t) it.get_group(0) * S;
            const int i = (int) it.get_local_id(0);
            const float xi = v[i];
            const float sum = sycl::reduce_over_group(it.get_group(), xi * xi, sycl::plus<float>());
            v[i] = xi * (1.0f / sycl::fmax(sycl::sqrt(sum), eps));
        });
    sync_if_needed(stream, event, "glm5_l2_heads");
}

void glm5_kda_out(const float* o, const float* g2, const float* w, float eps, float* y, int64_t n_heads,
                  int64_t n_tokens, void* stream) {
    constexpr int S = 128;
    if (n_tokens <= 0 || n_heads <= 0) return;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) (n_tokens * n_heads) * S, S), [=](sycl::nd_item<1> it) {
            const int64_t base = (int64_t) it.get_group(0) * S;
            const int i = (int) it.get_local_id(0);
            const float oi = o[base + i];
            const float ss = sycl::reduce_over_group(it.get_group(), oi * oi, sycl::plus<float>());
            const float inv = 1.0f / sycl::sqrt(ss / (float) S + eps);
            y[base + i] = oi * inv * w[i] * sigmoid(g2[base + i]);
        });
    sync_if_needed(stream, event, "glm5_kda_out");
}

void glm5_kda_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* out, int64_t n_heads, void* stream) {
    constexpr int S = 128;
    constexpr int COLS = 4;   // columns (sub-groups) a work-group
    if (n_heads <= 0 || n_heads > 65535) throw core::DeviceError("glm5_kda_step: " + std::to_string(n_heads) + " heads");
    const int H = (int) n_heads;
    const float scale = 1.0f / sycl::sqrt((float) S);
    // one sub-group a (head, column): its four state rows a lane in registers, as native_gdn_step
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<2>({(size_t) H, (size_t) S * WARP}, {1, (size_t) COLS * WARP}),
        [=, cols = COLS](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = (int) it.get_group(0);
            const int lane = (int) sg.get_local_linear_id();
            const int col = (int) it.get_group(1) * cols + (int) sg.get_group_linear_id();
            float s[4], kr[4], qr[4], dr[4];
            float kv = 0.0f;
            for (int r = 0; r < 4; ++r) {
                const int i = r * WARP + lane;
                dr[r] = sycl::exp(g[head * S + i]);
                s[r] = dr[r] * state[((size_t) i * H + head) * S + col];   // the decay first, as ggml
                kr[r] = k[head * S + i];
                qr[r] = q[head * S + i];
                kv += s[r] * kr[r];
            }
            kv = sycl::reduce_over_group(sg, kv, sycl::plus<float>());
            const float delta = (v[head * S + col] - kv) * beta[head];
            float o = 0.0f;
            for (int r = 0; r < 4; ++r) {
                s[r] += kr[r] * delta;
                o += s[r] * qr[r];
            }
            o = sycl::reduce_over_group(sg, o, sycl::plus<float>());
            if (lane == 0) out[head * S + col] = o * scale;
            for (int r = 0; r < 4; ++r) state[((size_t) (r * WARP + lane) * H + head) * S + col] = s[r];
        });
    sync_if_needed(stream, event, "glm5_kda_step");
}

void glm5_kda_scan(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                   float* out, int64_t n_heads, int64_t n_tokens, void* stream, float* const* snaps, int n_snaps) {
    constexpr int S = 128;
    // columns (sub-groups) a work-group.  b70-local: 16, not 4 - a lane's state rows lie H S floats apart, so a
    // sub-group's load touches 32 lines and uses 4 bytes of each; 16 neighbouring columns a work-group use them whole
    // (each column's arithmetic unchanged)
    static const int COLS = [] { const char* e = std::getenv("STRATA_KDA_SCAN_COLS"); return e ? std::atoi(e) : 16; }();
    if (COLS < 1 || S % COLS != 0 || COLS * WARP > 1024) throw core::DeviceError("glm5_kda_scan: STRATA_KDA_SCAN_COLS");
    if (n_heads <= 0 || n_heads > 65535) throw core::DeviceError("glm5_kda_scan: " + std::to_string(n_heads) + " heads");
    if (n_snaps < 0 || n_snaps > kGlm5MaxSnaps) throw core::DeviceError("glm5_kda_scan: too many snapshots");
    if (n_tokens <= 0) return;
    struct Snaps { float* p[kGlm5MaxSnaps]; } sn{};
    for (int i = 0; i < n_snaps; ++i) sn.p[i] = snaps[i];
    const int NS = (int) std::min<int64_t>(n_snaps, n_tokens);
    const int H = (int) n_heads;
    const int64_t T = n_tokens, TS = (int64_t) H * S;
    const float scale = 1.0f / sycl::sqrt((float) S);
    // b70-local: STRATA_KDA_SCAN_LDS (default on) - the work-group's 16 columns of the state through local memory: the
    // loads and stores (state, snapshots) go 16 columns side by side (64 contiguous bytes a key row) instead of a lane
    // a key row H S floats apart (32 lines a load).  Each column's arithmetic is the one below's, bit for bit.
    static const bool lds = [] { const char* e = std::getenv("STRATA_KDA_SCAN_LDS"); return e == nullptr || e[0] != '0'; }();
    if (lds) {
        constexpr int C16 = 16, PAD = C16 + 1, WG = C16 * WARP;
        const auto ev = queue_for(stream).submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> tile(sycl::range<1>((size_t) S * PAD), h);
            h.parallel_for(sycl::nd_range<2>({(size_t) H, (size_t) (S / C16) * WG}, {1, (size_t) WG}),
                           [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int head = (int) it.get_group(0);
                const int lane = (int) sg.get_local_linear_id();
                const int w = (int) sg.get_group_linear_id();
                const int c0 = (int) it.get_group(1) * C16, col = c0 + w;
                const int tid = (int) it.get_local_id(1);
                auto gidx = [&](int i, int c) { return ((size_t) i * H + head) * S + c0 + c; };
                // the tile's key rows: 16 columns a row, contiguous in memory
                for (int e = tid; e < S * C16; e += WG) tile[(e / C16) * PAD + e % C16] = state[gidx(e / C16, e % C16)];
                sycl::group_barrier(it.get_group());
                float s[4];
                for (int r = 0; r < 4; ++r) s[r] = tile[(r * WARP + lane) * PAD + w];
                const int64_t hs = (int64_t) head * S;
                for (int64_t t = 0; t < T; ++t) {
                    const float* gt = g + t * TS + hs;
                    const float* kt = k + t * TS + hs;
                    const float* qt = q + t * TS + hs;
                    float kr[4], qr[4];
                    float kv = 0.0f;
                    for (int r = 0; r < 4; ++r) {
                        const int i = r * WARP + lane;
                        s[r] = sycl::exp(gt[i]) * s[r];
                        kr[r] = kt[i];
                        qr[r] = qt[i];
                        kv += s[r] * kr[r];
                    }
                    kv = sycl::reduce_over_group(sg, kv, sycl::plus<float>());
                    const float delta = (v[t * TS + hs + col] - kv) * beta[t * H + head];
                    float o = 0.0f;
                    for (int r = 0; r < 4; ++r) {
                        s[r] += kr[r] * delta;
                        o += s[r] * qr[r];
                    }
                    o = sycl::reduce_over_group(sg, o, sycl::plus<float>());
                    if (lane == 0) out[t * TS + hs + col] = o * scale;
                    if (t < NS) {   // the state after this row through the tile, written 16 columns side by side
                        sycl::group_barrier(it.get_group());
                        for (int r = 0; r < 4; ++r) tile[(r * WARP + lane) * PAD + w] = s[r];
                        sycl::group_barrier(it.get_group());
                        float* sp = sn.p[t];
                        for (int e = tid; e < S * C16; e += WG) sp[gidx(e / C16, e % C16)] = tile[(e / C16) * PAD + e % C16];
                    }
                }
                sycl::group_barrier(it.get_group());
                for (int r = 0; r < 4; ++r) tile[(r * WARP + lane) * PAD + w] = s[r];
                sycl::group_barrier(it.get_group());
                for (int e = tid; e < S * C16; e += WG) state[gidx(e / C16, e % C16)] = tile[(e / C16) * PAD + e % C16];
            });
        });
        sync_if_needed(stream, ev, "glm5_kda_scan");
        return;
    }
    // one sub-group a (head, column): its four state rows a lane in registers, as native_gdn_step
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<2>({(size_t) H, (size_t) S * WARP}, {1, (size_t) COLS * WARP}),
        [=, cols = COLS](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = (int) it.get_group(0);
            const int lane = (int) sg.get_local_linear_id();
            const int col = (int) it.get_group(1) * cols + (int) sg.get_group_linear_id();
            float s[4];
            for (int r = 0; r < 4; ++r) s[r] = state[((size_t) (r * WARP + lane) * H + head) * S + col];
            const int64_t hs = (int64_t) head * S;
            for (int64_t t = 0; t < T; ++t) {   // glm5_kda_step's body a token, the column's state kept in registers
                const float* gt = g + t * TS + hs;
                const float* kt = k + t * TS + hs;
                const float* qt = q + t * TS + hs;
                float kr[4], qr[4];
                float kv = 0.0f;
                for (int r = 0; r < 4; ++r) {
                    const int i = r * WARP + lane;
                    s[r] = sycl::exp(gt[i]) * s[r];   // the decay first, as ggml
                    kr[r] = kt[i];
                    qr[r] = qt[i];
                    kv += s[r] * kr[r];
                }
                kv = sycl::reduce_over_group(sg, kv, sycl::plus<float>());
                const float delta = (v[t * TS + hs + col] - kv) * beta[t * H + head];
                float o = 0.0f;
                for (int r = 0; r < 4; ++r) {
                    s[r] += kr[r] * delta;
                    o += s[r] * qr[r];
                }
                o = sycl::reduce_over_group(sg, o, sycl::plus<float>());
                if (lane == 0) out[t * TS + hs + col] = o * scale;
                if (t < NS)   // the state after this row, to go back to (a speculative window's rows but its last)
                    for (int r = 0; r < 4; ++r) sn.p[t][((size_t) (r * WARP + lane) * H + head) * S + col] = s[r];
            }
            for (int r = 0; r < 4; ++r) state[((size_t) (r * WARP + lane) * H + head) * S + col] = s[r];
        });
    sync_if_needed(stream, event, "glm5_kda_scan");
}

void glm5_idx_steps(const int32_t* steps_in, int64_t n_tokens, int kpool, int top_k, int32_t* steps_out, void* stream) {
    if (n_tokens <= 0) return;
    if (kpool <= 1 || top_k <= 0) throw core::DeviceError("glm5_idx_steps: kpool above 1 and a positive top_k");
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n_tokens), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0];
        const int32_t pos = steps_in[t * kStepCount + kStepPos];
        const int32_t nb = pos >= 0 ? (pos + 1) / kpool : 0;
        int32_t* o = steps_out + t * kStepCount;
        o[kStepPos] = pos;
        o[kStepNBid] = nb;
        o[kStepNKv] = nb * kpool;
        o[kStepWidth] = nb * kpool < top_k ? nb * kpool : top_k;
    });
    sync_if_needed(stream, event, "glm5_idx_steps");
}

void glm5_idx_append(const float* k_raw, const float* g_raw, const float* ln_w, const float* ln_b, float ln_eps,
                     const int32_t* steps, int64_t n_tokens, float* cache, void* stream) {
    constexpr int D = 128;
    if (n_tokens <= 0) return;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * D, D), [=](sycl::nd_item<1> it) {
            const int64_t t = (int64_t) it.get_group(0);
            const int c = (int) it.get_local_id(0);
            const int32_t pos = steps[t * kStepCount + kStepPos];
            const float x = k_raw[t * D + c];
            // ggml_norm: the mean, then the variance of the centred values
            const float mean = sycl::reduce_over_group(it.get_group(), x, sycl::plus<float>()) / (float) D;
            const float xc = x - mean;
            const float var = sycl::reduce_over_group(it.get_group(), xc * xc, sycl::plus<float>()) / (float) D;
            if (pos < 0) return;
            float* row = cache + (int64_t) pos * 2 * D;
            row[c] = xc * (1.0f / sycl::sqrt(var + ln_eps)) * ln_w[c] + ln_b[c];
            row[D + c] = g_raw[t * D + c];
        });
    sync_if_needed(stream, event, "glm5_idx_append");
}

void glm5_idx_pool(const float* cache, const float* ape, int kpool, const int32_t* steps, int64_t n_tokens,
                   float* pooled, void* stream) {
    constexpr int D = 128, KMAX = 8;
    if (n_tokens <= 0) return;
    if (kpool <= 1 || kpool > KMAX) throw core::DeviceError("glm5_idx_pool: kpool " + std::to_string(kpool));
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n_tokens * D), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0] / D;
        const int c = (int) ((int64_t) id[0] % D);
        const int32_t pos = steps[t * kStepCount + kStepPos];
        if (pos < 0 || (pos + 1) % kpool != 0) return;   // no pool completes at this token
        const int64_t m = (pos + 1) / kpool - 1;
        float l[KMAX], mx = -INFINITY;
        for (int j = 0; j < kpool; ++j) {
            l[j] = cache[(m * kpool + j) * 2 * D + D + c] + ape[j * D + c];
            mx = sycl::fmax(mx, l[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < kpool; ++j) {
            l[j] = sycl::exp(l[j] - mx);
            sum += l[j];
        }
        const float inv = 1.0f / sum;
        float acc = 0.0f;
        for (int j = 0; j < kpool; ++j) acc += l[j] * inv * cache[(m * kpool + j) * 2 * D + c];
        pooled[m * D + c] = acc;
    });
    sync_if_needed(stream, event, "glm5_idx_pool");
}

void glm5_idx_scores(const float* pooled, const float* q, const float* w, int n_heads, const int32_t* steps,
                     int64_t n_tokens, int64_t max_blocks, float* scores, void* stream) {
    constexpr int D = 128, PER = D / WARP, SG = 8;   // sub-groups (blocks) a work-group
    if (n_tokens <= 0) return;
    if (n_heads <= 0 || max_blocks <= 0 || n_tokens > 65535)
        throw core::DeviceError("glm5_idx_scores: " + std::to_string(n_heads) + " heads, " + std::to_string(max_blocks) +
                                " blocks, " + std::to_string(n_tokens) + " tokens");
    const size_t gx = (size_t) ((max_blocks + SG - 1) / SG);
    const int H = n_heads;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<2>({(size_t) n_tokens, gx * SG * WARP}, {1, (size_t) SG * WARP}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t t = (int64_t) it.get_group(0);
            const int64_t b = (int64_t) it.get_group(1) * SG + (int64_t) sg.get_group_linear_id();
            const int64_t n_bid = steps[t * kStepCount + kStepNBid];
            if (b > n_bid || b >= max_blocks) return;   // uniform over the sub-group
            const int lane = (int) sg.get_local_linear_id();
            float score = 0.0f;
            if (b < n_bid) {
                float k[PER];
                for (int i = 0; i < PER; ++i) k[i] = pooled[b * D + (int64_t) lane * PER + i];
                for (int h = 0; h < H; ++h) {
                    const float* qh = q + (t * H + h) * D + (int64_t) lane * PER;
                    float d = 0.0f;
                    for (int i = 0; i < PER; ++i) d += qh[i] * k[i];
                    d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                    score += (d > 0.0f ? d : 0.0f) * w[t * H + h];
                }
            }
            if (lane == 0) scores[t * max_blocks + b] = score;
        });
    sync_if_needed(stream, event, "glm5_idx_scores");
}

void glm5_latent_store(const float* kv, const int32_t* steps, int64_t n_tokens, int64_t rank, uint16_t* latent,
                       void* stream) {
    if (n_tokens <= 0 || rank <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * rank)), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0] / rank, l = (int64_t) id[0] % rank;
        const int32_t pos = steps[t * kStepCount + kStepPos];
        if (pos >= 0) latent[(int64_t) pos * rank + l] = f16_from_f32(kv[t * rank + l]);
    });
    sync_if_needed(stream, event, "glm5_latent_store");
}

// b70-local (2026-10-27): the MLA's heads share one latent row per cell, but the kernel below gives every
// (token, head) its own work-group, so each cell's 1 KiB row was read 2 x n_heads times per token (scores, then
// values) by scalar loops: 3,024 prompt tokens took ~1 s a DSA block on the B70 (~10 s of a 49 s prompt).  This one
// takes G = 16 heads of a token per work-group and streams the cells through local memory in tiles with an online
// softmax, so a row is read once per 16 heads.  STRATA_MLA_HG=0 goes back to the per-head kernel.
static bool mla_hg_attn(const float* q, const uint16_t* latent, const int32_t* ids, int64_t cap, const int32_t* steps,
                        int H, float scale, int64_t n_tokens, float* out, void* stream, int KP) {
    constexpr int L = 512, WG = 256, G = 16, TJ = 16, PERD = L / (WG / G);   // 32 dims a thread in the output
    static const bool on = [] { const char* v = std::getenv("STRATA_MLA_HG"); return v != nullptr && v[0] == '1'; }();   // slower (opt-in)
    if (!on || H % G != 0) return false;
    const auto event = queue_for(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>((size_t) G * L), hd);
        sycl::local_accessor<uint16_t, 1> tile(sycl::range<1>((size_t) TJ * L), hd);
        sycl::local_accessor<float, 1> pr(sycl::range<1>((size_t) G * TJ), hd);
        sycl::local_accessor<float, 1> st(sycl::range<1>((size_t) 3 * G), hd);   // m, l, the rescale
        hd.parallel_for(sycl::nd_range<1>((size_t) (n_tokens * (H / G)) * WG, WG),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int64_t grp = (int64_t) it.get_group(0), t = grp / (H / G);
            const int h0 = (int) (grp % (H / G)) * G;
            const int lid = (int) it.get_local_id(0);
            const int64_t chosen = sycl::min<int64_t>(steps[t * kStepCount + kStepWidth], cap);
            const int32_t pos = steps[t * kStepCount + kStepPos];
            const int64_t tail0 = KP > 0 && pos >= 0 ? (int64_t) ((pos + 1) / KP) * KP : 0;
            const int64_t width = chosen + (KP > 0 && pos >= 0 ? (int64_t) pos + 1 - tail0 : 0);
            // this thread's output: head og, dims od + 16 k (neighbouring lanes on neighbouring halves: a lane's
            // own 32-dim run put 16 lanes 64 bytes apart, all on one bank of the local memory, 2.4x slower)
            const int og = lid / (WG / G), od = lid % (WG / G);
            float* oh = out + ((int64_t) t * H + h0 + og) * L + od;
            if (width <= 0) {
                for (int k = 0; k < PERD; ++k) oh[k * (WG / G)] = 0.0f;
                return;
            }
            for (int i = lid; i < G * L; i += WG) qs[i] = q[((int64_t) t * H + h0) * L + i];
            if (lid < G) { st[lid] = -INFINITY; st[G + lid] = 0.0f; }
            const int32_t* sel = ids + t * cap;
            float acc[PERD];
            for (int k = 0; k < PERD; ++k) acc[k] = 0.0f;
            const int sg_ = lid / TJ, sj = lid % TJ;   // this thread's score: head sg_, cell sj of the tile
            for (int64_t j0 = 0; j0 < width; j0 += TJ) {
                {   // the tile's rows: a thread 32 halves of one row
                    const int r = lid / (WG / TJ), c = (lid % (WG / TJ)) * (L / (WG / TJ));
                    const int64_t j = j0 + r;
                    if (j < width) {
                        const int64_t cl = j < chosen ? (int64_t) sel[j] : tail0 + (j - chosen);
                        const uint16_t* src = latent + cl * L + c;
                        for (int k = 0; k < L / (WG / TJ); ++k) tile[r * L + c + k] = src[k];
                    }
                }
                sycl::group_barrier(it.get_group());
                {
                    float d = -INFINITY;
                    if (j0 + sj < width) {
                        d = 0.0f;
                        const float* qq = &qs[sg_ * L];
                        const uint16_t* rr = &tile[sj * L];
                        for (int i = 0; i < L; ++i) {
                            const int k = (i + sj * 2) & (L - 1);   // spread the rows over the banks
                            d += qq[k] * f32_from_f16(rr[k]);
                        }
                        d *= scale;
                    }
                    pr[sg_ * TJ + sj] = d;
                }
                sycl::group_barrier(it.get_group());
                if (lid < G) {   // the online softmax, a thread a head
                    float mx = st[lid];
                    for (int j = 0; j < TJ; ++j) mx = sycl::fmax(mx, pr[lid * TJ + j]);
                    const float corr = sycl::exp(st[lid] - mx);
                    float sum = 0.0f;
                    for (int j = 0; j < TJ; ++j) {
                        const float e = pr[lid * TJ + j] == -INFINITY ? 0.0f : sycl::exp(pr[lid * TJ + j] - mx);
                        pr[lid * TJ + j] = e;
                        sum += e;
                    }
                    st[lid] = mx;
                    st[G + lid] = st[G + lid] * corr + sum;
                    st[2 * G + lid] = corr;
                }
                sycl::group_barrier(it.get_group());
                const float corr = st[2 * G + og];
                for (int k = 0; k < PERD; ++k) acc[k] *= corr;
                for (int j = 0; j < TJ; ++j) {
                    const float p = pr[og * TJ + j];
                    if (p == 0.0f) continue;
                    const uint16_t* rr = &tile[j * L + od];
                    for (int k = 0; k < PERD; ++k) acc[k] += p * f32_from_f16(rr[k * (WG / G)]);
                }
                sycl::group_barrier(it.get_group());
            }
            const float inv = 1.0f / st[G + og];
            for (int k = 0; k < PERD; ++k) oh[k * (WG / G)] = acc[k] * inv;
        });
    });
    sync_if_needed(stream, event, "glm5_mla_attn (heads grouped)");
    return true;
}

void glm5_mla_attn(const float* q, const uint16_t* latent, const int32_t* ids, int64_t cap, const int32_t* steps,
                   int n_heads, float scale, int64_t n_tokens, float* out, void* stream, int tail_kpool) {
    constexpr int L = 512, WG = 256, NSG = WG / WARP, PER = L / WARP;
    if (n_tokens <= 0) return;
    if (n_heads > 0 && cap > 0 && tail_kpool >= 0 && tail_kpool != 1 &&   // one token too: 1.52 -> see below
        glm5_mla_attn_xmx(q, latent, ids, cap, steps, n_heads, scale, n_tokens, out, stream, tail_kpool))
        return;
    if (n_tokens > 1 && n_heads > 0 && cap > 0 && tail_kpool >= 0 && tail_kpool != 1 &&
        mla_hg_attn(q, latent, ids, cap, steps, n_heads, scale, n_tokens, out, stream, tail_kpool))
        return;
    if (n_heads <= 0 || cap <= 0 || n_tokens * n_heads > 0x7fffffff || tail_kpool < 0 || tail_kpool == 1)
        throw core::DeviceError("glm5_mla_attn: " + std::to_string(n_heads) + " heads, cap " + std::to_string(cap) +
                                ", tail pool " + std::to_string(tail_kpool));
    const int H = n_heads, KP = tail_kpool;
    const auto event = queue_for(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>((size_t) (cap + (KP > 0 ? KP - 1 : 0))), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) (n_tokens * H) * WG, WG),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int64_t th = (int64_t) it.get_group(0), t = th / H;
            const int lid = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int warp = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int64_t chosen = sycl::min<int64_t>(steps[t * kStepCount + kStepWidth], cap);
            // the unfinished pool's cells up to the token, after the chosen ones: cell j - chosen + tail0
            const int32_t pos = steps[t * kStepCount + kStepPos];
            const int64_t tail0 = KP > 0 && pos >= 0 ? (int64_t) ((pos + 1) / KP) * KP : 0;
            const int64_t width = chosen + (KP > 0 && pos >= 0 ? (int64_t) pos + 1 - tail0 : 0);
            const float* qh = q + th * L;
            float* oh = out + th * L;
            if (width <= 0) {   // uniform over the work-group
                for (int d = lid; d < L; d += WG) oh[d] = 0.0f;
                return;
            }
            float qr[PER];
            for (int i = 0; i < PER; ++i) qr[i] = qh[lane * PER + i];
            const int32_t* sel = ids + t * cap;
            auto cell = [&](int64_t j) { return j < chosen ? (int64_t) sel[j] : tail0 + (j - chosen); };
            for (int64_t j = warp; j < width; j += NSG) {   // a sub-group a cell
                const uint16_t* row = latent + cell(j) * L + (int64_t) lane * PER;
                float d = 0.0f;
                for (int i = 0; i < PER; ++i) d += qr[i] * f32_from_f16(row[i]);
                d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                if (lane == 0) sc[j] = d * scale;
            }
            sycl::group_barrier(it.get_group());
            float mx = -INFINITY;
            for (int64_t j = lid; j < width; j += WG) mx = sycl::fmax(mx, sc[j]);
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t j = lid; j < width; j += WG) sum += sycl::exp(sc[j] - mx);
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float inv = 1.0f / sum;
            for (int64_t j = lid; j < width; j += WG) sc[j] = sycl::exp(sc[j] - mx) * inv;   // the probabilities
            sycl::group_barrier(it.get_group());
            for (int d = lid; d < L; d += WG) {
                float acc = 0.0f;
                for (int64_t j = 0; j < width; ++j) acc += sc[j] * f32_from_f16(latent[cell(j) * L + d]);
                oh[d] = acc;
            }
        });
    });
    sync_if_needed(stream, event, "glm5_mla_attn");
}

void glm5_swiglu_clamp(const float* gate, const float* up, float* out, int64_t n, float limit, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float gv = sycl::fmin(gate[i], limit);
        const float uv = sycl::clamp(up[i], -limit, limit);
        out[i] = gv / (1.0f + sycl::exp(-gv)) * uv;
    });
    sync_if_needed(stream, event, "glm5_swiglu_clamp");
}

}  // namespace strata::kernels
