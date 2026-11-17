// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/inkling2.cpp - see include/strata/kernels/inkling2.hpp (b70-local).
//
// Transcribed from llama.cpp's inkling2 graph (PR #32217 at beb656e, src/models/inkling2.cpp); MIT License, Copyright
// (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE.  Plain kernels first (a decode window is a few
// tokens): correctness against llama.cpp before speed.
#include "strata/kernels/inkling2.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_inkling2
#include "strata/kernels/device_arch_tag.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels {
namespace {

constexpr int WG = 256;
constexpr int kMaxStates = 16;     // the attention residuals' blocks + the current row (Inkling 2: at most 9)
constexpr int kMaxTaps = 8;
constexpr int kAttnChunk = 2048;   // the scores one pass of the attention keeps in local memory

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }
void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}

}  // namespace

void inkling2_attn_res(const float* blocks, int64_t block_stride, int n_blocks, const float* cur, const float* queries,
                       int nq, float* out, int64_t n_embd, int n_tokens, void* stream) {
    if (n_tokens <= 0 || nq <= 0) return;
    if (n_blocks < 0 || n_blocks + 1 > kMaxStates) throw core::DeviceError("inkling2_attn_res: too many blocks");
    const int64_t N = n_embd;
    const int S = n_blocks + 1;
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(kMaxStates), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * nq * WG, WG), [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_group(0), t = g % n_tokens, qi = g / n_tokens;
            const int lid = (int) it.get_local_id(0);
            const float* qv = queries + (int64_t) qi * N;
            float inv[kMaxStates], sc[kMaxStates];
            // each state's RMS and its score against the query
            for (int s = 0; s < S; ++s) {
                const float* x = s < n_blocks ? blocks + s * block_stride + (int64_t) t * N : cur + (int64_t) t * N;
                float ss = 0.0f, dq = 0.0f;
                for (int64_t i = lid; i < N; i += WG) {
                    const float v = x[i];
                    ss += v * v;
                    dq += v * qv[i];
                }
                ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                dq = sycl::reduce_over_group(it.get_group(), dq, sycl::plus<float>());
                inv[s] = sycl::rsqrt(ss / (float) N + 1e-6f);
                sc[s] = dq * inv[s] * sycl::rsqrt((float) N);
            }
            float mx = sc[0];
            for (int s = 1; s < S; ++s) mx = sycl::fmax(mx, sc[s]);
            float sum = 0.0f;
            for (int s = 0; s < S; ++s) { sc[s] = sycl::exp(sc[s] - mx); sum += sc[s]; }
            for (int s = 0; s < S; ++s) sc[s] = sc[s] / sum * inv[s];
            float* o = out + ((int64_t) qi * n_tokens + t) * N;
            for (int64_t i = lid; i < N; i += WG) {
                float acc = 0.0f;
                for (int s = 0; s < S; ++s) {
                    const float* x = s < n_blocks ? blocks + s * block_stride + (int64_t) t * N : cur + (int64_t) t * N;
                    acc += sc[s] * x[i];
                }
                o[i] = acc;
            }
            (void) red;
        });
    });
    sync_if_needed(stream, event, "inkling2_attn_res");
}

void inkling2_conv(const float* x, float* state, const float* w, const float* dyn, int heads, float* y,
                   int64_t n_channels, int n_taps, float scale, int n_tokens, void* stream) {
    if (n_tokens <= 0 || n_channels <= 0) return;
    if (n_taps < 2 || n_taps > kMaxTaps || (dyn != nullptr && (heads <= 0 || n_channels % heads != 0)))
        throw core::DeviceError("inkling2_conv: taps or heads");
    const int64_t C = n_channels, per_head = dyn != nullptr ? C / heads : 1;
    const int K = n_taps, T = n_tokens;
    // a work-item a channel, the tokens in order (a decode window is short)
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = (int64_t) id[0];
        float win[kMaxTaps];
        for (int j = 0; j < K - 1; ++j) win[j] = state[c * (K - 1) + j];
        const float* wc = w + c * K;
        const int64_t hd = c / per_head;
        for (int t = 0; t < T; ++t) {
            win[K - 1] = x[(int64_t) t * C + c];
            float sum = 0.0f;
            for (int j = 0; j < K; ++j) sum += win[j] * wc[j];
            if (dyn != nullptr) {
                const float* tap = dyn + ((int64_t) t * heads + hd) * K;
                for (int j = 0; j < K; ++j) sum += tap[j] * win[K - 1 - j];   // lag j: x_{t - j}
            }
            y[(int64_t) t * C + c] = (win[K - 1] + sum) * scale;
            for (int j = 0; j < K - 1; ++j) win[j] = win[j + 1];
        }
        for (int j = 0; j < K - 1; ++j) state[c * (K - 1) + j] = win[j];
    });
    sync_if_needed(stream, event, "inkling2_conv");
}

void inkling2_head_clip(float* x, const float* scale, float thr, float alpha, float floor_, const int32_t* steps,
                        int n_heads, int dim, int n_tokens, void* stream) {
    if (n_tokens <= 0 || n_heads <= 0) return;
    if (dim > 32 * 8) throw core::DeviceError("inkling2_head_clip: head too wide");
    // a sub-group of 32 a head, up to 8 values a lane
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * n_heads * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int g = (int) it.get_group(0), t = g / n_heads, hh = g % n_heads;
            const int lane = (int) it.get_local_id(0);
            float* v = x + ((int64_t) t * n_heads + hh) * dim;
            const float* sc = scale != nullptr ? scale + (int64_t) hh * dim : nullptr;
            float vals[8];
            float ss = 0.0f;
            for (int j = 0; j < 8; ++j) {
                const int i = lane + 32 * j;
                float a = i < dim ? v[i] : 0.0f;
                if (sc != nullptr && i < dim) a *= sc[i];
                vals[j] = a;
                ss += a * a;
            }
            ss = sycl::reduce_over_group(it.get_sub_group(), ss, sycl::plus<float>());
            const float norm = sycl::sqrt(ss) + 1e-6f;
            float f = sycl::fmin(sycl::fmax(thr / norm, 0.0f), 1.0f);
            if (alpha > 0.0f) {
                const float p = (float) steps[(int64_t) t * kStepCount + kStepPos] + 1.0f;
                f *= 1.0f + alpha * sycl::log(sycl::fmax(p / floor_, 1.0f));
            }
            for (int j = 0; j < 8; ++j) {
                const int i = lane + 32 * j;
                if (i < dim) v[i] = vals[j] * f;
            }
        });
    sync_if_needed(stream, event, "inkling2_head_clip");
}

void inkling2_head_rms(float* x, float eps, int n_heads, int dim, int n_tokens, void* stream) {
    if (n_tokens <= 0 || n_heads <= 0) return;
    if (dim > 32 * 8) throw core::DeviceError("inkling2_head_rms: head too wide");
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * n_heads * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int g = (int) it.get_group(0);
            const int lane = (int) it.get_local_id(0);
            float* v = x + (int64_t) g * dim;
            float ss = 0.0f;
            for (int i = lane; i < dim; i += 32) ss += v[i] * v[i];
            ss = sycl::reduce_over_group(it.get_sub_group(), ss, sycl::plus<float>());
            const float f = sycl::rsqrt(ss / (float) dim + eps);
            for (int i = lane; i < dim; i += 32) v[i] *= f;
        });
    sync_if_needed(stream, event, "inkling2_head_rms");
}

void inkling2_kv_store(const float* k, const float* v, const int32_t* steps, float* kc, float* vc, int64_t width,
                       int n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n_tokens * width), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / width, c = i % width;
        const int64_t cell = steps[t * kStepCount + kStepPos];
        kc[cell * width + c] = k[i];
        vc[cell * width + c] = v[i];
    });
    sync_if_needed(stream, event, "inkling2_kv_store");
}

void inkling2_attention(const float* q, const float* r, const float* sinks, const float* kc, const float* vc,
                        const float* table, const int32_t* steps, int n_head, int n_head_kv, int dim, int rel,
                        int extent, int window, float* out, int n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (dim != 128 || rel > 32 || n_head % n_head_kv != 0)
        throw core::DeviceError("inkling2_attention: heads of 128 and a bias rank of at most 32");
    constexpr int D = 128, AWG = 128;
    const int G = n_head / n_head_kv;
    const float qs = 1.0f / sycl::sqrt((float) D);
    // a work-group of 128 a (token, head): scores of a chunk of keys into local memory (a work-item a key), the chunk's
    // softmax folded into running (max, sum, acc[d]) per work-item d
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(D), h);
        sycl::local_accessor<float, 1> sr(sycl::range<1>(32), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>(kAttnChunk), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * n_head * AWG, AWG), [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_group(0), t = g / n_head, hh = g % n_head, kvh = hh / G;
            const int lid = (int) it.get_local_id(0);
            const int64_t pos = steps[(int64_t) t * kStepCount + kStepPos];
            const float* qv = q + ((int64_t) t * n_head + hh) * D;
            sq[lid] = qv[lid] * qs;
            if (lid < rel) sr[lid] = r[((int64_t) t * n_head + hh) * rel + lid];
            sycl::group_barrier(it.get_group());
            const int64_t lo = window > 0 ? sycl::max((int64_t) 0, pos - window + 1) : 0;
            const int64_t width = (int64_t) n_head_kv * D;
            float m = sinks[(int64_t) t * n_head + hh];   // the sink: a logit with a zero value
            float l = 1.0f, acc = 0.0f;
            for (int64_t c0 = lo; c0 <= pos; c0 += kAttnChunk) {
                const int64_t c1 = sycl::min(pos + 1, c0 + kAttnChunk);
                float cm = -INFINITY;
                for (int64_t j = c0 + lid; j < c1; j += AWG) {
                    const float* kr = kc + j * width + (int64_t) kvh * D;
                    float s = 0.0f;
                    for (int d = 0; d < D; ++d) s += sq[d] * kr[d];
                    const int64_t dist = pos - j;
                    if (dist < extent) {
                        const float* tb = table + dist * rel;
                        for (int i = 0; i < rel; ++i) s += sr[i] * tb[i];
                    }
                    sc[j - c0] = s;
                    cm = sycl::fmax(cm, s);
                }
                cm = sycl::reduce_over_group(it.get_group(), cm, sycl::maximum<float>());
                sycl::group_barrier(it.get_group());
                const float nm = sycl::fmax(m, cm), a = sycl::exp(m - nm);
                l *= a;
                acc *= a;
                float ls = 0.0f;
                for (int64_t j = c0 + lid; j < c1; j += AWG) {
                    const float p = sycl::exp(sc[j - c0] - nm);
                    sc[j - c0] = p;
                    ls += p;
                }
                l += sycl::reduce_over_group(it.get_group(), ls, sycl::plus<float>());
                sycl::group_barrier(it.get_group());
                for (int64_t j = c0; j < c1; ++j) acc += sc[j - c0] * vc[j * width + (int64_t) kvh * D + lid];
                m = nm;
                sycl::group_barrier(it.get_group());
            }
            out[((int64_t) t * n_head + hh) * D + lid] = acc / l;
        });
    });
    sync_if_needed(stream, event, "inkling2_attention");
}

void inkling2_sigmoid_mul(float* x, const float* g, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        x[i] *= 1.0f / (1.0f + sycl::exp(-g[i]));
    });
    sync_if_needed(stream, event, "inkling2_sigmoid_mul");
}

void inkling2_softcap(float* x, int64_t rows, int64_t cols, int64_t ld, float cap, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    const float ic = 1.0f / cap;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (rows * cols)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], rr = i / cols, c = i % cols;
        float* p = x + rr * ld + c;
        *p = cap * sycl::tanh(*p * ic);
    });
    sync_if_needed(stream, event, "inkling2_softcap");
}

}  // namespace strata::kernels
