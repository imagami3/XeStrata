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
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>
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

// b70-local: up to 8 tokens - the residual mix in parallel over the row: a work-group a (token, chunk of 512) sums each
// state's squares and its dot with each query (one pass over the states, every sum in registers: the state and query
// loops run to the compile-time bounds), a second kernel a (token, query, chunk) turns the sums into the softmax and
// writes its chunk of the mix.  One work-group a (token, query) with 2S reductions in turn took up to 183K clock ticks
// a call on the B70 (11-18, 8 blocks, one token) for 166 KB of reads.
namespace {
constexpr int kResCh = 512, kResQ = 3;
std::mutex res_mu;
std::vector<std::pair<sycl::queue*, std::pair<float*, size_t>>> res_bufs;
float* res_scratch(sycl::queue& q, size_t n) {   // kept for the graphs that use it, as att_scratch
    std::lock_guard<std::mutex> lk(res_mu);
    for (auto& b : res_bufs)
        if (b.first == &q && b.second.second >= n) return b.second.first;
    float* p = sycl::malloc_device<float>(n, q);
    res_bufs.push_back({&q, {p, n}});
    return p;
}
}  // namespace

static void inkling2_attn_res_small(const float* blocks, int64_t block_stride, int n_blocks, const float* cur,
                                    const float* queries, int nq, float* out, int64_t N, int T, void* stream) {
    const int S = n_blocks + 1, NC = (int) ((N + kResCh - 1) / kResCh), PW = kMaxStates * (1 + kResQ);
    sycl::queue& qu = queue_for(stream);
    float* part = res_scratch(qu, (size_t) T * NC * PW + 64);
    qu.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) WG / 8 * PW), h);
        h.parallel_for(sycl::nd_range<1>((size_t) T * NC * WG, WG), [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_group(0), t = g / NC, ch = g % NC, lid = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int sgi = (int) sg.get_group_linear_id(), nsg = (int) sg.get_group_linear_range();
            float acc[PW];
            for (int k = 0; k < PW; ++k) acc[k] = 0.0f;
            const int64_t i0 = (int64_t) ch * kResCh;
            for (int64_t i = i0 + lid; i < sycl::min(N, i0 + kResCh); i += WG) {
                float qv[kResQ];
                for (int qi = 0; qi < kResQ; ++qi) qv[qi] = qi < nq ? queries[(int64_t) qi * N + i] : 0.0f;
                for (int s = 0; s < kMaxStates; ++s) {
                    if (s >= S) break;
                    const float v = s < n_blocks ? blocks[s * block_stride + (int64_t) t * N + i] : cur[(int64_t) t * N + i];
                    acc[s * (1 + kResQ)] += v * v;
                    for (int qi = 0; qi < kResQ; ++qi) acc[s * (1 + kResQ) + 1 + qi] += v * qv[qi];
                }
            }
            for (int k = 0; k < PW; ++k) {
                const int sk = k / (1 + kResQ), jk = k % (1 + kResQ);
                if (sk >= S || (jk > 0 && jk - 1 >= nq)) continue;   // only the sums the mix reads
                const float r = sycl::reduce_over_group(sg, acc[k], sycl::plus<float>());
                if (sg.get_local_linear_id() == 0 && nsg <= WG / 8) red[(size_t) sgi * PW + k] = r;
            }
            sycl::group_barrier(it.get_group());
            if (lid < PW && lid / (1 + kResQ) < S && (lid % (1 + kResQ) == 0 || lid % (1 + kResQ) - 1 < nq)) {
                float v = 0.0f;
                for (int j = 0; j < nsg && j < WG / 8; ++j) v += red[(size_t) j * PW + lid];
                part[((size_t) t * NC + ch) * PW + lid] = v;
            }
        });
    });
    // the token's partials (NC x S x (1 + nq)) into local memory with one load a work-item, then every sum from there:
    // read from global memory one after another they cost a load's latency each (~160 a work-item, ~25 us)
    const auto ev = qu.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lp(sycl::range<1>((size_t) NC * PW), h);
        h.parallel_for(sycl::nd_range<1>((size_t) T * nq * NC * WG, WG), [=](sycl::nd_item<1> it) {
        const int g = (int) it.get_group(0), ch = g % NC, tq = g / NC, t = tq / nq, qi = tq % nq;
        const int lid = (int) it.get_local_id(0);
        for (int j = lid; j < NC * PW; j += WG) lp[j] = part[(size_t) t * NC * PW + j];
        sycl::group_barrier(it.get_group());
        float w[kMaxStates];
        float mx = -INFINITY;
        float inv_s[kMaxStates];
        for (int s = 0; s < kMaxStates; ++s) {
            if (s >= S) break;
            float ss = 0.0f, dq = 0.0f;
            for (int c = 0; c < NC; ++c) {
                ss += lp[c * PW + s * (1 + kResQ)];
                dq += lp[c * PW + s * (1 + kResQ) + 1 + qi];
            }
            inv_s[s] = sycl::rsqrt(ss / (float) N + 1e-6f);
            const float sc = dq * inv_s[s] * sycl::rsqrt((float) N);
            mx = sycl::fmax(mx, sc);
            w[s] = sc;
        }
        float sum = 0.0f, e[kMaxStates];
        for (int s = 0; s < kMaxStates; ++s) {
            if (s >= S) break;
            e[s] = sycl::exp(w[s] - mx);
            sum += e[s];
        }
        for (int s = 0; s < kMaxStates; ++s) {
            if (s >= S) break;
            e[s] = e[s] / sum * inv_s[s];
        }
        float* o = out + ((int64_t) qi * T + t) * N;
        const int64_t i0 = (int64_t) ch * kResCh;
        for (int64_t i = i0 + lid; i < sycl::min(N, i0 + kResCh); i += WG) {
            float a = 0.0f;
            for (int s = 0; s < kMaxStates; ++s) {
                if (s >= S) break;
                a += e[s] * (s < n_blocks ? blocks[s * block_stride + (int64_t) t * N + i] : cur[(int64_t) t * N + i]);
            }
            o[i] = a;
        }
        });
    });
    sync_if_needed(stream, ev, "inkling2_attn_res (small)");
}

void inkling2_attn_res(const float* blocks, int64_t block_stride, int n_blocks, const float* cur, const float* queries,
                       int nq, float* out, int64_t n_embd, int n_tokens, void* stream) {
    if (n_tokens <= 0 || nq <= 0) return;
    if (n_blocks < 0 || n_blocks + 1 > kMaxStates) throw core::DeviceError("inkling2_attn_res: too many blocks");
    // b70-local: STRATA_INK_RES_SMALL=0 - the one work-group a (token, query) kernel for every window
    static const bool small_on = [] { const char* v = std::getenv("STRATA_INK_RES_SMALL"); return v == nullptr || v[0] != '0'; }();
    if (small_on && n_tokens <= 8 && nq <= kResQ) {
        inkling2_attn_res_small(blocks, block_stride, n_blocks, cur, queries, nq, out, n_embd, n_tokens, stream);
        return;
    }
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

void inkling2_raw_rows(float* x, const float* src, const int32_t* raw, int64_t n_embd, int n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n_tokens * (size_t) N), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        if (raw[i / N] != 0) x[i] = src[i];
    });
    sync_if_needed(stream, event, "inkling2_raw_rows");
}

static void inkling2_attention_split(const float* q, const float* r, const float* sinks, const float* kc,
                                     const float* vc, const float* table, const int32_t* steps, int n_head,
                                     int n_head_kv, int rel, int extent, int window, float* out, int n_tokens,
                                     int splits, const float* gate, void* stream);

void inkling2_attention_gated(const float* q, const float* r, const float* sinks, const float* kc, const float* vc,
                              const float* table, const int32_t* steps, int n_head, int n_head_kv, int dim, int rel,
                              int extent, int window, float* out, const float* gate, int n_tokens, void* stream) {
    static const int split_env = [] { const char* v = std::getenv("STRATA_INK_ATT_SPLIT"); return v ? std::atoi(v) : -1; }();
    if (n_tokens > 0 && n_tokens <= 8 && split_env != 0 && dim == 128 && rel <= 32 && n_head % n_head_kv == 0) {
        const int splits = split_env > 0 ? split_env : (window > 0 ? 32 : 64);
        inkling2_attention_split(q, r, sinks, kc, vc, table, steps, n_head, n_head_kv, rel, extent, window, out,
                                 n_tokens, splits, gate, stream);
        return;
    }
    inkling2_attention(q, r, sinks, kc, vc, table, steps, n_head, n_head_kv, dim, rel, extent, window, out, n_tokens,
                       stream);
    if (gate != nullptr) inkling2_sigmoid_mul(out, gate, (int64_t) n_tokens * n_head * dim, stream);
}

void inkling2_attn_prep(const float* kr, const float* vr, float* conv_k, float* conv_v, const float* k_w,
                        const float* v_w, const float* kdyn, const float* vdyn, int n_taps, float conv_scale, float* q,
                        float* r, const float* q_clip, const float* r_clip, float qk_thr, float r_thr, float alpha,
                        float floor_, float eps, const int32_t* steps, float* kc, float* vc, int n_head, int n_head_kv,
                        int dim, int rel, int n_tokens, float* snap_k, float* snap_v, int64_t snap_stride, int n_snap,
                        void* stream) {
    if (n_tokens <= 0) return;
    constexpr int D = 128;
    if (dim != D || rel > D || n_taps > kMaxTaps || n_tokens > 8) throw core::DeviceError("inkling2_attn_prep: shapes");
    const int K = n_taps, T = n_tokens, NKV = n_head_kv, H = n_head;
    const int64_t KVW = (int64_t) NKV * D;
    // work-groups of 128: [0, NKV) k's heads, [NKV, 2 NKV) v's, then H of q, then H of r; each over the T tokens
    const auto ev = queue_for(stream).parallel_for(sycl::nd_range<1>((size_t) (2 * NKV + 2 * H) * D, D),
                                                   [=](sycl::nd_item<1> it) {
        const int g = (int) it.get_group(0), lid = (int) it.get_local_id(0);
        auto clip_factor = [&](float ss, float thr, int t, bool with_alpha) {
            float f = sycl::fmin(sycl::fmax(thr / (sycl::sqrt(ss) + 1e-6f), 0.0f), 1.0f);
            if (with_alpha && alpha > 0.0f) {
                const float p = (float) steps[(int64_t) t * kStepCount + kStepPos] + 1.0f;
                f *= 1.0f + alpha * sycl::log(sycl::fmax(p / floor_, 1.0f));
            }
            return f;
        };
        if (g < 2 * NKV) {   // k or v: the head's channels, a work-item a channel, the tokens in order
            const bool is_v = g >= NKV;
            const int hd = is_v ? g - NKV : g;
            const int64_t c = (int64_t) hd * D + lid;
            const float* x = is_v ? vr : kr;
            float* state = is_v ? conv_v : conv_k;
            const float* w = (is_v ? v_w : k_w) + c * K;
            const float* dyn = is_v ? vdyn : kdyn;
            float* snap = is_v ? snap_v : snap_k;
            float win[kMaxTaps];
            for (int j = 0; j < K - 1; ++j) win[j] = state[c * (K - 1) + j];
            for (int t = 0; t < T; ++t) {
                win[K - 1] = x[(int64_t) t * KVW + c];
                float sum = 0.0f;
                for (int j = 0; j < K; ++j) sum += win[j] * w[j];
                if (dyn != nullptr) {
                    const float* tap = dyn + ((int64_t) t * NKV + hd) * K;
                    for (int j = 0; j < K; ++j) sum += tap[j] * win[K - 1 - j];
                }
                float y = (win[K - 1] + sum) * conv_scale;
                for (int j = 0; j < K - 1; ++j) win[j] = win[j + 1];
                if (snap != nullptr && t < n_snap)
                    for (int j = 0; j < K - 1; ++j) snap[(int64_t) t * snap_stride + c * (K - 1) + j] = win[j];
                const float ss = sycl::reduce_over_group(it.get_group(), y * y, sycl::plus<float>());
                y *= is_v ? sycl::rsqrt(ss / (float) D + eps) : clip_factor(ss, qk_thr, t, false);
                const int64_t cell = steps[(int64_t) t * kStepCount + kStepPos];
                (is_v ? vc : kc)[cell * KVW + c] = y;
            }
            for (int j = 0; j < K - 1; ++j) state[c * (K - 1) + j] = win[j];
            return;
        }
        const bool is_r = g >= 2 * NKV + H;
        const int hh = is_r ? g - 2 * NKV - H : g - 2 * NKV;
        const int w = is_r ? rel : D;
        const float* sc = is_r ? r_clip : q_clip;
        for (int t = 0; t < T; ++t) {
            float* v = (is_r ? r + ((int64_t) t * H + hh) * rel : q + ((int64_t) t * H + hh) * D);
            float a = lid < w ? v[lid] : 0.0f;
            if (sc != nullptr && lid < w) a *= sc[(int64_t) hh * w + lid];
            const float ss = sycl::reduce_over_group(it.get_group(), a * a, sycl::plus<float>());
            const float f = clip_factor(ss, is_r ? r_thr : qk_thr, t, true);
            if (lid < w) v[lid] = a * f;
        }
    });
    sync_if_needed(stream, ev, "inkling2_attn_prep");
}

void inkling2_conv(const float* x, float* state, const float* w, const float* dyn, int heads, float* y,
                   int64_t n_channels, int n_taps, float scale, int n_tokens, void* stream, float* snap,
                   int64_t snap_stride, int n_snap) {
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
            if (t < n_snap)   // b70-local: the history after row t (speculative decoding's rollback)
                for (int j = 0; j < K - 1; ++j) snap[(int64_t) t * snap_stride + c * (K - 1) + j] = win[j];
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

// b70-local: a few tokens (the decode, speculative decoding's verify windows) - the attention split over the keys
// (flash-decoding): a work-group a (token, head, split) takes its split's keys, a sub-group a key's dot product (the
// key's row read contiguously), and writes its (max, sum, acc[D]); a second kernel joins the splits and the sink.  One
// work-group a (token, head) gave 48 work-groups for a token and read each key's row a work-item at a time: on the
// B70 (11-18, 2,356 cells) the attention was 43% of a decode layer.
namespace {
std::mutex att_mu;
std::vector<std::pair<sycl::queue*, std::pair<float*, size_t>>> att_bufs;
float* att_scratch(sycl::queue& q, size_t n) {
    std::lock_guard<std::mutex> lk(att_mu);
    for (auto& b : att_bufs)
        if (b.first == &q && b.second.second >= n) return b.second.first;
    float* p = sycl::malloc_device<float>(n, q);   // an older, smaller one stays for the graphs that use it
    att_bufs.push_back({&q, {p, n}});
    return p;
}
}  // namespace

static void inkling2_attention_split(const float* q, const float* r, const float* sinks, const float* kc,
                                     const float* vc, const float* table, const int32_t* steps, int n_head,
                                     int n_head_kv, int rel, int extent, int window, float* out, int n_tokens,
                                     int splits, const float* gate, void* stream) {
    constexpr int D = 128, AWG = 128, PW = D + 2;
    const int G = n_head / n_head_kv;
    const float qs = 1.0f / sycl::sqrt((float) D);
    sycl::queue& qu = queue_for(stream);
    const size_t nwg = (size_t) n_tokens * n_head * splits;
    float* part = att_scratch(qu, nwg * PW);
    qu.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(D), h);
        sycl::local_accessor<float, 1> sr(sycl::range<1>(32), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>(kAttnChunk), h);
        h.parallel_for(sycl::nd_range<1>(nwg * AWG, AWG), [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_group(0), sp = g % splits, th = g / splits, t = th / n_head, hh = th % n_head;
            const int kvh = hh / G, lid = (int) it.get_local_id(0);
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), nl = (int) sg.get_local_linear_range();
            const int sgi = (int) sg.get_group_linear_id(), nsg = (int) sg.get_group_linear_range();
            const int64_t pos = steps[(int64_t) t * kStepCount + kStepPos];
            sq[lid] = q[((int64_t) t * n_head + hh) * D + lid] * qs;
            if (lid < rel) sr[lid] = r[((int64_t) t * n_head + hh) * rel + lid];
            const int64_t lo = window > 0 ? sycl::max((int64_t) 0, pos - window + 1) : 0;
            const int64_t nk = pos - lo + 1, L = (nk + splits - 1) / splits;
            const int64_t j0 = lo + (int64_t) sp * L, j1 = sycl::min(pos + 1, j0 + L);
            const int64_t width = (int64_t) n_head_kv * D;
            sycl::group_barrier(it.get_group());
            float m = -INFINITY, l = 0.0f, acc = 0.0f;
            for (int64_t c0 = j0; c0 < j1; c0 += kAttnChunk) {
                const int64_t c1 = sycl::min(j1, c0 + kAttnChunk);
                for (int64_t j = c0 + sgi; j < c1; j += nsg) {   // a sub-group a key
                    const float* kr = kc + j * width + (int64_t) kvh * D;
                    float s = 0.0f;
                    for (int d = lane; d < D; d += nl) s += sq[d] * kr[d];
                    s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                    if (lane == 0) {
                        const int64_t dist = pos - j;
                        if (dist < extent) {
                            const float* tb = table + dist * rel;
                            for (int i = 0; i < rel; ++i) s += sr[i] * tb[i];
                        }
                        sc[j - c0] = s;
                    }
                }
                sycl::group_barrier(it.get_group());
                float cm = -INFINITY;
                for (int64_t j = c0 + lid; j < c1; j += AWG) cm = sycl::fmax(cm, sc[j - c0]);
                cm = sycl::reduce_over_group(it.get_group(), cm, sycl::maximum<float>());
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
            float* pp = part + (size_t) g * PW;
            if (lid == 0) { pp[0] = m; pp[1] = l; }
            pp[2 + lid] = acc;
        });
    });
    const auto ev = qu.parallel_for(sycl::nd_range<1>((size_t) n_tokens * n_head * D, D), [=](sycl::nd_item<1> it) {
        const int th = (int) it.get_group(0), lid = (int) it.get_local_id(0);
        const float* pp = part + (size_t) th * splits * PW;
        float M = sinks[th];   // the sink: a logit with a zero value
        for (int s = 0; s < splits; ++s) M = sycl::fmax(M, pp[(size_t) s * PW]);
        float den = sycl::exp(sinks[th] - M), num = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float ms = pp[(size_t) s * PW];
            if (ms == -INFINITY) continue;
            const float w = sycl::exp(ms - M);
            den += w * pp[(size_t) s * PW + 1];
            num += w * pp[(size_t) s * PW + 2 + lid];
        }
        const float o = num / den;
        out[(int64_t) th * D + lid] = gate ? o / (1.0f + sycl::exp(-gate[(int64_t) th * D + lid])) : o;
    });
    sync_if_needed(stream, ev, "inkling2_attention (split)");
}

void inkling2_attention(const float* q, const float* r, const float* sinks, const float* kc, const float* vc,
                        const float* table, const int32_t* steps, int n_head, int n_head_kv, int dim, int rel,
                        int extent, int window, float* out, int n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    // b70-local: up to 8 tokens through the split kernel (STRATA_INK_ATT_SPLIT=0: off; =N: N splits on every layer)
    static const int split_env = [] { const char* v = std::getenv("STRATA_INK_ATT_SPLIT"); return v ? std::atoi(v) : -1; }();
    if (n_tokens <= 8 && split_env != 0 && dim == 128 && rel <= 32 && n_head % n_head_kv == 0) {
        // 32 splits on the sliding layers (1024 cells), 64 on the full ones (B70, 11-18: 8 -> 32 on the sliding layers
        // took their attention part from 517K to 351K clock ticks at 2,356 cells)
        const int splits = split_env > 0 ? split_env : (window > 0 ? 32 : 64);
        inkling2_attention_split(q, r, sinks, kc, vc, table, steps, n_head, n_head_kv, rel, extent, window, out,
                                 n_tokens, splits, nullptr, stream);
        return;
    }
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

void inkling2_put_cols(float* dst, int64_t dst_ld, int64_t col0, const float* src, int64_t w, int64_t rows, void* stream) {
    if (rows <= 0 || w <= 0) return;
    const auto ev = queue_for(stream).parallel_for(sycl::range<1>((size_t) (rows * w)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / w, c = i % w;
        dst[t * dst_ld + col0 + c] = src[i];
    });
    sync_if_needed(stream, ev, "inkling2_put_cols");
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
