// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/ink_dspark.cpp - b70-local: see include/strata/kernels/ink_dspark.hpp.  Plain kernels: the drafter
// is a few rows a step.
#include "strata/kernels/ink_dspark.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>

namespace strata::kernels {
namespace {

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }
void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}
constexpr int WG = 256;

}  // namespace

void dspark_mix_norm(const float* taps, int n_taps, int64_t tap_stride, const float* w, const float* g, float* out,
                     int64_t n, int n_rows, float eps, void* stream) {
    if (n_rows <= 0) return;
    const auto ev = queue_for(stream).parallel_for(sycl::nd_range<1>((size_t) n_rows * WG, WG), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_group(0), lid = (int) it.get_local_id(0);
        float* o = out + (int64_t) t * n;
        float ss = 0.0f;
        for (int64_t i = lid; i < n; i += WG) {
            float v = 0.0f;
            const float* tp = taps + (int64_t) t * n + i;
            for (int j = 0; j < n_taps; ++j) { v += w[j] * *tp; tp += tap_stride; }
            o[i] = v;
            ss += v * v;
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = sycl::rsqrt(ss / (float) n + eps);
        for (int64_t i = lid; i < n; i += WG) o[i] = o[i] * r * g[i];
    });
    sync_if_needed(stream, ev, "dspark_mix_norm");
}

void dspark_head_norm_rope(float* x, int n_rows, int heads, int dim, const float* g, int64_t pos0, float theta,
                           float eps, void* stream) {
    if (n_rows <= 0) return;
    const int half = dim / 2;
    const auto ev = queue_for(stream).parallel_for(sycl::nd_range<1>((size_t) n_rows * heads * half, (size_t) half),
                                                   [=](sycl::nd_item<1> it) {
        const int64_t rh = (int64_t) it.get_group(0);
        const int i = (int) it.get_local_id(0);
        const int64_t t = rh / heads;
        float* v = x + rh * dim;
        const float a = v[i], b = v[i + half];
        const float ss = sycl::reduce_over_group(it.get_group(), a * a + b * b, sycl::plus<float>());
        const float r = sycl::rsqrt(ss / (float) dim + eps);
        const float an = a * r * g[i], bn = b * r * g[i + half];
        // NeoX: pair (i, i + dim/2), angle = pos * theta^(-2i/dim)
        const float inv = sycl::pow(theta, -2.0f * (float) i / (float) dim);
        const float ang = (float) (pos0 + t) * inv;
        const float c = sycl::cos(ang), s = sycl::sin(ang);
        v[i] = an * c - bn * s;
        v[i + half] = bn * c + an * s;
    });
    sync_if_needed(stream, ev, "dspark_head_norm_rope");
}

void dspark_store_kv(const float* k, const float* v, float* kc, float* vc, int64_t width, int n_rows, int64_t pos0,
                     int64_t ring, void* stream) {
    if (n_rows <= 0) return;
    const auto ev = queue_for(stream).parallel_for(sycl::range<1>((size_t) n_rows * width), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / width, c = i % width;
        const int64_t cell = (pos0 + t) % ring;
        kc[cell * width + c] = k[i];
        vc[cell * width + c] = v[i];
    });
    sync_if_needed(stream, ev, "dspark_store_kv");
}

void dspark_attention(const float* q, const float* kc, const float* vc, int64_t ring, int64_t lo, int64_t hi,
                      const float* kb, const float* vb, int n_q, int heads, int kv_heads, int dim, float* out,
                      void* stream) {
    if (n_q <= 0) return;
    constexpr int D = 128, CH = 128;
    if (dim != D) throw core::DeviceError("dspark_attention: heads of 128");
    const int G = heads / kv_heads;
    const int64_t nctx = hi > lo ? hi - lo : 0, nk = nctx + n_q;
    const float qs = 1.0f / sycl::sqrt((float) D);
    const auto ev = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qv(sycl::range<1>(D), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>(CH), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_q * heads * D, D), [=](sycl::nd_item<1> it) {
            const int64_t rh = (int64_t) it.get_group(0);
            const int d = (int) it.get_local_id(0);
            const int hd = (int) (rh % heads), kh = hd / G;
            qv[d] = q[rh * D + d] * qs;
            sycl::group_barrier(it.get_group());
            auto key = [&](int64_t j) -> const float* {
                return j < nctx ? kc + (((lo + j) % ring) * kv_heads + kh) * D : kb + ((j - nctx) * kv_heads + kh) * D;
            };
            auto val = [&](int64_t j) -> const float* {
                return j < nctx ? vc + (((lo + j) % ring) * kv_heads + kh) * D : vb + ((j - nctx) * kv_heads + kh) * D;
            };
            float m = -INFINITY, l = 0.0f, acc = 0.0f;
            for (int64_t c0 = 0; c0 < nk; c0 += CH) {
                const int n = (int) sycl::min<int64_t>(CH, nk - c0);
                float s = -INFINITY;
                if (d < n) {
                    const float* kr = key(c0 + d);
                    float dot = 0.0f;
                    for (int e = 0; e < D; ++e) dot += qv[e] * kr[e];
                    s = dot;
                }
                const float cm = sycl::reduce_over_group(it.get_group(), s, sycl::maximum<float>());
                const float mn = sycl::fmax(m, cm);
                const float p = d < n ? sycl::exp(s - mn) : 0.0f;
                sc[d] = p;
                const float ps = sycl::reduce_over_group(it.get_group(), p, sycl::plus<float>());
                const float corr = sycl::exp(m - mn);
                sycl::group_barrier(it.get_group());
                float a = 0.0f;
                for (int j = 0; j < n; ++j) a += sc[j] * val(c0 + j)[d];
                acc = acc * corr + a;
                l = l * corr + ps;
                m = mn;
                sycl::group_barrier(it.get_group());
            }
            out[rh * D + d] = acc / l;
        });
    });
    sync_if_needed(stream, ev, "dspark_attention");
}

void dspark_swiglu(const float* g, const float* u, float* out, int64_t n, void* stream) {
    const auto ev = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const float x = g[i];
        out[i] = x / (1.0f + sycl::exp(-x)) * u[i];
    });
    sync_if_needed(stream, ev, "dspark_swiglu");
}

void dspark_block_init(float* x, const float* anchor, const float* tok_norm, const float* mask, int64_t n, int n_rows,
                       float eps, void* stream) {
    const auto ev = queue_for(stream).parallel_for(sycl::nd_range<1>((size_t) n_rows * WG, WG), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_group(0), lid = (int) it.get_local_id(0);
        float* o = x + (int64_t) t * n;
        if (t > 0) {
            for (int64_t i = lid; i < n; i += WG) o[i] = mask[i];
            return;
        }
        float ss = 0.0f;
        for (int64_t i = lid; i < n; i += WG) ss += anchor[i] * anchor[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = sycl::rsqrt(ss / (float) n + eps);
        for (int64_t i = lid; i < n; i += WG) o[i] = anchor[i] * r * tok_norm[i];
    });
    sync_if_needed(stream, ev, "dspark_block_init");
}

void dspark_joint(float* joint, const float* state, const uint16_t* w1, const int32_t* prev, const float* hidden,
                  const float* context, int rank, int64_t n, void* stream) {
    const int64_t total = 2 * (int64_t) rank + 2 * n;
    const auto ev = queue_for(stream).parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        float v;
        if (i < rank) v = state[i];
        else if (i < 2 * rank) {
            const uint16_t hbits = w1[(int64_t) *prev * rank + (i - rank)];
            v = (float) sycl::bit_cast<sycl::half>(hbits);
        } else if (i < 2 * rank + n) v = hidden[i - 2 * rank];
        else v = context[i - 2 * rank - n];
        joint[i] = v;
    });
    sync_if_needed(stream, ev, "dspark_joint");
}

void dspark_gru(const float* j, float* state, float* latent, int rank, void* stream) {
    const auto ev = queue_for(stream).parallel_for(sycl::range<1>((size_t) rank), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        const float gate = 1.0f / (1.0f + sycl::exp(-j[i]));
        const float cand = sycl::tanh(j[rank + i]);
        state[i] = cand + gate * (state[i] - cand);
        latent[i] = sycl::tanh(j[2 * rank + i]);
    });
    sync_if_needed(stream, ev, "dspark_gru");
}

void dspark_top_prob(const float* logits, int64_t n, float cap, float* p, void* stream) {
    constexpr int TW = 1024;
    const auto ev = queue_for(stream).parallel_for(sycl::nd_range<1>(TW, TW), [=](sycl::nd_item<1> it) {
        const int lid = (int) it.get_local_id(0);
        float m = -INFINITY;
        auto v = [&](int64_t i) { return cap > 0.0f ? cap * sycl::tanh(logits[i] / cap) : logits[i]; };
        for (int64_t i = lid; i < n; i += TW) m = sycl::fmax(m, v(i));
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 0.0f;
        for (int64_t i = lid; i < n; i += TW) s += sycl::exp(v(i) - m);
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (lid == 0) p[0] = 1.0f / s;
    });
    sync_if_needed(stream, ev, "dspark_top_prob");
}

void dspark_zero(float* x, int64_t n, void* stream) {
    const auto ev = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { x[id[0]] = 0.0f; });
    sync_if_needed(stream, ev, "dspark_zero");
}

}  // namespace strata::kernels
