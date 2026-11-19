// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tests/core/inkling2_attn_xmx_test.cpp - b70-local: Inkling 2's prompt attention on the matrix engines
// (inkling2_attention_xmx) against the plain kernel (inkling2_attention), on random rows at Inkling 2 Small's shapes
// (48 heads of 128 over 8 KV heads, a bias rank of 16 over 1024 offsets): a sliding window that cuts the cells and a
// full one, a window that is not a multiple of 16 tokens, a large sink.  And the decode's windows (up to 8 tokens:
// inkling2_attention's split kernels) against a reference on the host in double precision.  Exits 77 without the
// matrix engines.
#include "strata/core/gpu.hpp"
#include "strata/kernels/inkling2.hpp"
#include "strata/kernels/qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

namespace k = strata::kernels;
namespace gpu = strata::gpu;

template<typename E>
E* up(const std::vector<E>& h) {
    E* d = (E*) gpu::alloc_device(h.size() * sizeof(E));
    if (d != nullptr) gpu::copy(d, h.data(), h.size() * sizeof(E));
    return d;
}

// the attention on the host, in double precision (inkling2.hpp's definition)
std::vector<float> host_attention(const std::vector<float>& q, const std::vector<float>& r, const std::vector<float>& sinks,
                                  const std::vector<float>& kc, const std::vector<float>& vc,
                                  const std::vector<float>& table, int T, int64_t pos0, int H, int NKV, int D, int REL,
                                  int EXT, int window) {
    std::vector<float> out((size_t) T * H * D);
    const int G = H / NKV;
    std::vector<double> sc;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const int64_t pos = pos0 + t, lo = window > 0 ? std::max<int64_t>(0, pos - window + 1) : 0;
            const float* qv = &q[((size_t) t * H + h) * D];
            sc.assign((size_t) (pos - lo + 1), 0.0);
            double m = sinks[(size_t) t * H + h];
            for (int64_t j = lo; j <= pos; ++j) {
                const float* kr = &kc[((size_t) j * NKV + h / G) * D];
                double s = 0.0;
                for (int d = 0; d < D; ++d) s += (double) qv[d] * kr[d];
                s /= std::sqrt((double) D);
                if (pos - j < EXT)
                    for (int i = 0; i < REL; ++i) s += (double) r[((size_t) t * H + h) * REL + i] * table[(size_t) (pos - j) * REL + i];
                sc[(size_t) (j - lo)] = s;
                m = std::max(m, s);
            }
            double den = std::exp(sinks[(size_t) t * H + h] - m);
            std::vector<double> o((size_t) D, 0.0);
            for (int64_t j = lo; j <= pos; ++j) {
                const double p = std::exp(sc[(size_t) (j - lo)] - m);
                den += p;
                const float* vr = &vc[((size_t) j * NKV + h / G) * D];
                for (int d = 0; d < D; ++d) o[(size_t) d] += p * vr[d];
            }
            for (int d = 0; d < D; ++d) out[((size_t) t * H + h) * D + d] = (float) (o[(size_t) d] / den);
        }
    return out;
}

// 1: pass, 0: fail, -1: the matrix kernel did not run
int run(int T, int64_t pos0, int window, float sink_bias, uint32_t seed) {
    constexpr int H = 48, NKV = 8, D = 128, REL = 16, EXT = 1024;
    const int64_t cells = pos0 + T;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto fill = [&](size_t n, float s) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * s; return v; };
    // q and k of ~unit RMS a value give scores of a few units after the 1/sqrt(D), as the clipped model's
    const auto q = fill((size_t) T * H * D, 1.5f), r = fill((size_t) T * H * REL, 0.5f);
    auto sinks = fill((size_t) T * H, 1.0f);
    for (auto& x : sinks) x += sink_bias;
    const auto kc = fill((size_t) cells * NKV * D, 1.0f), vc = fill((size_t) cells * NKV * D, 1.0f);
    const auto table = fill((size_t) EXT * REL, 0.3f);
    std::vector<int32_t> steps((size_t) T * k::kStepCount, 0);
    for (int t = 0; t < T; ++t) steps[(size_t) t * k::kStepCount + k::kStepPos] = (int32_t) (pos0 + t);
    float *dq = up(q), *dr = up(r), *ds = up(sinks), *dk = up(kc), *dv = up(vc), *dt = up(table);
    int32_t* dst = up(steps);
    float* o1 = (float*) gpu::alloc_device((size_t) T * H * D * 4);
    float* o2 = (float*) gpu::alloc_device((size_t) T * H * D * 4);
    k::inkling2_attention(dq, dr, ds, dk, dv, dt, dst, H, NKV, D, REL, EXT, window, o1, T, nullptr);
    const bool ran = T > 8 && k::inkling2_attention_xmx(dq, dr, ds, dk, dv, dt, dst, H, NKV, D, REL, EXT, window, o2, T, nullptr);
    gpu::device_sync();
    std::vector<float> a((size_t) T * H * D), b(a.size());
    gpu::copy(a.data(), o1, a.size() * 4);
    gpu::copy(b.data(), o2, b.size() * 4);
    for (void* p : {(void*) dq, (void*) dr, (void*) ds, (void*) dk, (void*) dv, (void*) dt, (void*) dst, (void*) o1, (void*) o2})
        gpu::free(p);
    if (T <= 8) {   // the decode's kernels against the host
        const auto ref = host_attention(q, r, sinks, kc, vc, table, T, pos0, H, NKV, D, REL, EXT, window);
        double md = 0.0, mr = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            md = std::max(md, (double) std::fabs(a[i] - ref[i]));
            mr = std::max(mr, (double) std::fabs(ref[i]));
        }
        const bool ok = md <= 2e-5 * std::max(mr, 1e-3);
        std::printf("  T %3d, cells %5lld, window %4d, sink %+5.1f: max |decode - host| %.3e (max |host| %.3e) %s\n", T,
                    (long long) cells, window, sink_bias, md, mr, ok ? "ok" : "FAIL");
        return ok ? 1 : 0;
    }
    if (!ran) return -1;
    double md = 0.0, mr = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        md = std::max(md, (double) std::fabs(a[i] - b[i]));
        mr = std::max(mr, (double) std::fabs(a[i]));
    }
    const bool ok = md <= 2e-4 * std::max(mr, 1e-3);
    std::printf("  T %3d, cells %5lld, window %4d, sink %+5.1f: max |xmx - plain| %.3e (max |plain| %.3e) %s\n", T,
                (long long) cells, window, sink_bias, md, mr, ok ? "ok" : "FAIL");
    return ok ? 1 : 0;
}

}  // namespace

int main() {
    int fail = 0;
    const struct { int T; int64_t pos0; int window; float sink; } cases[] = {
        {37, 1500, 1024, 0.0f},   // a sliding layer: the window cuts the cells; 37 tokens: a partial block
        {64, 0, 1024, 0.0f},      // the prompt's first window: fewer cells than the window
        {48, 3000, 0, 0.0f},      // a full layer
        {16, 200, 0, 12.0f},      // the sink takes most of the weight
        {512, 4000, 1024, 0.0f},  // a whole window
        {512, 4000, 0, 0.0f},
        {4, 300, 1024, 0.0f},     // the decode: a short context
        {4, 1500, 1024, 0.0f},    // a sliding layer past its window
        {8, 4000, 0, 0.0f},       // a full layer
        {1, 16, 0, 12.0f},
        {3, 900, 1024, 0.0f},
    };
    uint32_t seed = 1;
    for (const auto& c : cases) {
        const int r = run(c.T, c.pos0, c.window, c.sink, seed++);
        if (r < 0) {
            std::printf("no matrix engines for the kernel: skipped\n");
            return 77;
        }
        fail += r == 0;
    }
    std::printf("%s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
