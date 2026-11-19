// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tests/core/inkling2_attn_xmx_test.cpp - b70-local: Inkling 2's prompt attention on the matrix engines
// (inkling2_attention_xmx) against the plain kernel (inkling2_attention), on random rows at Inkling 2 Small's shapes
// (48 heads of 128 over 8 KV heads, a bias rank of 16 over 1024 offsets): a sliding window that cuts the cells and a
// full one, a window that is not a multiple of 16 tokens, a large sink.  Exits 77 without the matrix engines.
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
    const bool ran = k::inkling2_attention_xmx(dq, dr, ds, dk, dv, dt, dst, H, NKV, D, REL, EXT, window, o2, T, nullptr);
    gpu::device_sync();
    std::vector<float> a((size_t) T * H * D), b(a.size());
    gpu::copy(a.data(), o1, a.size() * 4);
    gpu::copy(b.data(), o2, b.size() * 4);
    for (void* p : {(void*) dq, (void*) dr, (void*) ds, (void*) dk, (void*) dv, (void*) dt, (void*) dst, (void*) o1, (void*) o2})
        gpu::free(p);
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
