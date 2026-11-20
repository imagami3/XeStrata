// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/q4_rows_parity.cpp - b70-local: kNativeQ4Rows (native_q4_0_to_rows' split layout) against GGUF-order
// Q4_0 (type 2) on the same random weights (random nibbles, fp16 scales of 2^-10 to 2^-4, either sign), memcmp:
//
//   * native_mmvq for ncols 1..8, in the exact and the upstream multi-column layouts, at shapes that take the one-row
//     layout (n_in <= 5120) and the work-group one (6144), with row counts that leave partial row groups;
//   * native_mmvq_pair, native_mmvq_batched (3 batches) and native_mmvq_multi (3 matrices): the same answer (taken
//     or not) as type 2, and then the same bits;
//   * dequant_bf16 / dequant_f16 / dequant_f32 of a slice of rows (from row 3 on): the same values as type 2's;
//   * native_q4_0_to_rows refuses an odd block count.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "parity_device.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace P = strata::parity;
namespace k = strata::kernels;

std::vector<float> down(const float* d, size_t n) {
    std::vector<float> h(n);
    P::copy_bytes_out(h.data(), d, n * sizeof(float));
    return h;
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
float* zeros(size_t n) {
    auto* p = static_cast<float*>(P::alloc_bytes(n * sizeof(float) + 64));
    const std::vector<float> z(n + 16, 0.0f);
    P::copy_bytes_in(p, z.data(), z.size() * sizeof(float));
    return p;
}

// random Q4_0 blocks: 16 random bytes of nibbles and an fp16 d of magnitude 2^-10 .. 2^-4
std::vector<uint8_t> random_q4_0(std::mt19937& rng, size_t blocks) {
    std::vector<uint8_t> h(blocks * 18);
    for (size_t b = 0; b < blocks; ++b) {
        const uint16_t d = (uint16_t) (((rng() & 1u) << 15) | ((5u + rng() % 7u) << 10) | (rng() & 0x3ffu));
        std::memcpy(&h[b * 18], &d, 2);
        for (int i = 0; i < 16; ++i) h[b * 18 + 2 + i] = (uint8_t) rng();
    }
    return h;
}

}  // namespace

int main() {
    std::mt19937 rng(20261120);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    void* st = P::stream();
    int bad = 0, cases = 0, nonzero = 0;
    auto check = [&](bool ok, const std::string& what) {
        ++cases;
        if (!ok) {
            std::printf("    *** %s differs from type 2 ***\n", what.c_str());
            ++bad;
        }
    };

    const int shapes[][2] = {{64, 9}, {128, 37}, {2048, 300}, {4608, 1000}, {6144, 513}};   // (n_in, n_out)
    constexpr int B = 3;   // matrices for batched / multi / pair
    for (const auto& sh : shapes) {
        const int n_in = sh[0], n_out = sh[1];
        const size_t mat_bytes = k::native_mmvq_weight_bytes(2, n_in, n_out);
        if (k::native_mmvq_weight_bytes(k::kNativeQ4Rows, n_in, n_out) != mat_bytes) {
            std::printf("    *** kNativeQ4Rows weight bytes differ from Q4_0's ***\n");
            ++bad;
        }
        const auto host = random_q4_0(rng, mat_bytes / 18 * B);
        void* w = P::alloc_bytes(host.size() + 64);
        void* wr = P::alloc_bytes(host.size() + 64);
        P::copy_bytes_in(w, host.data(), host.size());
        P::copy_bytes_in(wr, host.data(), host.size());
        for (int b = 0; b < B; ++b) k::native_q4_0_to_rows(static_cast<uint8_t*>(wr) + b * mat_bytes, n_in, n_out, st);
        const void* w_b[B] = {w, static_cast<uint8_t*>(w) + mat_bytes, static_cast<uint8_t*>(w) + 2 * mat_bytes};
        const void* wr_b[B] = {wr, static_cast<uint8_t*>(wr) + mat_bytes, static_cast<uint8_t*>(wr) + 2 * mat_bytes};

        // B batches of 8 columns of activations
        std::vector<float> x((size_t) B * 8 * n_in);
        for (auto& v : x) v = gauss(rng);
        auto* xd = static_cast<float*>(P::alloc_bytes(x.size() * sizeof(float) + 64));
        P::copy_bytes_in(xd, x.data(), x.size() * sizeof(float));
        void* xq = P::alloc_bytes(k::native_q8_1_rows_bytes(n_in, (int64_t) B * 8) + 64);
        k::native_quantize_q8_1_rows(xd, xq, n_in, (int64_t) B * 8, st);

        for (int exact = 1; exact >= 0; --exact) {
            k::native_mmvq_set_multi_exact(exact != 0);
            for (int nc = 1; nc <= 8; ++nc) {
                const std::string tag = std::to_string(n_in) + "x" + std::to_string(n_out) + " ncols=" +
                                        std::to_string(nc) + (exact ? " exact" : " upstream");
                const size_t ny = (size_t) nc * n_out;
                float* y = zeros(ny);
                float* yr = zeros(ny);
                k::native_mmvq(2, w, xq, y, n_in, n_out, nc, st);
                k::native_mmvq(k::kNativeQ4Rows, wr, xq, yr, n_in, n_out, nc, st);
                P::sync();
                const auto hy = down(y, ny);
                for (float v : hy) nonzero += v != 0.0f;
                check(same(hy, down(yr, ny)), "native_mmvq " + tag);

                // the pair
                float* y2 = zeros(ny);
                float* yr2 = zeros(ny);
                float* y1 = zeros(ny);
                float* yr1 = zeros(ny);
                const bool p = k::native_mmvq_pair(2, w_b[0], w_b[1], xq, y1, y2, n_in, n_out, nc, st);
                const bool pr = k::native_mmvq_pair(k::kNativeQ4Rows, wr_b[0], wr_b[1], xq, yr1, yr2, n_in, n_out, nc, st);
                P::sync();
                check(p == pr, "native_mmvq_pair taken " + tag);
                if (p && pr) check(same(down(y1, ny), down(yr1, ny)) && same(down(y2, ny), down(yr2, ny)),
                                   "native_mmvq_pair " + tag);

                // batched: batch b reads matrix b and its own nc columns
                const size_t x_stride = k::native_q8_1_bytes(n_in, nc);
                float* yb = zeros(ny * B);
                float* ybr = zeros(ny * B);
                const bool bt = k::native_mmvq_batched(2, w, mat_bytes, xq, x_stride, yb, ny, n_in, n_out, nc, B, st);
                const bool btr = k::native_mmvq_batched(k::kNativeQ4Rows, wr, mat_bytes, xq, x_stride, ybr, ny, n_in, n_out,
                                                        nc, B, st);
                P::sync();
                check(bt == btr, "native_mmvq_batched taken " + tag);
                if (bt && btr) check(same(down(yb, ny * B), down(ybr, ny * B)), "native_mmvq_batched " + tag);

                // multi: three matrices of different row counts over the same activations
                const int outs[B] = {n_out, n_out / 2 + 1, 7};
                float* ym[B];
                float* ymr[B];
                for (int b = 0; b < B; ++b) {
                    ym[b] = zeros((size_t) nc * outs[b]);
                    ymr[b] = zeros((size_t) nc * outs[b]);
                }
                const bool mt = k::native_mmvq_multi(2, B, w_b, ym, outs, xq, n_in, nc, st);
                const bool mtr = k::native_mmvq_multi(k::kNativeQ4Rows, B, wr_b, ymr, outs, xq, n_in, nc, st);
                P::sync();
                check(mt == mtr, "native_mmvq_multi taken " + tag);
                if (mt && mtr) {
                    bool ok = true;
                    for (int b = 0; b < B; ++b)
                        ok = ok && same(down(ym[b], (size_t) nc * outs[b]), down(ymr[b], (size_t) nc * outs[b]));
                    check(ok, "native_mmvq_multi " + tag);
                }
            }
        }
        k::native_mmvq_set_multi_exact(true);

        // the dequantizers, rows 3 .. n_out - 1 of the first matrix
        {
            const int64_t r0 = 3, nr = n_out - 3;
            const size_t n = (size_t) nr * n_in;
            auto* a = static_cast<uint16_t*>(P::alloc_bytes(n * 2 + 64));
            auto* b = static_cast<uint16_t*>(P::alloc_bytes(n * 2 + 64));
            auto* fa = static_cast<float*>(P::alloc_bytes(n * 4 + 64));
            auto* fb = static_cast<float*>(P::alloc_bytes(n * 4 + 64));
            std::vector<uint16_t> ha(n), hb(n);
            const char* names[2] = {"dequant_bf16", "dequant_f16"};
            for (int f = 0; f < 2; ++f) {
                if (f == 0) {
                    k::dequant_bf16(2, w, r0, nr, n_in, a, st);
                    k::dequant_bf16(k::kNativeQ4Rows, wr, r0, nr, n_in, b, st);
                } else {
                    k::dequant_f16(2, w, r0, nr, n_in, a, st);
                    k::dequant_f16(k::kNativeQ4Rows, wr, r0, nr, n_in, b, st);
                }
                P::sync();
                P::copy_bytes_out(ha.data(), a, n * 2);
                P::copy_bytes_out(hb.data(), b, n * 2);
                check(std::memcmp(ha.data(), hb.data(), n * 2) == 0,
                      std::string(names[f]) + " " + std::to_string(n_in) + "x" + std::to_string(n_out));
            }
            k::dequant_f32(2, w, r0, nr, n_in, fa, st);
            k::dequant_f32(k::kNativeQ4Rows, wr, r0, nr, n_in, fb, st);
            P::sync();
            const auto hfa = down(fa, n), hfb = down(fb, n);
            check(same(hfa, hfb), "dequant_f32 " + std::to_string(n_in) + "x" + std::to_string(n_out));
            for (float v : hfa) nonzero += v != 0.0f;
        }
        std::printf("  %5d x %5d: %s\n", n_in, n_out, bad ? "*** NO ***" : "bitwise");
    }

    bool threw = false;
    try {
        k::native_q4_0_to_rows(P::alloc_bytes(18 * 4 + 64), 32, 4, st);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ++cases;
    if (!threw || k::native_q4_0_rows_ok(32) || !k::native_q4_0_rows_ok(64)) {
        std::printf("    *** native_q4_0_to_rows accepted an odd block count ***\n");
        ++bad;
    }
    if (nonzero == 0) {
        std::printf("    *** every output was zero ***\n");
        ++bad;
    }

    std::printf("\nq4_rows: %d failures in %d cases\n", bad, cases);
    if (bad) return 1;
    std::printf("q4_rows_parity OK\n");
    return 0;
}
