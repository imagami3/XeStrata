// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/mmvq_type_bench.cpp - b70-local (11-20): the decode GEMV (native_mmvq, 1 token) of one matrix per
// weight type at Inkling 2's attention shapes, to compare bytes/s: Q6_K rows (what Inkling's attention uses today),
// Q4_0 (GGUF order), Q4_0 rows, Q8_0 rows, Q4_K, Q5_K.  Weights are random bytes with sane fp16 scales (speed only).
// usage: mmvq_type_bench [--reps N]
#include "strata/kernels/native_mmvq.hpp"
#include "parity_device.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace P = strata::parity;

int main(int argc, char** argv) {
    int reps = 200;
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    struct T { const char* name; int type; int conv; int belems; int bbytes; int scale_off; };   // scale_off: byte of the fp16 d in a block
    const T types[] = {{"Q6_K rows", 14, 1, 256, 210, 208}, {"Q6_K", 14, 0, 256, 210, 208}, {"Q4_0", 2, 0, 32, 18, 0},
                       {"Q4_0 rows", 2, 3, 32, 18, 0},
                       {"Q8_0 rows", 8, 2, 32, 34, 0}, {"Q4_K", 12, 0, 256, 144, 0}, {"Q5_K", 13, 0, 256, 176, 0}};
    const int shapes[][2] = {{4608, 6144}, {6144, 4608}, {4608, 1024}, {4608, 18432}};   // (n_in, n_out)
    std::mt19937 rng(1);
    void* st = P::stream();
    for (auto& sh : shapes) {
        const int n_in = sh[0], n_out = sh[1];
        std::vector<float> x(n_in); for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.f;
        float* xd = (float*) P::alloc_bytes((size_t) n_in * 4 + 64); P::copy_bytes_in(xd, x.data(), (size_t) n_in * 4);
        void* xq = P::alloc_bytes(k::native_q8_1_rows_bytes(n_in, 1) + 64); k::native_quantize_q8_1(xd, xq, n_in, 1, st);
        float* y = (float*) P::alloc_bytes((size_t) n_out * 4 + 64);
        for (auto& t : types) {
            const size_t bytes = (size_t) n_out * (n_in / t.belems) * t.bbytes;
            std::vector<uint8_t> h(bytes); for (auto& b : h) b = (uint8_t) rng();
            for (size_t b0 = 0; b0 + t.bbytes <= bytes; b0 += t.bbytes) { uint16_t d = 0x1400; std::memcpy(&h[b0 + t.scale_off], &d, 2); if (t.belems == 256 && t.type != 14) std::memcpy(&h[b0 + 2], &d, 2); }
            const int NC = std::max(2, (int) (512e6 / bytes));   // copies cycled so the reads come from memory, not L2
            std::vector<void*> ws(NC);
            int type = t.type;
            for (auto& w : ws) {
                w = P::alloc_bytes(bytes + 64); P::copy_bytes_in(w, h.data(), bytes);
                if (t.conv == 1) { k::native_q6_k_to_rows(w, n_in, n_out, st); type = k::kNativeQ6KRows; }
                if (t.conv == 2) { k::native_q8_0_to_rows(w, n_in, n_out, st); type = k::kNativeQ8Rows; }
                if (t.conv == 3) { k::native_q4_0_to_rows(w, n_in, n_out, st); type = k::kNativeQ4Rows; }
            }
            for (int i = 0; i < 10; ++i) k::native_mmvq(type, ws[i % NC], xq, y, n_in, n_out, 1, st);
            P::sync();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) k::native_mmvq(type, ws[i % NC], xq, y, n_in, n_out, 1, st);
            P::sync();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("%5d x %5d  %-10s %8.1f us  %6.1f GB/s  (%.2f MB)\n", n_in, n_out, t.name, us, bytes / us / 1e3, bytes / 1e6);

        }

    }
    return 0;
}
