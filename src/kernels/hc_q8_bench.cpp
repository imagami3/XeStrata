// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/hc_q8_bench.cpp - times fused_gr_read_multi's Q8_0 read (STRATA_HC_Q8) on synthetic weights of the
// real geometry (n_embd 2560, hc 4, hc_lr 320) for 1..8 tokens, and writes or checks its outputs (mixed, rs,
// inject_out, R_out of every token) against a file, so a rewrite of the read can be timed and compared alone.
// usage: hc_q8_bench [--write FILE | --check FILE] [--reps N]

#include "strata/kernels/fused_gr.hpp"
#include "parity_device.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace P = strata::parity;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;

uint16_t f2h(float f) {   // fp32 -> fp16 (round to nearest even), finite inputs
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u;
    int e = (int) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    uint32_t h = s | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}
uint16_t f2bf(float f) { uint32_t x; std::memcpy(&x, &f, 4); return (uint16_t) ((x + 0x7fffu + ((x >> 16) & 1u)) >> 16); }

// kNativeQ8Rows: each row's int8 values, then its fp16 scales (one per 32)
std::vector<uint8_t> q8_rows(int rows, int cols, std::mt19937& rng, float amp) {
    const size_t row = (size_t) cols + (size_t) cols / 32 * 2;
    std::vector<uint8_t> out(row * (size_t) rows);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.2f * amp, amp);
    for (int r = 0; r < rows; ++r) {
        uint8_t* q = out.data() + row * (size_t) r;
        for (int i = 0; i < cols; ++i) q[i] = (uint8_t) (int8_t) qd(rng);
        for (int b = 0; b < cols / 32; ++b) {
            const uint16_t h = f2h(sd(rng));
            q[cols + 2 * b] = (uint8_t) (h & 0xff);
            q[cols + 2 * b + 1] = (uint8_t) (h >> 8);
        }
    }
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    std::string wfile, cfile;
    int reps = 4000;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--write" && i + 1 < argc) wfile = argv[++i];
        else if (a == "--check" && i + 1 < argc) cfile = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto qd = q8_rows(LR, D, rng, 0.002f), qu = q8_rows(D, LR, rng, 0.004f);
    std::vector<float> wn(D);
    for (float& v : wn) v = 1.0f + 0.1f * nd(rng);
    std::vector<uint16_t> wi((size_t) HC * D);
    for (auto& v : wi) v = f2bf(0.01f * nd(rng));
    auto up = [](const void* h, size_t bytes) { void* p = P::alloc_bytes(bytes + 64); P::copy_bytes_in(p, h, bytes); return p; };
    const uint8_t* d_qd = (const uint8_t*) up(qd.data(), qd.size());
    const uint8_t* d_qu = (const uint8_t*) up(qu.data(), qu.size());
    const float* d_wn = (const float*) up(wn.data(), wn.size() * 4);
    const uint16_t* d_wi = (const uint16_t*) up(wi.data(), wi.size() * 2);
    // per token: R (in), R_out, bo_prev, inj_prev, lo, rs, inject_out, mixed
    const size_t per = (size_t) D + D + N + HC + LR + HC + HC + N;
    const int TM = k::kFusedGrMaxT;
    std::vector<float> host(per * TM);
    for (int t = 0; t < TM; ++t) {
        float* b = host.data() + per * t;
        for (int i = 0; i < D; ++i) b[i] = nd(rng);
        for (int i = 0; i < N; ++i) b[2 * D + i] = 0.5f * nd(rng);
        for (int i = 0; i < HC; ++i) b[2 * D + N + i] = nd(rng);
    }
    float* buf = (float*) up(host.data(), host.size() * 4);
    float* xn = (float*) P::alloc_bytes((size_t) TM * D * 4 + 64);
    k::FusedGrArgs a[k::kFusedGrMaxT];
    for (int t = 0; t < TM; ++t) {
        float* b = buf + per * t;
        k::FusedGrArgs& x = a[t];
        x.R = b; x.R_out = b + D; x.apply = true; x.bo_prev = b + 2 * D; x.inj_prev = b + 2 * D + N;
        x.lo = b + 2 * D + N + HC; x.rs = b + 2 * D + N + HC + LR; x.inject_out = b + 2 * D + N + HC + LR + HC;
        x.mixed = b + 2 * D + N + HC + LR + 2 * HC;
        x.w_norm = d_wn; x.w_inject = d_wi; x.q8_down = d_qd; x.q8_up = d_qu; x.eps = 1e-6f;
    }
    void* st = P::stream();
    std::vector<float> outs;
    for (int T : {1, 2, 4, 8}) {
        for (int i = 0; i < 2000; ++i) k::fused_gr_read_multi(a, T, xn, st);   // clocks up
        P::sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) k::fused_gr_read_multi(a, T, xn, st);
        P::sync();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("T=%d  %.2f us\n", T, us);
        std::vector<float> h(per * T);
        P::copy_bytes_out(h.data(), buf, h.size() * 4);
        for (int t = 0; t < T; ++t) {
            const float* b = h.data() + per * t;
            outs.insert(outs.end(), b + D, b + 2 * D);                                   // R_out
            outs.insert(outs.end(), b + 2 * D + N + HC + LR, b + 2 * D + N + HC + LR + 2 * HC);   // rs, inject_out
            outs.insert(outs.end(), b + 2 * D + N + HC + LR + 2 * HC, b + per);          // mixed
        }
    }
    if (!wfile.empty()) {
        FILE* f = std::fopen(wfile.c_str(), "wb");
        std::fwrite(outs.data(), 4, outs.size(), f);
        std::fclose(f);
        std::printf("wrote %zu floats\n", outs.size());
    }
    if (!cfile.empty()) {
        std::vector<float> ref(outs.size());
        FILE* f = std::fopen(cfile.c_str(), "rb");
        const size_t got = f ? std::fread(ref.data(), 4, ref.size(), f) : 0;
        if (f) std::fclose(f);
        if (got != ref.size()) { std::printf("check: reference has %zu of %zu floats\n", got, ref.size()); return 2; }
        double md = 0, mr = 0;
        size_t diff = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double d = std::fabs((double) outs[i] - (double) ref[i]);
            if (outs[i] != ref[i]) ++diff;
            if (d > md) md = d;
            mr = std::max(mr, std::fabs((double) ref[i]));
        }
        std::printf("check: %zu of %zu values differ, max |diff| %.3g (max |ref| %.3g)\n", diff, ref.size(), md, mr);
        return md <= 1e-3 * (mr > 1 ? mr : 1) ? 0 : 1;
    }
    return 0;
}
