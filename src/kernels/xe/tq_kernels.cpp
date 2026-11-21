// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/tq_kernels.cpp - b70-local (11-19): Inkling 2's trellis-coded experts (tq2: QTIP-style bit-shift trellis,
// 14-bit states, 2-bit symbols, 16x16 tiles of 66 bytes, the 1MAD computed code) for a few entries an expert (the decode
// and small windows).  The GGUF (w/mk_tq_mixed.py) keeps a matrix W [R, K] as R/16 bands of K/16 tiles; FileExpertSource
// re-packs each band's tiles in blocks of 16 (tq_repack) so that lane l of a sub-group reads dword i of tile l at
// [i * nt + l]: one 64-byte load for the 16 lanes instead of 16 scattered ones (B70: 175 -> 257 GB/s of codes).
//
// A tile's weight t (row t >> 4, column t & 15 of the tile) is values[s_t]: s_0 = bytes 0-1, s_t = (s_{t-1} >> 2) |
// (b_t << 12) with b_t the t-th 2-bit symbol (bytes 2..65, the first in the low bits), values[s] = (bytesum(s * 34038481 +
// 76625530) - 510) / 147.8.  W_rot[r, c] = weight * gain[r]; y = W_rot @ x', x' = Hb (sign * x), Hb the block-diagonal
// Sylvester Hadamard of 64 scaled 1/8.  The byte sum is one integer dot instruction (SPV_KHR_integer_dot_product, as
// cuda_intrinsics.hpp's dp4a); the generic byte loop the SYCL headers' dot_acc expands to was 2.5x slower on the B70.
//
// The blob (expert_source.cpp copy_from_files, tq layers): [256-byte header: int32 ff, int32 kTqMagic][gate codes ff/16
// bands of N/16 tiles][up codes][down codes N/16 bands of ff/16 tiles][f16 gains: gate ff, up ff, down N][the input's
// sign bits: N/8 bytes][the down input's: ff/8 bytes] (bit 1 = -1).
#include "strata/kernels/tq_kernels.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>


namespace strata::kernels {

namespace {


struct View {
    int64_t ff;
    const uint8_t* gate;
    const uint8_t* up;
    const uint8_t* down;
    const sycl::half* gains;
    const uint8_t* gu_sign;
    const uint8_t* d_sign;
};
inline View view(const uint8_t* blob, int64_t N) {
    View v;
    v.ff = *(const int32_t*) blob;
    const uint8_t* b = blob + 256;
    const size_t gu = (size_t) (v.ff / 16) * (size_t) (N / 16) * kTqTileBytes, dn = gu;
    v.gate = b;
    v.up = b + gu;
    v.down = b + 2 * gu;
    v.gains = (const sycl::half*) (b + 2 * gu + dn);
    v.gu_sign = (const uint8_t*) (v.gains + 2 * v.ff + N);
    v.d_sign = v.gu_sign + N / 8;
    return v;
}

// y rows of one role for every entry of every group: a work-group of S sub-groups takes one band (16 rows) of one expert,
// sub-group s the s-th slice of the band's 16-tile blocks; the slices' sums meet in local memory.  The input's rotation
// is done here: Hb's 64-block is H4 (x) H16 (Sylvester: popcount(i & j) splits by bits), a lane's tile covers 16 inputs of a
// block (a block = 4 tiles = 4 neighbouring lanes, blocks of tiles start at multiples of 16), so a lane does H16 on its 16
// values and the 4 lanes H4 by two exchanges.  GU: bands [0, ff/16) gate, [ff/16, 2 ff/16) up, input x [ent_tok[e]][N],
// output gate / up [e][FF]; else down: N/16 bands, input h [e][FF], output out[ent_dst[e]][N].
// PRE (STRATA_TQ_PREROT=1, b70-local 11-22): xin holds each entry's rotated input [entry][ldx] (tq_rot below: the
// same operations in the same order, so the sums are the same to the bit) and the band's work-groups only read it.
template <int S, int EPM, bool GU, bool PRE = false>
sycl::event gemv(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_idx, const float* xin, int64_t ldx, float* o0, float* o1,
                 int64_t N, int64_t FF) {
    const int64_t bands = GU ? 2 * FF / 16 : N / 16;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(S * EPM * 16, h);
        h.parallel_for(sycl::nd_range<2>({cg, (size_t) bands * S * 16}, {1, (size_t) S * 16}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sl = (int) sg.get_group_linear_id();
            const int64_t band = (int64_t) it.get_group(1);
            for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
                const View v = view((const uint8_t*) grp_ptr[g], N);
                const int64_t ffb = v.ff / 16;
                if (GU ? band >= 2 * ffb : false) continue;   // the same for the whole work-group: no barrier is skipped by part of it
                const bool up = GU && band >= ffb;
                const int64_t rb = up ? band - ffb : band;   // the band in its matrix
                const int64_t K = GU ? N : v.ff, NC = K / 16, nblk = (NC + 15) / 16, per = (nblk + S - 1) / S;
                const uint8_t* M = GU ? (up ? v.up : v.gate) : v.down;
                const uint8_t* bandp = M + (size_t) rb * (size_t) NC * kTqTileBytes;
                const uint8_t* sgn = GU ? v.gu_sign : v.d_sign;
                const int64_t blk0 = sl * per, blk1 = sycl::min(nblk, blk0 + per);
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                // a pass takes up to EPM entries (each code decoded once for them); a single entry its own pass, so that
                // it does not pay the second entry's products
                auto pass = [&](int e, auto ep) {
                    constexpr int EP = decltype(ep)::value;
                    const int n = sycl::min(EP, e1 - e);
                    float acc[EP][16];
#pragma unroll
                    for (int j = 0; j < EP; ++j)
#pragma unroll
                        for (int r = 0; r < 16; ++r) acc[j][r] = 0.0f;
                    for (int64_t bk = blk0; bk < blk1; ++bk) {   // uniform in the sub-group: every lane takes part in the exchanges
                        const int64_t c0 = bk * 16;
                        const int nt = (int) sycl::min<int64_t>(16, NC - c0);   // a multiple of 4 (widths are multiples of 64)
                        const bool live = lane < nt;
                        const int ln = live ? lane : 0;
                        const int64_t c = c0 + ln;
                        // the inputs: sign, H16 in the lane, H4 over lanes 4m..4m+3; 1/8 (orthonormal) and 1/147.8 (the code's scale)
                        float xv[EP][16];
                        float xs[EP];
                        if constexpr (PRE) {
#pragma unroll
                            for (int j = 0; j < EP; ++j) {
                                const float* xp = xin + (size_t) (e + (j < n ? j : 0)) * (size_t) ldx + c * 16;
                                xs[j] = 0.0f;
#pragma unroll
                                for (int i = 0; i < 16; ++i) { xv[j][i] = live ? xp[i] : 0.0f; xs[j] += xv[j][i]; }
                            }
                        } else {
                        const uint32_t sb = (uint32_t) sgn[2 * c] | ((uint32_t) sgn[2 * c + 1] << 8);
#pragma unroll
                        for (int j = 0; j < EP; ++j) {
                            const int64_t row = GU ? (int64_t) ent_idx[e + (j < n ? j : 0)] : (int64_t) (e + (j < n ? j : 0));
                            const float* xp = xin + (size_t) row * (size_t) ldx + c * 16;
#pragma unroll
                            for (int i = 0; i < 16; ++i) xv[j][i] = ((sb >> i) & 1) ? -xp[i] : xp[i];
#pragma unroll
                            for (int hh = 1; hh < 16; hh <<= 1)
#pragma unroll
                                for (int i = 0; i < 16; ++i)
                                    if (!(i & hh)) { const float a0 = xv[j][i], a1 = xv[j][i | hh]; xv[j][i] = a0 + a1; xv[j][i | hh] = a0 - a1; }
#pragma unroll
                            for (int m = 1; m < 4; m <<= 1)
#pragma unroll
                                for (int i = 0; i < 16; ++i) {
                                    const float o = sycl::permute_group_by_xor(sg, xv[j][i], m);
                                    xv[j][i] = (lane & m) ? o - xv[j][i] : xv[j][i] + o;
                                }
                            xs[j] = 0.0f;
#pragma unroll
                            for (int i = 0; i < 16; ++i) { xv[j][i] = live ? xv[j][i] * (0.125f / 147.8f) : 0.0f; xs[j] += xv[j][i]; }
                        }
                        }
                        if (!live) continue;
                        // the pointer steps by nt from the last read (AGENTS.md: IGC 2.27-2.30's GEP simplification misreads
                        // constant multiples of a runtime stride)
                        const uint32_t* B = (const uint32_t*) (bandp + (size_t) c0 * kTqTileBytes);
                        uint32_t d[16];
                        const uint32_t* p = B + lane;
#pragma unroll
                        for (int i = 0; i < 16; ++i) { d[i] = *p; p += nt; }
                        const uint32_t d16 = ((const uint16_t*) (p - lane))[lane];
#pragma unroll
                        for (int j = 0; j < EP; ++j)
#pragma unroll
                            for (int r = 0; r < 16; ++r) acc[j][r] -= 510.0f * xs[j];
                        auto tw = [&](int k) -> uint32_t { return k < 32 ? ((d[k >> 1] >> (16 * (k & 1))) & 0xFFFFu) : d16; };
                        uint32_t s = tw(0);
                        {
                            const float w = (float) tq_bsum(s * 34038481u + 76625530u);
#pragma unroll
                            for (int j = 0; j < EP; ++j) acc[j][0] += w * xv[j][0];
                        }
#pragma unroll
                        for (int wd = 0; wd < 32; ++wd) {
                            const uint32_t U = s | (tw(1 + wd) << 14);
#pragma unroll
                            for (int u = 1; u <= 8; ++u) {
                                const int t = wd * 8 + u;
                                if (t > 255) break;
                                const uint32_t z = tq_bits(U, 2 * u, 14);
                                const float w = (float) tq_bsum(z * 34038481u + 76625530u);
#pragma unroll
                                for (int j = 0; j < EP; ++j) acc[j][t >> 4] += w * xv[j][t & 15];
                            }
                            s = (U >> 16) & 0x3FFFu;
                        }
                    }
#pragma unroll
                    for (int j = 0; j < EP; ++j)
#pragma unroll
                        for (int r = 0; r < 16; ++r) {
                            const float t = sycl::reduce_over_group(sg, acc[j][r], sycl::plus<float>());
                            if (lane == r) red[(sl * EP + j) * 16 + r] = t;
                        }
                    sycl::group_barrier(it.get_group());
                    if (sl == 0) {
#pragma unroll
                        for (int j = 0; j < EP; ++j) {
                            if (j >= n) break;
                            float t = 0.0f;
                            for (int s2 = 0; s2 < S; ++s2) t += red[(s2 * EP + j) * 16 + lane];
                            const int64_t row = rb * 16 + lane;
                            if (GU) {
                                const float gain = (float) v.gains[(up ? v.ff : 0) + row];
                                (up ? o1 : o0)[(size_t) (e + j) * (size_t) FF + row] = t * gain;
                            } else {
                                const float gain = (float) v.gains[2 * v.ff + row];
                                o0[(size_t) ent_idx[e + j] * (size_t) N + row] = t * gain;
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                };
                for (int e = e0; e < e1;) {
                    if (EPM > 1 && e1 - e >= 2) { pass(e, std::integral_constant<int, EPM>{}); e += sycl::min(EPM, e1 - e); }
                    else { pass(e, std::integral_constant<int, 1>{}); e += 1; }
                }
            }
        });
    });
}

// STRATA_TQ_PAIR=0: every entry its own pass (else two entries of an expert share a pass, a single one has its own)
bool tq_pair_on() { static const bool on = [] { const char* v = std::getenv("STRATA_TQ_PAIR"); return !v || std::strtol(v, nullptr, 10) != 0; }(); return on; }

// b70-local (11-22): each entry's input rotated once (STRATA_TQ_PREROT=1), as gemv does it in every band's work-group:
// xr[e][c * 16 + i] = (0.125 / 147.8) (Hb (sign * x))[c * 16 + i], a sub-group of 16 lanes takes 16 tiles' inputs (lane l
// tile c = 16 b + l, its H4 partners the lanes 4m..4m+3 as there).  GU: x [ent_tok[e]][N] by the gate / up signs, K = N;
// else h [e][ld] by the down signs, K = the expert's ff (h's rows past it are left).  An entry's expert: the last group
// whose start is at or before it.
template <bool GU>
sycl::event tq_rot(sycl::queue& q, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                   const int32_t* ent_idx, int64_t cap_entries, const float* xin, int64_t ldx, float* xr, int64_t N, int64_t K) {
    const size_t subs = (size_t) ((K / 16 + 15) / 16);
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({(size_t) cap_entries, subs * 16}, {1, 16}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id();
            const int e = (int) it.get_group(0);
            const int ng = *n_groups;
            if (ng <= 0 || e < grp_start[0] || e >= grp_start[ng]) return;   // the whole sub-group
            int lo = 0, hi = ng - 1;
            while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (grp_start[mid] <= e) lo = mid; else hi = mid - 1; }
            const View v = view((const uint8_t*) grp_ptr[lo], N);
            const int64_t Kk = GU ? N : v.ff, NC = Kk / 16;
            const int64_t c = (int64_t) it.get_group(1) * 16 + lane;
            if ((int64_t) it.get_group(1) * 16 >= NC) return;   // the whole sub-group
            const bool live = c < NC;   // lanes 4m..4m+3 alike (widths are multiples of 64)
            const int64_t cc = live ? c : 0;
            const uint8_t* sgn = GU ? v.gu_sign : v.d_sign;
            const uint32_t sb = (uint32_t) sgn[2 * cc] | ((uint32_t) sgn[2 * cc + 1] << 8);
            const int64_t row = GU ? (int64_t) ent_idx[e] : (int64_t) e;
            const float* xp = xin + (size_t) row * (size_t) ldx + cc * 16;
            float xv[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) xv[i] = ((sb >> i) & 1) ? -xp[i] : xp[i];
#pragma unroll
            for (int hh = 1; hh < 16; hh <<= 1)
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    if (!(i & hh)) { const float a0 = xv[i], a1 = xv[i | hh]; xv[i] = a0 + a1; xv[i | hh] = a0 - a1; }
#pragma unroll
            for (int m = 1; m < 4; m <<= 1)
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float o = sycl::permute_group_by_xor(sg, xv[i], m);
                    xv[i] = (lane & m) ? o - xv[i] : xv[i] + o;
                }
            if (!live) return;
            float* xo = xr + (size_t) e * (size_t) (GU ? N : ldx) + c * 16;
#pragma unroll
            for (int i = 0; i < 16; ++i) xo[i] = xv[i] * (0.125f / 147.8f);
        });
    });
}

}  // namespace

// STRATA_TQ_PREROT=0: the rotation in each band's work-group again (B70 11-22, DSpark spec 3: code / agent / ja
// 55.9 / 52.7 / 42.1 -> 57.2 / 54.0 / 42.7 tok/s, the same text)
bool tq_prerot_on() { static const bool on = [] { const char* v = std::getenv("STRATA_TQ_PREROT"); return !v || std::strtol(v, nullptr, 10) != 0; }(); return on; }

// one entry an expert is the decode and nearly every small window (a window's tokens rarely share an expert); a big
// window (the prompt) on the XMX engines (tq_xmx.cpp), else two entries at once (each code decoded once for both)
sycl::event tq_expert_gu(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                         const int32_t* n_groups, const int32_t* ent_tok, int64_t cap_entries, const float* x, float* gate,
                         float* up, int64_t N, int64_t FF, float* xr) {
    if (N % 64 != 0 || FF % 64 != 0) throw std::runtime_error("tq experts: widths must be multiples of 64");
    if (tq_xmx_on(q, cap_entries, N, FF)) return tq_xmx_gu(q, cg, grp_ptr, grp_start, n_groups, ent_tok, cap_entries, x, gate, up, N, FF);
    if (xr != nullptr) {
        tq_rot<true>(q, grp_ptr, grp_start, n_groups, ent_tok, cap_entries, x, N, xr, N, N);
        if (tq_pair_on()) return gemv<6, 2, true, true>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, xr, N, gate, up, N, FF);
        return gemv<6, 1, true, true>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, xr, N, gate, up, N, FF);
    }
    if (tq_pair_on()) return gemv<6, 2, true>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, x, N, gate, up, N, FF);
    return gemv<6, 1, true>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, x, N, gate, up, N, FF);
}

sycl::event tq_expert_down(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, int64_t cap_entries, const float* h,
                           float* out, int64_t N, int64_t FF, float* hr) {
    if (tq_xmx_on(q, cap_entries, N, FF)) return tq_xmx_down(q, cg, grp_ptr, grp_start, n_groups, ent_dst, cap_entries, h, out, N, FF);
    if (hr != nullptr) {
        tq_rot<false>(q, grp_ptr, grp_start, n_groups, nullptr, cap_entries, h, FF, hr, N, FF);
        if (tq_pair_on()) return gemv<3, 2, false, true>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hr, FF, out, nullptr, N, FF);
        return gemv<3, 1, false, true>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hr, FF, out, nullptr, N, FF);
    }
    if (tq_pair_on()) return gemv<3, 2, false>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, h, FF, out, nullptr, N, FF);
    return gemv<3, 1, false>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, h, FF, out, nullptr, N, FF);
}

}  // namespace strata::kernels
