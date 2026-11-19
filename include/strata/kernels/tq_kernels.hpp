// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/tq_kernels.hpp - b70-local (11-19): Inkling 2's trellis-coded (tq2) experts on the GPU
// (src/kernels/xe/tq_kernels.cpp has the format and the blob layout).
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

inline constexpr int64_t kTqTileBytes = 66;            ///< a 16x16 tile: 2 bytes of start state, 255 2-bit symbols
inline constexpr int32_t kTqMagic = 0x32515454;        ///< "TTQ2", the blob header's second int32
/// The blob's bytes for an expert of width ff (multiple of 64) and model width N (multiple of 64), 256-aligned.
inline uint64_t tq_blob_bytes(int64_t N, int64_t ff) {
    const uint64_t codes = 3 * (uint64_t) (ff / 16) * (uint64_t) (N / 16) * (uint64_t) kTqTileBytes;
    return (256 + codes + 2 * (uint64_t) (2 * ff + N) + (uint64_t) (N / 8) + (uint64_t) (ff / 8) + 255) & ~(uint64_t) 255;
}
/// One band (16 rows) of nc tiles from the GGUF's order to the kernels' (blocks of 16 tiles, lane-interleaved dwords:
/// dword i of tile l of a block of nt tiles at [i * nt + l], its bytes 64-65 at halfword l after the 16 nt dwords).
inline void tq_repack_band(const uint8_t* src, uint8_t* dst, int64_t nc) {
    for (int64_t c0 = 0; c0 < nc; c0 += 16) {
        const int64_t nt = nc - c0 < 16 ? nc - c0 : 16;
        const uint8_t* s = src + c0 * kTqTileBytes;
        uint8_t* d = dst + c0 * kTqTileBytes;
        for (int64_t l = 0; l < nt; ++l) {
            for (int i = 0; i < 16; ++i)
                for (int b = 0; b < 4; ++b) d[(i * nt + l) * 4 + b] = s[l * kTqTileBytes + 4 * i + b];
            d[64 * nt + 2 * l] = s[l * kTqTileBytes + 64];
            d[64 * nt + 2 * l + 1] = s[l * kTqTileBytes + 65];
        }
    }
}
/// Gate and up rows of every entry (x: FP32 rows by ent_tok, N each) into gate / up [entry][FF].
sycl::event tq_expert_gu(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                  const int32_t* n_groups, const int32_t* ent_tok, int64_t cap_entries, const float* x, float* gate,
                  float* up, int64_t N, int64_t FF);
/// Down rows of every entry (h: [entry][FF], the GLU output) into out[ent_dst[entry]][N].
sycl::event tq_expert_down(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, int64_t cap_entries, const float* h,
                           float* out, int64_t N, int64_t FF);

}  // namespace strata::kernels
