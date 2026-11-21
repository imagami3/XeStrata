// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/tq_kernels.hpp - b70-local (11-19): Inkling 2's trellis-coded (tq2) experts on the GPU
// (src/kernels/xe/tq_kernels.cpp has the format and the blob layout).
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__) && !defined(__AMDGCN__)
extern SYCL_EXTERNAL unsigned __spirv_UDotKHR(unsigned a, unsigned b, int packed_format);   // 0: 4x8-bit packed
extern SYCL_EXTERNAL unsigned __spirv_BitFieldUExtract(unsigned base, unsigned offset, unsigned count);
extern SYCL_EXTERNAL int __spirv_SUDotAccSatKHR(int a, unsigned b, int acc, int packed_format);
#endif

namespace strata::kernels {

/// The sum of a word's 4 bytes: one integer dot instruction on the device (SPV_KHR_integer_dot_product, as
/// cuda_intrinsics.hpp's dp4a; the SYCL headers' dot_acc expands to a byte loop, 2.5x slower on the B70).
inline uint32_t tq_bsum(uint32_t v) {
#if defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__) && !defined(__AMDGCN__)
    return __spirv_UDotKHR(v, 0x01010101u, 0);
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    // NVIDIA (sm_61 and later): one dp4a, not four byte extracts and three adds (the decode's integer pipe)
    uint32_t r;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(v), "r"(0x01010101u), "r"(0u));
    return r;
#else
    return (v & 255u) + ((v >> 8) & 255u) + ((v >> 16) & 255u) + (v >> 24);
#endif
}

/// The sum of a word's 4 bytes less 510 (the 1MAD code's integer value, -510 .. 510): one integer dot instruction with
/// its accumulator on the device.
inline int32_t tq_bsum510(uint32_t v) {
#if defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__) && !defined(__AMDGCN__)
    return __spirv_SUDotAccSatKHR(0x01010101, v, -510, 0);
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    int32_t r;   // NVIDIA: one dp4a with its accumulator (bytes unsigned, the multiplier's signed 1s)
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(v), "r"(0x01010101), "r"(-510));
    return r;
#else
    return (int32_t) ((v & 255u) + ((v >> 8) & 255u) + ((v >> 16) & 255u) + (v >> 24)) - 510;
#endif
}

/// Bits [off, off + n) of v: one bit-field instruction on the device (a shift and a mask take two of the integer pipe,
/// which the decode fills).
inline uint32_t tq_bits(uint32_t v, uint32_t off, uint32_t n) {
#if defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__) && !defined(__AMDGCN__)
    return __spirv_BitFieldUExtract(v, off, n);
#else
    return (v >> off) & ((1u << n) - 1u);
#endif
}

/// (float) b for 0 <= b < 2^23 without an integer-to-float conversion (a quarter-rate instruction on NVIDIA): the bits
/// of 2^23 + b, less 2^23 (exact)
inline float tq_u2f(uint32_t b) {
    return sycl::bit_cast<float>(0x4B000000u | b) - 8388608.0f;
}

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

/// Whether a window of `cap_entries` entries takes the XMX kernels (tq_xmx.cpp; STRATA_TQ_XMX: 0 off, else the
/// entries it needs, default 64), as the device reports its matrix engines and local memory.
bool tq_xmx_on(sycl::queue& q, int64_t cap_entries, int64_t N, int64_t FF);
sycl::event tq_xmx_gu(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                      const int32_t* n_groups, const int32_t* ent_tok, int64_t cap_entries, const float* x, float* gate,
                      float* up, int64_t N, int64_t FF);
sycl::event tq_xmx_down(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                        const int32_t* n_groups, const int32_t* ent_dst, int64_t cap_entries, const float* h, float* out,
                        int64_t N, int64_t FF);

}  // namespace strata::kernels
