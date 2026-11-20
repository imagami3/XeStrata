// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

// Native GGUF Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL
// and IQ4_XS / CUDA Q8_1 adapters, pinned to llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, sm_120 generic MMVQ.
// All pointers are device pointers, at least 4-byte aligned, with no overlap.
// All calls enqueue on the explicit non-null CUDA stream; no allocation or wait.
// The translation unit must use --use_fast_math, as the pinned CUDA oracle does.
//
// Shapes use GGUF order: n_in is the contiguous reduction dimension, n_out is
// the weight row count, and ncols is the activation column/token count, 1..8
// (plan v0.3 P3). Columns are contiguous: activation column j is x + j * n_in
// (its Q8_1 blocks at j * n_in / 32) and output column j is y + j * n_out. See
// native_mmvq_set_multi_exact for how ncols > 1 relates to ncols == 1.
// n_in must be a positive multiple of 32 for quantization/Q4_0/Q5_0/Q8_0/IQ4_NL,
// 64 for Q2_0, or 256 for the other formats. n_out must be positive. Weights remain
// unmodified row-major GGUF blocks: Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210 and
// IQ4_XS=136 bytes per 256 elements; Q2_0 is 18 bytes per 64 elements.
// Q4_0/IQ4_NL=18, Q5_0=22, Q8_0=34 bytes per 32 elements.
// Q8_1 scratch has 36 bytes per 32 elements, with no extra row padding here.
// Input floats must be finite, and their block scales/sums representable in FP16.
std::size_t native_q8_1_bytes(int n_in, int ncols = 1);

// Layout for ncols > 1. false: llama.cpp's generic multi-column table (upstream), equal to ncols == 1 to
// float rounding, speed not yet measured. true (default): the ncols == 1 layout, every column bitwise equal to a
// single-column call. Set before
// graph capture; captured graphs keep the kernels they captured.
void native_mmvq_set_multi_exact(bool exact);
bool native_mmvq_multi_exact();

// One quantization may serve multiple weight matrices sharing the same input.
// Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it does
// not reconstruct that sum from the quantized integers.
void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols,
                          void* stream);
/// native_quantize_q8_1's layout for any number of rows (one a token per head, the MLA's per-head products): the
/// bytes of `rows` rows (0 and more: the offset of row `rows`), and their quantization (rows >= 1).  The products
/// then read up to 8 of them at a time.
std::size_t native_q8_1_rows_bytes(int n_in, int64_t rows);
void native_quantize_q8_1_rows(const float* x, void* x_q8_1, int n_in, int64_t rows, void* stream);
/// native_quantize_q8_1 of gate / (1 + exp(-gate)) * up (the shared expert's SwiGLU) in one launch, the same bytes as
/// the SwiGLU's stored result quantized (upstream be7889ed).
void native_swiglu_q8_1(const float* gate, const float* up, void* x_q8_1, int n_in, int ncols, void* stream);
/// out = gate / (1 + exp(-gate)) * up in FP32, native_swiglu_q8_1's expression unquantized (a BF16 down projection
/// reads floats); `out` may be `gate`.
void native_swiglu_f32(const float* gate, const float* up, float* out, int n_in, int ncols, void* stream);

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

// Convenience composition: caller owns scratch sized by native_q8_1_bytes.
void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q2_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q2_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

// Storage/dispatch helpers take stable GGML type IDs, avoiding a ggml runtime
// dependency in the engine: Q4_0=2, Q5_0=6, Q8_0=8, Q3_K=11, Q4_K=12, Q5_K=13,
// Q6_K=14, IQ4_NL=20, IQ4_XS=23, Q2_0=42. Unsupported IDs throw in the byte-count
// and launch helpers; only the
// capability query returns false.
bool native_mmvq_supported(int ggml_type) noexcept;
std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out);
void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream);
/// native_mmvq of two matrices of one type and shape over the same activations in one launch (upstream b08cf3e1's
/// pair), every output the two calls' bits.  Returns false, launching nothing, where that layout does not apply.
/// b70-local: `batches` products of one shape in one launch (batch b: weights + b * w_stride bytes, x + b * x_stride
/// bytes, y + b * y_stride floats), each row's arithmetic native_mmvq's; false (nothing launched) where native_mmvq
/// would not take its one-row layout or the type has none (STRATA_MMVQ_BATCHED=0: always false).
bool native_mmvq_batched(int ggml_type, const void* w, std::size_t w_stride, const void* x_q8_1, std::size_t x_stride,
                         float* y, std::size_t y_stride, int n_in, int n_out, int ncols, int batches, void* stream);
/// b70-local: `count` (1-8) products of the same q8_1 activations, one type and n_in, matrix i of n_out[i] rows into
/// y[i], in one launch, each row's arithmetic native_mmvq's; false (nothing launched) as native_mmvq_batched.
bool native_mmvq_multi(int ggml_type, int count, const void* const* w, float* const* y, const int* n_out,
                       const void* x_q8_1, int n_in, int ncols, void* stream);
bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream);

// Q6_K with each row's blocks split into four arrays: the row's ql (128 bytes a block), then its qh (64), its
// scales (16) and its d (2), each in block order.  A Q6_K block is 210 bytes, so in GGUF order every other block
// starts on a two-byte boundary and each 32-bit weight load becomes two 16-bit ones; split, every load is aligned
// and the kernel reads the row at about twice the speed.  The same lanes do the same arithmetic in the same order,
// so the results are bitwise those of GGUF-order Q6_K.  Not a GGML type: the engine's own ID, which every
// consumer (native_mmvq, dequant_f16/bf16/f32) must handle by name.  Never written to a file.
// XeStrata's own type IDs live from kXeTypeBase ("XS" << 16) on, apart from ggml's and upstream Strata's (below 100).
constexpr int kXeTypeBase = 0x58530000;
constexpr int kNativeQ6KRows = kXeTypeBase + 1;
// Rewrites Q6_K rows (n_in / 256 blocks each, an even count so every row starts 4-byte aligned) in place into the
// kNativeQ6KRows layout.  Throws on an odd block count; the caller then keeps GGUF order.
bool native_q6_k_rows_ok(int n_in) noexcept;
void native_q6_k_to_rows(void* weights, int n_in, int n_out, void* stream);
// Q8_0 with each row's blocks split into two arrays: the row's qs (32 bytes a block), then its d (2).  A Q8_0 block is
// 34 bytes, so in GGUF order its qs sit on two-byte boundaries and the kernel loads them 16 bits at a time; split,
// every load is aligned (the B70 read the 2560 x 248320 output head at 150 GB/s in GGUF order).  The same lanes do
// the same arithmetic in the same order, so the results are bitwise those of GGUF-order Q8_0.  The engine's own ID,
// as kNativeQ6KRows; never written to a file.
constexpr int kNativeQ8Rows = kXeTypeBase + 2;
// Rewrites Q8_0 rows (n_in / 32 blocks each, an even count so every row starts 4-byte aligned) in place into the
// kNativeQ8Rows layout.  Throws on an odd block count; the caller then keeps GGUF order.
bool native_q8_0_rows_ok(int n_in) noexcept;
void native_q8_0_to_rows(void* weights, int n_in, int n_out, void* stream);
// b70-local: Q4_0 with each row's blocks split the same way: the row's qs (16 bytes of nibbles a block), then its d
// (2).  An 18-byte Q4_0 block leaves its qs on two-byte boundaries in GGUF order (the B70 read a 4608 x 6144 matrix at
// ~125 GB/s); split, every load is aligned.  The same lanes do the same arithmetic in the same order, so the results
// are bitwise those of GGUF-order Q4_0 (type 2).  The engine's own ID, as kNativeQ6KRows; never written to a file.
constexpr int kNativeQ4Rows = kXeTypeBase + 3;
// Rewrites Q4_0 rows (n_in / 32 blocks each, an even count so every row starts 4-byte aligned) in place into the
// kNativeQ4Rows layout.  Throws on an odd block count; the caller then keeps GGUF order.
bool native_q4_0_rows_ok(int n_in) noexcept;
void native_q4_0_to_rows(void* weights, int n_in, int n_out, void* stream);

} // namespace strata::kernels

namespace strata::kernels {
/// b70-local (q6k_xmx.cpp): STRATA_Q6K_XMX=1 - the decode's Q6_K rows products for 2-8 tokens on XMX.
bool native_q6k_xmx_enabled();
/// b70-local (q6k_xmx.cpp): STRATA_Q8_XMX=1 - the decode's kNativeQ8Rows projections for 2-8 tokens on XMX (FP16 tokens,
/// not bitwise native_mmvq's); x: T rows of n_in floats (not quantized), y: T rows of n_out floats.
/// b70-local (q8_dpas.cpp): STRATA_Q8_DPAS=N - the decode's kNativeQ8Rows projections of N or more tokens (1: 2 or more)
/// on int8 XMX (ESIMD DPAS, the tokens quantized as q8_1); x: T rows of n_in floats.  false (nothing launched): not this
/// shape, or no int8 XMX on the device (STRATA_NO_XMX=1 included) - the caller takes native_mmvq.
bool native_q8_dpas_enabled();
int native_q8_dpas_min_tokens();
bool native_q8_dpas_ok(int n_in, int n_out, int T) noexcept;
bool native_q8_rows_dpas(const void* weights, const float* x, float* y, int n_in, int n_out, int T, void* stream);
/// b70-local (q8_dpas.cpp v2): STRATA_Q8_DPAS2=N - kNativeQ8Rows on DPAS from the q8_1 rows (0: off), up to 6 matrices
/// sharing x in one launch (weights as B, tokens as A, T <= 8).
int native_q8_dpas2_min_tokens();
bool native_q8_rows_dpas_xq(int count, const void* const* w, float* const* y, const int* n_out, const void* xq, int n_in, int T, void* stream);
/// v3: the gemvx layout (weights as A, tokens as B, T <= 16) from the q8_1 rows; same contract as _xq
bool native_q8_rows_dpas_xq3(int count, const void* const* w, float* const* y, const int* n_out, const void* xq, int n_in, int T, void* stream);
bool native_q8_xmx_enabled();
bool native_q8_rows_xmx_ok(int n_in, int n_out, int T) noexcept;
void native_q8_rows_xmx(const void* weights, const float* x, float* y, int n_in, int n_out, int T, void* stream);
/// x (T rows of n floats) -> xh (T rows of FP16)
void native_f32_to_f16_rows(const float* x, void* xh, int n, int T, void* stream);
/// y (T rows of n_out) = kNativeQ6KRows weights x the FP16 rows xh (8 rows of n_in allocated, rows past T finite)
void native_q6k_rows_xmx(const void* weights, const void* xh, float* y, int n_in, int n_out, int T, void* stream);
}  // namespace strata::kernels
