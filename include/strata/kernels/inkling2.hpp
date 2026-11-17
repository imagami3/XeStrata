// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/inkling2.hpp - Inkling 2's kernels (b70-local): the attention residuals, the causal short
// convolutions, the clips, the attention with its relative bias and sinks, the gate and the softcap.
//
// The arithmetic follows llama.cpp's inkling2 graph (src/models/inkling2.cpp of PR #32217 at beb656e; MIT License,
// Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE), checked against transformers'
// modeling_inkling2.py.  Rows are [T][...] floats; `steps` is the window's step records (qsa.hpp, kStepPos first).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The attention residual (Inkling2BlockAttentionResidual): for each token t and query q, the states s = the
/// n_blocks block rows `blocks + b * block_stride` (each [T][N]) and `cur` [T][N], each RMS-normed without a weight
/// (eps 1e-6); out[q][t] = sum_s softmax_s(normed_s . query_q / sqrt(N)) normed_s.  queries [nq][N] F32;
/// out [nq][T][N].
// b70-local: rows t with raw[t] != 0 (a picture's, Inkling 2's vision) are copied from `src` into `x`: they enter the
// model as the encoder made them, without the token embedding's norm.  raw: mapped, read when the kernel runs.
void inkling2_raw_rows(float* x, const float* src, const int32_t* raw, int64_t n_embd, int n_tokens, void* stream);

void inkling2_attn_res(const float* blocks, int64_t block_stride, int n_blocks, const float* cur, const float* queries,
                       int nq, float* out, int64_t n_embd, int n_tokens, void* stream);

/// The causal short convolution over the window's tokens with `n_taps - 1` rows of history per channel in `state`
/// [C][n_taps - 1] (oldest first; updated): y_t = (x_t + sum_j w[c][j] win_t[j] (+ the dynamic taps)) * scale, with
/// win_t = x_{t - n_taps + 1} .. x_t.  w [C][n_taps] (ggml's ssm_conv layout).  `dyn` (or null): per token
/// [heads][n_taps] taps, tap lag j of head h multiplying x_{t - j} for the head's C / heads channels.  x and y may
/// not alias.
/// b70-local: `snap` (or null): the history after each of rows 0 .. n_snap - 1 as well, row t's at snap + t *
/// snap_stride (state's layout), for speculative decoding to go back to an accepted row.
void inkling2_conv(const float* x, float* state, const float* w, const float* dyn, int heads, float* y,
                   int64_t n_channels, int n_taps, float scale, int n_tokens, void* stream, float* snap = nullptr,
                   int64_t snap_stride = 0, int n_snap = 0);

/// Per head of `dim` values: x *= scale (per head and value, or null), then the L2 clip x *= min(1, thr / (|x| + 1e-6)),
/// then (alpha > 0) x *= 1 + alpha log(max(1, (pos + 1) / floor)) with the token's position from `steps`.
void inkling2_head_clip(float* x, const float* scale, float thr, float alpha, float floor_, const int32_t* steps,
                        int n_heads, int dim, int n_tokens, void* stream);

/// Per head of `dim` values: x * rsqrt(mean(x^2) + eps) (no weight).
void inkling2_head_rms(float* x, float eps, int n_heads, int dim, int n_tokens, void* stream);

/// The window's keys and values into the layer's cache at their positions (cell = position): k, v [T][nkv * D];
/// cache [cells][nkv * D] F32.
void inkling2_kv_store(const float* k, const float* v, const int32_t* steps, float* kc, float* vc, int64_t width,
                       int n_tokens, void* stream);

/// The attention: q [T][H][D] (clipped, scaled), r [T][H][R], sinks [T][H], the cache [cells][nkv][D], the bias table
/// [extent][R] (row d: offset d); score_j = q . k_j / sqrt(D) + (d < extent ? r . table[d] : 0) with d = pos - j,
/// over j <= pos and (window > 0) d < window; softmax with the sink's logit in the denominator; out [T][H][D].
void inkling2_attention(const float* q, const float* r, const float* sinks, const float* kc, const float* vc,
                        const float* table, const int32_t* steps, int n_head, int n_head_kv, int dim, int rel,
                        int extent, int window, float* out, int n_tokens, void* stream);

/// x *= sigmoid(g), n values.
void inkling2_sigmoid_mul(float* x, const float* g, int64_t n, void* stream);

/// x = cap tanh(x / cap) over `cols` of each of `rows` rows `ld` apart.
void inkling2_softcap(float* x, int64_t rows, int64_t cols, int64_t ld, float cap, void* stream);

}  // namespace strata::kernels
