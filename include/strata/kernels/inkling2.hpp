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

/// b70-local: a few tokens' (n_tokens <= 8) attention inputs in one launch, inkling2_conv of k and v (with the dynamic
/// taps kdyn / vdyn or null, the histories' snapshots as inkling2_conv's), the clips of q, k and r, v's RMS norm and the
/// cache store - the same arithmetic as those kernels one after another.  kr / vr [T][nkv D] in, q [T][H][D] and
/// r [T][H][rel] clipped in place.
void inkling2_attn_prep(const float* kr, const float* vr, float* conv_k, float* conv_v, const float* k_w,
                        const float* v_w, const float* kdyn, const float* vdyn, int n_taps, float conv_scale, float* q,
                        float* r, const float* q_clip, const float* r_clip, float qk_thr, float r_thr, float alpha,
                        float floor_, float eps, const int32_t* steps, float* kc, float* vc, int n_head, int n_head_kv,
                        int dim, int rel, int n_tokens, float* snap_k, float* snap_v, int64_t snap_stride, int n_snap,
                        void* stream);

/// b70-local: inkling2_attention and then out *= sigmoid(gate) (gate [T][H][D]; null: none).
void inkling2_attention_gated(const float* q, const float* r, const float* sinks, const float* kc, const float* vc,
                              const float* table, const int32_t* steps, int n_head, int n_head_kv, int dim, int rel,
                              int extent, int window, float* out, const float* gate, int n_tokens, void* stream);

/// x = cap tanh(x / cap) over `cols` of each of `rows` rows `ld` apart.
void inkling2_softcap(float* x, int64_t rows, int64_t cols, int64_t ld, float cap, void* stream);
/// b70-local (tensor-parallel slices): dst[t * dst_ld + col0 + i] = src[t * w + i] for t < rows, i < w; src may be
/// mapped host memory
void inkling2_put_cols(float* dst, int64_t dst_ld, int64_t col0, const float* src, int64_t w, int64_t rows, void* stream);
// b70-local TP: every GPU's column slice of one or two outputs (d1 may be null) in one launch: stage st's slice of width w
// goes to columns st*w..; this GPU's (own_st) from own0/own1 ([rows][w], null: already in place), the others' from rem
// (helper j = the j-th other stage: rem + j*rem_stride, d1's at + rem_off1)
void inkling2_tp_place(float* d0, float* d1, int64_t ld, int64_t w, int64_t rows, const float* own0, const float* own1,
                       int own_st, const float* rem, int64_t rem_stride, int64_t rem_off1, int n_st, void* stream,
                       const uint32_t* flag = nullptr, int fstride = 0, int n_flags = 0, uint32_t value = 0);
///< (b70-local: flag non-null: first waits in the kernel for flag[i * fstride] >= value, the n_flags flags)
/// b70-local: inkling2_tp_place for uneven cuts: stage st's columns [off[st], off[st] + w[st]) of the ld-wide rows;
/// own0 / own1 [rows][w[own_st]], a helper's slot at rem + j * rem_stride holding d0's [rows][w] then d1's
struct TpPlaceCut { int64_t off[4], w[4]; };
void inkling2_tp_place_cut(float* d0, float* d1, int64_t ld, const TpPlaceCut& cut, int n_st, int64_t rows,
                           const float* own0, const float* own1, int own_st, const float* rem, int64_t rem_stride,
                           void* stream, const uint32_t* flag = nullptr, int fstride = 0, int n_flags = 0,
                           uint32_t value = 0);

}  // namespace strata::kernels
