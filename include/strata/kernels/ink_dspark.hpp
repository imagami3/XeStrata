// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// b70-local: the kernels of Inkling 2's DSpark drafter (mtp.safetensors; src/core/plain_dspark.cpp).  The arithmetic
// follows llama.cpp PR #32218 (src/models/dflash.cpp graph_inkling2, build_inkling2_head).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// out[t] = rms_norm(sum_j w[j] * taps[j * tap_stride + t * n]) * g, rows t < n_rows (the target's tap rows mixed
/// with one softmax row of fuse_weight, `w` on the device).
void dspark_mix_norm(const float* taps, int n_taps, int64_t tap_stride, const float* w, const float* g, float* out,
                     int64_t n, int n_rows, float eps, void* stream);

/// x [n_rows][heads][dim] in place: each head RMS-normed with g [dim], then NeoX-rotated at position pos0 + row.
void dspark_head_norm_rope(float* x, int n_rows, int heads, int dim, const float* g, int64_t pos0, float theta,
                           float eps, void* stream);

/// rows t < n_rows of k / v [n_rows][width] into the cache cells (pos0 + t) % ring.
void dspark_store_kv(const float* k, const float* v, float* kc, float* vc, int64_t width, int n_rows, int64_t pos0,
                     int64_t ring, void* stream);

/// The block's attention: query rows [n_q][heads][dim] over the context cells [lo, hi) (cell p at p % ring of kc / vc,
/// [ring][kv_heads][dim]) and then the block's own rows kb / vb [n_q][kv_heads][dim] (no mask: every row sees all);
/// out [n_q][heads][dim].
void dspark_attention(const float* q, const float* kc, const float* vc, int64_t ring, int64_t lo, int64_t hi,
                      const float* kb, const float* vb, int n_q, int heads, int kv_heads, int dim, float* out,
                      void* stream);

/// out = silu(g) * u, n values.
void dspark_swiglu(const float* g, const float* u, float* out, int64_t n, void* stream);

/// The block's input: row 0 = rms_norm(anchor) * tok_norm, rows 1 .. n_rows - 1 = mask.
void dspark_block_init(float* x, const float* anchor, const float* tok_norm, const float* mask, int64_t n, int n_rows,
                       float eps, void* stream);

/// The head's joint row [state (rank) | w1[*prev] (rank, F16 [vocab][rank]) | hidden (n) | context (n)].
void dspark_joint(float* joint, const float* state, const uint16_t* w1, const int32_t* prev, const float* hidden,
                  const float* context, int rank, int64_t n, void* stream);

/// j [3 rank]: gate = sigmoid(j0), cand = tanh(j1); state = cand + gate * (state - cand); latent = tanh(j2).
void dspark_gru(const float* j, float* state, float* latent, int rank, void* stream);

/// p[0] = the probability of logits' maximum under the softmax over n (one row, softcapped by cap > 0 first, as the
/// model's logits): the draft's confidence.
void dspark_top_prob(const float* logits, int64_t n, float cap, float* p, void* stream);

/// x = 0 (n floats).
void dspark_zero(float* x, int64_t n, void* stream);

}  // namespace strata::kernels
