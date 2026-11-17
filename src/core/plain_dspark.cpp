// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/plain_dspark.cpp - b70-local: Inkling 2's DSpark drafter (STRATA_INK_SPEC=dspark, STRATA_INK_DSPARK=<gguf>
// from tools' mk_dspark_gguf.py) on the last stage's GPU.
//
// The drafter (llama.cpp PR #32218 graph_inkling2): the target's rows at its tap layers (the inputs of layers 5, 11,
// .., 41 and the final normed rows) are mixed per drafter layer (softmax of fuse_weight's row) and RMS-normed into a
// context row; its K / V (k_proj_ctx / v_proj_ctx, K normed and NeoX-rotated at the row's position) go into the
// layer's cache at that position (ingest, after every window: rejected rows are rewritten by the next window and never
// read).  A draft: the block [the pending token's normed embedding, 7 masks] at the pending position on, through 5
// pre-norm layers (q/k head norms, NeoX rope, attention over the context before the pending position and the whole
// block, SwiGLU), the final norm; then the head, row by row: a GRU-like state over [state | markov_w1[prev] | row |
// mixed context of the position before], whose correction (hidden_out) adds to the row before the target's output
// head; the first maximum is the draft and the next row's prev.
#include "strata/core/plain_model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/ink_dspark.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/xmx_gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::core {
namespace k = strata::kernels;

namespace {
constexpr int kBlock = 8, kRnnRing = 16;
// STRATA_DSPARK_PROF=1: the drafter's parts timed (a sync at each), printed every 32 drafts
bool ds_prof() { static const bool on = std::getenv("STRATA_DSPARK_PROF") != nullptr; return on; }
double ds_t[6] = {};
long long ds_n = 0;
}

bool PlainModel::dspark_load(std::string& err) {
    const char* path = std::getenv("STRATA_INK_DSPARK");
    if (path == nullptr || path[0] == '\0') { err = "plain model: STRATA_INK_SPEC=dspark needs STRATA_INK_DSPARK=<drafter gguf>"; return false; }
    GgufModel m = GgufModel::open(path);
    const auto& meta = m.meta();
    auto u = [&](const char* key) -> int64_t {
        const auto* v = meta.get(key);
        return v ? (int64_t) v->num() : -1;
    };
    auto f = [&](const char* key) -> double {
        const auto* v = meta.get(key);
        return v ? v->num() : 0.0;
    };
    const int64_t N = spec_.n_embd;
    if (u("dspark.embedding_length") != N || u("dspark.block_size") != kBlock) {
        err = std::string("plain model: ") + path + " is not an Inkling 2 DSpark drafter for this model";
        return false;
    }
    const int nl = (int) u("dspark.block_count");
    ds_H_ = (int) u("dspark.head_count");
    ds_KVH_ = (int) u("dspark.head_count_kv");
    ds_HD_ = (int) u("dspark.key_length");
    ds_FF_ = (int) u("dspark.feed_forward_length");
    ds_rank_ = (int) u("dspark.markov_rank");
    ds_swa_ = (int) u("dspark.sliding_window");
    ds_theta_ = (float) f("dspark.rope_freq_base");
    ds_eps_ = (float) f("dspark.rms_eps");
    ds_teps_ = (float) f("dspark.target_rms_eps");
    const int64_t local = u("dspark.local_layers");
    if (const auto* tl = meta.get("dspark.target_layers"); tl != nullptr) {
        tap_layers_.clear();
        for (const auto& it : tl->items) tap_layers_.push_back((int) (int64_t) it.u);
    }
    if (nl < 1 || ds_HD_ != 128 || ds_H_ % ds_KVH_ != 0 || tap_layers_.size() != 8 || ds_rank_ < 1) {
        err = "plain model: the DSpark drafter's shapes are not the ones this engine runs";
        return false;
    }
    // fuse_weight's rows, softmaxed here once: per layer and one for the head's context
    const TensorInfo* tf = nullptr;
    size_t at = 0;
    if ((tf = m.find("dspark.fuse", &at)) == nullptr || tf->elements() != (uint64_t) ((nl + 1) * 8)) {
        err = "plain model: the drafter's fuse weights";
        return false;
    }
    std::vector<float> mix((size_t) ((nl + 1) * 8));
    std::memcpy(mix.data(), m.shard(at).tensor_data(*tf), mix.size() * 4);
    for (int r = 0; r <= nl; ++r) {
        float* w = mix.data() + r * 8;
        const float mx = *std::max_element(w, w + 8);
        float s = 0.0f;
        for (int j = 0; j < 8; ++j) s += (w[j] = std::exp(w[j] - mx));
        for (int j = 0; j < 8; ++j) w[j] /= s;
    }
    void* dmix = nullptr;
    if (!strata::gpu::alloc_device(&dmix, mix.size() * 4) || !strata::gpu::copy(dmix, mix.data(), mix.size() * 4)) {
        err = "plain model: no VRAM for the drafter";
        return false;
    }
    allocs_.push_back(dmix);
    ds_wmix_ = (const float*) dmix;
    const uint64_t KV = (uint64_t) ds_KVH_ * ds_HD_, Q = (uint64_t) ds_H_ * ds_HD_;
    if (!upload_f32(m, "dspark.mask_embed", N, ds_mask_, err) || !upload_f32(m, "dspark.norm", N, ds_norm_, err) ||
        !upload_f32(m, "dspark.rnn_norm", N, ds_rnn_norm_, err) || !upload_f32(m, "dspark.tok_norm", N, ds_tok_norm_, err) ||
        !upload(m, "dspark.rnn_proj", ds_rnn_, err) || !upload(m, "dspark.hidden_out", ds_hout_, err) ||
        !upload_f32(m, "dspark.rnn_proj_b", 3 * (uint64_t) ds_rank_, ds_rnn_b_, err) ||
        !upload_f32(m, "dspark.hidden_out_b", N, ds_hout_b_, err))
        return false;
    {   // markov_w1: F16 [vocab][rank], read a row at a time
        const TensorInfo* t = m.find("dspark.markov_w1", &at);
        if (t == nullptr || t->type != 1 || t->shape.size() != 2 || t->shape[0] != (uint64_t) ds_rank_) {
            err = "plain model: the drafter's markov_w1";
            return false;
        }
        void* d = nullptr;
        const uint64_t bytes = t->elements() * 2;
        if (!strata::gpu::alloc_device(&d, bytes) || !strata::gpu::copy(d, m.shard(at).tensor_data(*t), bytes)) {
            err = "plain model: no VRAM for the drafter's markov_w1";
            return false;
        }
        allocs_.push_back(d);
        ds_w1_ = (const uint16_t*) d;
    }
    ds_layers_.assign((size_t) nl, {});
    const int64_t cells = opt_.max_cells;
    for (int l = 0; l < nl; ++l) {
        DSparkLayer& L = ds_layers_[(size_t) l];
        const std::string p = "dspark." + std::to_string(l) + ".";
        L.local = (local >> l) & 1;
        // a sliding layer's cache is a ring: the window and a prompt window's rows past it
        L.ring = L.local ? std::min<int64_t>(cells, (int64_t) ds_swa_ + 2 * opt_.max_t + 16) : cells;
        if (!upload_f32(m, p + "ctx_norm", N, L.ctx_norm, err) || !upload_f32(m, p + "attn_norm", N, L.attn_norm, err) ||
            !upload_f32(m, p + "ffn_norm", N, L.ffn_norm, err) || !upload_f32(m, p + "q_norm", ds_HD_, L.q_norm, err) ||
            !upload_f32(m, p + "k_norm", ds_HD_, L.k_norm, err) || !upload_f32(m, p + "k_ctx_norm", ds_HD_, L.k_ctx_norm, err) ||
            !upload(m, p + "q", L.q, err) || !upload(m, p + "k", L.k, err) || !upload(m, p + "v", L.v, err) ||
            !upload(m, p + "o", L.o, err) || !upload(m, p + "k_ctx", L.kctx, err) || !upload(m, p + "v_ctx", L.vctx, err) ||
            !upload(m, p + "gate", L.gate, err) || !upload(m, p + "up", L.up, err) || !upload(m, p + "down", L.down, err))
            return false;
        if ((uint64_t) L.q.n_out != Q || (uint64_t) L.k.n_out != KV || (uint64_t) L.kctx.n_out != KV || L.gate.n_out != ds_FF_) {
            err = "plain model: the drafter's layer " + std::to_string(l) + " has other shapes";
            return false;
        }
        if (!strata::gpu::alloc_device(&L.kc, (size_t) (L.ring * (int64_t) KV) * 4) ||
            !strata::gpu::alloc_device(&L.vc, (size_t) (L.ring * (int64_t) KV) * 4)) {
            err = "plain model: no VRAM for the drafter's cache";
            return false;
        }
        allocs_.push_back(L.kc);
        allocs_.push_back(L.vc);
    }
    // the work buffers: a prompt window's rows for the context, the block's 8 rows for the draft
    const int64_t TR = std::max<int64_t>(opt_.max_t, kBlock);
    auto dev = [&](float** p, uint64_t floats) {
        void* d = nullptr;
        if (!strata::gpu::alloc_device(&d, std::max<uint64_t>(floats * 4, 256))) return false;
        allocs_.push_back(d);
        *p = (float*) d;
        return true;
    };
    const uint64_t wide = std::max<uint64_t>({(uint64_t) N, Q, (uint64_t) ds_FF_, (uint64_t) (2 * ds_rank_ + 2 * N)});
    void* xq = nullptr;
    if (!dev(&ds_ctx_, TR * N) || !dev(&ds_k_, TR * KV) || !dev(&ds_v_, TR * KV) || !dev(&ds_x_, kBlock * N) ||
        !dev(&ds_h_, kBlock * N) || !dev(&ds_q_, kBlock * Q) || !dev(&ds_att_, kBlock * Q) ||
        !dev(&ds_g_, kBlock * (uint64_t) ds_FF_) || !dev(&ds_u_, kBlock * (uint64_t) ds_FF_) ||
        !dev(&ds_frnn_, kRnnRing * N) || !dev(&ds_state_, ds_rank_) || !dev(&ds_lat_, ds_rank_) ||
        !dev(&ds_joint_, 2 * ds_rank_ + 2 * N) || !dev(&ds_jout_, 3 * ds_rank_) || !dev(&ds_corr_, N) ||
        !dev(&ds_logits_, (uint64_t) head_.n_out) ||
        !strata::gpu::alloc_device(&xq, k::native_q8_1_bytes((int) wide, kBlock))) {
        err = "plain model: no VRAM for the drafter's buffers";
        return false;
    }
    allocs_.push_back(xq);
    ds_xq_ = xq;
    // mapped: the pending token's embedding row and the drafts
    if (!strata::gpu::alloc_host((void**) &h_anchor_, (size_t) N * 4) || !strata::gpu::device_pointer((void**) &m_anchor_, h_anchor_) ||
        !strata::gpu::alloc_host((void**) &h_dprev_, 64) || !strata::gpu::device_pointer((void**) &m_dprev_, h_dprev_) ||
        !strata::gpu::alloc_host((void**) &h_dprob_, 64) || !strata::gpu::device_pointer((void**) &m_dprob_, h_dprob_)) {
        err = "plain model: no mapped memory for the drafter";
        return false;
    }
    dspark_ = true;
    std::fprintf(stderr, "strata plain: DSpark drafter %s: %d layers, %zu taps, rank %d\n", path, nl, tap_layers_.size(), ds_rank_);
    return true;
}

// rows of x (n_in wide) through w, 8 rows a launch
void PlainModel::dspark_mm(const W& w, const float* x, float* y, int rows) {
    // Q8_0 rows (kNativeQ8Rows) on the matrix engines, 2-8 rows a launch, where the device has them
    static const bool xmx = k::xmx_available(k::XmxType::f16);
    // STRATA_DSPARK_XMX=1 only: on the B70 (11-18) its drafts were wrong (2.7% accepted against 80% with
    // native_mmvq), the cause not found yet
    static const bool xmx_on = [] { const char* v = std::getenv("STRATA_DSPARK_XMX"); return v != nullptr && v[0] == '1'; }();
    if (xmx_on && xmx && w.type == k::kNativeQ8Rows && k::native_q8_rows_xmx_ok(w.n_in, w.n_out, kBlock)) {
        for (int r0 = 0; r0 < rows; r0 += kBlock) {
            const int n = std::min(kBlock, rows - r0);
            if (n >= 2) {
                k::native_q8_rows_xmx(w.data, x + (size_t) r0 * w.n_in, y + (size_t) r0 * w.n_out, w.n_in, w.n_out, n, cs_);
            } else {
                k::native_quantize_q8_1(x + (size_t) r0 * w.n_in, ds_xq_, w.n_in, 1, cs_);
                k::native_mmvq(w.type, w.data, ds_xq_, y + (size_t) r0 * w.n_out, w.n_in, w.n_out, 1, cs_);
            }
        }
        return;
    }
    for (int r0 = 0; r0 < rows; r0 += kBlock) {
        const int n = std::min(kBlock, rows - r0);
        k::native_quantize_q8_1(x + (size_t) r0 * w.n_in, ds_xq_, w.n_in, n, cs_);
        k::native_mmvq(w.type, w.data, ds_xq_, y + (size_t) r0 * w.n_out, w.n_in, w.n_out, n, cs_);
    }
}

// the window's T rows at pos .. pos + T - 1 into every layer's context cache and the head's context ring (the taps,
// every one in i_taps_ by now: forward copied the earlier stages' slots here)
bool PlainModel::dspark_ingest(int T, int64_t pos, std::string& err) {
    const auto ti = std::chrono::steady_clock::now();
    const int64_t N = spec_.n_embd, KV = (int64_t) ds_KVH_ * ds_HD_, stride = (int64_t) opt_.max_t * N;
    for (size_t l = 0; l < ds_layers_.size(); ++l) {
        const DSparkLayer& L = ds_layers_[l];
        k::dspark_mix_norm(i_taps_, 8, stride, ds_wmix_ + l * 8, L.ctx_norm, ds_ctx_, N, T, ds_eps_, cs_);
        dspark_mm(L.kctx, ds_ctx_, ds_k_, T);
        dspark_mm(L.vctx, ds_ctx_, ds_v_, T);
        k::dspark_head_norm_rope(ds_k_, T, ds_KVH_, ds_HD_, L.k_ctx_norm, pos, ds_theta_, ds_eps_, cs_);
        k::dspark_store_kv(ds_k_, ds_v_, L.kc, L.vc, KV, T, pos, L.ring, cs_);
    }
    k::dspark_mix_norm(i_taps_, 8, stride, ds_wmix_ + ds_layers_.size() * 8, ds_rnn_norm_, ds_ctx_, N, T, ds_eps_, cs_);
    const int keep = std::min(T, kRnnRing);   // only the last rows matter for the ring
    k::dspark_store_kv(ds_ctx_ + (size_t) (T - keep) * N, ds_ctx_ + (size_t) (T - keep) * N, ds_frnn_, ds_frnn_, N, keep,
                       pos + T - keep, kRnnRing, cs_);
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the drafter's context: ") + strata::gpu::last_error();
        return false;
    }
    if (ds_prof() && T <= kBlock) ds_t[5] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count();
    return true;
}

// n drafts after the pending token cur_ at cur_pos_ into drafts_
bool PlainModel::dspark_draft(int n, std::string& err) {
    drafts_.clear();
    const int64_t a = cur_pos_, N = spec_.n_embd;
    if (n <= 0 || a < 1) return true;
    n = std::min(n, kBlock - 1);
    const PlainModel& r = *root_;
    if (!k::cpu::native_row_to_float(r.emb_type_, r.emb_ + (size_t) cur_ * r.emb_row_, h_anchor_, N)) {
        err = "plain model: the drafter's anchor embedding";
        return false;
    }
    using Clk = std::chrono::steady_clock;
    auto lap = [&](int i, Clk::time_point& t0) {
        if (!ds_prof()) return;
        strata::gpu::stream_sync(cs_);
        const auto t1 = Clk::now();
        ds_t[i] += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
    };
    auto tp = Clk::now();
    k::dspark_block_init(ds_x_, m_anchor_, ds_tok_norm_, ds_mask_, N, kBlock, ds_teps_, cs_);
    for (const DSparkLayer& L : ds_layers_) {
        k::native_qsa_rms_norm_weighted(ds_x_, L.attn_norm, ds_h_, (int) N, kBlock, ds_eps_, cs_);
        dspark_mm(L.q, ds_h_, ds_q_, kBlock);
        dspark_mm(L.k, ds_h_, ds_k_, kBlock);
        dspark_mm(L.v, ds_h_, ds_v_, kBlock);
        k::dspark_head_norm_rope(ds_q_, kBlock, ds_H_, ds_HD_, L.q_norm, a, ds_theta_, ds_eps_, cs_);
        k::dspark_head_norm_rope(ds_k_, kBlock, ds_KVH_, ds_HD_, L.k_norm, a, ds_theta_, ds_eps_, cs_);
        const int64_t lo = L.local ? std::max<int64_t>(0, a - ds_swa_) : 0;   // the window ends at the anchor
        lap(0, tp);
        k::dspark_attention(ds_q_, L.kc, L.vc, L.ring, lo, a, ds_k_, ds_v_, kBlock, ds_H_, ds_KVH_, ds_HD_, ds_att_, cs_);
        lap(1, tp);
        dspark_mm(L.o, ds_att_, ds_h_, kBlock);
        k::add_inplace(ds_x_, ds_h_, kBlock * N, cs_);
        k::native_qsa_rms_norm_weighted(ds_x_, L.ffn_norm, ds_h_, (int) N, kBlock, ds_eps_, cs_);
        dspark_mm(L.gate, ds_h_, ds_g_, kBlock);
        dspark_mm(L.up, ds_h_, ds_u_, kBlock);
        k::dspark_swiglu(ds_g_, ds_u_, ds_g_, (int64_t) kBlock * ds_FF_, cs_);
        dspark_mm(L.down, ds_g_, ds_h_, kBlock);
        k::add_inplace(ds_x_, ds_h_, kBlock * N, cs_);
        lap(2, tp);
    }
    k::native_qsa_rms_norm_weighted(ds_x_, ds_norm_, ds_h_, (int) N, kBlock, ds_eps_, cs_);
    // the head, row by row
    k::dspark_zero(ds_state_, ds_rank_, cs_);
    h_dprev_[0] = cur_;
    const float* ctx = ds_frnn_ + (size_t) ((a - 1) % kRnnRing) * N;
    k::SamplerParams sp;
    sp.greedy = true;
    sp.temperature = 0.0f;
    sp.top_p = 1.0f;
    for (int i = 1; i <= n; ++i) {
        k::dspark_joint(ds_joint_, ds_state_, ds_w1_, m_dprev_ + (i - 1), ds_h_ + (size_t) i * N, ctx, ds_rank_, N, cs_);
        dspark_mm(ds_rnn_, ds_joint_, ds_jout_, 1);
        k::add_inplace(ds_jout_, ds_rnn_b_, 3 * ds_rank_, cs_);
        k::dspark_gru(ds_jout_, ds_state_, ds_lat_, ds_rank_, cs_);
        dspark_mm(ds_hout_, ds_lat_, ds_corr_, 1);
        k::add_inplace(ds_corr_, ds_hout_b_, N, cs_);
        k::add_inplace(ds_corr_, ds_h_ + (size_t) i * N, N, cs_);   // the target's head over row + correction
        lap(3, tp);
        k::native_quantize_q8_1(ds_corr_, ds_xq_, head_.n_in, 1, cs_);
        k::native_mmvq(head_.type, head_.data, ds_xq_, ds_logits_, head_.n_in, head_.n_out, 1, cs_);
        k::sample_tokens(ds_logits_, 1, (int) n_vocab_, nullptr, 0, sp, m_dprev_ + i, cs_);
        k::dspark_top_prob(ds_logits_, n_vocab_, (float) spec_.final_softcap, m_dprob_ + i, cs_);
        lap(4, tp);
    }
    if (ds_prof() && ++ds_n % 32 == 0)
        std::fprintf(stderr, "strata dspark prof (%lld drafts, ms each): block proj %.2f, attention %.2f, o+ffn %.2f, "
                             "gru %.2f, head %.2f; ingest %.2f\n", ds_n, ds_t[0] / ds_n, ds_t[1] / ds_n, ds_t[2] / ds_n,
                     ds_t[3] / ds_n, ds_t[4] / ds_n, ds_t[5] / ds_n);
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the drafts: ") + strata::gpu::last_error();
        return false;
    }
    // STRATA_DSPARK_MIN_P (default 0.3): the drafts end before the first whose confidence is under it (a window then
    // checks only drafts likely to hold: on Japanese text half the drafts failed and the window's cost ate the gain)
    static const float min_p = [] { const char* v = std::getenv("STRATA_DSPARK_MIN_P"); return v ? (float) std::atof(v) : 0.3f; }();
    for (int i = 1; i <= n; ++i) {
        if (((volatile float*) h_dprob_)[i] < min_p) break;
        drafts_.push_back((int32_t) ((volatile int32_t*) h_dprev_)[i]);
    }
    return true;
}

}  // namespace strata::core
