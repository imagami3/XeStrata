// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/plain_inkling2.cpp - Inkling 2 (inkling2) in PlainModel (plain_model.hpp), b70-local.
//
// The block follows llama.cpp's inkling2 graph (src/models/inkling2.cpp of PR #32217 at beb656e; MIT License,
// Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE) step for step:
//
//   residuals: the inputs of layers 0, B, 2B, .. (B = attn_res.block_size) are kept as blocks; a layer's attention
//              input (after layer 0) and its MoE input, and the head's, are softmax mixes of the RMS-normed blocks and
//              the running row by learned queries (inkling2_attn_res).  A layer that starts a block keeps the
//              attention's output as the running row (no residual add); the others add it.
//   attention: q, k, v, r (the relative-bias query), the sink logit and the output gate from the normed inputs (a
//              full layer: q / k / v each its own norm and residual query); k and v through 4-tap causal convolutions
//              (a full layer's with per-head dynamic taps from the k / v inputs); q, k, r L2-clipped, v RMS-normed;
//              a full layer scales q and r by 1 + alpha log(max(1, (p + 1) / floor)); the scores add r . table[p - j]
//              and the softmax a per-head sink; sliding layers see the last `sliding_window` keys
//   output:    x sigmoid(gate), attn_output, a causal convolution
//   MoE:       the sigmoid router (bias for the choice, weights over their sum x expert_weights_scale), the routed
//              experts with SITU (expert_act: 4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25)), a causal convolution
//   head:      the mix by the output query, output_norm, output, 30 tanh(z / 30), the unpadded vocabulary
//
// Experts: blk.N.ffn_gate_exps / ffn_up_exps / ffn_down_exps (w/split_gu.py splits PR #32217's fused gate_up).
// The prompt runs in windows; no MTP drafting yet; K/V in F32, a cell a position.
#include "strata/core/plain_model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/inkling2.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/swiglu_clamp.hpp"
#include "strata/kernels/xmx_gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>

namespace strata::core {
namespace {

namespace k = strata::kernels;

constexpr int kBF16 = 30;

bool imapped(size_t bytes, void** h, void** d) {
    if (!strata::gpu::alloc_host(h, bytes)) return false;
    std::memset(*h, 0, bytes);
    return strata::gpu::device_pointer(d, *h);
}

}  // namespace

bool PlainModel::load_iblock(const GgufModel& m, int64_t l, IBlock& b, std::string& err) {
    const ModelSpec& s = spec_;
    const uint64_t N = (uint64_t) s.n_embd, E = (uint64_t) s.n_expert;
    const int64_t H = s.n_head, NKV = s.n_head_kv[(size_t) l], D = s.key_length, R = s.n_rel, KT = s.n_shortconv;
    const std::string p = "blk." + std::to_string(l) + ".";
    auto bad = [&](const char* what) {
        err = "plain model: block " + std::to_string(l) + "'s " + what + " have other shapes";
        return false;
    };
    // b70-local (11-17): a router cut to the kept experts (GB10's RVQ group layout, [N, n_kept]): its columns padded back to
    // E, the bias of the absent experts -1e30 so the choice never takes them (their order is the groups', as the experts')
    auto upload_router_padded = [&](const GgufModel& mm, const std::string& pp, uint64_t NN, uint64_t EE, const float*& w,
                                    const float*& bias, std::string& e2) {
        size_t at = 0, atb = 0;
        const TensorInfo* tw = mm.find(pp + "ffn_gate_inp.weight", &at);
        const TensorInfo* tb = mm.find(pp + "exp_probs_b.bias", &atb);
        if (tw == nullptr || tb == nullptr || tw->elements() == NN * EE)
            return upload_f32(mm, pp + "ffn_gate_inp.weight", NN * EE, w, e2) &&
                   upload_f32(mm, pp + "exp_probs_b.bias", EE, bias, e2);
        const uint64_t k = tb->elements();
        if (tw->type != 0 || tb->type != 0 || k == 0 || k > EE || tw->elements() != NN * k || !mm.in_bounds(*tw, at) ||
            !mm.in_bounds(*tb, atb)) {
            e2 = "plain model: " + pp + "ffn_gate_inp.weight / exp_probs_b.bias are not F32 [N, k] / [k] with k <= E";
            return false;
        }
        std::vector<float> hw((size_t) (NN * EE), 0.0f), hb((size_t) EE, -1e30f);
        std::memcpy(hw.data(), mm.shard(at).tensor_data(*tw), (size_t) (NN * k * 4));
        std::memcpy(hb.data(), mm.shard(atb).tensor_data(*tb), (size_t) (k * 4));
        void* dw = nullptr;
        void* db = nullptr;
        if (!strata::gpu::alloc_device(&dw, hw.size() * 4) || !strata::gpu::alloc_device(&db, hb.size() * 4)) {
            e2 = "plain model: no VRAM for " + pp + "'s router";
            return false;
        }
        allocs_.push_back(dw);
        allocs_.push_back(db);
        if (!strata::gpu::copy(dw, hw.data(), hw.size() * 4) || !strata::gpu::copy(db, hb.data(), hb.size() * 4)) {
            e2 = "plain model: uploading " + pp + "'s router failed";
            return false;
        }
        w = (const float*) dw;
        bias = (const float*) db;
        return true;
    };
    b.swa = s.swa[(size_t) l];
    b.extent = b.swa ? s.n_swa : s.rel_extent;
    b.n_aq = l == 0 ? 0 : (b.swa ? 1 : 3);
    if (!upload(m, p + "attn_q.weight", b.q, err) || !upload(m, p + "attn_k.weight", b.k, err) ||
        !upload(m, p + "attn_v.weight", b.v, err) || !upload(m, p + "attn_output.weight", b.o, err) ||
        !upload(m, p + "attn_gate.weight", b.gate, err) || !upload(m, p + "attn_rel.weight", b.rel, err) ||
        !upload(m, p + "attn_sink_proj.weight", b.sink, err) ||
        !upload_f32(m, p + "attn_q_clip.weight", (uint64_t) (H * D), b.q_clip, err) ||
        !upload_f32(m, p + "attn_rel_clip.weight", (uint64_t) (H * R), b.rel_clip, err) ||
        !upload_f32(m, p + "attn_rel_bias.weight", (uint64_t) (R * b.extent), b.rel_table, err) ||
        !upload_f32(m, p + "attn_k_conv.weight", (uint64_t) (NKV * D * KT), b.k_conv, err) ||
        !upload_f32(m, p + "attn_v_conv.weight", (uint64_t) (NKV * D * KT), b.v_conv, err) ||
        !upload_f32(m, p + "attn_post_conv.weight", N * (uint64_t) KT, b.attn_post_conv, err) ||
        !upload_f32(m, p + "ffn_post_conv.weight", N * (uint64_t) KT, b.ffn_post_conv, err) ||
        !upload_f32(m, p + "ffn_res_query.weight", N, b.ffn_rq, err) ||
        !upload_f32(m, p + "ffn_norm.weight", N, b.ffn_norm, err) ||
        !upload_router_padded(m, p, N, E, b.router, b.router_b, err))
        return false;
    if (b.swa) {
        if (!upload_f32(m, p + "attn_norm.weight", N, b.attn_norm, err)) return false;
    } else if (!upload_f32(m, p + "attn_q_inp_norm.weight", N, b.q_norm, err) ||
               !upload_f32(m, p + "attn_k_inp_norm.weight", N, b.k_norm, err) ||
               !upload_f32(m, p + "attn_v_inp_norm.weight", N, b.v_norm, err) ||
               !upload(m, p + "attn_k_conv_proj.weight", b.kproj, err) ||
               !upload(m, p + "attn_v_conv_proj.weight", b.vproj, err)) {
        return false;
    }
    if (b.n_aq > 0 && !upload_f32(m, p + "attn_res_query.weight", N * (uint64_t) b.n_aq, b.attn_rq, err)) return false;
    if ((uint64_t) b.q.n_in != N || b.q.n_out != H * D || (uint64_t) b.k.n_in != N || b.k.n_out != NKV * D ||
        (uint64_t) b.v.n_in != N || b.v.n_out != NKV * D || b.o.n_in != H * D || (uint64_t) b.o.n_out != N ||
        (uint64_t) b.gate.n_in != N || b.gate.n_out != H * D || (uint64_t) b.rel.n_in != N || b.rel.n_out != H * R ||
        (uint64_t) b.sink.n_in != N || b.sink.n_out != H)
        return bad("attention projections");
    if (!b.swa && (b.kproj.n_out != KT * NKV || b.vproj.n_out != KT * NKV)) return bad("dynamic convolution taps");
    for (const W* w : {&b.q, &b.k, &b.v, &b.o, &b.gate, &b.rel, &b.sink, &b.kproj, &b.vproj})
        if (w->data != nullptr && w->type == kBF16) {
            err = "plain model: a BF16 Inkling 2 projection is not run yet (quantize the dense weights)";
            return false;
        }
    return true;
}

bool PlainModel::load_inkling2(std::string& err) {
    const ModelSpec& s = spec_;
    ink_ = true;
    if (s.key_length != 128 || s.value_length != 128 || s.n_rel > 32 || s.n_shortconv > 8 || s.n_expert_used > 15 ||
        s.gating != ExpertGating::Sigmoid || s.n_embd % 256 != 0 || s.n_dense_lead != 0) {
        err = "plain model: an Inkling 2 shape this engine's kernels are not built for";
        return false;
    }
    for (int64_t l = 0; l < s.n_layer; ++l)
        if (s.n_head_kv[(size_t) l] != s.n_head_kv[0]) {
            err = "plain model: Inkling 2 with K/V heads that differ between layers";
            return false;
        }
    nkv_ = s.n_head_kv[0];
    model_mtp_ = false;
    has_mtp_ = false;
    opt_.prompt_chunk = 0;   // the prompt runs in windows (no batched path yet)
    opt_.spec = 0;
    cs_ = strata::gpu::stream_create();
    if (cs_ == nullptr) { err = "plain model: stream create failed"; return false; }
    k::expert_situ_set((float) s.situ_gate, (float) s.situ_up);   // a process setting, before any graph

    const GgufModel& m = *model_;
    {   // the embedding stays in the mapping: a token's row is dequantized on the host
        size_t at = 0;
        const TensorInfo* t = m.find("token_embd.weight", &at);
        if (t == nullptr || t->shape.size() != 2 || t->shape[0] != (uint64_t) s.n_embd || !m.in_bounds(*t, at)) {
            err = "plain model: token_embd.weight is missing or of another width";
            return false;
        }
        emb_type_ = (int) t->type;
        n_vocab_ = s.n_vocab_unpadded;   // the tokenizer's; the padded rows are never picked
        if ((int64_t) t->shape[1] < n_vocab_) { err = "plain model: token_embd.weight has too few rows"; return false; }
        emb_row_ = k::cpu::native_row_bytes(emb_type_, s.n_embd);
        if (emb_row_ == 0) { err = "plain model: token_embd.weight's type has no host dequantizer"; return false; }
        emb_ = m.shard(at).tensor_data(*t);
    }
    if (first() && !upload_f32(m, "token_embd_norm.weight", (uint64_t) s.n_embd, i_tok_norm_, err)) return false;
    if (last()) {   // the head: the last stage's, its unpadded rows
        if (!upload(m, "output.weight", head_, err) ||
            !upload_f32(m, "output_norm.weight", (uint64_t) s.n_embd, out_norm_, err) ||
            !upload_f32(m, "output_res_query.weight", (uint64_t) s.n_embd, i_out_rq_, err))
            return false;
        if (head_.n_in != s.n_embd || head_.n_out < n_vocab_ || head_.type == kBF16) {
            err = "plain model: output.weight's shape or type";
            return false;
        }
        head_.n_out = (int) n_vocab_;
    }
    iblocks_.resize((size_t) s.n_layer);
    for (int64_t l = lb_; l < le_; ++l)
        if (!load_iblock(m, l, iblocks_[(size_t) l], err)) return false;
    i_nblocks_ = (s.n_layer + s.attn_res_block - 1) / s.attn_res_block;

    // per-sequence state: per layer the K/V cache (a cell a position) and the four convolutions' histories
    const int64_t cells = opt_.max_cells, KVW = nkv_ * s.key_length, N = s.n_embd, KT1 = s.n_shortconv - 1;
    i_conv_floats_ = (2 * KVW + 2 * N) * KT1;
    for (int64_t l = lb_; l < le_; ++l) {
        IBlock& b = iblocks_[(size_t) l];
        if (!strata::gpu::alloc_device(&b.kc, (size_t) (cells * KVW) * 4) ||
            !strata::gpu::alloc_device(&b.vc, (size_t) (cells * KVW) * 4) ||
            !strata::gpu::alloc_device(&b.conv, (size_t) i_conv_floats_ * 4)) {
            err = "plain model: no VRAM for the K/V cache (" + std::to_string(cells) + " cells)";
            return false;
        }
        allocs_.push_back(b.kc);
        allocs_.push_back(b.vc);
        allocs_.push_back(b.conv);
    }

    // the window's buffers
    const uint64_t T = (uint64_t) opt_.max_t, UN = (uint64_t) N, E = (uint64_t) s.n_expert;
    const uint64_t K = (uint64_t) s.n_expert_used, NH = (uint64_t) s.n_head, HD = NH * (uint64_t) s.key_length;
    const uint64_t KV = (uint64_t) KVW, R = NH * (uint64_t) s.n_rel, TAPS = (uint64_t) (s.n_shortconv * nkv_);
    hand_w_ = (1 + i_nblocks_) * N;   // a stage hands on the running row and the blocks
    if (!imapped(T * UN * 4, (void**) &h_emb_, (void**) &m_emb_) ||
        !imapped(T * k::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) ||
        !imapped(T * (NH + (uint64_t) nkv_) * 4, (void**) &h_pos_, (void**) &m_pos_) ||
        !imapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) ||
        (!first() && !imapped(T * (uint64_t) hand_w_ * 4, (void**) &h_in_, (void**) &m_in_)) ||
        (!last() && !imapped(T * (uint64_t) hand_w_ * 4, (void**) &h_hand_, (void**) &m_hand_))) {
        err = "plain model: mapped staging allocation failed";
        return false;
    }
    // (a row at a time: native_q8_1_bytes takes at most 8 rows, and a prompt window may be up to 256)
    const size_t xq = std::max(k::native_q8_1_bytes((int) UN, 1), k::native_q8_1_bytes((int) HD, 1)) * (size_t) T;
    auto dev = [&](auto** p, uint64_t bytes) {
        void* d = nullptr;
        if (!strata::gpu::alloc_device(&d, (size_t) std::max<uint64_t>(bytes, 256))) return false;
        allocs_.push_back(d);
        *p = (std::remove_reference_t<decltype(*p)>) d;
        return true;
    };
    if (!dev(&step_, T * k::kStepCount * 4) || !dev(&x_, T * UN * 4) || !dev(&hn_, T * UN * 4) ||
        !dev(&mix_, T * UN * 4) || !dev(&g_xq_, xq) || !dev(&i_blocks_, (uint64_t) i_nblocks_ * T * UN * 4) ||
        !dev(&i_in_, 3 * T * UN * 4) || !dev(&i_hk_, T * UN * 4) || !dev(&i_hv_, T * UN * 4) ||
        !dev(&i_q_, T * HD * 4) || !dev(&i_kr_, T * KV * 4) || !dev(&i_vr_, T * KV * 4) || !dev(&i_k_, T * KV * 4) ||
        !dev(&i_v_, T * KV * 4) || !dev(&i_r_, T * R * 4) || !dev(&i_sink_, T * NH * 4) ||
        !dev(&i_gate_, T * HD * 4) || !dev(&i_kp_, T * TAPS * 4) || !dev(&i_vp_, T * TAPS * 4) ||
        !dev(&i_att_, T * HD * 4) || !dev(&i_o_, T * UN * 4) || !dev(&rlog_, T * E * 4) || !dev(&ids_, T * K * 4) ||
        !dev(&w_, T * K * 4) || !dev(&moe_out_, T * UN * 4) ||
        !dev(&logits_, (last() ? std::min<uint64_t>(T, 8) : 1) * (uint64_t) n_vocab_ * 4)) {   // a big window's head: one row
        err = "plain model: no VRAM for the window's buffers";
        return false;
    }
    xq_ = g_xq_;   // the head's q8_1 rows (head_logits)
    // b70-local: a prompt window's projections in FP16 (gproj in record_iblock): the rows and 8 zero rows past them
    if (const char* v = std::getenv("STRATA_INK_Q6K_XMX"); T > 8 && (v == nullptr || v[0] != '0')) {   // =0: off
        const uint64_t wide = std::max<uint64_t>(UN, HD);
        if ((!dev(&i_xh_, (T + 8) * wide * 2) || !strata::gpu::memset(i_xh_, 0, (size_t) ((T + 8) * wide * 2)))) {
            err = "plain model: no VRAM for the prompt window's FP16 rows";
            return false;
        }
    }
    if (!strata::gpu::stream_sync(cs_)) { err = "plain model: the uploads failed"; return false; }
    if (!open_experts(err)) return false;
    exec_.assign((size_t) opt_.max_t + 1, nullptr);
    segs_.assign((size_t) opt_.max_t + 1, {});
    mexec_.assign((size_t) opt_.max_t + 1, nullptr);
    msegs_.assign((size_t) opt_.max_t + 1, {});
    if (!reset_inkling2(err)) return false;
    if (root_ == this) {
        int64_t n_swa = 0;
        for (int64_t l = 0; l < s.n_layer; ++l) n_swa += s.swa[(size_t) l] ? 1 : 0;
        std::fprintf(stderr, "strata plain: Inkling 2, %lld sliding and %lld full layers, %lld blocks, %lld cells; the "
                             "prompt in windows\n", (long long) n_swa, (long long) (s.n_layer - n_swa),
                     (long long) i_nblocks_, (long long) cells);
    }
    return true;
}

bool PlainModel::reset_inkling2(std::string& err) {
    bool ok = true;
    for (int64_t l = lb_; l < le_; ++l) {
        const IBlock& b = iblocks_[(size_t) l];
        ok = ok && strata::gpu::memset(b.conv, 0, (size_t) i_conv_floats_ * 4);
    }
    cur_ = -1;
    drafts_.clear();
    if (!ok) err = "plain model: resetting the state failed";
    return ok;
}

bool PlainModel::record_iblock(const IBlock& b, int64_t l, int T, uint32_t& ring,
                               std::vector<strata::gpu::Graph*>& segs, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, H = s.n_head, NKV = nkv_, D = s.key_length, KVW = NKV * D, R = s.n_rel;
    const int64_t E = s.n_expert, K = s.n_expert_used, KT = s.n_shortconv, B = s.attn_res_block;
    const float eps = (float) s.rms_eps;
    const int64_t bstride = (int64_t) opt_.max_t * N;
    // b70-local: a window past 8 tokens (an Inkling prompt window) runs the 8-row kernels (the projections' MMVQ, the
    // router's GEMV) in slices of 8 rows; g_xq_ (the rows' q8_1) is pointed at each slice while it is recorded
    static const bool q8x_on = [] { const char* v = std::getenv("STRATA_INK_Q8_XMX"); return v == nullptr || v[0] != '0'; }();
    auto gproj = [&](const W& w, const float* x, float* y, int TT, bool quantized, strata::gpu::Stream c) {
        if (TT <= 8) { this->gproj(w, x, y, TT, quantized, c); return; }
        // Q6_K rows on the matrix engines (q6k_xmx.cpp: 8 tokens a launch at about one token's cost), the tokens in
        // FP16: i_xh_ holds the window's rows and 8 zero rows past them
        if (i_xh_ != nullptr && w.type == k::kNativeQ6KRows && w.n_in % 256 == 0 && w.n_out % 16 == 0 &&
            k::xmx_available(k::XmxType::f16)) {
            k::native_f32_to_f16_rows(x, i_xh_, (int) w.n_in, TT, c);
            for (int t0 = 0; t0 < TT; t0 += 8)
                k::native_q6k_rows_xmx(w.data, (const uint16_t*) i_xh_ + (size_t) t0 * w.n_in, y + (size_t) t0 * w.n_out,
                                       (int) w.n_in, (int) w.n_out, std::min(8, TT - t0), c);
            return;
        }
        // Q8_0 rows on the matrix engines likewise (q6k_xmx.cpp native_q8_rows_xmx, 2-8 tokens a launch, FP16 tokens)
        if (w.type == k::kNativeQ8Rows && k::xmx_available(k::XmxType::f16) && k::native_q8_rows_xmx_ok((int) w.n_in, (int) w.n_out, 8) &&
            q8x_on) {
            for (int t0 = 0; t0 < TT; t0 += 8) {
                const int tn = std::min(8, TT - t0);
                if (tn >= 2)
                    k::native_q8_rows_xmx(w.data, x + (size_t) t0 * w.n_in, y + (size_t) t0 * w.n_out, (int) w.n_in, (int) w.n_out, tn, c);
                else {
                    if (quantized) k::native_quantize_q8_1(x + (size_t) t0 * w.n_in, g_xq_, (int) w.n_in, 1, c);
                    this->gproj(w, x + (size_t) t0 * w.n_in, y + (size_t) t0 * w.n_out, 1, false, c);
                }
            }
            return;
        }
        uint8_t* const xq0 = (uint8_t*) g_xq_;
        const size_t qrow = k::native_q8_1_bytes((int) w.n_in, 1);
        for (int t0 = 0; t0 < TT; t0 += 8) {
            const int tn = std::min(8, TT - t0);
            g_xq_ = (decltype(g_xq_)) (xq0 + (quantized ? (size_t) t0 * qrow : 0));
            this->gproj(w, x + (size_t) t0 * (size_t) w.n_in, y + (size_t) t0 * (size_t) w.n_out, tn, quantized, c);
        }
        g_xq_ = (decltype(g_xq_)) xq0;
    };
    float* conv_k = b.conv;
    float* conv_v = conv_k + KVW * (KT - 1);
    float* conv_a = conv_v + KVW * (KT - 1);
    float* conv_f = conv_a + N * (KT - 1);

    // ---- the attention inputs: the residual mixes of the blocks before this layer and the running row
    const float* in[3] = {x_, x_, x_};
    if (b.n_aq > 0) {
        const int nb = (int) ((l - 1) / B + 1);
        k::inkling2_attn_res(i_blocks_, bstride, nb, x_, b.attn_rq, b.n_aq, i_in_, N, T, cs);
        for (int j = 0; j < 3; ++j) in[j] = i_in_ + (b.swa ? 0 : j) * (int64_t) T * N;
    }
    if (b.swa) {
        k::native_qsa_rms_norm_weighted_q8_1(in[0], b.attn_norm, hn_, g_xq_, (int) N, T, eps, cs);
        gproj(b.k, hn_, i_kr_, T, true, cs);
        gproj(b.v, hn_, i_vr_, T, true, cs);
    } else {
        // k and v (and their dynamic taps) from their own norms first, then q's (its q8_1 stays in g_xq_)
        k::native_qsa_rms_norm_weighted_q8_1(in[1], b.k_norm, i_hk_, g_xq_, (int) N, T, eps, cs);
        gproj(b.k, i_hk_, i_kr_, T, true, cs);
        gproj(b.kproj, i_hk_, i_kp_, T, true, cs);
        k::native_qsa_rms_norm_weighted_q8_1(in[2], b.v_norm, i_hv_, g_xq_, (int) N, T, eps, cs);
        gproj(b.v, i_hv_, i_vr_, T, true, cs);
        gproj(b.vproj, i_hv_, i_vp_, T, true, cs);
        k::native_qsa_rms_norm_weighted_q8_1(in[0], b.q_norm, hn_, g_xq_, (int) N, T, eps, cs);
    }
    gproj(b.q, hn_, i_q_, T, true, cs);
    gproj(b.rel, hn_, i_r_, T, true, cs);
    gproj(b.sink, hn_, i_sink_, T, true, cs);
    gproj(b.gate, hn_, i_gate_, T, true, cs);

    // ---- k and v through their convolutions, the clips, the cache, the attention
    const float sk = 1.0f / std::sqrt((float) ((b.swa ? 1 : 2) * KT + 1));
    k::inkling2_conv(i_kr_, conv_k, b.k_conv, b.swa ? nullptr : i_kp_, (int) NKV, i_k_, KVW, (int) KT, sk, T, cs);
    k::inkling2_conv(i_vr_, conv_v, b.v_conv, b.swa ? nullptr : i_vp_, (int) NKV, i_v_, KVW, (int) KT, sk, T, cs);
    const float alpha = b.swa ? 0.0f : (float) s.log_scale_alpha, fl = (float) s.log_scale_floor;
    k::inkling2_head_clip(i_q_, b.q_clip, (float) s.qk_clip, alpha, fl, step_, (int) H, (int) D, T, cs);
    k::inkling2_head_clip(i_k_, nullptr, (float) s.qk_clip, 0.0f, fl, step_, (int) NKV, (int) D, T, cs);
    k::inkling2_head_rms(i_v_, eps, (int) NKV, (int) D, T, cs);
    k::inkling2_head_clip(i_r_, b.rel_clip, (float) s.rel_clip, alpha, fl, step_, (int) H, (int) R, T, cs);
    k::inkling2_kv_store(i_k_, i_v_, step_, b.kc, b.vc, KVW, T, cs);
    k::inkling2_attention(i_q_, i_r_, i_sink_, b.kc, b.vc, b.rel_table, step_, (int) H, (int) NKV, (int) D, (int) R,
                          (int) b.extent, b.swa ? (int) s.n_swa : 0, i_att_, T, cs);
    k::inkling2_sigmoid_mul(i_att_, i_gate_, (int64_t) T * H * D, cs);
    gproj(b.o, i_att_, i_o_, T, false, cs);
    const float s1 = 1.0f / std::sqrt((float) (KT + 1));
    k::inkling2_conv(i_o_, conv_a, b.attn_post_conv, nullptr, 0, mix_, N, (int) KT, s1, T, cs);
    if (l % B == 0) {   // a block starts: the layer's input is kept, the attention's output is the running row
        k::copy_from_mapped(i_blocks_ + (l / B) * bstride, x_, (int64_t) T * N, cs);
        k::copy_from_mapped(x_, mix_, (int64_t) T * N, cs);
    } else {
        k::add_inplace(x_, mix_, (int64_t) T * N, cs);
    }

    // ---- the MoE: its residual mix, the norm, the router, the experts, the convolution
    k::inkling2_attn_res(i_blocks_, bstride, (int) (l / B + 1), x_, b.ffn_rq, 1, i_in_, N, T, cs);
    k::native_qsa_rms_norm_weighted_q8_1(i_in_, b.ffn_norm, hn_, g_xq_, (int) N, T, eps, cs);
    for (int t0 = 0; t0 < T; t0 += 8)
        k::f32_gemv_fp32_mmvf_multi(hn_ + (size_t) t0 * N, N, b.router, rlog_ + (size_t) t0 * E, E, N, E, std::min(8, T - t0), cs);
    const float scale = s.expert_weights_scale > 0 ? (float) s.expert_weights_scale : 1.0f;
    k::glm5_router_sigmoid(rlog_, b.router_b, T, (int) E, (int) K, true, scale, ids_, w_, cs);
    ++ring;
    win_.publish(ring, 0, 0, T, hn_, ids_, w_, cs);
    if (win_.segmented()) {   // a graph ends where the host must answer the ring
        strata::gpu::Graph* g = nullptr;
        if (!strata::gpu::end_capture(cs, &g)) {
            err = std::string("plain model: segment capture: ") + strata::gpu::last_error();
            return false;
        }
        segs.push_back(g);
        if (!strata::gpu::begin_capture(cs)) { err = "plain model: begin capture failed"; return false; }
    }
    if (!win_.experts(s.moe_index(l), ring, 0, 0, T, cs, err)) return false;
    k::native_moe_combine_multi(win_.parts(), w_, nullptr, moe_out_, N, K, T, cs);
    k::inkling2_conv(moe_out_, conv_f, b.ffn_post_conv, nullptr, 0, mix_, N, (int) KT, s1, T, cs);
    k::add_inplace(x_, mix_, (int64_t) T * N, cs);
    return true;
}

bool PlainModel::record_window_inkling2(int T, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd;
    const float eps = (float) s.rms_eps;
    const int64_t bstride = (int64_t) opt_.max_t * N;
    k::copy_i32_from_mapped(step_, m_step_, (int64_t) T * k::kStepCount, cs);
    if (first()) {   // the embeddings, RMS-normed with token_embd_norm
        k::copy_from_mapped(mix_, m_emb_, (int64_t) T * N, cs);
        k::native_qsa_rms_norm_weighted(mix_, i_tok_norm_, x_, (int) N, T, eps, cs);
    } else {         // the running row, then the blocks, as the stage before handed them on
        k::copy_from_mapped(x_, m_in_, (int64_t) T * N, cs);
        for (int64_t j = 0; j < i_nblocks_; ++j)
            k::copy_from_mapped(i_blocks_ + j * bstride, m_in_ + (1 + j) * (int64_t) T * N, (int64_t) T * N, cs);
    }
    uint32_t ring = 0;
    for (int64_t l = lb_; l < le_; ++l)
        if (!record_iblock(iblocks_[(size_t) l], l, T, ring, segs_[(size_t) T], cs, err)) return false;
    if (!last()) {
        k::copy_from_mapped(m_hand_, x_, (int64_t) T * N, cs);
        for (int64_t j = 0; j < i_nblocks_; ++j)
            k::copy_from_mapped(m_hand_ + (1 + j) * (int64_t) T * N, i_blocks_ + j * bstride, (int64_t) T * N, cs);
        return true;
    }
    k::inkling2_attn_res(i_blocks_, bstride, (int) i_nblocks_, x_, i_out_rq_, 1, i_in_, N, T, cs);
    k::native_qsa_rms_norm_weighted(i_in_, out_norm_, hn_, (int) N, T, eps, cs);
    if (T > 8) {   // b70-local: a prompt window: its last row's logits only, in that row's place (forward picks that row)
        float* lr = logits_;   // row 0 (the buffer holds 8 rows); forward samples it into the last token's place
        k::native_quantize_q8_1(hn_ + (size_t) (T - 1) * N, xq_, head_.n_in, 1, cs);
        k::native_mmvq(head_.type, head_.data, xq_, lr, head_.n_in, head_.n_out, 1, cs);
        k::inkling2_softcap(lr, 1, n_vocab_, n_vocab_, (float) s.final_softcap, cs);
        return true;
    }
    head_logits(hn_, T, cs);   // the tokens are picked after the graph (forward), with the run's sampler
    k::inkling2_softcap(logits_, T, n_vocab_, n_vocab_, (float) s.final_softcap, cs);
    return true;
}

}  // namespace strata::core
