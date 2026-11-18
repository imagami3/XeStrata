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
#include "strata/kernels/verify_kernels.hpp"
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
#include <initializer_list>
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
    // b70-local TP: q, gate and o as this GPU's slice of their rows
    const TpCut cq = tp_cut(l, H * D, false), co = tp_cut(l, (int64_t) N, true);
    const int ts = tp_ ? ep_stage_ : 0;
    const int64_t q0r = tp_ ? cq.off[ts] : 0, qs = tp_ ? cq.w[ts] : H * D, o0r = tp_ ? co.off[ts] : 0,
                  os = tp_ ? co.w[ts] : (int64_t) N;
    if (!upload(m, p + "attn_q.weight", b.q, err, false, q0r, tp_ ? qs : -1) || !upload(m, p + "attn_k.weight", b.k, err) ||
        !upload(m, p + "attn_v.weight", b.v, err) ||
        !upload(m, p + "attn_output.weight", b.o, err, false, o0r, tp_ ? os : -1) ||
        !upload(m, p + "attn_gate.weight", b.gate, err, false, q0r, tp_ ? qs : -1) ||
        !upload(m, p + "attn_rel.weight", b.rel, err) ||
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
    // b70-local: the router read as BF16 (half the bytes; the checkpoint's router is BF16, so the values are the same)
    // when every F32 value is a BF16 value (STRATA_INK_ROUTER_BF16=0: F32)
    if (const char* rb = std::getenv("STRATA_INK_ROUTER_BF16"); rb == nullptr || rb[0] != '0') {
        const uint64_t n = N * E;
        std::vector<uint32_t> f((size_t) n);
        std::vector<uint16_t> h((size_t) n);
        bool exact = strata::gpu::copy(f.data(), b.router, n * 4);
        for (uint64_t i = 0; i < n && exact; ++i) {
            exact = (f[(size_t) i] & 0xFFFFu) == 0;
            h[(size_t) i] = (uint16_t) (f[(size_t) i] >> 16);
        }
        void* d = nullptr;
        if (exact && strata::gpu::alloc_device(&d, n * 2)) {
            allocs_.push_back(d);
            if (!strata::gpu::copy(d, h.data(), n * 2)) { err = "plain model: uploading the BF16 router failed"; return false; }
            b.router16 = (const uint16_t*) d;
            // the F32 copy freed: every window reads the BF16 one (7 MB a layer, the drafter's head's room)
            void* f32 = const_cast<float*>(b.router);
            const auto it = std::find(allocs_.begin(), allocs_.end(), f32);
            if (it != allocs_.end()) {
                allocs_.erase(it);
                strata::gpu::free(f32);
                b.router = nullptr;
            }
        } else if (!exact && l == 0) {
            std::fprintf(stderr, "[strata] %sffn_gate_inp.weight is not exact in BF16: the router stays F32\n", p.c_str());
        }
    }
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
    if ((uint64_t) b.q.n_in != N || b.q.n_out != qs || (uint64_t) b.k.n_in != N || b.k.n_out != NKV * D ||
        (uint64_t) b.v.n_in != N || b.v.n_out != NKV * D || b.o.n_in != H * D || b.o.n_out != os ||
        (uint64_t) b.gate.n_in != N || b.gate.n_out != qs || (uint64_t) b.rel.n_in != N || b.rel.n_out != H * R ||
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
    if (const char* v = std::getenv("STRATA_INK_SPEC"); v == nullptr || v[0] == '\0' || v[0] == '0')
        opt_.spec = 0;   // no MTP layer: speculative decoding only with STRATA_INK_SPEC (b70-local)
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
    if (tp_) {   // b70-local TP: this GPU's slices of the other stages' layers too
        tpw_.resize((size_t) s.n_layer);
        for (int64_t l = 0; l < s.n_layer; ++l) {
            TPW& t = tpw_[(size_t) l];
            if (l >= lb_ && l < le_) {
                const IBlock& b = iblocks_[(size_t) l];
                t.q = b.q;
                t.gate = b.gate;
                t.o = b.o;
                continue;
            }
            const std::string p = "blk." + std::to_string(l) + ".";
            const TpCut cq = tp_cut(l, s.n_head * s.key_length, false), co = tp_cut(l, s.n_embd, true);
            const int64_t q0r = cq.off[ep_stage_], qs = cq.w[ep_stage_], o0r = co.off[ep_stage_], os = co.w[ep_stage_];
            if (!upload(m, p + "attn_q.weight", t.q, err, false, q0r, qs) ||
                !upload(m, p + "attn_gate.weight", t.gate, err, false, q0r, qs) ||
                !upload(m, p + "attn_output.weight", t.o, err, false, o0r, os))
                return false;
        }
    }
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
    // b70-local: speculative decoding (STRATA_INK_SPEC): the histories after a window's rows but its last, every stage's
    if (const char* v = std::getenv("STRATA_INK_SPEC"); v != nullptr && v[0] != '\0' && v[0] != '0' && opt_.spec > 1) {
        opt_.spec = std::min(opt_.spec, std::min(opt_.max_t, 8));
        n_snap_ = opt_.spec - 1;
        ink_spec_ = last();
        for (int64_t l = lb_; l < le_; ++l) {
            IBlock& b = iblocks_[(size_t) l];
            if (!strata::gpu::alloc_device(&b.conv_snap, (size_t) (n_snap_ * i_conv_floats_) * 4)) {
                err = "plain model: no VRAM for the convolutions' copies (STRATA_INK_SPEC)";
                return false;
            }
            allocs_.push_back(b.conv_snap);
        }
    }

    // the window's buffers
    const uint64_t T = (uint64_t) opt_.max_t, UN = (uint64_t) N, E = (uint64_t) s.n_expert;
    const uint64_t K = (uint64_t) s.n_expert_used, NH = (uint64_t) s.n_head, HD = NH * (uint64_t) s.key_length;
    const uint64_t KV = (uint64_t) KVW, R = NH * (uint64_t) s.n_rel, TAPS = (uint64_t) (s.n_shortconv * nkv_);
    hand_w_ = (1 + i_nblocks_) * N;   // a stage hands on the running row and the blocks
    if (!imapped(T * UN * 4, (void**) &h_emb_, (void**) &m_emb_) || !imapped(T * 4, (void**) &h_raw_, (void**) &m_raw_) ||
        !imapped(T * k::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) ||
        !imapped(T * (NH + (uint64_t) nkv_) * 4, (void**) &h_pos_, (void**) &m_pos_) ||
        !imapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) ||
        (!first() && !imapped(T * (uint64_t) hand_w_ * 4, (void**) &h_in_, (void**) &m_in_)) ||
        (!last() && !imapped(T * (uint64_t) hand_w_ * 4, (void**) &h_hand_, (void**) &m_hand_))) {
        err = "plain model: mapped staging allocation failed";
        return false;
    }
    // b70-local EP: the relay's rows (an owner's: the other GPUs' sums; a helper's: the owner's rows in, its sum out)
    if (tp_) {   // the widest q / gate slice (every layer's cuts have the same widths, by owner or helper)
        const TpCut c = tp_cut(0, (int64_t) HD, false), co = tp_cut(0, (int64_t) UN, true);
        tp_wmax_qg_ = 0;
        for (int st = 0; st < ep_n_; ++st) tp_wmax_qg_ = std::max({tp_wmax_qg_, c.w[st], (co.w[st] + 1) / 2});
        const bool even = tp_wmax_qg_ * ep_n_ == (int64_t) HD && c.w[0] == c.w[ep_n_ - 1] && co.w[0] == co.w[ep_n_ - 1];
        const char* v = std::getenv("STRATA_INK_EP_DIRECT");
        if (!even && v != nullptr && v[0] == '0') {
            err = "plain model: uneven TP cuts (STRATA_INK_TP_QG / _O) need the direct relay";
            return false;
        }
    }
    const uint64_t TQS = tp_ ? 2 * (uint64_t) tp_wmax_qg_ : 0, AXW = tp_ ? std::max(UN, HD) : UN,
                   AYW = std::max(UN, TQS);   // TP: a helper's input (o's: H x D) and its two q / gate slices out
    if (ep_ && (!imapped((uint64_t) (ep_n_ - 1) * T * UN * 4, (void**) &h_rem_, (void**) &m_rem_) ||
                !imapped(T * AXW * 4, (void**) &h_ax_, (void**) &m_ax_) ||
                !imapped(T * K * 4, (void**) &h_aids_, (void**) &m_aids_) ||
                !imapped(T * K * 4, (void**) &h_aw_, (void**) &m_aw_) ||
                !imapped(T * AYW * 4, (void**) &h_ay_, (void**) &m_ay_) ||
                !imapped(64, (void**) &h_ain_, (void**) &m_ain_) || !imapped(64, (void**) &h_aout_, (void**) &m_aout_))) {
        err = "plain model: the expert relay's mapped buffers";
        return false;
    }
    if (tp_ && (!imapped(T * AXW * 4, (void**) &h_tx_, (void**) &m_tx_) ||
                !imapped(64, (void**) &h_tseq_, (void**) &m_tseq_) || !imapped(64, (void**) &h_tflag_, (void**) &m_tflag_) ||
                !imapped((uint64_t) (ep_n_ - 1) * T * TQS * 4, (void**) &h_trem_, (void**) &m_trem_))) {
        err = "plain model: the TP relay's mapped buffers";
        return false;
    }
    if (ep_) {   // b70-local EP direct: the region (STRATA_INK_EP_DIRECT=0: the host relays)
        const char* v = std::getenv("STRATA_INK_EP_DIRECT");
        ep_direct_ = v == nullptr || v[0] != '0';
    }
    if (ep_ && ep_direct_) {
        size_t off = 4096;
        auto take = [&](uint64_t bytes) { const size_t o = off; off += (size_t) ((bytes + 255) / 256 * 256); return o; };
        const size_t o_tx = take(T * AXW * 4), o_ex = take(T * UN * 4), o_ids = take(T * K * 4), o_w = take(T * K * 4),
                     o_trem = take((uint64_t) (ep_n_ - 1) * T * std::max<uint64_t>(TQS, 1) * 4),
                     o_rem = take((uint64_t) (ep_n_ - 1) * T * UN * 4);
        ep_reg_bytes_ = (off + 65535) / 65536 * 65536;
        void* dm = nullptr;
        if (!imapped(ep_reg_bytes_, &ep_reg_, &dm) || dm != ep_reg_) {
            err = "plain model: the direct relay's region";
            return false;
        }
        uint8_t* b8 = (uint8_t*) ep_reg_;
        epr_.tseq = (uint32_t*) b8;
        epr_.eseq = (uint32_t*) (b8 + 64);
        epr_.flag = (uint32_t*) (b8 + 128);
        epr_.tx = (float*) (b8 + o_tx);
        epr_.ex = (float*) (b8 + o_ex);
        epr_.eids = (int32_t*) (b8 + o_ids);
        epr_.ew = (float*) (b8 + o_w);
        epr_.trem = (float*) (b8 + o_trem);
        epr_.rem = (float*) (b8 + o_rem);
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
        !dev(&w_, T * K * 4) || !dev(&moe_out_, T * UN * 4) || (tp_ && !dev(&i_tps_, T * TQS * 4)) ||
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
    const char* spec_kind = std::getenv("STRATA_INK_SPEC");
    const bool want_dspark = spec_kind != nullptr && std::strcmp(spec_kind, "dspark") == 0 && opt_.spec > 1;
    if (std::getenv("STRATA_DUMP_TAPS") != nullptr || want_dspark) {   // b70-local: the drafter's taps (forward writes them)
        const char* v = std::getenv("STRATA_TAP_LAYERS");
        const std::string list = v ? v : "5,11,17,23,29,35,41,48";
        for (size_t at = 0; at < list.size();) {
            tap_layers_.push_back(std::atoi(list.c_str() + at));
            const size_t c = list.find(',', at);
            if (c == std::string::npos) break;
            at = c + 1;
        }
        if (!dev(&i_taps_, (uint64_t) tap_layers_.size() * T * UN * 4)) { err = "plain model: no VRAM for the taps"; return false; }
    }
    if (want_dspark && last() && !dspark_load(err)) return false;   // before the expert tier takes the VRAM
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

void glm5_decode_stamp(const void* model, int64_t l, int i, strata::gpu::Stream cs);

// b70-local: a window's projection through w: MMVQ up to 8 rows, a prompt window's rows on the matrix engines or in
// slices of 8 (g_xq_ holds the rows' q8_1 when `quantized`)
void PlainModel::gproj_win(const W& w, const float* x, float* y, int TT, bool quantized, strata::gpu::Stream c) {
    static const bool q8x_on = [] { const char* v = std::getenv("STRATA_INK_Q8_XMX"); return v == nullptr || v[0] != '0'; }();
    {
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
    }
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
    auto gproj = [&](const W& w, const float* x, float* y, int TT, bool quantized, strata::gpu::Stream c) {
        gproj_win(w, x, y, TT, quantized, c);
    };
    float* conv_k = b.conv;
    float* conv_v = conv_k + KVW * (KT - 1);
    float* conv_a = conv_v + KVW * (KT - 1);
    float* conv_f = conv_a + N * (KT - 1);
    // b70-local: the histories after rows 0 .. ns - 1 (speculative decoding), at the same offsets in each snapshot row
    const int ns = b.conv_snap != nullptr ? std::min(T - 1, n_snap_) : 0;
    const int64_t sst = i_conv_floats_;
    float* snap_k = ns > 0 ? b.conv_snap : nullptr;
    float* snap_v = ns > 0 ? b.conv_snap + (conv_v - b.conv) : nullptr;
    float* snap_a = ns > 0 ? b.conv_snap + (conv_a - b.conv) : nullptr;
    float* snap_f = ns > 0 ? b.conv_snap + (conv_f - b.conv) : nullptr;

    // ---- the attention inputs: the residual mixes of the blocks before this layer and the running row
    glm5_decode_stamp(this, l, 0, cs);
    if (const int j = tap_slot(l); j >= 0 && i_taps_ != nullptr)   // b70-local: the layer's input, a drafter tap
        k::copy_from_mapped(i_taps_ + (size_t) j * opt_.max_t * N, x_, (int64_t) T * N, cs);
    const float* in[3] = {x_, x_, x_};
    if (b.n_aq > 0) {
        const int nb = (int) ((l - 1) / B + 1);
        k::inkling2_attn_res(i_blocks_, bstride, nb, x_, b.attn_rq, b.n_aq, i_in_, N, T, cs);
        for (int j = 0; j < 3; ++j) in[j] = i_in_ + (b.swa ? 0 : j) * (int64_t) T * N;
    }
    // b70-local: a few tokens' projections that read the same q8_1 rows (g_xq_) in one launch (native_mmvq_multi,
    // each row's arithmetic native_mmvq's; STRATA_MMVQ_MULTI=0: one by one)
    auto gmulti = [&](std::initializer_list<std::pair<const W*, float*>> ps, const float* x) {
        if (T <= 8 && ps.size() <= 8) {
            const void* ws[8];
            float* ys[8];
            int no[8], c = 0;
            const W& w0 = *ps.begin()->first;
            bool same = true;
            for (const auto& [w, y] : ps) {
                same = same && w->type == w0.type && w->n_in == w0.n_in && w->data != nullptr;
                ws[c] = w->data; ys[c] = y; no[c++] = (int) w->n_out;
            }
            if (same && k::native_mmvq_multi(w0.type, c, ws, ys, no, g_xq_, (int) w0.n_in, T, cs)) return;
        }
        for (const auto& [w, y] : ps) gproj(*w, x, y, T, true, cs);
    };
    // b70-local TP: this GPU's rows of q and gate (or o) and the other GPUs', relayed, each in its columns
    const int64_t tqs = tp_ ? H * D / ep_n_ : 0, tos = tp_ ? (int64_t) N / ep_n_ : 0;
    const bool epd = ep_direct_;
    float* const tp_rem = epd ? epr_.trem : m_trem_;
    auto tp_others = [&](float* dst, int64_t ld, int64_t w, int64_t off) {   // the relay's slices of every other GPU
        int j = 0;
        for (int st = 0; st < ep_n_; ++st) {
            if (st == ep_stage_) continue;
            k::inkling2_put_cols(dst, ld, st * w, tp_rem + (size_t) j * T * 2 * tqs + (size_t) off, w, T, cs);
            ++j;
        }
    };
    // published first (the other GPUs start while this one runs its other projections), finished after them
    auto tp_pub = [&](const float* xin) {
        k::doorbell_publish(xin, ids_, w_, (int64_t) T * N, 0, epd ? epr_.tx : m_tx_, nullptr, nullptr,
                            epd ? epr_.tseq : m_tseq_, cs);
    };
    // b70-local: EP direct: the waits inside the kernels that read what they wait for (STRATA_INK_EP_FUSEWAIT=0: apart)
    static const bool fw_env = [] { const char* v = std::getenv("STRATA_INK_EP_FUSEWAIT"); return v == nullptr || v[0] != '0'; }();
    const bool fw = epd && fw_env;
    auto tp_wait = [&] {   // the other GPUs' slices of this phase there
        if (epd) k::wait_flags_ge(epr_.flag, kEpFlagStride, ep_n_ - 1, ++ep_hc_, cs);
        else k::wait_flag_ge(m_tflag_, ++tp_ring_, cs);
    };
    // b70-local: one kernel places the slices (STRATA_INK_TP_PLACE=0: one put_cols a slice); one token: this GPU's
    // rows straight into their columns
    static const bool tp_place = [] { const char* v = std::getenv("STRATA_INK_TP_PLACE"); return v == nullptr || v[0] != '0'; }();
    // b70-local: the direct relay's slots: [T][2 x the widest slice]; the cuts may be uneven (tp_cut)
    const int64_t slot = (int64_t) T * 2 * tp_wmax_qg_;
    auto pcut = [&](const TpCut& c) {
        k::TpPlaceCut p{};
        for (int st = 0; st < 4; ++st) { p.off[st] = c.off[st]; p.w[st] = c.w[st]; }
        return p;
    };
    auto tp_qg = [&](const float* xin, bool quantized) {
        if (tp_place && epd) {
            const TpCut c = tp_cut(l, H * D, false);
            const int64_t wq = c.w[ep_stage_], oq = c.off[ep_stage_];
            const bool direct = T == 1;
            float* q0 = direct ? i_q_ + (size_t) oq : i_tps_;
            float* g0 = direct ? i_gate_ + (size_t) oq : i_tps_ + (size_t) T * wq;
            gproj(b.q, xin, q0, T, quantized, cs);
            gproj(b.gate, xin, g0, T, true, cs);
            glm5_decode_stamp(this, l, 12, cs);
            if (!fw) {
                tp_wait();
                glm5_decode_stamp(this, l, 13, cs);
            }
            k::inkling2_tp_place_cut(i_q_, i_gate_, H * D, pcut(c), ep_n_, T, direct ? nullptr : q0, direct ? nullptr : g0,
                                     ep_stage_, epr_.trem, slot, cs, fw ? epr_.flag : nullptr, kEpFlagStride, ep_n_ - 1,
                                     fw ? ++ep_hc_ : 0);
            return;
        }
        if (tp_place) {
            const bool direct = T == 1;
            float* q0 = direct ? i_q_ + (size_t) ep_stage_ * tqs : i_tps_;
            float* g0 = direct ? i_gate_ + (size_t) ep_stage_ * tqs : i_tps_ + (size_t) T * tqs;
            gproj(b.q, xin, q0, T, quantized, cs);
            gproj(b.gate, xin, g0, T, true, cs);
            glm5_decode_stamp(this, l, 12, cs);
            if (fw) {
                k::inkling2_tp_place(i_q_, i_gate_, H * D, tqs, T, direct ? nullptr : q0, direct ? nullptr : g0, ep_stage_,
                                     tp_rem, (int64_t) T * 2 * tqs, (int64_t) T * tqs, ep_n_, cs, epr_.flag, kEpFlagStride,
                                     ep_n_ - 1, ++ep_hc_);
                return;
            }
            tp_wait();
            glm5_decode_stamp(this, l, 13, cs);
            k::inkling2_tp_place(i_q_, i_gate_, H * D, tqs, T, direct ? nullptr : q0, direct ? nullptr : g0, ep_stage_,
                                 tp_rem, (int64_t) T * 2 * tqs, (int64_t) T * tqs, ep_n_, cs);
            return;
        }
        gproj(b.q, xin, i_tps_, T, quantized, cs);
        gproj(b.gate, xin, i_tps_ + (size_t) T * tqs, T, true, cs);
        k::inkling2_put_cols(i_q_, H * D, ep_stage_ * tqs, i_tps_, tqs, T, cs);
        k::inkling2_put_cols(i_gate_, H * D, ep_stage_ * tqs, i_tps_ + (size_t) T * tqs, tqs, T, cs);
        tp_wait();
        tp_others(i_q_, H * D, tqs, 0);
        tp_others(i_gate_, H * D, tqs, (int64_t) T * tqs);
    };
    if (b.swa) {
        k::native_qsa_rms_norm_weighted_q8_1(in[0], b.attn_norm, hn_, g_xq_, (int) N, T, eps, cs);
        if (tp_) {
            tp_pub(hn_);
            gmulti({{&b.k, i_kr_}, {&b.v, i_vr_}, {&b.rel, i_r_}, {&b.sink, i_sink_}}, hn_);
            tp_qg(hn_, true);
        } else
        gmulti({{&b.k, i_kr_}, {&b.v, i_vr_}, {&b.q, i_q_}, {&b.rel, i_r_}, {&b.sink, i_sink_}, {&b.gate, i_gate_}}, hn_);
    } else {
        // k and v (and their dynamic taps) from their own norms first, then q's (its q8_1 stays in g_xq_); TP: q's norm
        // first and published, its q8_1 made again after k's and v's
        if (tp_) {
            k::native_qsa_rms_norm_weighted_q8_1(in[0], b.q_norm, hn_, g_xq_, (int) N, T, eps, cs);
            tp_pub(hn_);
        }
        k::native_qsa_rms_norm_weighted_q8_1(in[1], b.k_norm, i_hk_, g_xq_, (int) N, T, eps, cs);
        gmulti({{&b.k, i_kr_}, {&b.kproj, i_kp_}}, i_hk_);
        k::native_qsa_rms_norm_weighted_q8_1(in[2], b.v_norm, i_hv_, g_xq_, (int) N, T, eps, cs);
        gmulti({{&b.v, i_vr_}, {&b.vproj, i_vp_}}, i_hv_);
        if (tp_) {
            const size_t qrow = k::native_q8_1_bytes((int) N, 1);   // the q8_1 of q's rows again, 8 rows a launch
            for (int t0 = 0; t0 < T; t0 += 8)
                k::native_quantize_q8_1(hn_ + (size_t) t0 * N, (uint8_t*) g_xq_ + (size_t) t0 * qrow, (int) N,
                                        std::min(8, T - t0), cs);
            gmulti({{&b.rel, i_r_}, {&b.sink, i_sink_}}, hn_);
            tp_qg(hn_, true);
        } else {
            k::native_qsa_rms_norm_weighted_q8_1(in[0], b.q_norm, hn_, g_xq_, (int) N, T, eps, cs);
        gmulti({{&b.q, i_q_}, {&b.rel, i_r_}, {&b.sink, i_sink_}, {&b.gate, i_gate_}}, hn_);
        }
    }
    glm5_decode_stamp(this, l, 1, cs);

    // ---- k and v through their convolutions, the clips, the cache, the attention
    const float sk = 1.0f / std::sqrt((float) ((b.swa ? 1 : 2) * KT + 1));
    const float alpha = b.swa ? 0.0f : (float) s.log_scale_alpha, fl = (float) s.log_scale_floor;
    // b70-local: a few tokens' inputs in one launch (STRATA_INK_PREP_FUSED=0: the kernels one by one), the gate in the
    // attention's last kernel
    static const bool prep_fused = [] { const char* v = std::getenv("STRATA_INK_PREP_FUSED"); return v == nullptr || v[0] != '0'; }();
    if (prep_fused && T <= 8 && D == 128) {
        k::inkling2_attn_prep(i_kr_, i_vr_, conv_k, conv_v, b.k_conv, b.v_conv, b.swa ? nullptr : i_kp_,
                              b.swa ? nullptr : i_vp_, (int) KT, sk, i_q_, i_r_, b.q_clip, b.rel_clip, (float) s.qk_clip,
                              (float) s.rel_clip, alpha, fl, eps, step_, b.kc, b.vc, (int) H, (int) NKV, (int) D, (int) R,
                              T, snap_k, snap_v, sst, ns, cs);
    } else {
        k::inkling2_conv(i_kr_, conv_k, b.k_conv, b.swa ? nullptr : i_kp_, (int) NKV, i_k_, KVW, (int) KT, sk, T, cs, snap_k, sst, ns);
        k::inkling2_conv(i_vr_, conv_v, b.v_conv, b.swa ? nullptr : i_vp_, (int) NKV, i_v_, KVW, (int) KT, sk, T, cs, snap_v, sst, ns);
        k::inkling2_head_clip(i_q_, b.q_clip, (float) s.qk_clip, alpha, fl, step_, (int) H, (int) D, T, cs);
        k::inkling2_head_clip(i_k_, nullptr, (float) s.qk_clip, 0.0f, fl, step_, (int) NKV, (int) D, T, cs);
        k::inkling2_head_rms(i_v_, eps, (int) NKV, (int) D, T, cs);
        k::inkling2_head_clip(i_r_, b.rel_clip, (float) s.rel_clip, alpha, fl, step_, (int) H, (int) R, T, cs);
        k::inkling2_kv_store(i_k_, i_v_, step_, b.kc, b.vc, KVW, T, cs);
    }
    k::inkling2_attention_gated(i_q_, i_r_, i_sink_, b.kc, b.vc, b.rel_table, step_, (int) H, (int) NKV, (int) D, (int) R,
                                (int) b.extent, b.swa ? (int) s.n_swa : 0, i_att_, i_gate_, T, cs);
    glm5_decode_stamp(this, l, 2, cs);
    if (tp_) {   // b70-local TP: o's rows, this GPU's and the others'
        glm5_decode_stamp(this, l, 14, cs);
        k::doorbell_publish(i_att_, ids_, w_, (int64_t) T * H * D, 0, epd ? epr_.tx : m_tx_, nullptr, nullptr,
                            epd ? epr_.tseq : m_tseq_, cs);
        if (tp_place && epd) {
            const TpCut c = tp_cut(l, (int64_t) N, true);
            const bool direct = T == 1;
            float* o0 = direct ? i_o_ + (size_t) c.off[ep_stage_] : i_tps_;
            gproj(b.o, i_att_, o0, T, false, cs);
            glm5_decode_stamp(this, l, 10, cs);
            if (!fw) {
                tp_wait();
                glm5_decode_stamp(this, l, 11, cs);
            }
            k::inkling2_tp_place_cut(i_o_, nullptr, N, pcut(c), ep_n_, T, direct ? nullptr : o0, nullptr, ep_stage_,
                                     epr_.trem, slot, cs, fw ? epr_.flag : nullptr, kEpFlagStride, ep_n_ - 1,
                                     fw ? ++ep_hc_ : 0);
        } else if (tp_place) {
            const bool direct = T == 1;
            float* o0 = direct ? i_o_ + (size_t) ep_stage_ * tos : i_tps_;
            gproj(b.o, i_att_, o0, T, false, cs);
            glm5_decode_stamp(this, l, 10, cs);
            if (fw) {
                k::inkling2_tp_place(i_o_, nullptr, N, tos, T, direct ? nullptr : o0, nullptr, ep_stage_, tp_rem,
                                     (int64_t) T * 2 * tqs, 0, ep_n_, cs, epr_.flag, kEpFlagStride, ep_n_ - 1, ++ep_hc_);
            } else {
            tp_wait();
            glm5_decode_stamp(this, l, 11, cs);
            k::inkling2_tp_place(i_o_, nullptr, N, tos, T, direct ? nullptr : o0, nullptr, ep_stage_, tp_rem,
                                 (int64_t) T * 2 * tqs, 0, ep_n_, cs);
            }
        } else {
        gproj(b.o, i_att_, i_tps_, T, false, cs);
        k::inkling2_put_cols(i_o_, N, ep_stage_ * tos, i_tps_, tos, T, cs);
        tp_wait();
        tp_others(i_o_, N, tos, 0);
        }
    } else
    gproj(b.o, i_att_, i_o_, T, false, cs);
    const float s1 = 1.0f / std::sqrt((float) (KT + 1));
    k::inkling2_conv(i_o_, conv_a, b.attn_post_conv, nullptr, 0, mix_, N, (int) KT, s1, T, cs, snap_a, sst, ns);
    if (l % B == 0) {   // a block starts: the layer's input is kept, the attention's output is the running row
        k::copy_from_mapped(i_blocks_ + (l / B) * bstride, x_, (int64_t) T * N, cs);
        k::copy_from_mapped(x_, mix_, (int64_t) T * N, cs);
    } else {
        k::add_inplace(x_, mix_, (int64_t) T * N, cs);
    }

    glm5_decode_stamp(this, l, 3, cs);
    // ---- the MoE: its residual mix, the norm, the router, the experts, the convolution
    k::inkling2_attn_res(i_blocks_, bstride, (int) (l / B + 1), x_, b.ffn_rq, 1, i_in_, N, T, cs);
    glm5_decode_stamp(this, l, 6, cs);   // b70-local profile: the MoE input's residual mix done
    k::native_qsa_rms_norm_weighted_q8_1(i_in_, b.ffn_norm, hn_, g_xq_, (int) N, T, eps, cs);
    for (int t0 = 0; t0 < T; t0 += 8) {
        if (b.router16 != nullptr)
            k::bf16_gemv_fp32_mmvf_multi(hn_ + (size_t) t0 * N, N, b.router16, rlog_ + (size_t) t0 * E, E, N, E, std::min(8, T - t0), cs);
        else
            k::f32_gemv_fp32_mmvf_multi(hn_ + (size_t) t0 * N, N, b.router, rlog_ + (size_t) t0 * E, E, N, E, std::min(8, T - t0), cs);
    }
    glm5_decode_stamp(this, l, 7, cs);   // b70-local profile: the norm and the router's product done
    const float scale = s.expert_weights_scale > 0 ? (float) s.expert_weights_scale : 1.0f;
    k::glm5_router_sigmoid(rlog_, b.router_b, T, (int) E, (int) K, true, scale, ids_, w_, cs);
    glm5_decode_stamp(this, l, 4, cs);
    ++ring;
    glm5_decode_stamp(this, l, 15, cs);
    if (ep_direct_ && win_.ep())   // b70-local EP direct: the rows, ids and weights straight to the helpers (their kernels read them)
        k::doorbell_publish(hn_, ids_, w_, (int64_t) T * N, (int64_t) T * K, epr_.ex, epr_.eids, epr_.ew, epr_.eseq, cs);
    win_.publish(ring, 0, 0, T, hn_, ids_, w_, cs, !(ep_direct_ && win_.ep()));
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
    if (win_.ep()) {   // b70-local EP: the other GPUs' sums of their experts, in GPU order (the relay's rows)
        glm5_decode_stamp(this, l, 8, cs);   // b70-local profile: this GPU's experts done (the wait starts)
        static const bool fwe = [] { const char* v = std::getenv("STRATA_INK_EP_FUSEWAIT"); return v == nullptr || v[0] != '0'; }();
        if (ep_direct_ && fwe)
            k::add_rows_inplace_after(moe_out_, epr_.rem, (int64_t) T * N, ep_n_ - 1, (int64_t) T * N, epr_.flag,
                                      kEpFlagStride, ep_n_ - 1, ++ep_hc_, cs);
        else {
        if (ep_direct_) k::wait_flags_ge(epr_.flag, kEpFlagStride, ep_n_ - 1, ++ep_hc_, cs);
        else k::wait_flag_ge(win_.dev_flag(), ring, cs);
        glm5_decode_stamp(this, l, 9, cs);
        static const bool add1 = [] { const char* v = std::getenv("STRATA_INK_EP_ADD1"); return v == nullptr || v[0] != '0'; }();
        if (ep_direct_)
            k::add_rows_inplace(moe_out_, epr_.rem, (int64_t) T * N, ep_n_ - 1, (int64_t) T * N, cs);
        else if (add1)   // one kernel: the sums straight from the mapped rows, in GPU order
            k::add_rows_inplace(moe_out_, m_rem_, (int64_t) T * N, ep_n_ - 1, (int64_t) T * N, cs);
        else
        for (int j = 0; j + 1 < ep_n_; ++j) {
            k::copy_from_mapped(i_o_, m_rem_ + (size_t) j * T * N, (int64_t) T * N, cs);
            k::add_inplace(moe_out_, i_o_, (int64_t) T * N, cs);
        }
        }
    }
    k::inkling2_conv(moe_out_, conv_f, b.ffn_post_conv, nullptr, 0, mix_, N, (int) KT, s1, T, cs, snap_f, sst, ns);
    k::add_inplace(x_, mix_, (int64_t) T * N, cs);
    glm5_decode_stamp(this, l, 5, cs);
    return true;
}

bool PlainModel::record_assist(const PlainModel& own, int T, strata::gpu::Graph** g, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, K = s.n_expert_used;
    if (!setup_device_plan(err)) return false;
    if (!win_.device_plan() || !win_.ep()) {
        err = "plain model: the expert relay needs the device plan on every GPU";
        return false;
    }
    strata::gpu::Stream cs = cs_;
    if (!strata::gpu::begin_capture(cs)) { err = "plain model: begin capture failed"; return false; }
    uint32_t r = 0, P = 0;
    const int64_t HD = s.n_head * s.key_length, tqs = tp_ ? HD / ep_n_ : 0, tos = tp_ ? N / ep_n_ : 0;
    // b70-local EP direct: the owner's region (registered in this context): its sequence numbers in, this GPU's slot out
    const bool epd = own.ep_direct_;
    const EpReg& R = own.epr_;
    const int j = ep_stage_ < own.ep_stage_ ? ep_stage_ : ep_stage_ - 1;
    uint32_t* const myflag = epd ? R.flag + (size_t) j * kEpFlagStride : nullptr;
    uint32_t tc = 0;
    for (int64_t l = own.lb_; l < own.le_; ++l) {
        if (!s.layer_routes(l)) continue;
        ++r;
        if (tp_ && epd) {
            const TPW& t = tpw_[(size_t) l];
            float* const out = R.trem + (size_t) j * T * 2 * (size_t) own.tp_wmax_qg_;
            const int64_t wq = t.q.n_out, wo = t.o.n_out;   // this GPU's cut of the layer (tp_cut)
            const bool fwh = std::getenv("STRATA_INK_EP_FUSEWAIT") == nullptr || std::getenv("STRATA_INK_EP_FUSEWAIT")[0] != '0';
            if (fwh) {
                const k::MappedCopy c{hn_, R.tx, (int64_t) T * N};
                k::copy_from_mapped_multi_after(&c, 1, R.tseq, 0, 1, ++tc, cs);
            } else {
            k::wait_flag_ge(R.tseq, ++tc, cs);
            k::copy_from_mapped(hn_, R.tx, (int64_t) T * N, cs);
            }
            gproj_win(t.q, hn_, i_tps_, T, false, cs);
            gproj_win(t.gate, hn_, i_tps_ + (size_t) T * wq, T, false, cs);
            k::doorbell_publish(i_tps_, ids_, w_, (int64_t) T * 2 * wq, 0, out, nullptr, nullptr, myflag, cs);
            if (fwh) {
                const k::MappedCopy c{i_att_, R.tx, (int64_t) T * HD};
                k::copy_from_mapped_multi_after(&c, 1, R.tseq, 0, 1, ++tc, cs);
                if (T == 1) glm5_decode_stamp(this, l, 4, cs);
            } else {
            k::wait_flag_ge(R.tseq, ++tc, cs);
            if (T == 1) glm5_decode_stamp(this, l, 4, cs);
            k::copy_from_mapped(i_att_, R.tx, (int64_t) T * HD, cs);
            }
            gproj_win(t.o, i_att_, i_tps_, T, false, cs);
            k::doorbell_publish(i_tps_, ids_, w_, (int64_t) T * wo, 0, out, nullptr, nullptr, myflag, cs);
        } else if (tp_) {   // TP: this GPU's rows of the layer's q and gate, then of o
            const TPW& t = tpw_[(size_t) l];
            k::wait_flag_ge(m_ain_, ++P, cs);
            k::copy_from_mapped(hn_, m_ax_, (int64_t) T * N, cs);
            gproj_win(t.q, hn_, i_tps_, T, false, cs);
            gproj_win(t.gate, hn_, i_tps_ + (size_t) T * tqs, T, false, cs);
            k::doorbell_publish(i_tps_, ids_, w_, (int64_t) T * 2 * tqs, 0, m_ay_, nullptr, nullptr, m_aout_, cs);
            k::wait_flag_ge(m_ain_, ++P, cs);
            k::copy_from_mapped(i_att_, m_ax_, (int64_t) T * HD, cs);
            gproj_win(t.o, i_att_, i_tps_, T, false, cs);
            k::doorbell_publish(i_tps_, ids_, w_, (int64_t) T * tos, 0, m_ay_, nullptr, nullptr, m_aout_, cs);
        }
        static const bool fwx = [] { const char* v = std::getenv("STRATA_INK_EP_FUSEWAIT"); return v == nullptr || v[0] != '0'; }();
        if (epd && fwx) {
            const k::MappedCopy c[3] = {{hn_, R.ex, (int64_t) T * N}, {ids_, R.eids, (int64_t) T * K}, {w_, R.ew, (int64_t) T * K}};
            k::copy_from_mapped_multi_after(c, 3, R.eseq, 0, 1, (uint32_t) r, cs);
            if (T == 1) glm5_decode_stamp(this, l, 0, cs);
        } else if (epd) {
            k::wait_flag_ge(R.eseq, (uint32_t) r, cs);
            if (T == 1) glm5_decode_stamp(this, l, 0, cs);
            k::copy_from_mapped(hn_, R.ex, (int64_t) T * N, cs);
            k::copy_i32_from_mapped(ids_, R.eids, (int64_t) T * K, cs);
            k::copy_from_mapped(w_, R.ew, (int64_t) T * K, cs);
        } else {
        k::wait_flag_ge(m_ain_, ++P, cs);
        if (T == 1) glm5_decode_stamp(this, l, 0, cs);
        k::copy_from_mapped(hn_, m_ax_, (int64_t) T * N, cs);
        k::copy_i32_from_mapped(ids_, m_aids_, (int64_t) T * K, cs);
        k::copy_from_mapped(w_, m_aw_, (int64_t) T * K, cs);
        }
        win_.publish(r, 0, 0, T, hn_, ids_, w_, cs, false);
        if (T == 1) glm5_decode_stamp(this, l, 1, cs);
        if (!win_.experts(s.moe_index(l), r, 0, 0, T, cs, err)) return false;
        k::native_moe_combine_multi(win_.parts(), w_, nullptr, moe_out_, N, K, T, cs);
        if (T == 1) glm5_decode_stamp(this, l, 2, cs);
        if (epd)
            k::doorbell_publish(moe_out_, ids_, w_, (int64_t) T * N, 0, R.rem + (size_t) j * T * N, nullptr, nullptr,
                                myflag, cs);
        else
        k::doorbell_publish(moe_out_, ids_, w_, (int64_t) T * N, 0, m_ay_, m_aids_, m_aw_, m_aout_, cs);
        if (T == 1) glm5_decode_stamp(this, l, 3, cs);
    }
    if (!strata::gpu::end_capture(cs, g)) {
        err = std::string("plain model: the relay's capture: ") + strata::gpu::last_error();
        return false;
    }
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
        k::inkling2_raw_rows(x_, mix_, m_raw_, N, T, cs);   // b70-local: a picture's rows as they came (no norm)
    } else {         // the running row, then the blocks, as the stage before handed them on
        k::copy_from_mapped(x_, m_in_, (int64_t) T * N, cs);
        for (int64_t j = 0; j < i_nblocks_; ++j)
            k::copy_from_mapped(i_blocks_ + j * bstride, m_in_ + (1 + j) * (int64_t) T * N, (int64_t) T * N, cs);
    }
    uint32_t ring = 0;
    tp_ring_ = 0;
    ep_hc_ = 0;
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
    if (const int j = tap_slot(s.n_layer); j >= 0 && i_taps_ != nullptr)   // b70-local: the final rows, a drafter tap
        k::copy_from_mapped(i_taps_ + (size_t) j * opt_.max_t * N, hn_, (int64_t) T * N, cs);
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
