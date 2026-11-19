// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/cpu/expert_layout.hpp - plan v0.3 P6: where each routed expert lives in experts.bin.
//
// A Q2_0 pack (tools/strata_pack.py) has one blob size for every layer, `BLOB`, in the Strata expert form.  A
// native pack (tools/iq_pack.py, the IQ2_XS / IQ3_XXS files) keeps each expert's raw GGUF slices, so the blob
// size and the formats change from layer to layer; `native_experts.txt` says how.  Everything that touches an
// expert blob - the arena, the VRAM tier, the prompt path, the CPU pool, the GPU window - asks this table.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

struct ExpertLayout {
    bool native = false;
    int64_t n_layers = 0, n_expert = NE;
    std::vector<NativeFmt> fmt;           ///< per layer (native packs)
    std::vector<uint64_t> offset, bytes;  ///< per layer: where its 512 blobs start, bytes per blob
    /// Plan v0.3 P6: per layer, the absolute offsets of the gate / up / down tensors in their GGUF files, so the
    /// arena can be filled from the GGUF itself when the pack has no experts.bin (3 x n_layers, 0 = unknown).
    std::vector<uint64_t> gguf_off;
    /// Per layer, the GGUF files (names beside the --native shard) that hold its gate / up / down tensors when
    /// the model's shards split the layers (Swift's GGUFs: layers 13-47 in shard 2, and layer 13's down tensor in
    /// shard 1) (3 x n_layers).  Empty = the --native shard itself.
    std::vector<std::string> gguf_file;
    /// Per layer of this table, the model's block that holds it (`blk.<block>.ffn_*_exps`): a model whose first blocks
    /// have no routed experts (GLM-5.3's three dense ones) numbers its MoE layers apart from its blocks.  Empty: the
    /// same numbers.
    std::vector<int64_t> block;
    /// NVFP4 layers (fmt.tail_off != 0): per layer, n_expert x {s_gate, s_up, s_down, 0}, the blob's tail (the GGUF's
    /// `blk.<b>.ffn_*_exps.scale` tensors, ModelOpt's weight_scale_2; 1 where a role has none).  Empty for others.
    std::vector<std::vector<float>> tail;
    /// b70-local RVQ2X8 layers: per layer, n_expert x tail_raw_bytes[layer] bytes (per-row f16 gains of gate, up, down,
    /// then the layer's two F32 books [2][256][8]), copied to the blob's tail.  Empty for others.
    std::vector<std::vector<uint8_t>> tail_raw;
    std::vector<uint64_t> tail_raw_bytes;
    /// b70-local, STRATA_PTQ1_AS_Q2=1: per layer, 1 when its three tensors are PTQ1_0 (143) in the GGUF and are
    /// re-packed into Q2_0 (42) as they are read (fmt says 42; a file slice is 28/36 of the blob's).  Empty: none.
    std::vector<char> ptq1_q2;
    bool from_ptq1(int64_t layer) const { return !ptq1_q2.empty() && ptq1_q2[(size_t) layer] != 0; }
    /// b70-local RVQ2X8 groups (11-17, GB10's GGUF layout: blk.N.ffn_*_exps{,_g1,..}, the router cut to the kept experts in
    /// group order): every expert has its own width and blob.  Per (layer, expert), n_expert a layer: e_ff (0 = absent,
    /// the router never picks it), e_bytes (its blob), e_src (3 absolute file offsets of its gate / up / down codes),
    /// e_gsrc (3 of its gains); per layer, book_src (the layer's books).  bytes[l] is the layer's largest blob.
    /// The offsets carry their file in the top byte (mixed_src): shard s of mixed_files, the GGUF's split parts.
    bool mixed = false;
    std::vector<int32_t> e_ff;
    std::vector<uint64_t> e_bytes, e_src, e_gsrc, book_src;
    std::vector<std::string> mixed_files;   ///< the shards' paths, in shard order
    static constexpr int kMixedShardShift = 56;
    static uint64_t mixed_src(size_t shard, uint64_t off) { return ((uint64_t) shard << kMixedShardShift) | off; }
    std::vector<int32_t> n_kept;          ///< per layer: experts present (the router's columns in the file)
    /// b70-local tq layers (fmt.tq): the rotation's signs of each input width K (the GGUF's tq.sign.<K>), one bit an
    /// input (1 = -1), copied into every blob of that width
    std::vector<std::pair<int64_t, std::vector<uint8_t>>> tq_signs;
    const std::vector<uint8_t>* tq_sign_bits(int64_t K) const {
        for (const auto& p : tq_signs) if (p.first == K) return &p.second;
        return nullptr;
    }
    uint64_t expert_bytes(int64_t layer, int64_t expert) const {
        return mixed ? e_bytes[(size_t) (layer * n_expert + expert)] : blob_bytes(layer);
    }
    int version = 0;                      ///< native_experts.txt's header version (0 = none given)
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< experts.bin size

    int64_t block_of(int64_t layer) const { return block.empty() ? layer : block[(size_t) layer]; }
    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
    }
    /// Writes the blob's tail (NVFP4 layers) after its three role slices; nothing for a layer without one.
    void write_tail(int64_t layer, int64_t expert, uint8_t* blob) const {
        if (!native || (size_t) layer >= fmt.size() || fmt[(size_t) layer].tail_off == 0) return;
        if ((size_t) layer < tail_raw.size() && !tail_raw[(size_t) layer].empty()) {
            const uint64_t tb = tail_raw_bytes[(size_t) layer];
            std::memcpy(blob + fmt[(size_t) layer].tail_off, tail_raw[(size_t) layer].data() + tb * (uint64_t) expert, tb);
            return;
        }
        const std::vector<float>& t = tail[(size_t) layer];
        std::memcpy(blob + fmt[(size_t) layer].tail_off, t.data() + 4 * (size_t) expert, 4 * sizeof(float));
    }
};

/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere; STRATA_FORCE_AVX2=1 answers no (for tests).
bool cpu_avx512_ok();
/// Whether this CPU (and its OS) runs the AVX2 kernels (AVX, AVX2, FMA, F16C): the floor of every expert kernel
/// (q2_avx2.cpp, iq_avx2.cpp, and ggml-cpu in the portable build).  STRATA_FORCE_AVX2 does not change it.
bool cpu_avx2_ok();
/// STRATA_IQ256_GATHER, the AVX-2 i-quant kernels' gathered grid decode (iq_avx2.cpp, upstream a7336175): -1 unset
/// (auto, see cpu_gather_fast_here), 0 the scalar decode on every core, 1 the gathered one on every core.
int iq256_gather_setting();
/// Whether this CPU's performance cores gather the IQ grid entries faster than they assemble them from scalar loads,
/// from what CPUID reports: an Intel CPU with AVX-VNNI (Alder Lake / Sapphire Rapids or newer; the older Intel cores
/// gather slowly or under the Downfall microcode, AMD Zen 2/3 gather slowly) that is hybrid (CPUID 7.0:EDX[15]).
/// A CPU of one core type is not told apart by CPUID from an E-core-only part, so it keeps the scalar decode.
bool cpu_gather_fast();
/// cpu_gather_fast() and the CALLING THREAD runs on a performance core: CPUID 1Ah core type 40h (the E-cores, 20h,
/// gather slower than they assemble).  Probed once per thread: the pool pins each worker to one core.
bool cpu_gather_fast_here();
/// Whether the AVX-VNNI kernels run here: the build has them (STRATA_HAVE_AVXVNNI), the CPU has AVX2 and AVX-VNNI
/// (CPUID 7.1 EAX bit 4).  STRATA_NO_AVXVNNI=1 answers no (to compare with the AVX2 kernels).
bool cpu_avxvnni_ok();
/// The CPU's brand string (CPUID 0x80000002..4), for messages; "unknown" when it has none.
std::string cpu_name();
/// Q2_0 GGUF rows / activation quantizer on the kernels this CPU has.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1);
void act_quant_any(const float* x, int n, ActQ& a);

/// The process-wide layout (canonical Q2_0 until `expert_layout_load` finds a native pack).
const ExpertLayout& expert_layout();
/// Reads `<pack_dir>/native_experts.txt` when it exists (a native pack), else sets the canonical layout.
/// Versions up to kExpertLayoutVersion are read; a newer one is refused (a newer packer wrote it).
bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
/// Replaces the process-wide layout (expert_layout_from_gguf builds one from a model's GGUF).
void expert_layout_set(const ExpertLayout& layout);
/// The newest native_experts.txt this engine reads.  v4 = v3 plus the per-role shard column `gate,up,down`,
/// written only when some layer's roles are in different shards (every other pack stays v3, byte for byte).
/// v5 = v4 plus `hadamard.txt` beside it: Hadamard-folded experts (hadamard.hpp), written only for such models,
/// so an engine that cannot rotate the activations refuses the pack instead of running it wrong.
inline constexpr int kExpertLayoutVersion = 5;

}  // namespace strata::kernels::cpu

namespace strata::kernels::cpu {
/// b70-local (STRATA_PTQ1_AS_Q2=1): PTQ1_0 blocks (128 trits as 26 B of base-3 digits + fp16 d) into Q2_0 blocks
/// (64 values as fp16 d + 16 B of 2-bit codes, value = code - 1): each 128 becomes two 64-blocks with the same d,
/// the exact inverse of tools' make_t3p.py, so the result is the Q2_0 container's bytes.  `n128` blocks: 28 n128
/// bytes read, 36 n128 written.
inline void ptq1_to_q2(const uint8_t* src, uint8_t* dst, uint64_t n128) {
    static const auto tab = [] {   // tab[byte][n] = the n-th trit (0..2) of a digit byte
        struct T { uint8_t t[256][5]; } r{};
        static const uint16_t pw[5] = {1, 3, 9, 27, 81};
        for (int b = 0; b < 256; ++b)
            for (int n = 0; n < 5; ++n) r.t[b][n] = (uint8_t) ((((uint16_t) (b * pw[n]) & 255) * 3) >> 8);
        return r;
    }();
    for (uint64_t k = 0; k < n128; ++k, src += 28, dst += 36) {
        uint8_t v[128];
        for (int i = 0; i < 16; ++i)
            for (int n = 0; n < 5; ++n) v[16 * n + i] = tab.t[src[i]][n];
        for (int i = 0; i < 8; ++i)
            for (int n = 0; n < 5; ++n) v[80 + 8 * n + i] = tab.t[src[16 + i]][n];
        for (int i = 0; i < 2; ++i)
            for (int n = 0; n < 4; ++n) v[120 + 2 * n + i] = tab.t[src[24 + i]][n];
        for (int h = 0; h < 2; ++h) {
            uint8_t* q = dst + 18 * h;
            q[0] = src[26];
            q[1] = src[27];
            for (int i = 0; i < 16; ++i) {
                const uint8_t* w = v + 64 * h + 4 * i;
                q[2 + i] = (uint8_t) (w[0] | (w[1] << 2) | (w[2] << 4) | (w[3] << 6));
            }
        }
    }
}
}  // namespace strata::kernels::cpu
