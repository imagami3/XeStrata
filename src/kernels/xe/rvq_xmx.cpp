// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/rvq_xmx.cpp - b70-local (11-17): Inkling 2's RVQ2X8 experts with per-expert widths (NativeFmt::mixed)
// on the XMX engines, for windows with many entries an expert (the prompt).  A work-group (one sub-group) takes 64 rows
// of one expert's projection and up to 8 * MT of its entries: per 32 inputs it decodes the 64 x 32 weights
// (book1[c >> 8] + book2[c & 255]) into local memory as FP16, gathers the entries' 32 inputs as FP16, and multiplies
// on the XMX engines (8 x 16 x 16 tiles, FP32 sums); the row gains go on the FP32 sums.  The blob is the mixed one
// (iq_kernels.cpp native_expert_grouped_rvq): [256-byte header: int32 ff][gate ff rows][up ff rows][down N rows of
// ff/8 codes][f16 gains 2ff + N][books F32 2 x 256 x 8].  Widths are multiples of 64 (576 / 1152 / 1728 / 2304).
#include "strata/core/runtime.hpp"
#include "strata/kernels/xmx_gemm.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstdint>
#include <cstdlib>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using half = sycl::half;

constexpr int TM = 8, TN = 16, TK = 16;   // one XMX tile
constexpr int RT = 64;                    // rows a work-group (4 tiles of 16)
constexpr int KC = 32;                    // inputs a step (4 codes a row)
constexpr int NB = 2 * 256 * 8;           // the books' floats

#define STRATA_RVQ_XMX_PROPERTIES                                                                                      \
    auto get(syclex::properties_tag) const {                                                                           \
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<16>};         \
    }

template<typename P>
inline auto lptr(P& acc) { return acc.template get_multi_ptr<sycl::access::decorated::no>(); }

// DOWN false: the gate and up rows (x: the tokens' FP32 rows, N wide; out: gate / up [entry][FF]).
// DOWN true: the down rows (x: h [entry][FF], K = ff; out: [ent_dst][N]).
template<int MT, bool DOWN>
struct RvqXmx {
    const unsigned long long* grp_ptr;
    const int32_t* grp_start;
    const int32_t* n_groups;
    const int32_t* ent_tok;
    const int32_t* ent_dst;
    const float* x;
    float* gate;      // DOWN: the output
    float* up;
    int64_t N, FF;
    int n_rt;         // row tiles of one projection: FF / 64 (gate and up: 2 * n_rt in the grid), N / 64 (down)
    sycl::local_accessor<half, 1> wt;    // [RT][KC]: B column-major (a row's KC inputs contiguous)
    sycl::local_accessor<half, 1> xa;    // [8 * MT][KC]: A row-major
    sycl::local_accessor<float, 1> ot;   // [8 * MT][RT]
    sycl::local_accessor<float, 1> bk;   // the books

    void operator()(sycl::nd_item<2> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int rt = (int) it.get_group(1);
        const bool is_up = !DOWN && rt >= n_rt;
        const int64_t r0 = (int64_t) (is_up ? rt - n_rt : rt) * RT;
        const size_t gu_row = (size_t) N / 8 * 2;   // bytes of a gate / up row
        constexpr int EP = TM * MT;
        half* const wtp = lptr(wt).get();
        half* const xap = lptr(xa).get();
        float* const otp = lptr(ot).get();
        float* const bkp = lptr(bk).get();
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int64_t ff = *(const int32_t*) blob;
            if (!DOWN && r0 >= ff) continue;            // the same for the whole work-group
            const uint8_t* b = blob + 256;
            const size_t dn_o = 2 * gu_row * (size_t) ff, d_row = (size_t) ff / 8 * 2, tl = dn_o + d_row * (size_t) N;
            const uint16_t* codes = (const uint16_t*) (b + (DOWN ? dn_o : (is_up ? gu_row * (size_t) ff : 0)));
            const int64_t cpr = DOWN ? ff / 8 : N / 8;   // codes a row
            const int64_t K = DOWN ? ff : N;
            const half* gains = (const half*) (b + tl) + (DOWN ? 2 * ff : (is_up ? ff : 0));
            const float* book = (const float*) (b + tl + 2 * (size_t) (2 * ff + N));
            sycl::group_barrier(it.get_group());
            for (int i = lane; i < NB; i += 16) bkp[i] = book[i];
            sycl::group_barrier(it.get_group());
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += EP) {
                const int n = sycl::min(EP, e1 - e);
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c[MT][4];
                for (int i = 0; i < MT; ++i)
                    for (int j = 0; j < 4; ++j) mx::joint_matrix_fill(sg, c[i][j], 0.0f);
                for (int64_t k0 = 0; k0 < K; k0 += KC) {
                    // the weights: 64 rows x 4 codes, 16 codes a lane
                    for (int i = 0; i < 16; ++i) {
                        const int idx = lane + 16 * i, rr = idx >> 2, cc = idx & 3;
                        const uint32_t code = codes[(r0 + rr) * cpr + k0 / 8 + cc];
                        const float* b1 = bkp + (code >> 8) * 8;
                        const float* b2 = bkp + 2048 + (code & 255u) * 8;
                        half* w = wtp + rr * KC + cc * 8;
                        for (int j = 0; j < 8; ++j) w[j] = (half) (b1[j] + b2[j]);
                    }
                    // the entries' inputs: EP x 32, 2 * MT * 8 a lane
                    for (int i = 0; i < 2 * EP; ++i) {
                        const int idx = lane + 16 * i, er = idx >> 5, kk = idx & 31;
                        float v = 0.0f;
                        if (er < n) v = DOWN ? x[(size_t) (e + er) * FF + k0 + kk] : x[(size_t) ent_tok[e + er] * N + k0 + kk];
                        xap[er * KC + kk] = (half) v;
                    }
                    sycl::group_barrier(it.get_group());
                    for (int kk = 0; kk < KC; kk += TK) {
                        mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> a[MT];
                        mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> bm[4];
                        for (int i = 0; i < MT; ++i) mx::joint_matrix_load(sg, a[i], lptr(xa) + i * TM * KC + kk, KC);
                        for (int j = 0; j < 4; ++j) mx::joint_matrix_load(sg, bm[j], lptr(wt) + j * TN * KC + kk, KC);
                        for (int i = 0; i < MT; ++i)
                            for (int j = 0; j < 4; ++j) mx::joint_matrix_mad(sg, c[i][j], a[i], bm[j], c[i][j]);
                    }
                    sycl::group_barrier(it.get_group());
                }
                for (int i = 0; i < MT; ++i)
                    for (int j = 0; j < 4; ++j)
                        mx::joint_matrix_store(sg, c[i][j], lptr(ot) + i * TM * RT + j * TN, RT, mx::layout::row_major);
                sycl::group_barrier(it.get_group());
                for (int idx = lane; idx < n * RT; idx += 16) {
                    const int er = idx / RT, rr = idx % RT;
                    const int64_t r = r0 + rr;
                    const float v = otp[er * RT + rr] * (float) gains[r];
                    if (DOWN) gate[(size_t) ent_dst[e + er] * N + r] = v;
                    else (is_up ? up : gate)[(size_t) (e + er) * FF + r] = v;
                }
                sycl::group_barrier(it.get_group());
            }
        }
    }
    STRATA_RVQ_XMX_PROPERTIES
};

// v2: a work-group of 8 sub-groups takes 128 rows (16 a sub-group, one XMX column tile each) and up to 8 * MT entries;
// per 64 inputs its 128 lanes decode the 128 x 64 weights (8 codes a lane) into local memory, and each sub-group
// multiplies its 16 rows.  The books sit in local memory for the group.  Default register file (MT accumulators).
constexpr int NSG2 = 8, RT2 = NSG2 * TN, KC2 = 64;
template<int MT, bool DOWN>
struct RvqXmx2 {
    const unsigned long long* grp_ptr;
    const int32_t* grp_start;
    const int32_t* n_groups;
    const int32_t* ent_tok;
    const int32_t* ent_dst;
    const float* x;
    float* gate;
    float* up;
    int64_t N, FF;
    int n_rt;
    sycl::local_accessor<half, 1> wt;    // [RT2][KC2], a row's KC2 inputs contiguous (B column-major)
    sycl::local_accessor<half, 1> xa;    // [8 * MT][KC2]
    sycl::local_accessor<float, 1> ot;   // [8 * MT][RT2]
    sycl::local_accessor<float, 1> bk;

    void operator()(sycl::nd_item<2> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id();
        const int tid = (int) it.get_local_linear_id();
        const int rt = (int) it.get_group(1);
        const bool is_up = !DOWN && rt >= n_rt;
        const int64_t r0 = (int64_t) (is_up ? rt - n_rt : rt) * RT2;
        const size_t gu_row = (size_t) N / 8 * 2;
        constexpr int EP = TM * MT, WG = NSG2 * 16;
        half* const wtp = lptr(wt).get();
        half* const xap = lptr(xa).get();
        float* const otp = lptr(ot).get();
        float* const bkp = lptr(bk).get();
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int64_t ff = *(const int32_t*) blob;
            if (!DOWN && r0 >= ff) continue;
            const int64_t rows = DOWN ? N : ff;
            const uint8_t* b = blob + 256;
            const size_t dn_o = 2 * gu_row * (size_t) ff, d_row = (size_t) ff / 8 * 2, tl = dn_o + d_row * (size_t) N;
            const uint16_t* codes = (const uint16_t*) (b + (DOWN ? dn_o : (is_up ? gu_row * (size_t) ff : 0)));
            const int64_t cpr = DOWN ? ff / 8 : N / 8;
            const int64_t K = DOWN ? ff : N;
            const half* gains = (const half*) (b + tl) + (DOWN ? 2 * ff : (is_up ? ff : 0));
            const float* book = (const float*) (b + tl + 2 * (size_t) (2 * ff + N));
            sycl::group_barrier(it.get_group());
            for (int i = tid; i < NB; i += WG) bkp[i] = book[i];
            sycl::group_barrier(it.get_group());
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += EP) {
                const int n = sycl::min(EP, e1 - e);
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c[MT];
                for (int i = 0; i < MT; ++i) mx::joint_matrix_fill(sg, c[i], 0.0f);
                for (int64_t k0 = 0; k0 < K; k0 += KC2) {
                    // 128 rows x 8 codes: lane tid takes codes tid, tid + 128, ...: row (idx >> 3), code (idx & 7)
                    for (int i = 0; i < 8; ++i) {
                        const int idx = tid + WG * i, rr = idx >> 3, cc = idx & 7;
                        half* w = wtp + rr * KC2 + cc * 8;
                        if (r0 + rr < rows) {
                            const uint32_t code = codes[(r0 + rr) * cpr + k0 / 8 + cc];
                            const float* b1 = bkp + (code >> 8) * 8;
                            const float* b2 = bkp + 2048 + (code & 255u) * 8;
                            sycl::vec<half, 8> h8;
                            for (int j = 0; j < 8; ++j) h8[j] = (half) (b1[j] + b2[j]);
                            *reinterpret_cast<sycl::vec<half, 8>*>(w) = h8;
                        } else {
                            *reinterpret_cast<sycl::vec<half, 8>*>(w) = sycl::vec<half, 8>((half) 0.0f);
                        }
                    }
                    // the entries' 64 inputs
                    for (int idx = tid; idx < EP * KC2; idx += WG) {
                        const int er = idx / KC2, kk = idx % KC2;
                        float v = 0.0f;
                        if (er < n) v = DOWN ? x[(size_t) (e + er) * FF + k0 + kk] : x[(size_t) ent_tok[e + er] * N + k0 + kk];
                        xap[idx] = (half) v;
                    }
                    sycl::group_barrier(it.get_group());
                    for (int kk = 0; kk < KC2; kk += TK) {
                        mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> bm;
                        mx::joint_matrix_load(sg, bm, lptr(wt) + sgid * TN * KC2 + kk, KC2);
                        for (int i = 0; i < MT; ++i) {
                            mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> a;
                            mx::joint_matrix_load(sg, a, lptr(xa) + i * TM * KC2 + kk, KC2);
                            mx::joint_matrix_mad(sg, c[i], a, bm, c[i]);
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
                for (int i = 0; i < MT; ++i)
                    mx::joint_matrix_store(sg, c[i], lptr(ot) + i * TM * RT2 + sgid * TN, RT2, mx::layout::row_major);
                sycl::group_barrier(it.get_group());
                for (int idx = tid; idx < n * RT2; idx += WG) {
                    const int er = idx / RT2, rr = idx % RT2;
                    const int64_t r = r0 + rr;
                    if (r >= rows) continue;
                    const float v = otp[er * RT2 + rr] * (float) gains[r];
                    if (DOWN) gate[(size_t) ent_dst[e + er] * N + r] = v;
                    else (is_up ? up : gate)[(size_t) (e + er) * FF + r] = v;
                }
                sycl::group_barrier(it.get_group());
            }
        }
    }
};

template<int MT, bool DOWN>
sycl::event launch2(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                    const int32_t* n_groups, const int32_t* ent_tok, const int32_t* ent_dst, const float* x, float* o1,
                    float* o2, int64_t N, int64_t FF) {
    const int n_rt = (int) (((DOWN ? N : FF) + RT2 - 1) / RT2);
    const size_t tiles = (size_t) (DOWN ? n_rt : 2 * n_rt);
    return q.submit([&](sycl::handler& h) {
        RvqXmx2<MT, DOWN> k{grp_ptr, grp_start, n_groups, ent_tok, ent_dst, x, o1, o2, N, FF, n_rt,
                            sycl::local_accessor<half, 1>(sycl::range<1>(RT2 * KC2), h),
                            sycl::local_accessor<half, 1>(sycl::range<1>(TM * MT * KC2), h),
                            sycl::local_accessor<float, 1>(sycl::range<1>(TM * MT * RT2), h),
                            sycl::local_accessor<float, 1>(sycl::range<1>(NB), h)};
        h.parallel_for(sycl::nd_range<2>({cg, tiles * NSG2 * 16}, {1, (size_t) NSG2 * 16}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] { k(it); });
    });
}

template<int MT, bool DOWN>
sycl::event launch(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_tok, const int32_t* ent_dst, const float* x, float* o1,
                   float* o2, int64_t N, int64_t FF) {
    const int n_rt = (int) ((DOWN ? N : FF) / RT);
    const size_t tiles = (size_t) (DOWN ? n_rt : 2 * n_rt);
    return q.submit([&](sycl::handler& h) {
        RvqXmx<MT, DOWN> k{grp_ptr, grp_start, n_groups, ent_tok, ent_dst, x, o1, o2, N, FF, n_rt,
                           sycl::local_accessor<half, 1>(sycl::range<1>(RT * KC), h),
                           sycl::local_accessor<half, 1>(sycl::range<1>(TM * MT * KC), h),
                           sycl::local_accessor<float, 1>(sycl::range<1>(TM * MT * RT), h),
                           sycl::local_accessor<float, 1>(sycl::range<1>(NB), h)};
        h.parallel_for(sycl::nd_range<2>({cg, tiles * 16}, {1, 16}), k);
    });
}

}  // namespace

/// Whether the mixed RVQ experts of a window of `cap_entries` entries take the XMX kernels (STRATA_RVQ_XMX: 0 off,
/// else the entries a window needs, default 64).
bool rvq_xmx_on(int64_t cap_entries, int64_t N, int64_t FF) {
    static const long min_e = [] { const char* v = std::getenv("STRATA_RVQ_XMX"); return v ? std::strtol(v, nullptr, 10) : 64L; }();
    // B70, Inkling 2 86 GB on 3 cards, 256-token prompt windows (~5 entries an expert): v2 prefill 77 -> 92-97 tok/s
    // against the plain kernel; v1 (one sub-group a work-group, STRATA_RVQ_XMX_V=1) was 4x slower than the plain one
    // the matrix engines as the device reports them (FP16 8 x 16 x 16, 16-wide sub-groups); else the plain kernel
    return min_e > 0 && cap_entries >= min_e && N % RT == 0 && FF % RT == 0 && N % KC == 0 && xmx_available(XmxType::f16);
}

/// gate / up ([entry][FF] each) of the mixed RVQ experts' groups, on the XMX engines
sycl::event rvq_xmx_gu(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                       const int32_t* n_groups, const int32_t* ent_tok, const float* x, float* gate, float* up,
                       int64_t N, int64_t FF) {
    static const int mt = [] { const char* v = std::getenv("STRATA_RVQ_XMX_MT"); return v ? (int) std::strtol(v, nullptr, 10) : 2; }();
    static const int ver = [] { const char* v = std::getenv("STRATA_RVQ_XMX_V"); return v ? (int) std::strtol(v, nullptr, 10) : 2; }();
    if (ver == 2) return mt == 1 ? launch2<1, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, x, gate, up, N, FF)
                                 : launch2<2, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, x, gate, up, N, FF);
    if (mt == 4) return launch<4, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, x, gate, up, N, FF);
    if (mt == 1) return launch<1, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, x, gate, up, N, FF);
    return launch<2, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, x, gate, up, N, FF);
}

/// out[ent_dst] ([.][N]) = down(h [entry][FF]) of the mixed RVQ experts' groups, on the XMX engines
sycl::event rvq_xmx_down(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                         const int32_t* n_groups, const int32_t* ent_dst, const float* h, float* out, int64_t N,
                         int64_t FF) {
    static const int mt = [] { const char* v = std::getenv("STRATA_RVQ_XMX_MT"); return v ? (int) std::strtol(v, nullptr, 10) : 2; }();
    static const int ver = [] { const char* v = std::getenv("STRATA_RVQ_XMX_V"); return v ? (int) std::strtol(v, nullptr, 10) : 2; }();
    if (ver == 2) return mt == 1 ? launch2<1, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, h, out, nullptr, N, FF)
                                 : launch2<2, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, h, out, nullptr, N, FF);
    if (mt == 4) return launch<4, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, h, out, nullptr, N, FF);
    if (mt == 1) return launch<1, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, h, out, nullptr, N, FF);
    return launch<2, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, h, out, nullptr, N, FF);
}

}  // namespace strata::kernels
