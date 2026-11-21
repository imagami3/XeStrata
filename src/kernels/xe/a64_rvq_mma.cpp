// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// GB10 local port (aarch64 + CUDA): rvq_xmx.cpp's v2 kernel for NVIDIA tensor cores through SYCL joint_matrix
// (CUDA backend: 32-wide sub-groups, FP16 16 x 16 x 16 tiles, FP32 sums). Inkling 2's mixed RVQ2X8 experts for windows
// with many entries an expert (the prompt). A work-group of 8 sub-groups takes 128 rows of one expert's projection
// (16 a sub-group) and up to 16 * MT of its entries; per 64 inputs its 256 lanes decode the 128 x 64 weights
// (book1[c >> 8] + book2[c & 255]) into local memory as FP16, gather the entries' 64 inputs as FP16, and each
// sub-group multiplies its 16 rows. The blob layout and the outputs are rvq_xmx.cpp's.
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
using half = sycl::half;

constexpr int TM = 16, TN = 16, TK = 16;     // one CUDA tensor-core tile (m16n16k16)
constexpr int NSG = 8, SG = 32, WG = NSG * SG;
constexpr int RT = NSG * TN;                 // 128 rows a work-group
constexpr int KC = 64;                       // inputs a step (8 codes a row)
constexpr int NB = 2 * 256 * 8;              // the books' floats

template<typename P>
inline auto lptr(P& acc) { return acc.template get_multi_ptr<sycl::access::decorated::no>(); }

template<int MT, bool DOWN>
struct RvqMma {
    const unsigned long long* grp_ptr;
    const int32_t* grp_start;
    const int32_t* n_groups;
    const int32_t* ent_tok;
    const int32_t* ent_dst;
    const float* x;
    float* gate;      // DOWN: the output
    float* up;
    int64_t N, FF;
    int n_rt;
    sycl::local_accessor<half, 1> wt;    // [RT][KC]: B column-major (a row's KC inputs contiguous)
    sycl::local_accessor<half, 1> xa;    // [TM * MT][KC]: A row-major
    sycl::local_accessor<float, 1> ot;   // [TM * MT][RT]
    sycl::local_accessor<float, 1> bk;   // the books

    void operator()(sycl::nd_item<2> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id();
        const int tid = (int) it.get_local_linear_id();
        const int rt = (int) it.get_group(1);
        const bool is_up = !DOWN && rt >= n_rt;
        const int64_t r0 = (int64_t) (is_up ? rt - n_rt : rt) * RT;
        const size_t gu_row = (size_t) N / 8 * 2;
        constexpr int EP = TM * MT;
        half* const wtp = lptr(wt).get();
        half* const xap = lptr(xa).get();
        float* const otp = lptr(ot).get();
        float* const bkp = lptr(bk).get();
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int64_t ff = *(const int32_t*) blob;
            if (!DOWN && r0 >= ff) continue;   // the same for the whole work-group
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
                for (int64_t k0 = 0; k0 < K; k0 += KC) {
                    // 128 rows x 8 codes = 1024 codes, 4 a lane: row (idx >> 3), code (idx & 7)
                    for (int i = 0; i < (RT * (KC / 8)) / WG; ++i) {
                        const int idx = tid + WG * i, rr = idx >> 3, cc = idx & 7;
                        half* w = wtp + rr * KC + cc * 8;
                        if (r0 + rr < rows && k0 + cc * 8 < K) {
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
                    for (int idx = tid; idx < EP * KC; idx += WG) {
                        const int er = idx / KC, kk = idx % KC;
                        float v = 0.0f;
                        if (er < n && k0 + kk < K) v = DOWN ? x[(size_t) (e + er) * FF + k0 + kk] : x[(size_t) ent_tok[e + er] * N + k0 + kk];
                        xap[idx] = (half) v;
                    }
                    sycl::group_barrier(it.get_group());
                    for (int kk = 0; kk < KC; kk += TK) {
                        mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> bm;
                        mx::joint_matrix_load(sg, bm, lptr(wt) + sgid * TN * KC + kk, KC);
                        for (int i = 0; i < MT; ++i) {
                            mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> a;
                            mx::joint_matrix_load(sg, a, lptr(xa) + i * TM * KC + kk, KC);
                            mx::joint_matrix_mad(sg, c[i], a, bm, c[i]);
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
                for (int i = 0; i < MT; ++i)
                    mx::joint_matrix_store(sg, c[i], lptr(ot) + i * TM * RT + sgid * TN, RT, mx::layout::row_major);
                sycl::group_barrier(it.get_group());
                for (int idx = tid; idx < n * RT; idx += WG) {
                    const int er = idx / RT, rr = idx % RT;
                    const int64_t r = r0 + rr;
                    if (r >= rows) continue;
                    const float v = otp[er * RT + rr] * (float) gains[r];
                    if (DOWN) gate[(size_t) ent_dst[e + er] * N + r] = v;
                    else (is_up ? up : gate)[(size_t) (e + er) * FF + r] = v;
                }
                sycl::group_barrier(it.get_group());
            }
        }
    }
};

template<int MT, bool DOWN>
sycl::event launch(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_tok, const int32_t* ent_dst, const float* x, float* o1,
                   float* o2, int64_t N, int64_t FF) {
    const int n_rt = (int) (((DOWN ? N : FF) + RT - 1) / RT);
    const size_t tiles = (size_t) (DOWN ? n_rt : 2 * n_rt);
    return q.submit([&](sycl::handler& h) {
        RvqMma<MT, DOWN> k{grp_ptr, grp_start, n_groups, ent_tok, ent_dst, x, o1, o2, N, FF, n_rt,
                           sycl::local_accessor<half, 1>(sycl::range<1>(RT * KC), h),
                           sycl::local_accessor<half, 1>(sycl::range<1>(TM * MT * KC), h),
                           sycl::local_accessor<float, 1>(sycl::range<1>(TM * MT * RT), h),
                           sycl::local_accessor<float, 1>(sycl::range<1>(NB), h)};
        h.parallel_for(sycl::nd_range<2>({cg, tiles * WG}, {1, (size_t) WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] { k(it); });
    });
}

int mt_setting() {
    static const int mt = [] { const char* v = std::getenv("STRATA_RVQ_XMX_MT"); return v ? (int) std::strtol(v, nullptr, 10) : 2; }();
    return mt;
}

}  // namespace

/// Whether a window's mixed RVQ experts take the tensor-core kernels (STRATA_RVQ_XMX: 0 off, else the entries a window
/// needs, default 64 - below it the LUT kernels of rvq_lut.cpp).
bool rvq_xmx_on(int64_t cap_entries, int64_t N, int64_t FF) {
    static const long min_e = [] { const char* v = std::getenv("STRATA_RVQ_XMX"); return v ? std::strtol(v, nullptr, 10) : 64L; }();
    return min_e > 0 && cap_entries >= min_e && N % KC == 0 && FF % 64 == 0;
}

sycl::event rvq_xmx_gu(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                       const int32_t* n_groups, const int32_t* ent_tok, const float* x, float* gate, float* up,
                       int64_t N, int64_t FF) {
    return mt_setting() == 1 ? launch<1, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, x, gate, up, N, FF)
                             : launch<2, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, x, gate, up, N, FF);
}

sycl::event rvq_xmx_down(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                         const int32_t* n_groups, const int32_t* ent_dst, const float* h, float* out, int64_t N,
                         int64_t FF) {
    return mt_setting() == 1 ? launch<1, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, h, out, nullptr, N, FF)
                             : launch<2, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, h, out, nullptr, N, FF);
}

}  // namespace strata::kernels
