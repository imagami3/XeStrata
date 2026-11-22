// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/a64_tq_mma.cpp - GB10 port (11-21) of tq_xmx.cpp for NVIDIA tensor cores through SYCL joint_matrix
// (CUDA: 32-wide sub-groups; m8n32k16 FP16 tiles, FP32 sums): a sub-group takes 2 bands (32 rows; lane l decodes tile
// l % 16 of band l / 16), a work-group 2 sub-groups (64 rows, as on the B70); the rotation's H4 exchanges stay within 4
// lanes.  Everything else (the blob, the codes, the scratch, the outputs) is tq_xmx.cpp's.  Original header:
// b70-local (11-19): Inkling 2's trellis-coded (tq2) experts on the XMX engines, for windows
// with many entries an expert (the prompt).  The blob and the codes' order are tq_kernels.cpp's.  First the entries'
// inputs are rotated into a scratch row each, as FP16 (x' = Hb (sign * x): a work-item a 16 inputs, H16 in it and H4
// over 4 neighbouring lanes).  Then a work-group of 4 sub-groups takes 64 rows (4 bands) of one expert's projection and
// up to 8 * NP of its entries: per 256 inputs each lane decodes one tile (its band, its 16 inputs: the decode kernel's
// coalesced loads) into local memory as FP16 once, and each sub-group multiplies its band on the XMX engines for every
// 8 of the entries (8 x 16 x 16 tiles, the inputs read from global memory, FP32 sums kept in local memory between the
// steps); the row gains (times the code's 1/147.8) go on the FP32 sums.  The decode is the larger cost (B70, 64 experts
// x 8 entries: without it 4.4 ms of 10.2), so one decode serves as many entries as it can: 64 experts x 32 entries 52 ms
// with a decode every 8 entries and the inputs gathered into local memory, 17.5 ms then (x 4 entries 12.1 -> 8.6 ms).
// A weight goes into local memory as its code's integer value (-510 .. 510, exact in FP16: the byte sum's dot
// instruction subtracts the 510, one conversion to FP16): x 8 / 32 entries 8.3 / 17.5 -> 6.8 / 15.2 ms against the
// value in FP32 converted.  Keeping the sums of 2 or 4 passes in registers a step (each weight tile loaded once for
// them) was slower at 8 entries (7.6 / 8.1 ms) and faster at 32 (14.6 / 13.3).
#include "strata/kernels/tq_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <unordered_map>


namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using half = sycl::half;

constexpr int TM = 8, TN = 32, TK = 16;   // one CUDA tensor-core tile (m8n32k16): 8 entries x 32 rows (2 bands)
constexpr int SG = 32;                    // NVIDIA sub-group
constexpr int NSG = 2, RT = NSG * TN;     // sub-groups a work-group, its rows (64)
constexpr int KC = 256;                   // inputs a step: 16 tiles of a band


template<typename P>
inline auto lptr(P& acc) { return acc.template get_multi_ptr<sycl::access::decorated::no>(); }

struct Buf { half* p = nullptr; size_t n = 0; };
std::mutex buf_mu;
std::unordered_map<size_t, Buf>& bufs() { static std::unordered_map<size_t, Buf> m; return m; }
// the rotated inputs of this queue's device (slot 0: x' for gate / up, slot 1: h' for down); an older, smaller buffer
// stays allocated for the graphs recorded with it
half* scratch(sycl::queue& q, size_t n, int slot) {
    std::lock_guard<std::mutex> lk(buf_mu);
    Buf& b = bufs()[std::hash<sycl::device>{}(q.get_device()) * 4 + (size_t) slot];
    if (b.n < n) {
        b.p = sycl::malloc_device<half>(n, q);
        b.n = b.p ? n : 0;
    }
    if (b.p == nullptr) throw std::runtime_error("tq experts (XMX): no scratch for the rotated inputs");
    return b.p;
}

struct View {
    int64_t ff;
    const uint8_t* gate;
    const uint8_t* up;
    const uint8_t* down;
    const half* gains;
    const uint8_t* gu_sign;
    const uint8_t* d_sign;
};
inline View view(const uint8_t* blob, int64_t N) {
    View v;
    v.ff = *(const int32_t*) blob;
    const uint8_t* b = blob + 256;
    const size_t gu = (size_t) (v.ff / 16) * (size_t) (N / 16) * kTqTileBytes;
    v.gate = b;
    v.up = b + gu;
    v.down = b + 2 * gu;
    v.gains = (const half*) (b + 3 * gu);
    v.gu_sign = (const uint8_t*) (v.gains + 2 * v.ff + N);
    v.d_sign = v.gu_sign + N / 8;
    return v;
}

// out[e][0, K) = Hb (sign * in[row(e)]) for every entry: K = N (gate / up: row = ent_tok[e]) or the expert's ff (down:
// row = e); a work-item 16 inputs; lanes past K still take part in the exchanges (K is a multiple of 64)
sycl::event rotate(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_tok, const float* in, int64_t ld_in, half* out,
                   int64_t ld_out, int64_t N, int64_t KMAX, bool down) {
    const size_t items = (size_t) ((KMAX / 16 + SG - 1) / SG * SG);
    // GB10 port: at most 512 group slots (cg is the entry capacity; the loop below walks g += the range), as TqMma2's
    static const size_t gmax = [] { const char* e = std::getenv("STRATA_TQ_MMA2_GMAX"); return e ? (size_t) std::atol(e) : (size_t) 512; }();
    const size_t cgs = gmax > 0 ? std::min(cg, gmax) : cg;
    return q.parallel_for(sycl::nd_range<2>({cgs, items}, {1, (size_t) SG}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t c = (int64_t) it.get_global_id(1);
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const View v = view((const uint8_t*) grp_ptr[g], N);
            const int64_t K = down ? v.ff : N;
            const bool live = c * 16 < K;
            const int64_t cc = live ? c : 0;
            const uint8_t* sgn = down ? v.d_sign : v.gu_sign;
            const uint32_t sb = (uint32_t) sgn[2 * cc] | ((uint32_t) sgn[2 * cc + 1] << 8);
            for (int e = grp_start[g]; e < grp_start[g + 1]; ++e) {
                const float* xp = in + (size_t) (down ? e : ent_tok[e]) * (size_t) ld_in + cc * 16;
                float xv[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) xv[i] = ((sb >> i) & 1) ? -xp[i] : xp[i];
#pragma unroll
                for (int hh = 1; hh < 16; hh <<= 1)
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        if (!(i & hh)) { const float a0 = xv[i], a1 = xv[i | hh]; xv[i] = a0 + a1; xv[i | hh] = a0 - a1; }
#pragma unroll
                for (int m = 1; m < 4; m <<= 1)
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float o = sycl::permute_group_by_xor(sg, xv[i], m);
                        xv[i] = (lane & m) ? o - xv[i] : xv[i] + o;
                    }
                if (live) {
                    sycl::vec<float, 16> v;
#pragma unroll
                    for (int i = 0; i < 16; ++i) v[i] = xv[i] * 0.125f;
                    *reinterpret_cast<sycl::vec<half, 16>*>(out + (size_t) e * (size_t) ld_out + c * 16) = v.convert<half>();
                }
            }
        }
    });
}

// DOWN false: gate and up (xr: x' [entry][N]; out gate / up [entry][FF]); DOWN true: down (xr: h' [entry][FF], K = ff;
// out [ent_dst][N])
template<int NP, bool DOWN, int GRF>
struct TqMma {
    const unsigned long long* grp_ptr;
    const int32_t* grp_start;
    const int32_t* n_groups;
    const int32_t* ent_dst;
    const half* xr;   // the rotated inputs, FP16, a row an entry (8 rows of room past the last)
    int64_t ldx;
    float* o0;
    float* o1;
    int64_t N, FF;
    int n_rt;   // row tiles of one projection (gate / up: FF / 64, both in the grid; down: N / 64)
    sycl::local_accessor<half, 1> wt;    // [RT][KC]: a row's KC inputs contiguous (B column-major)
    sycl::local_accessor<float, 1> ot;   // [8 * NP][RT]: the sums of the entries of a decode, across the k steps

    void operator()(sycl::nd_item<2> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int tid = (int) it.get_local_linear_id();
        const int rt = (int) it.get_group(1);
        const bool up = !DOWN && rt >= n_rt;
        const int64_t r0 = (int64_t) (up ? rt - n_rt : rt) * RT;
        constexpr int EP = TM * NP, WG = NSG * SG;   // EP: the entries one decode of the weights serves
        half* const wtp = lptr(wt).get();
        float* const otp = lptr(ot).get();
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const View v = view((const uint8_t*) grp_ptr[g], N);
            const int64_t ff = v.ff;
            if (!DOWN && r0 >= ff) continue;   // the same for the whole work-group
            const int64_t K = DOWN ? ff : N, NC = K / 16;
            const uint8_t* M = DOWN ? v.down : (up ? v.up : v.gate);
            const int bl = lane >> 4, l16 = lane & 15;   // this lane's band of the sub-group's two, its tile
            const int64_t band = r0 / 16 + 2 * sgid + bl;
            const uint8_t* bandp = M + (size_t) band * (size_t) NC * kTqTileBytes;
            const half* gains = v.gains + (DOWN ? 2 * ff : (up ? ff : 0));
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += EP) {
                const int n = sycl::min(EP, e1 - e), np = (n + TM - 1) / TM;
                // the sums live in local memory (each sub-group its own 16 columns): a decode of the weights serves up
                // to NP passes of 8 entries, and the decode is the cost (1 to 8 entries an expert take the same time)
                for (int idx = lane; idx < np * TM * TN; idx += SG) otp[(idx / TN) * RT + sgid * TN + idx % TN] = 0.0f;
                for (int64_t k0 = 0; k0 < K; k0 += KC) {
                    // the weights: lane = tile (k0 / 16 + lane) of the sub-group's band, as FP16 (value, no gain)
                    const int64_t c0 = k0 / 16;
                    const int nt = (int) sycl::min<int64_t>(16, NC - c0);   // a multiple of 4
                    half* w = wtp + (size_t) (sgid * TN + bl * 16) * KC + l16 * 16;
                    if (l16 < nt) {
                        const uint32_t* p = (const uint32_t*) (bandp + (size_t) c0 * kTqTileBytes) + l16;
                        uint32_t d[16];
#pragma unroll
                        for (int i = 0; i < 16; ++i) { d[i] = *p; p += nt; }   // steps by nt (AGENTS.md: IGC's GEP simplification)
                        const uint32_t d16 = ((const uint16_t*) (p - l16))[l16];
                        auto tw = [&](int k) -> uint32_t { return k < 32 ? ((d[k >> 1] >> (16 * (k & 1))) & 0xFFFFu) : d16; };
                        // the code's integer value (-510 .. 510, exact in FP16; its 1/147.8 goes on the row gains)
                        auto val = [](uint32_t z) { return tq_bsum510(z * 34038481u + 76625530u); };
                        // a row's 16 values, then one conversion and one 32-byte store (inserting FP16 lanes one by one
                        // into a vector was 10x slower)
                        auto put = [&](const int32_t* f, int r) {
                            sycl::vec<int32_t, 16> v;
#pragma unroll
                            for (int i = 0; i < 16; ++i) v[i] = f[i];
                            *reinterpret_cast<sycl::vec<half, 16>*>(w + r * KC) = v.convert<half>();
                        };
                        uint32_t s = tw(0);
                        int32_t row[16];
                        row[0] = val(s);
#pragma unroll
                        for (int wd = 0; wd < 32; ++wd) {
                            const uint32_t U = s | (tw(1 + wd) << 14);
#pragma unroll
                            for (int u = 1; u <= 8; ++u) {
                                const int t = wd * 8 + u;
                                if (t > 255) break;
                                row[t & 15] = val(tq_bits(U, 2 * u, 14));
                                if ((t & 15) == 15) put(row, t >> 4);
                            }
                            s = (U >> 16) & 0x3FFFu;
                        }
                    } else {
#pragma unroll
                        for (int r = 0; r < 16; ++r) *reinterpret_cast<sycl::vec<half, 16>*>(w + r * KC) = sycl::vec<half, 16>((half) 0.0f);
                    }
                    sycl::group_barrier(it.get_group());
                    for (int i = 0; i < np; ++i) {
                        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c;
                        mx::joint_matrix_load(sg, c, lptr(ot) + i * TM * RT + sgid * TN, RT, mx::layout::row_major);
                        // the entries' rotated inputs straight from global memory (rows past n: other entries' or the
                        // room's, whose products land in rows that are not stored)
                        const half* xe = xr + (size_t) (e + i * TM) * (size_t) ldx + k0;
                        const int kc = (int) sycl::min<int64_t>(KC, K - k0);   // a multiple of 64
                        for (int kk = 0; kk < kc; kk += TK) {
                            mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> bm;
                            mx::joint_matrix_load(sg, bm, lptr(wt) + sgid * TN * KC + kk, KC);
                            mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> a;
                            mx::joint_matrix_load(sg, a, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                  sycl::access::decorated::no>(xe + kk), (size_t) ldx);
                            mx::joint_matrix_mad(sg, c, a, bm, c);
                        }
                        mx::joint_matrix_store(sg, c, lptr(ot) + i * TM * RT + sgid * TN, RT, mx::layout::row_major);
                    }
                    sycl::group_barrier(it.get_group());
                }
                for (int idx = tid; idx < n * RT; idx += WG) {
                    const int er = idx / RT, rr = idx % RT;
                    const int64_t r = r0 + rr;
                    const float y = otp[er * RT + rr] * ((float) gains[r] * (1.0f / 147.8f));
                    if (DOWN) o0[(size_t) ent_dst[e + er] * (size_t) N + r] = y;
                    else (up ? o1 : o0)[(size_t) (e + er) * (size_t) FF + r] = y;
                }
                sycl::group_barrier(it.get_group());
            }
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<SG>}; }
};

template<int NP, bool DOWN, int GRF>
sycl::event launch(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_dst, const half* xr, int64_t ldx, float* o0, float* o1,
                   int64_t N, int64_t FF) {
    const int n_rt = (int) ((DOWN ? N : FF) / RT);
    const size_t tiles = (size_t) (DOWN ? n_rt : 2 * n_rt);
    return q.submit([&](sycl::handler& h) {
        TqMma<NP, DOWN, GRF> k{grp_ptr, grp_start, n_groups, ent_dst, xr, ldx, o0, o1, N, FF, n_rt,
                          sycl::local_accessor<half, 1>(sycl::range<1>(RT * KC), h),
                          sycl::local_accessor<float, 1>(sycl::range<1>(TM * NP * RT), h)};
        h.parallel_for(sycl::nd_range<2>({cg, tiles * NSG * SG}, {1, (size_t) NSG * SG}), k);
    });
}

size_t slm_need(int np) { return (size_t) (RT * KC) * sizeof(half) + (size_t) (TM * np * RT) * sizeof(float); }

// the passes of 8 entries one decode serves: STRATA_TQ_XMX_NP (1, 2, 4 or 8), else 8 where the local memory holds it
// (48 KB), else less (B70, 64 experts x 32 / 64 entries: NP 1 32.8 / -, 2 22.7 / -, 4 17.5 / 34.3, 8 17.5 / 29.3 ms)
int passes(sycl::queue& q) {
    static const long want = [] { const char* v = std::getenv("STRATA_TQ_XMX_NP"); return v ? std::strtol(v, nullptr, 10) : 4L; }();
    const size_t lm = q.get_device().get_info<sycl::info::device::local_mem_size>();
    int np = want >= 8 ? 8 : want >= 4 ? 4 : want >= 2 ? 2 : 1;
    while (np > 1 && slm_need(np) > lm) np /= 2;
    return np;
}

template<bool DOWN>
sycl::event launch_np(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                      const int32_t* n_groups, const int32_t* ent_dst, const half* xr, int64_t ldx, float* o0, float* o1,
                      int64_t N, int64_t FF) {
    switch (passes(q)) {
        case 8: return launch<8, DOWN, 128>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, xr, ldx, o0, o1, N, FF);
        case 4: return launch<4, DOWN, 128>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, xr, ldx, o0, o1, N, FF);
        case 2: return launch<2, DOWN, 128>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, xr, ldx, o0, o1, N, FF);
        default: return launch<1, DOWN, 128>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, xr, ldx, o0, o1, N, FF);
    }
}

// GB10 port, STRATA_TQ_MMA2 (11-22, default on): llama.cpp ink2v's tq_prefill_mma layout on the same blob and inputs.  A work-group
// of 4 sub-groups (128 lanes) takes 64 rows (4 bands; sub-group s computes band s) of one expert's projection and up to 64
// entries; per 64 inputs (4 tiles a band) the 16 tiles' codes are rebuilt as 66-byte images in local memory (the blob keeps
// 16-tile blocks as dword i of tile l at [i * nt + l], the 2-byte tails after them), each lane decodes 2 tile rows (16
// weights each, the code's integer value in FP16) into the weight tile, the entries' rotated inputs are staged in local
// memory, and each sub-group multiplies its band on the tensor cores (m16n16k16) with the sums in registers for the whole
// K.  The next step's codes and inputs are loaded while a step decodes and multiplies.  About 20 KB of local memory a
// work-group (the old kernel 48 KB: 4 sub-groups an SM), so the work-groups of an SM overlap decodes and products.
constexpr int M2_NE = 64, M2_KS = 64, M2_LD = M2_KS + 8, M2_TPS = M2_KS / 16;
constexpr size_t m2_lm_floats() { return (size_t) RT * M2_LD + 264 + 64; }   // sA + sB halves, codes (528 u16), gains
template<bool DOWN>
struct TqMma2 {
    const unsigned long long* grp_ptr;
    const int32_t* grp_start;
    const int32_t* n_groups;
    const int32_t* ent_dst;
    const half* xr;
    int64_t ldx;
    float* o0;
    float* o1;
    int64_t N, FF;
    int n_rt;
    sycl::local_accessor<float, 1> lm;

    // GB10 port: 128 lanes, at least STRATA_TQ_MMA2_MINB work-groups an SM (launch bounds: <= 168 registers for 3; 214 without
    // gave 2 an SM, 4 spilled 140 bytes; 4,713-token prompt 457-463 / 479-482 / 464-468 tok/s)
#ifndef STRATA_TQ_MMA2_MINB
#define STRATA_TQ_MMA2_MINB 3
#endif
    [[sycl::reqd_work_group_size(1, 128)]] [[intel::min_work_groups_per_cu(STRATA_TQ_MMA2_MINB)]]
    void operator()(sycl::nd_item<2> it) const {
        using namespace sycl;
        const sub_group sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id();
        const int tid = (int) it.get_local_linear_id();
        const int rt = (int) it.get_group(1);
        const bool up = !DOWN && rt >= n_rt;
        const int64_t r0 = (int64_t) (up ? rt - n_rt : rt) * RT;
        float* const lmp = lptr(lm).get();
        half* const sA = reinterpret_cast<half*>(lmp);            // [64 rows][LD]
        half* const sB = sA + RT * M2_LD;                          // [64 entries][LD]
        float* const sC = lmp;                                     // [64 entries][64 rows], after the last step
        uint16_t* const sT = reinterpret_cast<uint16_t*>(lmp + RT * M2_LD);   // 16 tiles x 33 u16
        float* const sgn = lmp + RT * M2_LD + 264;                 // [64] row gains / 147.8
        auto lp = [](auto* p) { return address_space_cast<access::address_space::local_space, access::decorated::no>(p); };
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const View v = view((const uint8_t*) grp_ptr[g], N);
            const int64_t ff = v.ff;
            if (!DOWN && r0 >= ff) continue;   // the same for the whole work-group
            const int64_t K = DOWN ? ff : N, NC = K / 16;
            const uint8_t* M = DOWN ? v.down : (up ? v.up : v.gate);
            const half* gains = v.gains + (DOWN ? 2 * ff : (up ? ff : 0));
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            group_barrier(it.get_group());   // the previous group's epilogue is done with sC / the gains
            if (tid < RT) sgn[tid] = (float) gains[r0 + tid] * (1.0f / 147.8f);
            // lanes 0..63: a dword-quad (band tid / 16, dword tid % 16 of the step's 4 tiles); 64..67: a band's 4 tails
            auto load_codes = [&](int64_t k0, vec<uint32_t, 4>& cq) {
                const int64_t c0 = k0 / 16, blk = c0 / 16;
                const int l0 = (int) (c0 % 16);
                const int nt = (int) sycl::min<int64_t>(16, NC - blk * 16);   // a multiple of 4
                cq = vec<uint32_t, 4>(0u);
                if (tid < 68) {
                    const int b = tid < 64 ? (tid >> 4) : tid - 64;
                    const uint8_t* bp = M + ((size_t) (r0 / 16 + b) * (size_t) NC + (size_t) blk * 16) * kTqTileBytes;
                    // the blob is only 4-byte aligned in its slot: dword and 2-byte loads (the old kernel's)
                    if (tid < 64) {
                        const uint32_t* w = reinterpret_cast<const uint32_t*>(bp) + (size_t) (tid & 15) * nt + l0;
                        cq = vec<uint32_t, 4>(w[0], w[1], w[2], w[3]);
                    } else {
                        const uint16_t* t = reinterpret_cast<const uint16_t*>(bp + 64 * (size_t) nt) + l0;
                        cq = vec<uint32_t, 4>((uint32_t) t[0] | ((uint32_t) t[1] << 16), (uint32_t) t[2] | ((uint32_t) t[3] << 16), 0u, 0u);
                    }
                }
            };
            for (int e = e0; e < e1; e += M2_NE) {
                const int ne = sycl::min(M2_NE, e1 - e);
                const int nf = (ne + 15) / 16;   // the fragments with entries (the others' products are not stored)
                // one of this lane's 4 input quads (returned by value: an array by reference went to the stack)
                auto load_x1 = [&](int64_t k0, int q4) -> vec<uint32_t, 4> {
                    const int qq = tid + q4 * 128, n = qq >> 3, c = qq & 7;
                    if (n < ne) return *reinterpret_cast<const vec<uint32_t, 4>*>(xr + (size_t) (e + n) * (size_t) ldx + k0 + c * 8);
                    return vec<uint32_t, 4>(0u);
                };
                mx::joint_matrix<sub_group, float, mx::use::accumulator, 16, 16> acc[4];
#pragma unroll
                for (int f = 0; f < 4; ++f) mx::joint_matrix_fill(sg, acc[f], 0.0f);
                vec<uint32_t, 4> cq;
                load_codes(0, cq);
                vec<uint32_t, 4> cx0 = load_x1(0, 0), cx1 = load_x1(0, 1), cx2 = load_x1(0, 2), cx3 = load_x1(0, 3);
                for (int64_t k0 = 0; k0 < K; k0 += M2_KS) {
                    // the codes as 66-byte tile images: tile (band b, j): u16 [2i, 2i + 1] = dword i, [32] = the tail
                    if (tid < 64) {
                        const int b = tid >> 4, i = tid & 15;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            uint16_t* tb = sT + (b * 4 + j) * 33;
                            tb[2 * i] = (uint16_t) (cq[j] & 0xFFFFu);
                            tb[2 * i + 1] = (uint16_t) (cq[j] >> 16);
                        }
                    } else if (tid < 68) {
                        const int b = tid - 64;
                        sT[(b * 4 + 0) * 33 + 32] = (uint16_t) (cq[0] & 0xFFFFu);
                        sT[(b * 4 + 1) * 33 + 32] = (uint16_t) (cq[0] >> 16);
                        sT[(b * 4 + 2) * 33 + 32] = (uint16_t) (cq[1] & 0xFFFFu);
                        sT[(b * 4 + 3) * 33 + 32] = (uint16_t) (cq[1] >> 16);
                    }
                    {
                        const int c = tid & 7;
                        *reinterpret_cast<vec<uint32_t, 4>*>(sB + ((tid >> 3) + 0) * M2_LD + c * 8) = cx0;
                        *reinterpret_cast<vec<uint32_t, 4>*>(sB + ((tid >> 3) + 16) * M2_LD + c * 8) = cx1;
                        *reinterpret_cast<vec<uint32_t, 4>*>(sB + ((tid >> 3) + 32) * M2_LD + c * 8) = cx2;
                        *reinterpret_cast<vec<uint32_t, 4>*>(sB + ((tid >> 3) + 48) * M2_LD + c * 8) = cx3;
                    }
                    group_barrier(it.get_group());
                    if (k0 + M2_KS < K) {   // in flight meanwhile
                        load_codes(k0 + M2_KS, cq);
                        cx0 = load_x1(k0 + M2_KS, 0); cx1 = load_x1(k0 + M2_KS, 1); cx2 = load_x1(k0 + M2_KS, 2); cx3 = load_x1(k0 + M2_KS, 3);
                    }
                    // decode: unit u = (band b, tile jt, tile row r), the row's 16 weights from its 64-bit window
#pragma unroll
                    for (int q2 = 0; q2 < 2; ++q2) {
                        const int u = tid + q2 * 128, b = u >> 6, jt = (u >> 4) & 3, r = u & 15;
                        const uint16_t* tb = sT + (b * 4 + jt) * 33;
                        auto SB = [&](int kk) -> uint32_t { return (uint32_t) tb[1 + 2 * kk] | ((uint32_t) tb[2 + 2 * kk] << 16); };
                        const uint32_t sbr = SB(r);
                        const uint32_t lo = r == 0 ? (((uint32_t) tb[0] & 0x3FFFu) | (sbr << 14)) : ((SB(r - 1) >> 18) | (sbr << 14));
                        const uint32_t hi = (sbr >> 18) | (r < 15 ? (SB(r + 1) << 14) : 0u);
                        const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                        uint32_t o[8];
#pragma unroll
                        for (int c = 0; c < 16; c += 2) {
                            const uint32_t z0 = (uint32_t) (ch >> (2 * c)) & 0x3FFFu, z1 = (uint32_t) (ch >> (2 * c + 2)) & 0x3FFFu;
                            const uint32_t b0 = tq_bsum(z0 * 34038481u + 76625530u), b1 = tq_bsum(z1 * 34038481u + 76625530u);
                            // FP16 bits of 1024 + b, minus 1534: b - 510 exactly (the code's integer value)
                            vec<half, 2> h2 = bit_cast<vec<half, 2>>((0x64006400u | b0) | (b1 << 16));
                            h2 -= vec<half, 2>((half) 1534.0f);
                            o[c / 2] = bit_cast<uint32_t>(h2);
                        }
                        vec<uint32_t, 4>* dp = reinterpret_cast<vec<uint32_t, 4>*>(sA + (b * 16 + r) * M2_LD + jt * 16);
                        dp[0] = vec<uint32_t, 4>(o[0], o[1], o[2], o[3]);
                        dp[1] = vec<uint32_t, 4>(o[4], o[5], o[6], o[7]);
                    }
                    group_barrier(it.get_group());
#pragma unroll
                    for (int kk = 0; kk < M2_KS; kk += 16) {
                        mx::joint_matrix<sub_group, half, mx::use::a, 16, 16, mx::layout::row_major> a;
                        mx::joint_matrix_load(sg, a, lp(sA + (sgid * 16) * M2_LD + kk), M2_LD);
#pragma unroll
                        for (int f = 0; f < 4; ++f) {
                            if (f < nf) {   // uniform in the work-group; f stays a constant (no break: acc stays in registers)
                                mx::joint_matrix<sub_group, half, mx::use::b, 16, 16, mx::layout::col_major> bm;
                                mx::joint_matrix_load(sg, bm, lp(sB + (f * 16) * M2_LD + kk), M2_LD);
                                mx::joint_matrix_mad(sg, acc[f], a, bm, acc[f]);
                            }
                        }
                    }
                    group_barrier(it.get_group());
                }
                // the sums: sC[entry][row], element (row m of band sgid, entry n of fragment f)
#pragma unroll
                for (int f = 0; f < 4; ++f)
                    if (f < nf) mx::joint_matrix_store(sg, acc[f], lp(sC + (f * 16) * RT + sgid * 16), RT, mx::layout::col_major);
                group_barrier(it.get_group());
                for (int idx = tid; idx < ne * RT; idx += 128) {
                    const int er = idx / RT, rr = idx % RT;
                    const int64_t r = r0 + rr;
                    const float y = sC[er * RT + rr] * sgn[rr];
                    if (DOWN) o0[(size_t) ent_dst[e + er] * (size_t) N + r] = y;
                    else (up ? o1 : o0)[(size_t) (e + er) * (size_t) FF + r] = y;
                }
                group_barrier(it.get_group());
            }
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<SG>}; }
};

// GB10 port (11-22): the tensor-core steps of TqMma2 in PTX.  This DPC++'s joint_matrix_load for a / b takes a generic
// address (wmma.load without .shared: 89 generic 32-bit loads a step in SASS, no ldmatrix); ldmatrix.x4 from the shared
// window and mma.sync m16n8k16 are the instructions llama.cpp's tq_prefill_mma uses.  The products are the same
// HMMA.16816.F32 (joint_matrix's m16n16k16 is two of them) over the same k order, so the sums are the same bits.
static inline __attribute__((always_inline)) uint32_t smem_u32(const void* p) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    uint32_t a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
#else
    return (uint32_t) (uintptr_t) p;
#endif
}
static inline __attribute__((always_inline)) void ldsm_x4(uint32_t addr, uint32_t* r) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
#else
    r[0] = r[1] = r[2] = r[3] = addr;
#endif
}
static inline __attribute__((always_inline)) void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    c[0] += (float) (a[0] + b0 + b1);
#endif
}
template<bool DOWN>
struct TqMma3 {
    const unsigned long long* grp_ptr;
    const int32_t* grp_start;
    const int32_t* n_groups;
    const int32_t* ent_dst;
    const half* xr;
    int64_t ldx;
    float* o0;
    float* o1;
    int64_t N, FF;
    int n_rt;
    sycl::local_accessor<float, 1> lm;

    // GB10 port: 128 lanes, at least STRATA_TQ_MMA2_MINB work-groups an SM (launch bounds: <= 168 registers for 3; 214 without
    // gave 2 an SM, 4 spilled 140 bytes; 4,713-token prompt 457-463 / 479-482 / 464-468 tok/s)
#ifndef STRATA_TQ_MMA2_MINB
#define STRATA_TQ_MMA2_MINB 3
#endif
    [[sycl::reqd_work_group_size(1, 128)]] [[intel::min_work_groups_per_cu(STRATA_TQ_MMA2_MINB)]]
    void operator()(sycl::nd_item<2> it) const {
        using namespace sycl;
        const sub_group sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id();
        const int tid = (int) it.get_local_linear_id();
        const int rt = (int) it.get_group(1);
        const bool up = !DOWN && rt >= n_rt;
        const int64_t r0 = (int64_t) (up ? rt - n_rt : rt) * RT;
        float* const lmp = lptr(lm).get();
        half* const sA = reinterpret_cast<half*>(lmp);            // [64 rows][LD]
        half* const sB = sA + RT * M2_LD;                          // [64 entries][LD]
        float* const sC = lmp;                                     // [64 entries][64 rows], after the last step
        uint16_t* const sT = reinterpret_cast<uint16_t*>(lmp + RT * M2_LD);   // 16 tiles x 33 u16
        float* const sgn = lmp + RT * M2_LD + 264;                 // [64] row gains / 147.8
        const int lane = (int) sg.get_local_linear_id();
        // ldmatrix row addresses (shared window): A (row-major 16 x 16 of band sgid): lanes 0..15 rows 0..15 at k 0, 16..31
        // the same rows at k 8 (a0..a3); B ([entry][k], the col operand): lanes 8i..8i+7 entries (i / 2) * 8 + 0..7 at k
        // (i % 2) * 8 (b0, b1 of entries 0..7, then of 8..15)
        const uint32_t aA = smem_u32(sA + (sgid * 16 + (lane & 15)) * M2_LD + (lane >> 4) * 8);
        const uint32_t aB = smem_u32(sB + ((lane >> 4) * 8 + (lane & 7)) * M2_LD + ((lane >> 3) & 1) * 8);
        for (int g = (int) it.get_group(0); g < *n_groups; g += (int) it.get_group_range(0)) {
            const View v = view((const uint8_t*) grp_ptr[g], N);
            const int64_t ff = v.ff;
            if (!DOWN && r0 >= ff) continue;   // the same for the whole work-group
            const int64_t K = DOWN ? ff : N, NC = K / 16;
            const uint8_t* M = DOWN ? v.down : (up ? v.up : v.gate);
            const half* gains = v.gains + (DOWN ? 2 * ff : (up ? ff : 0));
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            group_barrier(it.get_group());   // the previous group's epilogue is done with sC / the gains
            if (tid < RT) sgn[tid] = (float) gains[r0 + tid] * (1.0f / 147.8f);
            // lanes 0..63: a dword-quad (band tid / 16, dword tid % 16 of the step's 4 tiles); 64..67: a band's 4 tails
            auto load_codes = [&](int64_t k0, vec<uint32_t, 4>& cq) {
                const int64_t c0 = k0 / 16, blk = c0 / 16;
                const int l0 = (int) (c0 % 16);
                const int nt = (int) sycl::min<int64_t>(16, NC - blk * 16);   // a multiple of 4
                cq = vec<uint32_t, 4>(0u);
                if (tid < 68) {
                    const int b = tid < 64 ? (tid >> 4) : tid - 64;
                    const uint8_t* bp = M + ((size_t) (r0 / 16 + b) * (size_t) NC + (size_t) blk * 16) * kTqTileBytes;
                    // the blob is only 4-byte aligned in its slot: dword and 2-byte loads (the old kernel's)
                    if (tid < 64) {
                        const uint32_t* w = reinterpret_cast<const uint32_t*>(bp) + (size_t) (tid & 15) * nt + l0;
                        cq = vec<uint32_t, 4>(w[0], w[1], w[2], w[3]);
                    } else {
                        const uint16_t* t = reinterpret_cast<const uint16_t*>(bp + 64 * (size_t) nt) + l0;
                        cq = vec<uint32_t, 4>((uint32_t) t[0] | ((uint32_t) t[1] << 16), (uint32_t) t[2] | ((uint32_t) t[3] << 16), 0u, 0u);
                    }
                }
            };
            for (int e = e0; e < e1; e += M2_NE) {
                const int ne = sycl::min(M2_NE, e1 - e);
                const int nf = (ne + 15) / 16;   // the fragments with entries (the others' products are not stored)
                // one of this lane's 4 input quads (returned by value: an array by reference went to the stack)
                auto load_x1 = [&](int64_t k0, int q4) -> vec<uint32_t, 4> {
                    const int qq = tid + q4 * 128, n = qq >> 3, c = qq & 7;
                    if (n < ne) return *reinterpret_cast<const vec<uint32_t, 4>*>(xr + (size_t) (e + n) * (size_t) ldx + k0 + c * 8);
                    return vec<uint32_t, 4>(0u);
                };
                float acc[32];   // [fragment f][n half][4]: mma.m16n8k16's C fragments
#pragma unroll
                for (int i = 0; i < 32; ++i) acc[i] = 0.0f;
                vec<uint32_t, 4> cq;
                load_codes(0, cq);
                vec<uint32_t, 4> cx0 = load_x1(0, 0), cx1 = load_x1(0, 1), cx2 = load_x1(0, 2), cx3 = load_x1(0, 3);
                for (int64_t k0 = 0; k0 < K; k0 += M2_KS) {
                    // the codes as 66-byte tile images: tile (band b, j): u16 [2i, 2i + 1] = dword i, [32] = the tail
                    if (tid < 64) {
                        const int b = tid >> 4, i = tid & 15;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            uint16_t* tb = sT + (b * 4 + j) * 33;
                            tb[2 * i] = (uint16_t) (cq[j] & 0xFFFFu);
                            tb[2 * i + 1] = (uint16_t) (cq[j] >> 16);
                        }
                    } else if (tid < 68) {
                        const int b = tid - 64;
                        sT[(b * 4 + 0) * 33 + 32] = (uint16_t) (cq[0] & 0xFFFFu);
                        sT[(b * 4 + 1) * 33 + 32] = (uint16_t) (cq[0] >> 16);
                        sT[(b * 4 + 2) * 33 + 32] = (uint16_t) (cq[1] & 0xFFFFu);
                        sT[(b * 4 + 3) * 33 + 32] = (uint16_t) (cq[1] >> 16);
                    }
                    {
                        const int c = tid & 7;
                        *reinterpret_cast<vec<uint32_t, 4>*>(sB + ((tid >> 3) + 0) * M2_LD + c * 8) = cx0;
                        *reinterpret_cast<vec<uint32_t, 4>*>(sB + ((tid >> 3) + 16) * M2_LD + c * 8) = cx1;
                        *reinterpret_cast<vec<uint32_t, 4>*>(sB + ((tid >> 3) + 32) * M2_LD + c * 8) = cx2;
                        *reinterpret_cast<vec<uint32_t, 4>*>(sB + ((tid >> 3) + 48) * M2_LD + c * 8) = cx3;
                    }
                    group_barrier(it.get_group());
                    if (k0 + M2_KS < K) {   // in flight meanwhile
                        load_codes(k0 + M2_KS, cq);
                        cx0 = load_x1(k0 + M2_KS, 0); cx1 = load_x1(k0 + M2_KS, 1); cx2 = load_x1(k0 + M2_KS, 2); cx3 = load_x1(k0 + M2_KS, 3);
                    }
                    // decode: unit u = (band b, tile jt, tile row r), the row's 16 weights from its 64-bit window
#pragma unroll
                    for (int q2 = 0; q2 < 2; ++q2) {
                        const int u = tid + q2 * 128, b = u >> 6, jt = (u >> 4) & 3, r = u & 15;
                        const uint16_t* tb = sT + (b * 4 + jt) * 33;
                        auto SB = [&](int kk) -> uint32_t { return (uint32_t) tb[1 + 2 * kk] | ((uint32_t) tb[2 + 2 * kk] << 16); };
                        const uint32_t sbr = SB(r);
                        const uint32_t lo = r == 0 ? (((uint32_t) tb[0] & 0x3FFFu) | (sbr << 14)) : ((SB(r - 1) >> 18) | (sbr << 14));
                        const uint32_t hi = (sbr >> 18) | (r < 15 ? (SB(r + 1) << 14) : 0u);
                        const uint64_t ch = (uint64_t) lo | ((uint64_t) hi << 32);
                        uint32_t o[8];
#pragma unroll
                        for (int c = 0; c < 16; c += 2) {
                            const uint32_t z0 = (uint32_t) (ch >> (2 * c)) & 0x3FFFu, z1 = (uint32_t) (ch >> (2 * c + 2)) & 0x3FFFu;
                            const uint32_t b0 = tq_bsum(z0 * 34038481u + 76625530u), b1 = tq_bsum(z1 * 34038481u + 76625530u);
                            // FP16 bits of 1024 + b, minus 1534: b - 510 exactly (the code's integer value)
                            vec<half, 2> h2 = bit_cast<vec<half, 2>>((0x64006400u | b0) | (b1 << 16));
                            h2 -= vec<half, 2>((half) 1534.0f);
                            o[c / 2] = bit_cast<uint32_t>(h2);
                        }
                        vec<uint32_t, 4>* dp = reinterpret_cast<vec<uint32_t, 4>*>(sA + (b * 16 + r) * M2_LD + jt * 16);
                        dp[0] = vec<uint32_t, 4>(o[0], o[1], o[2], o[3]);
                        dp[1] = vec<uint32_t, 4>(o[4], o[5], o[6], o[7]);
                    }
                    group_barrier(it.get_group());
#pragma unroll
                    for (int kk = 0; kk < M2_KS; kk += 16) {
                        uint32_t a[4];
                        ldsm_x4(aA + (uint32_t) (kk * 2), a);
#pragma unroll
                        for (int f = 0; f < 4; ++f) {
                            if (f < nf) {   // uniform in the work-group; f stays a constant (no break: acc stays in registers)
                                uint32_t b[4];
                                ldsm_x4(aB + (uint32_t) ((f * 16 * M2_LD + kk) * 2), b);
                                mma16816(acc + f * 8, a, b[0], b[1]);       // entries f * 16 + 0..7
                                mma16816(acc + f * 8 + 4, a, b[2], b[3]);   // entries f * 16 + 8..15
                            }
                        }
                    }
                    group_barrier(it.get_group());
                }
                // the sums: sC[entry][row], element (row m of band sgid, entry n of fragment f)
#pragma unroll
                for (int f = 0; f < 4; ++f)
                    if (f < nf)
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            // C fragment: c0, c1 (row g, entries 2t, 2t + 1), c2, c3 (row g + 8, the same entries)
                            const float* c = acc + f * 8 + hh * 4;
                            const int n = f * 16 + hh * 8 + 2 * (lane & 3), m = sgid * 16 + (lane >> 2);
                            sC[n * RT + m] = c[0];
                            sC[(n + 1) * RT + m] = c[1];
                            sC[n * RT + m + 8] = c[2];
                            sC[(n + 1) * RT + m + 8] = c[3];
                        }
                group_barrier(it.get_group());
                for (int idx = tid; idx < ne * RT; idx += 128) {
                    const int er = idx / RT, rr = idx % RT;
                    const int64_t r = r0 + rr;
                    const float y = sC[er * RT + rr] * sgn[rr];
                    if (DOWN) o0[(size_t) ent_dst[e + er] * (size_t) N + r] = y;
                    else (up ? o1 : o0)[(size_t) (e + er) * (size_t) FF + r] = y;
                }
                group_barrier(it.get_group());
            }
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<SG>}; }
};

template<bool DOWN>
sycl::event launch2(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                    const int32_t* n_groups, const int32_t* ent_dst, const half* xr, int64_t ldx, float* o0, float* o1,
                    int64_t N, int64_t FF) {
    const int n_rt = (int) ((DOWN ? N : FF) / RT);
    const size_t tiles = (size_t) (DOWN ? n_rt : 2 * n_rt);
    return q.submit([&](sycl::handler& h) {
        // STRATA_TQ_MMA2_PTX=0: joint_matrix (TqMma2), else ldmatrix + mma.sync (TqMma3); the same bits
        static const bool ptx = [] { const char* e = std::getenv("STRATA_TQ_MMA2_PTX"); return !e || std::atoi(e) != 0; }();
        static const size_t gmax = [] { const char* e = std::getenv("STRATA_TQ_MMA2_GMAX"); return e ? (size_t) std::atol(e) : (size_t) 512; }();
        const size_t cgs = gmax > 0 ? std::min(cg, gmax) : cg;
        if (ptx) {
            TqMma3<DOWN> k{grp_ptr, grp_start, n_groups, ent_dst, xr, ldx, o0, o1, N, FF, n_rt,
                           sycl::local_accessor<float, 1>(sycl::range<1>(m2_lm_floats()), h)};
            h.parallel_for(sycl::nd_range<2>({cgs, tiles * 128}, {1, (size_t) 128}), k);
            return;
        }
        TqMma2<DOWN> k{grp_ptr, grp_start, n_groups, ent_dst, xr, ldx, o0, o1, N, FF, n_rt,
                       sycl::local_accessor<float, 1>(sycl::range<1>(m2_lm_floats()), h)};
        // the group slots: cg is the window's entry capacity (16,384 at 2,048 tokens: 1.18 M work-groups, nearly all
        // empty), but a layer has at most its expert count of groups and a work-group walks g += its range: 512 slots
        // (STRATA_TQ_MMA2_GMAX; 0: cg)
        h.parallel_for(sycl::nd_range<2>({cgs, tiles * 128}, {1, (size_t) 128}), k);
    });
}
// default on (STRATA_TQ_MMA2=0: the B70 layout above).  GB10, public TQ1, a 4,719-token prompt: 152 -> 343-349 tok/s; the
// logits of a 4,713-token prompt and 3 tokens after it are the same bits
bool tq_mma2_on() { static const bool on = [] { const char* e = std::getenv("STRATA_TQ_MMA2"); return !e || std::atoi(e) != 0; }(); return on; }

}  // namespace

bool tq_xmx_on(sycl::queue& q, int64_t cap_entries, int64_t N, int64_t FF) {
    // STRATA_TQ_XMX: 0 off, else the entries a window needs (default 64: a window of 8 tokens' top-8; the decode and the
    // speculative windows take tq_kernels.cpp's kernels)
    static const long min_e = [] { const char* v = std::getenv("STRATA_TQ_XMX"); return v ? std::strtol(v, nullptr, 10) : 64L; }();
    if (min_e <= 0 || cap_entries < min_e || N % RT != 0 || FF % RT != 0) return false;   // GB10: tensor cores always
    return q.get_device().get_info<sycl::info::device::local_mem_size>() >= slm_need(1);
}

sycl::event tq_xmx_gu(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                      const int32_t* n_groups, const int32_t* ent_tok, int64_t cap_entries, const float* x, float* gate,
                      float* up, int64_t N, int64_t FF) {
    half* xr = scratch(q, (size_t) (cap_entries + TM) * (size_t) N, 0);
    rotate(q, cg, grp_ptr, grp_start, n_groups, ent_tok, x, N, xr, N, N, N, false);
    // the default register file: 16 entries' sums in registers in the large one ran 1.6x slower (half the threads; B70,
    // 64 experts x 5 entries); the sums are in local memory now
    if (tq_mma2_on()) return launch2<false>(q, cg, grp_ptr, grp_start, n_groups, nullptr, xr, N, gate, up, N, FF);
    return launch_np<false>(q, cg, grp_ptr, grp_start, n_groups, nullptr, xr, N, gate, up, N, FF);
}

sycl::event tq_xmx_down(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start,
                        const int32_t* n_groups, const int32_t* ent_dst, int64_t cap_entries, const float* h, float* out,
                        int64_t N, int64_t FF) {
    half* hr = scratch(q, (size_t) (cap_entries + TM) * (size_t) FF, 1);
    rotate(q, cg, grp_ptr, grp_start, n_groups, nullptr, h, FF, hr, FF, N, FF, true);
    if (tq_mma2_on()) return launch2<true>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hr, FF, out, nullptr, N, FF);
    return launch_np<true>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, hr, FF, out, nullptr, N, FF);
}

}  // namespace strata::kernels
