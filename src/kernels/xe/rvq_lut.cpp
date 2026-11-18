// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/rvq_lut.cpp - b70-local (11-18): Inkling 2's RVQ2X8 experts for one token (the decode) by lookup
// tables.  A weight row's 8 weights under code c are book1[c >> 8] + book2[c & 255], and the books are the layer's
// (blk.N.ffn_exps.rvq_book, the same in every blob), so for an input x the dot of a row with x is the sum over its codes
// of T1[j][c >> 8] + T2[j][c & 255], T[j][i] = book_i . x[8j .. 8j + 8].  A work-group takes 16 codes (128 inputs) of
// the rows: it builds those codes' table (16 x 512 FP32, 32 KiB of local memory) once and then reads two table entries
// a code for every row of every expert it covers; the per-chunk sums go to a partial buffer and a second kernel adds
// them up with the row gains.  The old kernel (iq_kernels.cpp rvq_dot_multi) read 64 bytes of books and 32 of x from
// the caches for each 2-byte code.  On the B70 (2026-11-18, Inkling 2 v3Fj 86 GB, 3 cards) the decode's experts went
// from 1.0M to 0.56M clock ticks a layer and the decode from 23.9 to 29.6 tok/s; chunks of 8, 24 and 32 codes were
// no faster (32: slower), and the lookups' bank conflicts are about 15% of the rest.
// Gate and up: each entry's own token's table (a window of a few tokens: speculative decoding's verify windows).
// Down: each entry's own h.  Widths (ff) must be multiples of 64 (8 codes) and n_embd of 128 (a chunk); a chunk past a narrower expert's
// width is skipped.  Devices without 33 KiB of local memory or work-groups of 256 keep the old kernel.
#include "strata/core/runtime.hpp"
#include "strata/kernels/canon_math.hpp"
#include "strata/kernels/swiglu_clamp.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace strata::kernels {
namespace {

constexpr int CC = 16;          // codes a chunk
constexpr int CI = CC * 8;      // inputs a chunk
constexpr int NE = 512;         // table entries a code (book1's 256, book2's 256)
constexpr int WG = 256;         // work-items a work-group
constexpr int RPT_GU = 2, RPT_D = 6;   // rows a work-item
// b70-local: chunks a work-group takes one after another, its rows' sums kept in registers across them: the partial
// sums written and read back were about 45% of the weights' bytes with a chunk a work-group (STRATA_RVQ_LUT_CPW)
int lut_cpw() { static const int v = [] { const char* e = std::getenv("STRATA_RVQ_LUT_CPW"); return e ? std::max(1, std::atoi(e)) : 1; }(); return v; }

// where a blob's parts are (iq_kernels.cpp native_expert_grouped_rvq: the mixed layout or the fixed one)
struct Geo {
    int64_t N, FF;
    bool mixed;
    size_t gu_row, up_off, down_off, d_row, tail;
};
struct View {
    const uint8_t* b;
    int64_t ff;
    size_t up_o, dn_o, d_row;
    const sycl::half* gains;
    const float* book;
};
inline View view(const Geo& G, const uint8_t* blob) {
    View v;
    if (G.mixed) {
        v.ff = *(const int32_t*) blob;
        v.b = blob + 256;
        v.up_o = G.gu_row * (size_t) v.ff;
        v.dn_o = 2 * v.up_o;
        v.d_row = (size_t) v.ff / 8 * 2;
        const size_t tl = v.dn_o + v.d_row * (size_t) G.N;
        v.gains = (const sycl::half*) (v.b + tl);
        v.book = (const float*) (v.b + tl + 2 * (size_t) (2 * v.ff + G.N));
    } else {
        v.ff = G.FF;
        v.b = blob;
        v.up_o = G.up_off;
        v.dn_o = G.down_off;
        v.d_row = G.d_row;
        v.gains = (const sycl::half*) (blob + G.tail);
        v.book = (const float*) (blob + G.tail + 2 * (size_t) (2 * G.FF + G.N));
    }
    return v;
}

// the table of nc codes from xs (local, nc * 8 inputs): work-item t takes book rows t and t + 256
inline void build(const float* book, const float* xs, float* T, int nc, int t) {
    for (int h = 0; h < 2; ++h) {
        const int i = t + h * WG;
        float bk[8];
        const sycl::vec<float, 4> b0 = *(const sycl::vec<float, 4>*) (book + (size_t) i * 8);
        const sycl::vec<float, 4> b1 = *(const sycl::vec<float, 4>*) (book + (size_t) i * 8 + 4);
        for (int k = 0; k < 4; ++k) { bk[k] = b0[k]; bk[4 + k] = b1[k]; }
        for (int c = 0; c < nc; ++c) {
            float d = 0.0f;
            for (int k = 0; k < 8; ++k) d += bk[k] * xs[c * 8 + k];
            T[c * NE + i] = d;
        }
    }
}

// one row's sum over its chunk's codes, already in registers (nv vectors of 8 codes)
using V4 = sycl::vec<uint32_t, 4>;
inline float lookup_regs(const V4* w, int nv, const float* T) {
    float acc = 0.0f;
    for (int v = 0; v < CC / 8; ++v) {
        if (v >= nv) break;
        for (int k = 0; k < 4; ++k) {
            const uint32_t lo = w[v][k] & 0xffffu, hi = w[v][k] >> 16;
            const float* t0 = T + (v * 8 + 2 * k) * NE;
            const float* t1 = t0 + NE;
            acc += t0[lo >> 8] + t0[256 + (lo & 255u)];
            acc += t1[hi >> 8] + t1[256 + (hi & 255u)];
        }
    }
    return acc;
}

// p[0] + p[stride] + ... + p[(n - 1) * stride] in that order, the loads 8 at a time ahead of their adds: the sums of
// the partials ran a load's latency an addend (a few work-items an EU to hide it: 60 us for 36 partials of 4.6K rows)
// b70-local: each entry's blob facts for the sums' kernels, written by the gate/up kernel (one work-item an entry, beside
// its lookups): the gains and the width, so a sum's work-item reads them at once instead of finding its entry's group
// and blob one dependent load after another (n_groups, grp_start..., grp_ptr, the header: ~40 us a kernel on the B70)
struct EntMeta { unsigned long long gains; int32_t ff, pad; };
// a flat kernel of n work-items in work-groups of 256 (the runtime's own choice for a sycl::range was slower)
inline size_t flat_n(size_t n) { return (n + WG - 1) / WG * WG; }
inline float sum_strided(const float* p, size_t stride, int n) {
    constexpr int U = 8;
    float s = 0.0f;
    int c = 0;
    for (; c + U <= n; c += U) {
        float v[U];
        #pragma unroll
        for (int j = 0; j < U; ++j) v[j] = p[(size_t) (c + j) * stride];
        #pragma unroll
        for (int j = 0; j < U; ++j) s += v[j];
    }
    for (; c < n; ++c) s += p[(size_t) c * stride];
    return s;
}

struct Buf { float* p = nullptr; size_t n = 0; };
std::mutex buf_mu;
std::unordered_map<size_t, Buf>& bufs() { static std::unordered_map<size_t, Buf> m; return m; }

// a partial sums' buffer of this queue's device, one for gate / up (slot 0) and one for down (slot 1); kept for the
// graphs recorded with it
float* partials(sycl::queue& q, size_t n, int slot) {
    std::lock_guard<std::mutex> lk(buf_mu);
    Buf& b = bufs()[std::hash<sycl::device>{}(q.get_device()) * 4 + (size_t) slot];
    if (b.n < n) {
        b.p = sycl::malloc_device<float>(n, q);   // an older, smaller one stays allocated for the graphs that use it
        b.n = b.p ? n : 0;
    }
    return b.p;
}

}  // namespace

bool rvq_lut_on(sycl::queue& q, int64_t cap_entries, int64_t N, int64_t FF, bool mixed, size_t up_off, size_t down_off,
                size_t d_row) {
    static const bool on = [] { const char* v = std::getenv("STRATA_RVQ_LUT"); return v == nullptr || v[0] != '0'; }();
    // up to STRATA_RVQ_LUT_MAX entries (default 32: a window of 4 tokens' top-8; the prompt's big windows take XMX)
    static const int64_t most = [] { const char* v = std::getenv("STRATA_RVQ_LUT_MAX"); return v ? std::atoll(v) : 32; }();
    if (!on || cap_entries > most || N % CI != 0 || FF % 64 != 0) return false;
    // the table and the inputs in local memory, and work-groups of 256, as the device reports
    const sycl::device d = q.get_device();
    if (d.get_info<sycl::info::device::local_mem_size>() < (CC * NE + CI) * sizeof(float) ||
        d.get_info<sycl::info::device::max_work_group_size>() < (size_t) WG)
        return false;
    if (!mixed && (up_off % 16 != 0 || down_off % 16 != 0 || d_row % 16 != 0)) return false;
    return true;
}

// gate and up: [entry][FF] for the entries' rows (gains on), each from its own token's row of x (ent_tok).  A
// work-group builds the table of one token at a time and covers the entries of that token: a window of T tokens
// (speculative decoding's verify windows) builds T tables and reads each (expert, token) pair's codes once.
sycl::event rvq_lut_gu(sycl::queue& q, const unsigned long long* grp_ptr, const int32_t* grp_start,
                       const int32_t* n_groups, const int32_t* ent_tok, int64_t cap_entries, const float* x, float* gate,
                       float* up, int64_t N, int64_t FF, bool mixed, size_t gu_row, size_t up_off, size_t tail, float* h_out) {
    const Geo G{N, FF, mixed, gu_row, up_off, 0, 0, tail};
    const int nch0 = (int) (N / CI);
    const int64_t rows = 2 * FF;
    // up to 8 entries (a token's top-8): chunk groups, each entry's row sums in registers across a group's chunks
    constexpr int ME = 8;
    const int cpw = cap_entries <= ME ? lut_cpw() : 1, nch = (nch0 + cpw - 1) / cpw;   // nch: the partial sets
    const size_t need = (size_t) nch * (size_t) cap_entries * (size_t) rows;
    float* part = partials(q, need, 0);
    if (part == nullptr) throw std::runtime_error("rvq_lut_gu: no partial buffer");
    const size_t nrb = (size_t) ((rows + WG * RPT_GU - 1) / (WG * RPT_GU));
    EntMeta* meta = reinterpret_cast<EntMeta*>(partials(q, (size_t) cap_entries * 4, 2));
    if (meta == nullptr) throw std::runtime_error("rvq_lut_gu: no entry buffer");
    const bool use_meta = cpw == 1 && cap_entries <= WG;
    if (cpw > 1) q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> T(sycl::range<1>(CC * NE), h);
        sycl::local_accessor<float, 1> xs(sycl::range<1>(CI), h);
        h.parallel_for(sycl::nd_range<2>({(size_t) nch, nrb * WG}, {1, (size_t) WG}), [=](sycl::nd_item<2> it) {
            const int ng = *n_groups;
            if (ng <= 0) return;
            const int e0 = grp_start[0], e1 = grp_start[ng];
            const int cg = (int) it.get_group(0), t = (int) it.get_local_id(1);
            const int64_t rb = (int64_t) it.get_group(1) * WG * RPT_GU;
            float* Tp = T.get_multi_ptr<sycl::access::decorated::no>().get();
            float* xp = xs.get_multi_ptr<sycl::access::decorated::no>().get();
            const float* book = view(G, (const uint8_t*) grp_ptr[0]).book;
            float acc[ME][RPT_GU];
            for (int j = 0; j < ME; ++j)
                for (int k = 0; k < RPT_GU; ++k) acc[j][k] = 0.0f;
            int tmax = 0;
            for (int e = e0; e < e1; ++e) tmax = sycl::max(tmax, ent_tok[e] + 1);
            for (int ch = cg * cpw; ch < sycl::min(nch0, (cg + 1) * cpw); ++ch)
                for (int tok = 0; tok < tmax; ++tok) {
                    bool any = false;
                    for (int e = e0; e < e1 && !any; ++e) any = ent_tok[e] == tok;
                    if (!any) continue;
                    sycl::group_barrier(it.get_group());
                    if (t < CI) xp[t] = x[(size_t) tok * N + (size_t) ch * CI + t];
                    sycl::group_barrier(it.get_group());
                    build(book, xp, Tp, CC, t);
                    sycl::group_barrier(it.get_group());
                    int g = 0;
                    for (int j = 0; j < ME; ++j) {
                        const int e = e0 + j;
                        if (e >= e1) break;
                        while (g + 1 < ng && grp_start[g + 1] <= e) ++g;
                        if (ent_tok[e] != tok) continue;
                        const View v = view(G, (const uint8_t*) grp_ptr[g]);
                        for (int k = 0; k < RPT_GU; ++k) {
                            const int64_t r = rb + (int64_t) k * WG + t;
                            if (r >= rows) continue;
                            const bool is_up = r >= FF;
                            const int64_t rr = is_up ? r - FF : r;
                            if (rr >= v.ff) continue;
                            const uint8_t* codes = v.b + (is_up ? v.up_o : 0) + (size_t) rr * G.gu_row + (size_t) ch * CC * 2;
                            V4 w[CC / 8];
                            for (int q8 = 0; q8 < CC / 8; ++q8) w[q8] = *(const V4*) (codes + (size_t) q8 * 16);
                            acc[j][k] += lookup_regs(w, CC / 8, Tp);
                        }
                    }
                }
            for (int j = 0; j < ME; ++j) {
                const int e = e0 + j;
                if (e >= e1) break;
                for (int k = 0; k < RPT_GU; ++k) {
                    const int64_t r = rb + (int64_t) k * WG + t;
                    if (r < rows) part[((size_t) cg * (size_t) cap_entries + (size_t) e) * (size_t) rows + (size_t) r] = acc[j][k];
                }
            }
        });
    });
    else q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> T(sycl::range<1>(CC * NE), h);
        sycl::local_accessor<float, 1> xs(sycl::range<1>(CI), h);
        h.parallel_for(sycl::nd_range<2>({(size_t) nch, nrb * WG}, {1, (size_t) WG}), [=](sycl::nd_item<2> it) {
            const int ng = *n_groups;
            if (ng <= 0) return;
            const int e0 = grp_start[0], e1 = grp_start[ng];
            const int ch = (int) it.get_group(0), t = (int) it.get_local_id(1);
            const int64_t rb = (int64_t) it.get_group(1) * WG * RPT_GU;
            if (use_meta && ch == 0 && rb == 0 && t < cap_entries) {
                EntMeta m{0, 0, 0};
                if (t >= e0 && t < e1) {
                    int g = 0;
                    while (g + 1 < ng && grp_start[g + 1] <= t) ++g;
                    const View v = view(G, (const uint8_t*) grp_ptr[g]);
                    m.gains = (unsigned long long) (uintptr_t) v.gains;
                    m.ff = (int32_t) v.ff;
                }
                meta[t] = m;
            }
            float* Tp = T.get_multi_ptr<sycl::access::decorated::no>().get();
            float* xp = xs.get_multi_ptr<sycl::access::decorated::no>().get();
            const float* book = view(G, (const uint8_t*) grp_ptr[0]).book;   // the layer's (every blob's the same)
            int tmax = 0;
            for (int e = e0; e < e1; ++e) tmax = sycl::max(tmax, ent_tok[e] + 1);
            for (int tok = 0; tok < tmax; ++tok) {
                bool any = false;
                for (int e = e0; e < e1 && !any; ++e) any = ent_tok[e] == tok;
                if (!any) continue;   // the same answer in every work-item: the barriers stay uniform
                sycl::group_barrier(it.get_group());   // the last token's lookups are done with the table
                if (t < CI) xp[t] = x[(size_t) tok * N + (size_t) ch * CI + t];
                sycl::group_barrier(it.get_group());
                build(book, xp, Tp, CC, t);
                sycl::group_barrier(it.get_group());
                int g = 0;
                for (int e = e0; e < e1; ++e) {
                    while (g + 1 < ng && grp_start[g + 1] <= e) ++g;
                    if (ent_tok[e] != tok) continue;
                    const View v = view(G, (const uint8_t*) grp_ptr[g]);
                    for (int k = 0; k < RPT_GU; ++k) {
                        const int64_t r = rb + (int64_t) k * WG + t;
                        if (r >= rows) continue;
                        const bool is_up = r >= FF;
                        const int64_t rr = is_up ? r - FF : r;
                        if (rr >= v.ff) continue;
                        const uint8_t* codes = v.b + (is_up ? v.up_o : 0) + (size_t) rr * G.gu_row + (size_t) ch * CC * 2;
                        V4 w[CC / 8];
                        for (int q8 = 0; q8 < CC / 8; ++q8) w[q8] = *(const V4*) (codes + (size_t) q8 * 16);
                        part[((size_t) ch * (size_t) cap_entries + (size_t) e) * (size_t) rows + (size_t) r] =
                            lookup_regs(w, CC / 8, Tp);
                    }
                }
            }
        });
    });
    if (h_out != nullptr) {   // b70-local: the sums, their gains and the activation in one kernel, into h [entry][FF]
        const ExpertAct lim = expert_act();
        const size_t nt = (size_t) cap_entries * (size_t) FF;
        return q.parallel_for(sycl::nd_range<1>(flat_n(nt), WG), [=](sycl::nd_item<1> id) {
            if (id.get_global_id(0) >= nt) return;
            const int64_t i = (int64_t) id.get_global_id(0), e = i / FF, rr = i % FF;
            const sycl::half* gains;
            int64_t ff;
            if (use_meta) {
                const EntMeta m = meta[e];
                ff = m.ff;
                gains = (const sycl::half*) (uintptr_t) m.gains;
                if (rr >= ff) return;   // ff 0: not an entry of this window
            } else {
                const int ng = *n_groups;
                if (ng <= 0 || e < grp_start[0] || e >= grp_start[ng]) return;
                int g = 0;
                while (g + 1 < ng && grp_start[g + 1] <= e) ++g;
                const View v = view(G, (const uint8_t*) grp_ptr[g]);
                ff = v.ff;
                gains = v.gains;
                if (rr >= ff) return;
            }
            const size_t stride = (size_t) cap_entries * (size_t) rows;
            const float* p0 = part + (size_t) e * (size_t) rows + (size_t) rr;
            const float sg = sum_strided(p0, stride, nch), su = sum_strided(p0 + FF, stride, nch);
            float gg = sg * (float) gains[rr], uu = su * (float) gains[ff + rr];
            if (lim.cg > 0.0f) { h_out[e * FF + rr] = canon::situ_mul(gg, uu, lim.cg, lim.cu); return; }
            swiglu_clamp(gg, uu, lim.lim);
            h_out[e * FF + rr] = (gg / (1.0f + sycl::exp(-gg))) * uu;
        });
    }
    const size_t nt = (size_t) cap_entries * (size_t) rows;
    return q.parallel_for(sycl::nd_range<1>(flat_n(nt), WG), [=](sycl::nd_item<1> id) {
        if (id.get_global_id(0) >= nt) return;
        const int64_t i = (int64_t) id.get_global_id(0), e = i / rows, r = i % rows;
        const int ng = *n_groups;
        if (ng <= 0 || e < grp_start[0] || e >= grp_start[ng]) return;
        int g = 0;
        while (g + 1 < ng && grp_start[g + 1] <= e) ++g;
        const View v = view(G, (const uint8_t*) grp_ptr[g]);
        const bool is_up = r >= FF;
        const int64_t rr = is_up ? r - FF : r;
        if (rr >= v.ff) return;
        const float s = sum_strided(part + (size_t) e * (size_t) rows + (size_t) r, (size_t) cap_entries * (size_t) rows, nch);
        (is_up ? up : gate)[(size_t) e * FF + rr] = s * (float) v.gains[(is_up ? v.ff : 0) + rr];
    });
}

// down: out[ent_dst[e]][N] from each entry's h [entry][FF] (its first ff)
sycl::event rvq_lut_down(sycl::queue& q, const unsigned long long* grp_ptr, const int32_t* grp_start,
                         const int32_t* n_groups, const int32_t* ent_dst, int64_t cap_entries, const float* hh,
                         float* out, int64_t N, int64_t FF, bool mixed, size_t gu_row, size_t down_off, size_t d_row,
                         size_t tail) {
    const Geo G{N, FF, mixed, gu_row, 0, down_off, d_row, tail};
    const int nch = (int) ((FF / 8 + CC - 1) / CC), cpw = lut_cpw(), nchg = (nch + cpw - 1) / cpw;
    const size_t need = (size_t) nchg * (size_t) cap_entries * (size_t) N;
    float* part = partials(q, need, 1);
    if (part == nullptr) throw std::runtime_error("rvq_lut_down: no partial buffer");
    // the entries' facts the gate/up kernel wrote (rvq_lut_gu, on the same terms)
    const EntMeta* meta = reinterpret_cast<const EntMeta*>(partials(q, (size_t) cap_entries * 4, 2));
    const bool use_meta = meta != nullptr && lut_cpw() == 1 && cap_entries <= WG;
    const size_t nrb = (size_t) ((N + WG * RPT_D - 1) / (WG * RPT_D));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> T(sycl::range<1>(CC * NE), h);
        sycl::local_accessor<float, 1> xs(sycl::range<1>(CI), h);
        h.parallel_for(sycl::nd_range<2>({(size_t) cap_entries * (size_t) nchg, nrb * WG}, {1, (size_t) WG}),
                       [=](sycl::nd_item<2> it) {
            const int ng = *n_groups;
            const int e = (int) it.get_group(0) / nchg, cg = (int) it.get_group(0) % nchg, t = (int) it.get_local_id(1);
            if (ng <= 0 || e < grp_start[0] || e >= grp_start[ng]) return;
            int g = 0;
            while (g + 1 < ng && grp_start[g + 1] <= e) ++g;
            const View v = view(G, (const uint8_t*) grp_ptr[g]);
            const int ncode = (int) (v.ff / 8), nch_e = (ncode + CC - 1) / CC;
            const int c0 = cg * cpw, c1 = sycl::min(nch_e, c0 + cpw);
            if (c0 >= c1) return;
            const int64_t rb = (int64_t) it.get_group(1) * WG * RPT_D;
            float acc[RPT_D];
            for (int k = 0; k < RPT_D; ++k) acc[k] = 0.0f;
            float* Tp = T.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int ch = c0; ch < c1; ++ch) {
                const int nc = sycl::min(CC, ncode - ch * CC), nv = nc / 8;
                sycl::group_barrier(it.get_group());   // the last chunk's lookups are done with the table
                if (t < nc * 8) xs[t] = hh[(size_t) e * FF + (size_t) ch * CI + t];
                // every row's codes loaded before the table is built (their loads in flight meanwhile)
                V4 w[RPT_D][CC / 8];
                for (int k = 0; k < RPT_D; ++k) {
                    const int64_t r = rb + (int64_t) k * WG + t;
                    if (r >= N) continue;
                    const uint8_t* codes = v.b + v.dn_o + (size_t) r * v.d_row + (size_t) ch * CC * 2;
                    for (int q8 = 0; q8 < CC / 8; ++q8)
                        if (q8 < nv) w[k][q8] = *(const V4*) (codes + (size_t) q8 * 16);
                }
                sycl::group_barrier(it.get_group());
                build(v.book, xs.get_multi_ptr<sycl::access::decorated::no>().get(), Tp, nc, t);
                sycl::group_barrier(it.get_group());
                for (int k = 0; k < RPT_D; ++k) acc[k] += lookup_regs(w[k], nv, Tp);
            }
            for (int k = 0; k < RPT_D; ++k) {
                const int64_t r = rb + (int64_t) k * WG + t;
                if (r < N) part[((size_t) cg * (size_t) cap_entries + (size_t) e) * (size_t) N + (size_t) r] = acc[k];
            }
        });
    });
    const size_t nt = (size_t) cap_entries * (size_t) N;
    return q.parallel_for(sycl::nd_range<1>(flat_n(nt), WG), [=](sycl::nd_item<1> id) {
        if (id.get_global_id(0) >= nt) return;
        const int64_t i = (int64_t) id.get_global_id(0), e = i / N, r = i % N;
        const int ng = *n_groups;
        int64_t ff;
        const sycl::half* gains;
        if (use_meta) {   // the gate/up kernel's facts of the entry (rvq_lut_gu)
            const EntMeta m = meta[e];
            ff = m.ff;
            gains = (const sycl::half*) (uintptr_t) m.gains;
            if (ff <= 0) return;
        } else {
            if (ng <= 0 || e < grp_start[0] || e >= grp_start[ng]) return;
            int g = 0;
            while (g + 1 < ng && grp_start[g + 1] <= e) ++g;
            const View v = view(G, (const uint8_t*) grp_ptr[g]);
            ff = v.ff;
            gains = v.gains;
        }
        const int nc_e = (int) ((ff / 8 + CC - 1) / CC), ng_e = (nc_e + cpw - 1) / cpw;
        const float s = sum_strided(part + (size_t) e * (size_t) N + (size_t) r, (size_t) cap_entries * (size_t) N, ng_e);
        out[(size_t) ent_dst[e] * N + r] = s * (float) gains[2 * ff + r];
    });
}

}  // namespace strata::kernels
