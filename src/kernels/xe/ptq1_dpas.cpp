// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// b70-local (experiment): PTQ1_0 (ggml 143) expert rows on Xe2's matrix engine with no base-3 arithmetic per weight.  A byte
// of PTQ1_0 holds 5 ternary digits; a 256-entry table turns it into those digits as 2-bit signed fields (-1 = 0b11, 0 = 0b00,
// 1 = 0b01), so three qs bytes and one qh digit make a dword of 16 2-bit weights, which DPAS takes as its B operand (s2) as
// it is.  A block's 128 weights come out in a fixed order of their own; the activations are put in that order instead.
// Step 1 here: the B operand's layout for s2 weights x s8 activations, found on the device (ptq1_dpas_probe).
#include "strata/core/runtime.hpp"
#include "strata/kernels/canon_math.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/swiglu_clamp.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::kernels {
namespace {
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using sycl::ext::intel::esimd::xmx::dpas_argument_type;
sycl::queue& Q(void* stream) { return stream ? *static_cast<sycl::queue*>(stream) : core::Runtime::get().compute(); }
}  // namespace

// map[w * 16 + f] = the (column n, K index k) that 2-bit field f of B's dword w meets in dpas<8, 8> s2 x s8, as n * 64 + k
// (-1: none, -2: more than one column); rows[...] = how many of the 8 A rows saw it (8 when the layout is as expected).
void ptq1_dpas_probe(int* map, int* rows, void* stream) {
    sycl::queue& q = Q(stream);
    int* dm = sycl::malloc_device<int>(512 * 2, q);
    q.parallel_for(sycl::nd_range<1>(512, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        const int id = (int) it.get_global_id(0), w = id / 16, f = id % 16;
        esimd::simd<int, 32> b = 0;                       // B: K 32 x N 16 of 2-bit values = 32 dwords
        b[w] = 1 << (2 * f);                              // one field = +1
        esimd::simd<int, 64> a;                           // A: 8 rows x K 32 int8 = 8 rows x 8 dwords; a[m][k] = k + 1
#pragma unroll
        for (int m = 0; m < 8; ++m)
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                const uint32_t k0 = (uint32_t) (4 * d + 1);
                a[m * 8 + d] = (int) (k0 | (k0 + 1) << 8 | (k0 + 2) << 16 | (k0 + 3) << 24);
            }
        esimd::simd<int, 128> c = 0;
        c = xmx::dpas<8, 8, int, int, int, int, dpas_argument_type::s2, dpas_argument_type::s8>(c, b, a);
        int res = -1, seen = 0;
        for (int n = 0; n < 16; ++n) {
            const int v = c[n];
            if (v != 0) res = res == -1 ? n * 64 + (v - 1) : -2;
        }
        if (res >= 0) {
            const int n = res / 64, v = res % 64 + 1;
            for (int m = 0; m < 8; ++m) seen += (int) c[m * 16 + n] == v ? 1 : 0;
        }
        esimd::simd<int, 2> o;
        o[0] = res;
        o[1] = seen;
        esimd::block_store<int, 2>(dm + id * 2, o);
    });
    q.wait();
    int h[1024];
    q.memcpy(h, dm, sizeof h).wait();
    for (int i = 0; i < 512; ++i) { map[i] = h[2 * i]; rows[i] = h[2 * i + 1]; }
    sycl::free(dm, q);
}

namespace {
// the K order the weights come out in: position p (0..127) of a block's DPAS order holds value ptq1_perm(p) of the block.
// Dword j (0..7) of a row = fields 0-4: digits 0-4 of qs byte 3j, 5-9: of byte 3j+1, 10-14: of byte 3j+2, 15: digit j&3
// of qh byte j>>2.  qs byte m < 16 holds values 16n + m, byte 16 + m values 80 + 8n + m, qh byte h values 120 + 2n + h.
inline int ptq1_perm(int p) {
    const int j = p / 16, f = p % 16;
    if (f == 15) return 120 + 2 * (j & 3) + (j >> 2);
    const int byte = 3 * j + f / 5, n = f % 5;
    return byte < 16 ? 16 * n + byte : 80 + 8 * n + (byte - 16);
}
// byte -> its 5 digits as 2-bit signed fields (digit 0 -> -1 = 0b11, 1 -> 0, 2 -> +1 = 0b01), digit n at bits 2n
uint32_t ptq1_lut_entry(int q) {
    static const int p3[5] = {1, 3, 9, 27, 81};
    uint32_t e = 0;
    for (int n = 0; n < 5; ++n) {
        const int d = (((q * p3[n]) & 255) * 3) >> 8;
        e |= (uint32_t) (d == 0 ? 3 : d == 1 ? 0 : 1) << (2 * n);
    }
    return e;
}

// x: T rows of K floats -> xa int8 [K/32 chunks][8 tokens][32] in the DPAS order, xs [K/32][8] (q8: amax / 127 per 32)
void ptq1_xq(sycl::queue& q, const float* x, int nt, int K, int8_t* xa, float* xs) {
    const int C = K / 32;
    q.parallel_for(sycl::nd_range<1>((size_t) C * 8 * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        auto sg = it.get_sub_group();
        const int w = (int) it.get_group(0), t = w % 8, c = w / 8, l = (int) it.get_local_id(0);
        const int P = c * 32 + l, g = P / 128;
        const float v = t < nt ? x[(size_t) t * K + (size_t) g * 128 + ptq1_perm(P % 128)] : 0.f;
        const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
        const float d = amax / 127.f, id_ = d > 0.f ? 1.f / d : 0.f;
        xa[((size_t) c * 8 + t) * 32 + l] = (int8_t) (int) sycl::rint(v * id_);
        if (l == 0) xs[c * 8 + t] = d;
    });
}

// W: N rows of K/128 PTQ1_0 blocks.  16 rows a work-group of KS threads that split the blocks; partials meet in SLM.
template<int KS>
void ptq1_gemv(sycl::queue& q, const int* xa, const float* xs, const uint32_t* lut, const uint8_t* W, int N, int K, int nt, float* y) {
    constexpr int RB = 16;
    q.parallel_for(sycl::nd_range<1>((size_t) (N / RB) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<1024 + KS * 128 * 4>();
        const int ks = (int) it.get_local_id(0), n0 = (int) it.get_group(0) * RB;
        // the table into SLM (256 dwords), each thread a share
        for (int i = ks * 16; i < 256; i += KS * 16) esimd::slm_block_store<uint32_t, 16>(i * 4, esimd::block_load<uint32_t, 16>(lut + i));
        esimd::barrier();
        const int G = K / 128;
        const uint32_t rs = (uint32_t) G * 28;
        const esimd::simd<uint32_t, 16> roff(0u, rs);
        esimd::simd<float, 8 * 16> acc = 0.f;
        for (int g = ks; g < G; g += KS) {
            const uint32_t* base = reinterpret_cast<const uint32_t*>(W + (size_t) n0 * rs + (size_t) g * 28);
            esimd::simd<uint32_t, 16> wd[7];
#pragma unroll
            for (int i = 0; i < 7; ++i) wd[i] = esimd::gather<uint32_t, 16>(base + i, roff);
#define PTQ1_LK(bb) esimd::slm_gather<uint32_t, 16>(esimd::simd<uint32_t, 16>(((wd[(bb) >> 2] >> (8 * ((bb) & 3))) & 0xFFu) * 4u))
            const esimd::simd<uint32_t, 16> h0 = PTQ1_LK(24), h1 = PTQ1_LK(25);
            esimd::simd<uint16_t, 16> dh = esimd::simd<uint16_t, 16>(wd[6] >> 16);
            const esimd::simd<float, 16> d = esimd::convert<float>(dh.template bit_cast_view<sycl::half>().read());
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                esimd::simd<int, 32> b;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int j = 2 * c + h;
                    const esimd::simd<uint32_t, 16> hq = (j >> 2) ? h1 : h0;
                    esimd::simd<uint32_t, 16> code = PTQ1_LK(3 * j) | (PTQ1_LK(3 * j + 1) << 10) | (PTQ1_LK(3 * j + 2) << 20) | (((hq >> (2 * (j & 3))) & 3u) << 30);
                    b.template select<16, 1>(16 * h) = code.template bit_cast_view<int>().read();
                }
                const esimd::simd<int, 64> a = esimd::block_load<int, 64>(xa + ((size_t) g * 4 + c) * 64);
                const esimd::simd<float, 8> sx = esimd::block_load<float, 8>(xs + (g * 4 + c) * 8);
                esimd::simd<int, 8 * 16> cc = 0;
                cc = xmx::dpas<8, 8, int, int, int, int, dpas_argument_type::s2, dpas_argument_type::s8>(cc, b, a);
#pragma unroll
                for (int m = 0; m < 8; ++m)
                    acc.template select<16, 1>(m * 16) += esimd::convert<float>(cc.template select<16, 1>(m * 16).read()) * (d * (float) sx[m]);
            }
        }
#undef PTQ1_LK
        if (KS > 1) {
            esimd::slm_block_store<float, 128>(1024 + ks * 128 * 4, acc);
            esimd::barrier();
            if (ks == 0) {
                for (int j = 1; j < KS; ++j) acc += esimd::slm_block_load<float, 128>(1024 + j * 128 * 4);
            } else {
                return;
            }
        }
        for (int t = 0; t < nt; ++t) esimd::block_store<float, 16>(y + (size_t) t * N + n0, acc.template select<16, 1>(t * 16).read());
    });
}
struct Scratch { sycl::queue* q; int8_t* xa; float* xs; uint32_t* lut; };
}  // namespace

bool ptq1_rows_dpas_ok(int n_in, int n_out, int T) noexcept { return n_out % 16 == 0 && n_in % 128 == 0 && n_in <= 32768 && T >= 1 && T <= 8; }

// y[t][n] = W[n] . x[t] for T <= 8 tokens of K floats; W = N rows of PTQ1_0 (K / 128 blocks a row, 4-byte aligned rows)
bool ptq1_rows_dpas(const void* weights, const float* x, float* y, int n_in, int n_out, int T, int ks, void* stream) {
    if (!ptq1_rows_dpas_ok(n_in, n_out, T)) return false;
    sycl::queue& q = Q(stream);
    static std::mutex mu;
    static std::vector<Scratch> pool;
    Scratch* s = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& p : pool)
            if (p.q == &q) s = &p;
        if (s == nullptr) {
            uint32_t h[256];
            for (int i = 0; i < 256; ++i) h[i] = ptq1_lut_entry(i);
            uint32_t* lut = sycl::malloc_device<uint32_t>(256, q);
            q.memcpy(lut, h, sizeof h).wait();
            pool.push_back({&q, sycl::malloc_device<int8_t>(32768 * 8, q), sycl::malloc_device<float>(32768 / 32 * 8, q), lut});
            s = &pool.back();
        }
    }
    ptq1_xq(q, x, T, n_in, s->xa, s->xs);
    const int* xa = reinterpret_cast<const int*>(s->xa);
    const uint8_t* W = static_cast<const uint8_t*>(weights);
    if (ks <= 0) ks = n_in / 128 >= 32 ? 8 : 4;
    if (ks >= 16) ptq1_gemv<16>(q, xa, s->xs, s->lut, W, n_out, n_in, T, y);
    else if (ks >= 8) ptq1_gemv<8>(q, xa, s->xs, s->lut, W, n_out, n_in, T, y);
    else if (ks >= 4) ptq1_gemv<4>(q, xa, s->xs, s->lut, W, n_out, n_in, T, y);
    else if (ks >= 2) ptq1_gemv<2>(q, xa, s->xs, s->lut, W, n_out, n_in, T, y);
    else ptq1_gemv<1>(q, xa, s->xs, s->lut, W, n_out, n_in, T, y);
    return true;
}

// ---------------------------------------------------------------- the routed experts (native_expert_grouped's PTQ1_0 path)
// The same groups and entries as the dp4a kernels: a group = one expert's blob (gate rows, up rows, down rows, PTQ1_0) and
// its entries [grp_start[g], grp_start[g + 1]), ent_tok = an entry's token, ent_dst = its output row.  Entries' activations
// in the DPAS order: xa [entry][K/32][32] int8 + xs [entry][K/32].
namespace {
// X (q8_1 rows of tokens, natural order) -> the entries' rows in the DPAS order, quantized again per 32 (from d * q)
void ent_xq(sycl::queue& q, const uint8_t* X, const int32_t* ent_tok, const int32_t* grp_start, const int32_t* n_groups,
            int64_t cap_entries, int K, int8_t* xa, float* xs) {
    const int C = K / 32;
    q.parallel_for(sycl::nd_range<1>((size_t) cap_entries * C * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        auto sg = it.get_sub_group();
        const int w = (int) it.get_group(0), e = w / C, c = w % C, l = (int) it.get_local_id(0);
        if (e < grp_start[0] || e >= grp_start[*n_groups]) return;   // uniform in the sub-group
        const int P = c * 32 + l, v = (P / 128) * 128 + ptq1_perm(P % 128);
        const uint8_t* blk = X + ((size_t) ent_tok[e] * (size_t) C + (size_t) (v / 32)) * 36;
        const float val = (float) *(const sycl::half*) blk * (float) (int8_t) blk[4 + v % 32];
        const float amax = sycl::reduce_over_group(sg, sycl::fabs(val), sycl::maximum<float>());
        const float d = amax / 127.f, id_ = d > 0.f ? 1.f / d : 0.f;
        xa[(size_t) e * K + P] = (int8_t) (int) sycl::rint(val * id_);
        if (l == 0) xs[(size_t) e * C + c] = d;
    });
}

// one 16-row tile of PTQ1_0 rows (row stride rs bytes, K/128 blocks) against <= 8 entries' DPAS-order rows: acc[m][n]
// a block of 16 rows (row stride in roff): a lane reads its row's 16 + 8 + 4 bytes (3 requests)
SYCL_ESIMD_FUNCTION inline void load_blk(const uint8_t* p, const esimd::simd<uint32_t, 16>& roff, esimd::simd<uint32_t, 16>* wd) {
    const uint32_t* base = reinterpret_cast<const uint32_t*>(p);
    esimd::simd<uint32_t, 64> v4 = esimd::gather<uint32_t, 64, 4>(base, roff);
    esimd::simd<uint32_t, 32> v2 = esimd::gather<uint32_t, 32, 2>(base + 4, roff);
#pragma unroll
    for (int i = 0; i < 4; ++i) wd[i] = v4.template select<16, 1>(16 * i);
    wd[4] = v2.template select<16, 1>(0);
    wd[5] = v2.template select<16, 1>(16);
    wd[6] = esimd::gather<uint32_t, 16>(base + 6, roff);
}

// one 16-row tile of PTQ1_0 rows (row stride rs bytes, K/128 blocks) against a group's n (<= 8) entries' DPAS-order rows
// from entry e: acc[m][row].  A thread takes a contiguous run of the blocks; the next block's loads go out before the
// current one is decoded (two in flight).  Rows m >= n of A stay 0 and are not read (a verify window's expert has 1-2).
template<int KS>
SYCL_ESIMD_FUNCTION inline esimd::simd<float, 128> tile_dot(const uint8_t* w0, uint32_t rs, int G, int ks, const int* xa,
                                                            const float* xs, int e, int n, int K, const uint32_t* lut, bool glut) {
    const int C = K / 32;
    const esimd::simd<uint32_t, 16> roff(0u, rs);
    esimd::simd<float, 128> acc = 0.f;
    esimd::simd<int, 64> a = 0;
    esimd::simd<float, 8> sx = 0.f;
    const int g0 = ks * G / KS, g1 = (ks + 1) * G / KS;
    esimd::simd<uint32_t, 16> wd[7], wn[7];
    if (g0 < g1) load_blk(w0 + (size_t) g0 * 28, roff, wd);
    for (int g = g0; g < g1; ++g) {
        if (g + 1 < g1) load_blk(w0 + (size_t) (g + 1) * 28, roff, wn);
        // byte bb of the 16 rows -> its table entry (the 5 digits as 2-bit fields)
#define PTQ1_BY(bb) ((wd[(bb) >> 2] >> (8 * ((bb) & 3))) & 0xFFu)
#define PTQ1_LK(bb) (glut ? esimd::gather<uint32_t, 16>(lut, PTQ1_BY(bb) << 2) : esimd::slm_gather<uint32_t, 16>(PTQ1_BY(bb) << 2))
        const esimd::simd<uint32_t, 16> h0 = PTQ1_LK(24), h1 = PTQ1_LK(25);
        esimd::simd<uint16_t, 16> dh = esimd::simd<uint16_t, 16>(wd[6] >> 16);
        const esimd::simd<float, 16> d = esimd::convert<float>(dh.template bit_cast_view<sycl::half>().read());
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            esimd::simd<int, 32> b;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int j = 2 * c + h;
                const esimd::simd<uint32_t, 16> hq = (j >> 2) ? h1 : h0;
                esimd::simd<uint32_t, 16> code = PTQ1_LK(3 * j) | (PTQ1_LK(3 * j + 1) << 10) | (PTQ1_LK(3 * j + 2) << 20) | (((hq >> (2 * (j & 3))) & 3u) << 30);
                b.template select<16, 1>(16 * h) = code.template bit_cast_view<int>().read();
            }
            const int ch = g * 4 + c;
#pragma unroll
            for (int m = 0; m < 8; ++m)
                if (m < n) {
                    a.template select<8, 1>(m * 8) = esimd::block_load<int, 8>(xa + ((size_t) (e + m) * C + ch) * 8);
                    sx[m] = xs[(size_t) (e + m) * C + ch];
                }
            esimd::simd<int, 128> cc = 0;
            cc = xmx::dpas<8, 8, int, int, int, int, dpas_argument_type::s2, dpas_argument_type::s8>(cc, b, a);
#pragma unroll
            for (int m = 0; m < 8; ++m)
                if (m < n) acc.template select<16, 1>(m * 16) += esimd::convert<float>(cc.template select<16, 1>(m * 16).read()) * (d * (float) sx[m]);
        }
#undef PTQ1_LK
#undef PTQ1_BY
#pragma unroll
        for (int i = 0; i < 7; ++i) wd[i] = wn[i];
    }
    return acc;
}

// rows of every group's blob part (gate+up: 2 n_ff rows, row r < n_ff gate r else up r - n_ff; down: n_embd rows) x its
// entries; out row of entry e: gate/up -> e (n_ff floats each), down -> ent_dst[e] (n_embd floats)
template<int KS, bool DOWN>
sycl::event ex_rows(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int8_t* xa8, const float* xs, const uint32_t* lut, NativeExpertLayout L,
                    float* o0, float* o1) {
    // STRATA_PTQ1_DPAS_TPW: 16-row tiles a work-group takes one after another (the table's copy into SLM shared)
    static const int tpw_env = [] { const char* e = std::getenv("STRATA_PTQ1_DPAS_TPW"); return e ? std::atoi(e) : 0; }();
    const int rows = DOWN ? (int) L.n_embd : (int) (2 * L.n_ff), tiles16 = rows / 16;
    int tpw = tpw_env > 0 ? tpw_env : 1;
    while (tiles16 % tpw) --tpw;
    const int tiles = tiles16 / tpw;
    const int K = DOWN ? (int) L.n_ff : (int) L.n_embd;
    const int* xa = reinterpret_cast<const int*>(xa8);
    // STRATA_PTQ1_DPAS_GLUT=1: the table read from global memory (cached) instead of each work-group's SLM copy (slower)
    static const bool glut_env = [] { const char* e = std::getenv("STRATA_PTQ1_DPAS_GLUT"); return e != nullptr && e[0] == '1'; }();
    const bool glut = glut_env;
    return q.parallel_for(sycl::nd_range<1>(cg * (size_t) tiles * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<1024 + KS * 128 * 4>();
        const int ks = (int) it.get_local_id(0), gid = (int) it.get_group(0), gi = gid / tiles, tile = gid % tiles;
        if (!glut) {
            for (int i = ks * 16; i < 256; i += KS * 16) esimd::slm_block_store<uint32_t, 16>(i * 4, esimd::block_load<uint32_t, 16>(lut + i));
            esimd::barrier();
        }
        const int G = K / 128;
        const int ng = *n_groups;
        for (int tt = 0; tt < tpw; ++tt) {
        const int n0 = (tile * tpw + tt) * 16;
        const bool up = !DOWN && n0 >= L.n_ff;
        const int r0 = up ? n0 - (int) L.n_ff : n0;
        const uint32_t rs = (uint32_t) (DOWN ? L.d_row : up ? L.up_row : L.gu_row);
        const size_t woff = DOWN ? L.down_off + (size_t) r0 * L.d_row : up ? L.up_off + (size_t) r0 * L.up_row : (size_t) r0 * L.gu_row;
        for (int g = gi; g < ng; g += (int) cg) {
            const uint8_t* w0 = reinterpret_cast<const uint8_t*>(grp_ptr[g]) + woff;
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += 8) {
                const int n = e1 - e < 8 ? e1 - e : 8;
                esimd::simd<float, 128> acc = tile_dot<KS>(w0, rs, G, ks, xa, xs, e, n, K, lut, glut);
                if constexpr (KS > 1) {
                    esimd::barrier();
                    esimd::slm_block_store<float, 128>(1024 + ks * 128 * 4, acc);
                    esimd::barrier();
                    if (ks == 0)
                        for (int j = 1; j < KS; ++j) acc += esimd::slm_block_load<float, 128>(1024 + j * 128 * 4);
                }
                if (ks == 0)
                    for (int m = 0; m < n; ++m) {
                        float* dst = DOWN ? o0 + (size_t) ent_dst[e + m] * L.n_embd + r0 : (up ? o1 : o0) + (size_t) (e + m) * L.n_ff + r0;
                        esimd::block_store<float, 16>(dst, acc.template select<16, 1>(m * 16).read());
                    }
            }
        }
        }
    });
}

// SwiGLU of the entries' gate/up into the down's DPAS order (ha [e][n_ff/32][32] + hs), a sub-group = one chunk
void ex_swiglu(sycl::queue& q, const float* gate, const float* up, const int32_t* grp_start, const int32_t* n_groups,
               int64_t cap_entries, int64_t n_ff, float lim, int8_t* ha, float* hs) {
    const long long nh = (long long) cap_entries * n_ff;
    const int C2 = (int) (n_ff / 32);
    q.parallel_for(sycl::nd_range<1>((size_t) ((nh + 255) / 256) * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const long long i = (long long) it.get_global_linear_id();
        if (i >= nh) return;
        const long long e = i / n_ff;
        if (e < grp_start[0] || e >= grp_start[*n_groups]) return;
        auto sg = it.get_sub_group();
        const int P = (int) (i % n_ff), v = (P / 128) * 128 + ptq1_perm(P % 128);
        float g = gate[e * n_ff + v], u = up[e * n_ff + v];
        swiglu_clamp(g, u, lim);
        const float xi = canon::silu_mul(g, u);
        const float amax = sycl::reduce_over_group(sg, sycl::fabs(xi), sycl::maximum<float>());
        const float d = amax / 127.f, id_ = d > 0.f ? 1.f / d : 0.f;
        ha[i] = (int8_t) (int) sycl::rint(xi * id_);
        if (P % 32 == 0) hs[e * C2 + P / 32] = d;
    });
}

constexpr int64_t kPtq1MaxEntries = 8192;
struct ExScratch { sycl::queue* q; int64_t cap; int8_t* xa; float* xs; int8_t* ha; float* hs; uint32_t* lut; };
}  // namespace

bool ptq1_experts_dpas_enabled() {
    static const bool on = [] { const char* e = std::getenv("STRATA_PTQ1_DPAS"); return e != nullptr && e[0] == '1'; }();
    return on;
}
bool ptq1_experts_dpas_ok(sycl::queue& q, const NativeExpertLayout& L, int64_t cap_entries) {
    if (cap_entries > kPtq1MaxEntries) return false;
    return ptq1_experts_dpas_enabled() && L.gu_type == 143 && L.d_type == 143 && (L.up_type < 0 || L.up_type == 143) &&
           L.had_block == 0 && L.tail_off == 0 && L.n_embd % 128 == 0 && L.n_ff % 128 == 0 && L.n_embd <= 8192 && L.n_ff <= 8192 && iq_mmq_usable(q);
}

sycl::event ptq1_experts_dpas(sycl::queue& q, const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                              const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, size_t cg, int64_t cap_entries,
                              const void* x_q8_1, float* gate, float* up, float* out, float lim) {
    // scratch for kPtq1MaxEntries entries (+ 8: a tile reads 8 entries' rows whatever the group's count), made once per
    // queue on a queue of its own: the call may be being recorded into a command graph, where nothing may wait
    static std::mutex mu;
    static std::vector<ExScratch> pool;
    ExScratch* s = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& p : pool)
            if (p.q == &q) s = &p;
        if (s == nullptr) {
            sycl::queue q2(q.get_context(), q.get_device());
            uint32_t h[768];   // the table, then it shifted by 10 and by 20 bits
            for (int i = 0; i < 256; ++i) { h[i] = ptq1_lut_entry(i); h[256 + i] = h[i] << 10; h[512 + i] = h[i] << 20; }
            const int64_t c = kPtq1MaxEntries + 8, E = 8192;   // rows of up to 8192 values (~150 MB a queue)
            ExScratch n{&q, kPtq1MaxEntries, sycl::malloc_device<int8_t>((size_t) c * E, q2), sycl::malloc_device<float>((size_t) c * (E / 32), q2),
                        sycl::malloc_device<int8_t>((size_t) c * E, q2), sycl::malloc_device<float>((size_t) c * (E / 32), q2), sycl::malloc_device<uint32_t>(768, q2)};
            q2.memcpy(n.lut, h, sizeof h).wait();
            q2.memset(n.xs, 0, (size_t) c * (E / 32) * sizeof(float));
            q2.memset(n.hs, 0, (size_t) c * (E / 32) * sizeof(float)).wait();
            pool.push_back(n);
            s = &pool.back();
        }
    }
    // STRATA_PTQ1_DPAS_SKIP (measuring only, wrong results): bit 1 the conversion, 2 gate/up, 4 SwiGLU, 8 down
    static const int skip = [] { const char* e = std::getenv("STRATA_PTQ1_DPAS_SKIP"); return e ? std::atoi(e) : 0; }();
    if (!(skip & 1)) ent_xq(q, static_cast<const uint8_t*>(x_q8_1), ent_tok, grp_start, n_groups, cap_entries, (int) L.n_embd, s->xa, s->xs);
    static const int ks_env = [] { const char* e = std::getenv("STRATA_PTQ1_DPAS_KS"); return e ? std::atoi(e) : 0; }();
    const int ksg = ks_env > 0 ? ks_env : 4, ksd = ks_env > 0 ? ks_env : 2;
    if (skip & 2) {}
    else if (ksg >= 8) ex_rows<8, false>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, s->xa, s->xs, s->lut, L, gate, up);
    else if (ksg >= 4) ex_rows<4, false>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, s->xa, s->xs, s->lut, L, gate, up);
    else ex_rows<2, false>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, s->xa, s->xs, s->lut, L, gate, up);
    if (!(skip & 4)) ex_swiglu(q, gate, up, grp_start, n_groups, cap_entries, L.n_ff, lim, s->ha, s->hs);
    if (skip & 8) return q.ext_oneapi_submit_barrier();
    if (ksd >= 8) return ex_rows<8, true>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, s->ha, s->hs, s->lut, L, out, nullptr);
    if (ksd >= 4) return ex_rows<4, true>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, s->ha, s->hs, s->lut, L, out, nullptr);
    return ex_rows<2, true>(q, cg, grp_ptr, grp_start, n_groups, ent_dst, s->ha, s->hs, s->lut, L, out, nullptr);
}

// ---------------------------------------------------------------- Q2_0 (ggml 42) experts on DPAS (STRATA_Q2_DPAS=1)
// A Q2_0 block = fp16 d + 16 bytes of 2-bit codes, weight = (code - 1) * d, the lowest bits first: a dword of codes is 16
// consecutive weights, the B operand's layout as it is (u2).  sum (code - 1) x = sum code x - sum x; sum x comes from a
// second DPAS against all-ones codes.  The activations are the q8_1 rows as they are (natural order, d per 32).
namespace {
template<int KS>
SYCL_ESIMD_FUNCTION inline esimd::simd<float, 128> tile_dot_q2(const uint8_t* w0, uint32_t rs, int NP, int ks, const uint8_t* const* act, int n) {
    const esimd::simd<uint32_t, 16> roff(0u, rs);
    esimd::simd<float, 128> acc = 0.f;
    esimd::simd<int, 64> a = 0;
    esimd::simd<float, 8> sx = 0.f;
    const esimd::simd<int, 32> ones = (int) 0x55555555;
    const esimd::simd<uint32_t, 8> aoff(0u, 4u);
    const int p0 = ks * NP / KS, p1 = (ks + 1) * NP / KS;
    for (int p = p0; p < p1; ++p) {   // a pair of blocks = 36 bytes = 9 dwords a row
        const uint32_t* base = reinterpret_cast<const uint32_t*>(w0 + (size_t) p * 36);
        esimd::simd<uint32_t, 128> v8 = esimd::gather<uint32_t, 128, 8>(base, roff);
        esimd::simd<uint32_t, 16> w8 = esimd::gather<uint32_t, 16>(base + 8, roff);
        esimd::simd<uint32_t, 16> c[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) c[k] = (v8.template select<16, 1>(16 * k) >> 16) | (v8.template select<16, 1>(16 * (k + 1)) << 16);
#pragma unroll
        for (int k = 0; k < 3; ++k) c[4 + k] = v8.template select<16, 1>(16 * (5 + k));
        c[7] = w8;
        esimd::simd<uint16_t, 16> h0 = esimd::simd<uint16_t, 16>(v8.template select<16, 1>(0) & 0xFFFFu);
        esimd::simd<uint16_t, 16> h1 = esimd::simd<uint16_t, 16>(v8.template select<16, 1>(64) >> 16);
        const esimd::simd<float, 16> d0 = esimd::convert<float>(h0.template bit_cast_view<sycl::half>().read());
        const esimd::simd<float, 16> d1 = esimd::convert<float>(h1.template bit_cast_view<sycl::half>().read());
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            esimd::simd<int, 32> b;
            b.template select<16, 1>(0) = c[2 * q].template bit_cast_view<int>().read();
            b.template select<16, 1>(16) = c[2 * q + 1].template bit_cast_view<int>().read();
            const int ch = p * 4 + q;
#pragma unroll
            for (int m = 0; m < 8; ++m)
                if (m < n) {
                    const uint8_t* blk = act[m] + (size_t) ch * 36;
                    a.template select<8, 1>(m * 8) = esimd::gather<int, 8>(reinterpret_cast<const int*>(blk + 4), aoff);
                    sx[m] = (float) *reinterpret_cast<const sycl::half*>(blk);
                }
            esimd::simd<int, 128> cc = 0, cs = 0;
            cc = xmx::dpas<8, 8, int, int, int, int, dpas_argument_type::u2, dpas_argument_type::s8>(cc, b, a);
            cs = xmx::dpas<8, 8, int, int, int, int, dpas_argument_type::u2, dpas_argument_type::s8>(cs, ones, a);
            const esimd::simd<float, 16> d = q < 2 ? d0 : d1;
#pragma unroll
            for (int m = 0; m < 8; ++m)
                if (m < n) acc.template select<16, 1>(m * 16) += esimd::convert<float>((cc.template select<16, 1>(m * 16) - cs.template select<16, 1>(m * 16)).read()) * (d * (float) sx[m]);
        }
    }
    return acc;
}

// gate+up (DOWN false: act = token rows of X through ent_tok, out e * n_ff) or down (act = rows e of hq, out ent_dst[e])
template<int KS, bool DOWN>
sycl::event q2_rows(sycl::queue& q, size_t cg, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_tok, const int32_t* ent_dst, const uint8_t* act, NativeExpertLayout L, float* o0, float* o1) {
    const int rows = DOWN ? (int) L.n_embd : (int) (2 * L.n_ff), tiles = rows / 16;
    const int K = DOWN ? (int) L.n_ff : (int) L.n_embd;
    return q.parallel_for(sycl::nd_range<1>(cg * (size_t) tiles * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<KS * 128 * 4>();
        const int ks = (int) it.get_local_id(0), gid = (int) it.get_group(0), gi = gid / tiles, tile = gid % tiles;
        const int n0 = tile * 16, NP = K / 128;
        const bool up = !DOWN && n0 >= L.n_ff;
        const int r0 = up ? n0 - (int) L.n_ff : n0;
        const uint32_t rs = (uint32_t) (DOWN ? L.d_row : up ? L.up_row : L.gu_row);
        const size_t woff = DOWN ? L.down_off + (size_t) r0 * L.d_row : up ? L.up_off + (size_t) r0 * L.up_row : (size_t) r0 * L.gu_row;
        const size_t arow = (size_t) (K / 32) * 36;
        const int ng = *n_groups;
        for (int g = gi; g < ng; g += (int) cg) {
            const uint8_t* w0 = reinterpret_cast<const uint8_t*>(grp_ptr[g]) + woff;
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += 8) {
                const int n = e1 - e < 8 ? e1 - e : 8;
                const uint8_t* ap[8];
#pragma unroll
                for (int m = 0; m < 8; ++m) ap[m] = act + (size_t) (m < n ? (DOWN ? e + m : ent_tok[e + m]) : 0) * arow;
                esimd::simd<float, 128> acc = tile_dot_q2<KS>(w0, rs, NP, ks, ap, n);
                if constexpr (KS > 1) {
                    esimd::barrier();
                    esimd::slm_block_store<float, 128>(ks * 128 * 4, acc);
                    esimd::barrier();
                    if (ks == 0)
                        for (int j = 1; j < KS; ++j) acc += esimd::slm_block_load<float, 128>(j * 128 * 4);
                }
                if (ks == 0)
                    for (int m = 0; m < n; ++m) {
                        float* dst = DOWN ? o0 + (size_t) ent_dst[e + m] * L.n_embd + r0 : (up ? o1 : o0) + (size_t) (e + m) * L.n_ff + r0;
                        esimd::block_store<float, 16>(dst, acc.template select<16, 1>(m * 16).read());
                    }
            }
        }
    });
}
}  // namespace

bool q2_experts_dpas_ok(sycl::queue& q, const NativeExpertLayout& L) {
    static const bool on = [] { const char* e = std::getenv("STRATA_Q2_DPAS"); return e != nullptr && e[0] == '1'; }();
    return on && L.gu_type == 42 && L.d_type == 42 && (L.up_type < 0 || L.up_type == 42) && L.had_block == 0 && L.tail_off == 0 &&
           L.n_embd % 128 == 0 && L.n_ff % 128 == 0 && L.gu_row % 4 == 0 && L.up_row % 4 == 0 && L.d_row % 4 == 0 && iq_mmq_usable(q);
}
sycl::event q2_gu_dpas(sycl::queue& q, const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                       const int32_t* n_groups, const int32_t* ent_tok, size_t cg, const void* x_q8_1, float* gate, float* up) {
    static const int ks = [] { const char* e = std::getenv("STRATA_Q2_DPAS_KS"); return e ? std::atoi(e) : 4; }();
    const auto* X = static_cast<const uint8_t*>(x_q8_1);
    if (ks >= 8) return q2_rows<8, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, X, L, gate, up);
    if (ks >= 4) return q2_rows<4, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, X, L, gate, up);
    return q2_rows<2, false>(q, cg, grp_ptr, grp_start, n_groups, ent_tok, nullptr, X, L, gate, up);
}
sycl::event q2_down_dpas(sycl::queue& q, const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                         const int32_t* n_groups, const int32_t* ent_dst, size_t cg, const void* hq, float* out) {
    static const int ks = [] { const char* e = std::getenv("STRATA_Q2_DPAS_KSD"); return e ? std::atoi(e) : 2; }();
    const auto* H = static_cast<const uint8_t*>(hq);
    if (ks >= 8) return q2_rows<8, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, H, L, out, nullptr);
    if (ks >= 4) return q2_rows<4, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, H, L, out, nullptr);
    return q2_rows<2, true>(q, cg, grp_ptr, grp_start, n_groups, nullptr, ent_dst, H, L, out, nullptr);
}

}  // namespace strata::kernels
