// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// b70-local: the decode's kNativeQ8Rows projections for 2-16 tokens (the MTP window) on Xe2's matrix engine, int8 x int8
// DPAS (ESIMD).  The weights are the A operand as they lie: a row's 32 int8 values of a block are 8 dwords, 8 rows a DPAS
// (systolic depth 8, repeat 8); the tokens are the B operand, quantized here to int8 per 32 values (as q8_1, symmetric,
// amax / 127) in the VNNI order the B operand wants ([block][dword of the 32][token][4 bytes]).  One DPAS gives 8 rows x
// 16 tokens of int32 block sums, scaled by the row's and the token's block scales into FP32.  native_mmvq (dp4a) redoes
// each token's arithmetic: on the B70 a 4096 x 8192 Q8_0 product took 57 / 82 / 103 us at 1 / 6 / 8 tokens; this one
// reads the same bytes once for all 16.  The int sums are exact; the activation quantization is q8_1's, so the results
// differ from native_mmvq's only by float rounding order.  The approach is ~/llm/w4a8's gemvx (vLLM XPU, 11-02).
#include "strata/kernels/native_mmvq.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/iq_mmq.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using sycl::ext::intel::esimd::xmx::dpas_argument_type;

sycl::queue& Q(void* stream) { return stream ? *static_cast<sycl::queue*>(stream) : core::Runtime::get().compute(); }

// x: T rows of K floats -> xb int8 [G][8][16][4] (block g, dword d of its 32 values, token t, byte), xs float [G][16]
void q8vnni(sycl::queue& q, const float* x, int nt, int K, int8_t* xb, float* xs) {
    const int G = K / 32;
    q.parallel_for(sycl::nd_range<1>((size_t) 16 * G * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        auto sg = it.get_sub_group();
        const int w = (int) it.get_group(0), t = w % 16, g = w / 16, l = (int) it.get_local_id(0);
        const float v = t < nt ? x[(size_t) t * K + (size_t) g * 32 + l] : 0.f;
        const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
        const float d = amax / 127.f, id_ = d > 0.f ? 1.f / d : 0.f;
        xb[(((size_t) g * 8 + l / 4) * 16 + t) * 4 + (l & 3)] = (int8_t) (int) sycl::rint(v * id_);
        if (l == 0) xs[g * 16 + t] = d;
    });
}

// W: kNativeQ8Rows (row n: bpr blocks' 32 int8 values, then bpr FP16 scales).  16 rows a work-group of KS threads that
// split the blocks; partial sums meet in SLM.
template<int KS>
void gemvx_q8rows(sycl::queue& q, const int* xb, const float* xs, const uint8_t* W, int N, int K, int nt, float* y) {
    constexpr int RB = 16;
    q.parallel_for(sycl::nd_range<1>((size_t) (N / RB) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<KS * 2 * 128 * 4>();
        const int ks = (int) it.get_local_id(0), n0 = (int) it.get_group(0) * RB;
        const int G = K / 32;
        const size_t rs = (size_t) G * 34;   // row stride in bytes
        esimd::simd<float, 8 * 16> acc[2];
        acc[0] = 0.f;
        acc[1] = 0.f;
        for (int g = ks; g < G; g += KS) {
            esimd::simd<int, 128> b = esimd::block_load<int, 128>(xb + (size_t) g * 128);
            esimd::simd<float, 16> sx = esimd::block_load<float, 16>(xs + g * 16);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                esimd::simd<int, 64> a;
                const uint8_t* r0 = W + (size_t) (n0 + i * 8) * rs;
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    a.template select<8, 1>(r * 8) = esimd::block_load<int, 8>(reinterpret_cast<const int*>(r0 + (size_t) r * rs + (size_t) g * 32));
                // the 8 rows' scales of block g in one gather (16-bit elements: their dwords, the half picked by g's parity)
                const esimd::simd<uint32_t, 8> off(0u, (uint32_t) rs);
                const esimd::simd<uint32_t, 8> wd = esimd::gather<uint32_t, 8>(reinterpret_cast<const uint32_t*>(r0 + (size_t) G * 32 + (size_t) (g & ~1) * 2), off);
                esimd::simd<uint16_t, 8> hb = (g & 1) ? esimd::simd<uint16_t, 8>(wd >> 16) : esimd::simd<uint16_t, 8>(wd & 0xFFFFu);
                const esimd::simd<float, 8> sw = esimd::convert<float>(hb.template bit_cast_view<sycl::half>().read());
                esimd::simd<int, 8 * 16> c = 0;
                c = xmx::dpas<8, 8, int, int, int, int, dpas_argument_type::s8, dpas_argument_type::s8>(c, b, a);
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    acc[i].template select<16, 1>(r * 16) += esimd::convert<float>(c.template select<16, 1>(r * 16).read()) * (sx * (float) sw[r]);
            }
        }
        if (KS > 1) {
            esimd::slm_block_store<float, 128>((ks * 2 + 0) * 128 * 4, acc[0]);
            esimd::slm_block_store<float, 128>((ks * 2 + 1) * 128 * 4, acc[1]);
            esimd::barrier();
        }
        for (int r = ks; r < RB; r += KS) {
            esimd::simd<float, 16> v = 0.f;
            if (KS > 1) {
                for (int j = 0; j < KS; ++j) v += esimd::slm_block_load<float, 16>(((j * 2 + r / 8) * 128 + (r % 8) * 16) * 4);
            } else {
                v = acc[r / 8].template select<16, 1>((r % 8) * 16);
            }
            for (int t = 0; t < nt; ++t) y[(size_t) t * N + n0 + r] = (float) v[t];
        }
    });
}

struct Scratch { sycl::queue* q; int8_t* xb; float* xs; size_t cap; };

}  // namespace

bool native_q8_dpas_enabled() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q8_DPAS"); return e ? std::atoi(e) : 0; }();
    return v != 0;
}
int native_q8_dpas_min_tokens() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q8_DPAS"); return e ? std::atoi(e) : 0; }();
    return v > 1 ? v : 2;   // STRATA_Q8_DPAS=N: windows of N or more tokens (1: from 2)
}
bool native_q8_dpas_ok(int n_in, int n_out, int T) noexcept {
    return n_out % 16 == 0 && n_in % 32 == 0 && n_in <= 32768 && T >= 1 && T <= 16;
}

bool native_q8_rows_dpas(const void* weights, const float* x, float* y, int n_in, int n_out, int T, void* stream) {
    if (!native_q8_dpas_ok(n_in, n_out, T)) return false;
    sycl::queue& q = Q(stream);
    if (!iq_mmq_usable(q)) return false;   // no int8 XMX of the shape DPAS<8, 8> needs (or STRATA_NO_XMX=1): native_mmvq
    static std::mutex mu;
    static std::vector<Scratch> pool;
    Scratch* s = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& p : pool)
            if (p.q == &q) s = &p;
        if (s == nullptr) {
            const size_t cap = 32768;
            pool.push_back({&q, sycl::malloc_device<int8_t>(cap * 16, q), sycl::malloc_device<float>(cap / 32 * 16, q), cap});
            s = &pool.back();
        }
    }
    q8vnni(q, x, T, n_in, s->xb, s->xs);
    const int G = n_in / 32;
    const int* xb = reinterpret_cast<const int*>(s->xb);
    const uint8_t* W = static_cast<const uint8_t*>(weights);
    // threads a 16-row group that split the blocks: STRATA_Q8_DPAS_KS overrides (tuning); default from the block count
    static const int ks_env = [] { const char* e = std::getenv("STRATA_Q8_DPAS_KS"); return e ? std::atoi(e) : 0; }();
    const int ks = ks_env > 0 ? ks_env : G >= 256 ? 8 : G >= 64 ? 4 : 2;
    if (ks >= 16) gemvx_q8rows<16>(q, xb, s->xs, W, n_out, n_in, T, y);
    else if (ks >= 8) gemvx_q8rows<8>(q, xb, s->xs, W, n_out, n_in, T, y);
    else if (ks >= 4) gemvx_q8rows<4>(q, xb, s->xs, W, n_out, n_in, T, y);
    else if (ks >= 2) gemvx_q8rows<2>(q, xb, s->xs, W, n_out, n_in, T, y);
    else gemvx_q8rows<1>(q, xb, s->xs, W, n_out, n_in, T, y);
    return true;
}

// ---------------------------------------------------------------- v2 (STRATA_Q8_DPAS2=N: windows of N or more tokens)
// The weights as the B operand (16 rows; a row's 32 int8 of a block = 8 dwords, one 32-byte read a lane lands in B's
// layout as it is), the tokens as A (<= 8 rows) straight from the q8_1 rows the decode quantized anyway (no second
// quantization, no extra launch), up to 6 matrices that share the input in one launch.
namespace {
struct Q8Mats { const uint8_t* w[6]; float* y[6]; int n_out[6]; int tile0[7]; };
template<int KS>
void gemvq_q8rows_multi(sycl::queue& q, Q8Mats M, int count, const uint8_t* xq, int K, int nt) {
    const int tiles = M.tile0[count];
    q.parallel_for(sycl::nd_range<1>((size_t) tiles * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<KS * 128 * 4>();
        const int ks = (int) it.get_local_id(0), t = (int) it.get_group(0);
        int i = 0;
#pragma unroll
        for (int j = 1; j < 6; ++j) i += (j < count && t >= M.tile0[j]) ? 1 : 0;
        const int n0 = (t - M.tile0[i]) * 16, G = K / 32;
        const uint32_t rs = (uint32_t) G * 34;
        const uint8_t* W = M.w[i] + (size_t) n0 * rs;
        const esimd::simd<uint32_t, 16> roff(0u, rs);
        const esimd::simd<uint32_t, 8> aoff(0u, 4u);
        esimd::simd<float, 128> acc = 0.f;
        esimd::simd<int, 64> a = 0;
        esimd::simd<float, 8> sx = 0.f;
        const int g0 = ks * G / KS, g1 = (ks + 1) * G / KS;
        for (int g = g0; g < g1; ++g) {
            const esimd::simd<int, 128> b = esimd::gather<int, 128, 8>(reinterpret_cast<const int*>(W + (size_t) g * 32), roff);
            const esimd::simd<uint32_t, 16> wd = esimd::gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(W + (size_t) G * 32 + (size_t) (g & ~1) * 2), roff);
            esimd::simd<uint16_t, 16> hb = (g & 1) ? esimd::simd<uint16_t, 16>(wd >> 16) : esimd::simd<uint16_t, 16>(wd & 0xFFFFu);
            const esimd::simd<float, 16> sw = esimd::convert<float>(hb.template bit_cast_view<sycl::half>().read());
#pragma unroll
            for (int m = 0; m < 8; ++m)
                if (m < nt) {
                    const uint8_t* blk = xq + ((size_t) m * G + g) * 36;
                    a.template select<8, 1>(m * 8) = esimd::gather<int, 8>(reinterpret_cast<const int*>(blk + 4), aoff);
                    sx[m] = (float) *reinterpret_cast<const sycl::half*>(blk);
                }
            esimd::simd<int, 128> cc = 0;
            cc = xmx::dpas<8, 8, int, int, int, int, dpas_argument_type::s8, dpas_argument_type::s8>(cc, b, a);
#pragma unroll
            for (int m = 0; m < 8; ++m)
                if (m < nt) acc.template select<16, 1>(m * 16) += esimd::convert<float>(cc.template select<16, 1>(m * 16).read()) * (sw * (float) sx[m]);
        }
        if constexpr (KS > 1) {
            esimd::slm_block_store<float, 128>(ks * 128 * 4, acc);
            esimd::barrier();
            if (ks != 0) return;
            for (int j = 1; j < KS; ++j) acc += esimd::slm_block_load<float, 128>(j * 128 * 4);
        }
        float* y = M.y[i];
        const int no = M.n_out[i];
        for (int m = 0; m < nt; ++m) esimd::block_store<float, 16>(y + (size_t) m * no + n0, acc.template select<16, 1>(m * 16).read());
    });
}
}  // namespace

int native_q8_dpas2_min_tokens() {
    static const int v = [] { const char* e = std::getenv("STRATA_Q8_DPAS2"); return e ? std::atoi(e) : 0; }();
    return v > 0 ? (v > 1 ? v : 2) : 0;   // 0: off
}

// y[i][t][n] = W_i[n] . x[t] for `count` (<= 6) kNativeQ8Rows matrices sharing x: x as q8_1 rows (n_in / 32 blocks a token)
bool native_q8_rows_dpas_xq(int count, const void* const* w, float* const* y, const int* n_out, const void* xq, int n_in, int T, void* stream) {
    if (count < 1 || count > 6 || T < 1 || T > 8 || n_in % 64 != 0) return false;
    sycl::queue& q = Q(stream);
    if (!iq_mmq_usable(q)) return false;
    Q8Mats M{};
    int tl = 0;
    for (int i = 0; i < count; ++i) {
        if (n_out[i] % 16 != 0) return false;
        M.w[i] = static_cast<const uint8_t*>(w[i]); M.y[i] = y[i]; M.n_out[i] = n_out[i]; M.tile0[i] = tl; tl += n_out[i] / 16;
    }
    for (int i = count; i <= 6; ++i) M.tile0[i] = tl;
    static const int ks_env = [] { const char* e = std::getenv("STRATA_Q8_DPAS2_KS"); return e ? std::atoi(e) : 0; }();
    const int G = n_in / 32, ks = ks_env > 0 ? ks_env : G >= 256 ? 8 : 4;
    const auto* X = static_cast<const uint8_t*>(xq);
    if (ks >= 16) gemvq_q8rows_multi<16>(q, M, count, X, n_in, T);
    else if (ks >= 8) gemvq_q8rows_multi<8>(q, M, count, X, n_in, T);
    else if (ks >= 4) gemvq_q8rows_multi<4>(q, M, count, X, n_in, T);
    else gemvq_q8rows_multi<2>(q, M, count, X, n_in, T);
    return true;
}

// ---------------------------------------------------------------- v3: gemvx (weights as A, 16 tokens as B) from q8_1 rows
// B's VNNI order [dword d of the block][token] is a 32-byte read per token lane: one gather (8 dwords a lane) of the q8_1
// rows.  Lanes past T read token 0 (their columns are not stored).  Up to 6 matrices sharing x in one launch.
namespace {
template<int KS>
void gemvx_q8rows_xq(sycl::queue& q, Q8Mats M, int count, const uint8_t* xq, int K, int nt) {
    constexpr int RB = 16;
    const int tiles = M.tile0[count];
    q.parallel_for(sycl::nd_range<1>((size_t) tiles * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<KS * 2 * 128 * 4>();
        const int ks = (int) it.get_local_id(0), t = (int) it.get_group(0);
        int i = 0;
#pragma unroll
        for (int j = 1; j < 6; ++j) i += (j < count && t >= M.tile0[j]) ? 1 : 0;
        const int n0 = (t - M.tile0[i]) * RB, G = K / 32;
        const size_t rs = (size_t) G * 34;
        const uint8_t* W = M.w[i];
        esimd::simd<uint32_t, 16> toff;
#pragma unroll
        for (int l = 0; l < 16; ++l) toff[l] = (uint32_t) ((l < nt ? l : 0) * G * 36);
        esimd::simd<float, 8 * 16> acc[2];
        acc[0] = 0.f;
        acc[1] = 0.f;
        const int g0 = ks * G / KS, g1 = (ks + 1) * G / KS;
        for (int g = g0; g < g1; ++g) {
            const uint8_t* xb = xq + (size_t) g * 36;
            const esimd::simd<int, 128> b = esimd::gather<int, 128, 8>(reinterpret_cast<const int*>(xb + 4), toff);
            const esimd::simd<uint32_t, 16> xd = esimd::gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(xb), toff);
            esimd::simd<uint16_t, 16> xh = esimd::simd<uint16_t, 16>(xd & 0xFFFFu);
            const esimd::simd<float, 16> sx = esimd::convert<float>(xh.template bit_cast_view<sycl::half>().read());
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                esimd::simd<int, 64> a;
                const uint8_t* r0 = W + (size_t) (n0 + h * 8) * rs;
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    a.template select<8, 1>(r * 8) = esimd::block_load<int, 8>(reinterpret_cast<const int*>(r0 + (size_t) r * rs + (size_t) g * 32));
                const esimd::simd<uint32_t, 8> off(0u, (uint32_t) rs);
                const esimd::simd<uint32_t, 8> wd = esimd::gather<uint32_t, 8>(reinterpret_cast<const uint32_t*>(r0 + (size_t) G * 32 + (size_t) (g & ~1) * 2), off);
                esimd::simd<uint16_t, 8> hb = (g & 1) ? esimd::simd<uint16_t, 8>(wd >> 16) : esimd::simd<uint16_t, 8>(wd & 0xFFFFu);
                const esimd::simd<float, 8> sw = esimd::convert<float>(hb.template bit_cast_view<sycl::half>().read());
                esimd::simd<int, 8 * 16> c = 0;
                c = xmx::dpas<8, 8, int, int, int, int, dpas_argument_type::s8, dpas_argument_type::s8>(c, b, a);
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    acc[h].template select<16, 1>(r * 16) += esimd::convert<float>(c.template select<16, 1>(r * 16).read()) * (sx * (float) sw[r]);
            }
        }
        if constexpr (KS > 1) {
            esimd::slm_block_store<float, 128>((ks * 2 + 0) * 128 * 4, acc[0]);
            esimd::slm_block_store<float, 128>((ks * 2 + 1) * 128 * 4, acc[1]);
            esimd::barrier();
        }
        float* y = M.y[i];
        const int no = M.n_out[i];
        for (int r = ks; r < RB; r += KS) {
            esimd::simd<float, 16> v = 0.f;
            if constexpr (KS > 1) {
                for (int j = 0; j < KS; ++j) v += esimd::slm_block_load<float, 16>(((j * 2 + r / 8) * 128 + (r % 8) * 16) * 4);
            } else {
                v = acc[r / 8].template select<16, 1>((r % 8) * 16);
            }
            for (int tt = 0; tt < nt; ++tt) y[(size_t) tt * no + n0 + r] = (float) v[tt];
        }
    });
}
}  // namespace

bool native_q8_rows_dpas_xq3(int count, const void* const* w, float* const* y, const int* n_out, const void* xq, int n_in, int T, void* stream) {
    if (count < 1 || count > 6 || T < 1 || T > 16 || n_in % 64 != 0) return false;
    sycl::queue& q = Q(stream);
    if (!iq_mmq_usable(q)) return false;
    Q8Mats M{};
    int tl = 0;
    for (int i = 0; i < count; ++i) {
        if (n_out[i] % 16 != 0) return false;
        M.w[i] = static_cast<const uint8_t*>(w[i]); M.y[i] = y[i]; M.n_out[i] = n_out[i]; M.tile0[i] = tl; tl += n_out[i] / 16;
    }
    for (int i = count; i <= 6; ++i) M.tile0[i] = tl;
    static const int ks_env = [] { const char* e = std::getenv("STRATA_Q8_DPAS_KS"); return e ? std::atoi(e) : 0; }();
    const int G = n_in / 32, ks = ks_env > 0 ? ks_env : G >= 256 ? 8 : G >= 64 ? 4 : 2;
    const auto* X = static_cast<const uint8_t*>(xq);
    if (ks >= 8) gemvx_q8rows_xq<8>(q, M, count, X, n_in, T);
    else if (ks >= 4) gemvx_q8rows_xq<4>(q, M, count, X, n_in, T);
    else gemvx_q8rows_xq<2>(q, M, count, X, n_in, T);
    return true;
}

}  // namespace strata::kernels
