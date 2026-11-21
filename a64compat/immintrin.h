// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
/* aarch64 build of XeStrata on GB10 (local feasibility port): x86 SIMD intrinsics through SIMDe on the host.
   The CPU expert kernels compiled this way are never run with STRATA_GPU_ONLY_EXPERTS=1. */
#pragma once
#if defined(__SYCL_DEVICE_ONLY__)
/* SYCL device pass (spir64 / nvptx): only the host code uses these; keep the names declared */
static inline void _mm_pause(void) {}
static inline void _mm_sfence(void) {}
static inline void _mm_lfence(void) {}
static inline void _mm_mfence(void) {}
static inline void _mm_clflush(const void *) {}
static inline unsigned long long __rdtsc(void) { return 0; }
static inline void _umonitor(void *) {}
static inline unsigned char _umwait(unsigned int, unsigned long long) { return 0; }
#define _mm_prefetch(p, h) ((void) (p))
#else
#include <stdint.h>
#include <string.h>
#define SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_NO_NATIVE   /* portable C bodies: some NEON mappings need constant shift counts the callers pass at run time */
#include <simde/x86/avx512.h>
#include <simde/x86/fma.h>
#include <simde/x86/f16c.h>
#ifndef _mm_pause
static inline void _mm_pause(void) { __asm__ __volatile__("yield"); }
#endif
/* WAITPKG: monitor is a no-op, umwait returns at once (the callers loop) */
static inline void _umonitor(void * p) { (void) p; }
static inline unsigned char _umwait(unsigned int c, unsigned long long t) { (void) c; (void) t; __asm__ __volatile__("yield"); return 0; }
/* TSC: the ARM generic timer (cntvct_el0, about 1 GHz on GB10's timer frequency register) */
static inline unsigned long long __rdtsc(void) { uint64_t v; __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v)); return v; }
#ifndef _mm512_cvtepu16_epi64
static inline simde__m512i a64_mm512_cvtepu16_epi64(simde__m128i a) {
    uint16_t s[8]; simde_mm_storeu_si128((simde__m128i *) s, a);
    int64_t d[8]; for (int i = 0; i < 8; ++i) d[i] = s[i];
    return simde_mm512_loadu_si512(d);
}
#define _mm512_cvtepu16_epi64(a) a64_mm512_cvtepu16_epi64(a)
#endif
/* missing from SIMDe: mask moves, the 32-bit gather, scalar F16C */
#ifndef _cvtu64_mask64
#define _cvtu64_mask64(x) ((simde__mmask64) (x))
#endif
#ifndef _cvtmask64_u64
#define _cvtmask64_u64(x) ((uint64_t) (x))
#endif
#ifndef _mm512_i32gather_epi32
static inline simde__m512i a64_mm512_i32gather_epi32(simde__m512i idx, const void * base, int scale) {
    int32_t ix[16], r[16]; memcpy(ix, &idx, 64);
    for (int i = 0; i < 16; ++i) memcpy(&r[i], (const char *) base + (int64_t) ix[i] * scale, 4);
    simde__m512i o; memcpy(&o, r, 64); return o;
}
#define _mm512_i32gather_epi32(idx, base, scale) a64_mm512_i32gather_epi32((idx), (base), (scale))
#endif
#ifndef _cvtsh_ss
static inline float a64_cvtsh_ss(unsigned short h) { __fp16 f; memcpy(&f, &h, 2); return (float) f; }
static inline unsigned short a64_cvtss_sh(float x, int r) { (void) r; __fp16 f = (__fp16) x; unsigned short h; memcpy(&h, &f, 2); return h; }
#define _cvtsh_ss(h) a64_cvtsh_ss(h)
#define _cvtss_sh(x, r) a64_cvtss_sh((x), (r))
#endif
#endif
