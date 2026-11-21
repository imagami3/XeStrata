// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// GB10 local feasibility port (aarch64 + CUDA only): stand-ins for the Intel-only kernels (ESIMD DPAS in q8_dpas.cpp
// and ptq1_dpas.cpp, joint_matrix sub-group-16 in rvq_xmx.cpp), which do not compile for NVPTX.  Every *_ok / *_on /
// *_enabled reports false, so the callers take their portable paths; the launchers abort if reached anyway.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/tq_kernels.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
[[noreturn]] void a64_no(const char* f) {
    std::fprintf(stderr, "XeStrata GB10 port: %s is Intel-only (not built for CUDA)\n", f);
    std::abort();
}
}  // namespace

bool native_q8_dpas_enabled() { return false; }
int native_q8_dpas_min_tokens() { return 1 << 30; }
bool native_q8_dpas_ok(int, int, int) noexcept { return false; }
bool native_q8_rows_dpas(const void*, const float*, float*, int, int, int, void*) { return false; }
int native_q8_dpas2_min_tokens() { return 1 << 30; }
bool native_q8_rows_dpas_xq(int, const void* const*, float* const*, const int*, const void*, int, int, void*) { return false; }
bool native_q8_rows_dpas_xq3(int, const void* const*, float* const*, const int*, const void*, int, int, void*) { return false; }

bool ptq1_experts_dpas_ok(sycl::queue&, const NativeExpertLayout&, int64_t) { return false; }
sycl::event ptq1_experts_dpas(sycl::queue&, const NativeExpertLayout&, const unsigned long long*, const int32_t*,
                              const int32_t*, const int32_t*, const int32_t*, size_t, int64_t,
                              const void*, float*, float*, float*, float) { a64_no(__func__); }
bool q2_experts_dpas_ok(sycl::queue&, const NativeExpertLayout&) { return false; }
sycl::event q2_gu_dpas(sycl::queue&, const NativeExpertLayout&, const unsigned long long*, const int32_t*,
                       const int32_t*, const int32_t*, size_t, const void*, float*, float*) { a64_no(__func__); }
sycl::event q2_down_dpas(sycl::queue&, const NativeExpertLayout&, const unsigned long long*, const int32_t*,
                         const int32_t*, const int32_t*, size_t, const void*, float*) { a64_no(__func__); }

// rvq_xmx_on / rvq_xmx_gu / rvq_xmx_down: a64_rvq_mma.cpp (tensor cores through joint_matrix)

// q6k_xmx.cpp (joint_matrix with Intel's shapes)
bool native_q8_xmx_enabled() { return false; }
bool native_q8_rows_xmx_ok(int, int, int) noexcept { return false; }
void native_q8_rows_xmx(const void*, const float*, float*, int, int, int, void*) { a64_no(__func__); }
bool native_q6k_xmx_enabled() { return false; }
void native_f32_to_f16_rows(const float* x, void* xh, int n, int T, void* stream) {
    sycl::half* h = static_cast<sycl::half*>(xh);
    const size_t total = (size_t) n * (size_t) T;
    auto& q = core::Runtime::get().stream(stream);   // the engine's stream handle (null: the compute queue, waited on)
    const sycl::event e = q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) { h[i] = (sycl::half) x[i]; });
    if (!stream) core::Runtime::get().wait(e, "native_f32_to_f16_rows");
}
void native_q6k_rows_xmx(const void*, const void*, float*, int, int, int, void*) { a64_no(__func__); }
}  // namespace strata::kernels

