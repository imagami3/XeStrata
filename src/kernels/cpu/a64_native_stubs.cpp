// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// GB10 local feasibility port: the native (ggml-cpu) CPU experts are not built on aarch64 (STRATA_NATIVE_EXPERTS=OFF);
// with every expert in device memory (STRATA_GPU_ONLY_EXPERTS=1) nothing calls these.
#include "strata/kernels/cpu/canon_expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::cpu {
bool canon_layer(const NativeFmt&) { return false; }
size_t native_quantize_rows(int, const float*, void*, int64_t, int64_t, const float*) {
    std::fprintf(stderr, "XeStrata GB10 port: native_quantize_rows needs the native CPU experts (not built on aarch64)\n");
    std::abort();
}
}  // namespace strata::kernels::cpu
