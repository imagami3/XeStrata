// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/swiglu_clamp.hpp - the routed experts' SwiGLU limit.
//
// GLM-5.3-Flash clamps its experts' SwiGLU (glm5-next.swiglu_clamp_exp, ggml_swiglu_clamp): with a limit L > 0 the gate
// is cut above at L and the up value to [-L, L] before silu(gate) * up.  Every other model has none (0).  A process
// setting, as the GDN output gate (gdn_set_silu_gate): set at load, before any graph is recorded; the GPU kernels read
// it when they are launched, so a recorded graph keeps the value it was recorded with.
#pragma once

#include <atomic>
#include <cmath>

#include "strata/kernels/canon_math.hpp"

namespace strata::kernels {

inline std::atomic<float> g_expert_swiglu_limit{0.0f};

inline void expert_swiglu_set_limit(float limit) {
    g_expert_swiglu_limit.store(limit > 0.0f ? limit : 0.0f, std::memory_order_relaxed);
}
inline float expert_swiglu_limit() { return g_expert_swiglu_limit.load(std::memory_order_relaxed); }

/// gate <= lim, -lim <= up <= lim when lim > 0 (host and device).
inline void swiglu_clamp(float& gate, float& up, float lim) {
    if (lim > 0.0f) {
        gate = gate < lim ? gate : lim;
        up = up < -lim ? -lim : (up > lim ? lim : up);
    }
}

/// The CPU kernels' silu(gate) * up after the clamp: with lim 0, the expression they always used.
inline float swiglu_host(float gate, float up, float lim) {
    swiglu_clamp(gate, up, lim);
    return (gate / (1.f + std::exp(-gate))) * up;
}

/// b70-local: the routed experts' activation, a process setting like the limit above.  Inkling 2 (SITU): softcaps
/// cg tanh(g / cg) and cu tanh(u / cu), times sigmoid(g) (canon::situ_mul); every other model: the clamped SwiGLU.
struct ExpertAct {
    float lim = 0.0f;   ///< the SwiGLU clamp (0: none)
    float cg = 0.0f;    ///< SITU's gate softcap (0: SwiGLU)
    float cu = 0.0f;    ///< ... and its up softcap
};
inline std::atomic<float> g_expert_situ_gate{0.0f};
inline std::atomic<float> g_expert_situ_up{0.0f};
inline void expert_situ_set(float cg, float cu) {
    g_expert_situ_gate.store(cg > 0.0f && cu > 0.0f ? cg : 0.0f, std::memory_order_relaxed);
    g_expert_situ_up.store(cg > 0.0f && cu > 0.0f ? cu : 0.0f, std::memory_order_relaxed);
}
inline ExpertAct expert_act() {
    return {expert_swiglu_limit(), g_expert_situ_gate.load(std::memory_order_relaxed),
            g_expert_situ_up.load(std::memory_order_relaxed)};
}
/// The activation with the canonical arithmetic (canon_math.hpp; host and device give the same bits).
inline float expert_act_mul(float gate, float up, const ExpertAct& a) {
    if (a.cg > 0.0f) return canon::situ_mul(gate, up, a.cg, a.cu);
    swiglu_clamp(gate, up, a.lim);
    return canon::silu_mul(gate, up);
}
/// The CPU kernels' form: SITU canonical, the SwiGLU as they always computed it.
inline float swiglu_host(float gate, float up, const ExpertAct& a) {
    if (a.cg > 0.0f) return canon::situ_mul(gate, up, a.cg, a.cu);
    return swiglu_host(gate, up, a.lim);
}

}  // namespace strata::kernels
