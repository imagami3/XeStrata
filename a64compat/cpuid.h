// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
/* aarch64: no CPUID; every x86 feature reads as absent */
#pragma once
#define __cpuid(l, a, b, c, d) do { (a) = (b) = (c) = (d) = 0; (void) (l); } while (0)
#define __cpuid_count(l, s, a, b, c, d) do { (a) = (b) = (c) = (d) = 0; (void) (l); (void) (s); } while (0)
