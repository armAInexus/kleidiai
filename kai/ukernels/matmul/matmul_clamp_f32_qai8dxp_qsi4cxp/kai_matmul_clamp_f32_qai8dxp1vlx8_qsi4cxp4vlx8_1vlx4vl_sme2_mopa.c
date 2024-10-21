//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural feature check

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo1vlx4_sme2_sdot.h"

#endif  // Architectural feature check
