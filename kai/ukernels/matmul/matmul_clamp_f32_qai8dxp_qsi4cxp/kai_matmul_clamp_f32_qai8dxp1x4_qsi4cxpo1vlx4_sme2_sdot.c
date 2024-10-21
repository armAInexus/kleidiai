//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_SME2)
#error "SME2 extension required to compile this micro-kernel"
#else  // Architectural feature check
#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo1vlx4_sme2_sdot.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

#endif  // Architectural feature check
