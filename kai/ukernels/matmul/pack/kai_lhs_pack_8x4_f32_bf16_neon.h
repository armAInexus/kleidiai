//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>
#include <stdint.h>
#include "kai/kai_common.h"

size_t kai_get_m_step_lhs_pack_8x4_f32_bf16_neon(size_t mr);

size_t kai_get_lhs_offset_lhs_pack_8x4_f32_bf16_neon(size_t m_idx, size_t lhs_stride);

size_t kai_get_lhs_packed_offset_lhs_pack_8x4_f32_bf16_neon(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);

size_t kai_get_lhs_packed_size_lhs_pack_8x4_f32_bf16_neon(size_t m, size_t k, size_t mr, size_t kr, size_t sr);

void kai_run_lhs_pack_8x4_f32_bf16_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start,
    const void* lhs, size_t lhs_stride, void* lhs_packed
);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus