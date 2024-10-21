//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);

inline static size_t kai_k_roundedup(size_t k, size_t kr) {
    // Since we pack a float and int32 value at the end of the row,
    // we must make sure that k is a multiple of 4 for alignment
    size_t kr_roundedup4 = kai_roundup(kr, 4);
    // Round k up to a multiple of kr
    return kai_roundup(k, kr_roundedup4);
}

size_t kai_get_n_step_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0(size_t k, size_t nr, size_t kr) {
    const size_t k_internal = kai_k_roundedup(k, kr);

    KAI_ASSERT((k_internal % 2) == 0);

    return nr * ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0(size_t n_idx, size_t k, size_t nr, size_t kr) {
    KAI_ASSERT((n_idx % nr) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0(k, nr, kr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0(size_t n, size_t k, size_t nr, size_t kr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0(k, nr, kr);
}

void kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_params* params) {
}
