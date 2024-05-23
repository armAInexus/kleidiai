//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0.h"

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);

inline static size_t kai_k_roundedup(size_t k, size_t kr, size_t sr) {
    // Since we pack a float and int32 value at the end of the row,
    // we must make sure that k is a multiple of 4 for alignment
    size_t kr_sr_roundedup4 = kai_roundup(kr * sr, 4);
    return kai_roundup(k, kr_sr_roundedup4);
}

inline static size_t kai_rhs_packed_stride(size_t k, size_t kr, size_t nr, size_t sr) {
    const size_t k_internal = kai_k_roundedup(k, kr, sr);

    KAI_ASSERT((k_internal % 2) == 0);

    return nr * ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs);
}

size_t kai_get_n_step_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_ASSERT((n_idx % nr) == 0);

    return (n_idx / nr) * kai_rhs_packed_stride(k, kr, nr, sr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(size_t n, size_t k, size_t nr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_rhs_packed_stride(k, kr, nr, sr);
}

void kai_run_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const int32_t* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0_params* params) {
    KAI_ASSERT((k % 2) == 0);
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(bias == NULL);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT((kr % sr) == 0);

    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->rhs_zero_point == 8);
    KAI_ASSERT(params->lhs_zero_point == 1);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const size_t rhs_zero_point = params->rhs_zero_point;
    const size_t rhs_stride = k / 2;
    const size_t rhs_packed_stride = kai_rhs_packed_stride(k, kr, nr, sr);
    const size_t k_internal = kai_k_roundedup(k, kr, sr);

    for (size_t y = 0; y < n; y += nr) {
        const uint8_t* src_row = rhs + y * rhs_stride;
        uint8_t* dst_row = (uint8_t*)rhs_packed + (y / nr) * rhs_packed_stride;

        int32_t* sums = (int32_t*)(dst_row + nr * (k_internal / 2));

        // Initialize to zero the RHS reduction sums
        memset(sums, 0, nr * sizeof(int32_t));

        for (size_t x = 0; x < k_internal; x += (kr * sr)) {
            for (size_t s = 0; s < sr; ++s) {
                for (size_t i = 0; i < nr; ++i) {
                    for (size_t kr_idx = 0; kr_idx < kr / sr; kr_idx += 2) {
                        const size_t k_idx_start0 = (x / 2) + kr_idx / 2 + s * (kr / sr) / 2;
                        const size_t k_idx_start1 = k_idx_start0 + (kr / 2);

                        const size_t src_addr_byte0 = i * rhs_stride + k_idx_start0;
                        const size_t src_addr_byte1 = i * rhs_stride + k_idx_start1;

                        uint8_t byte0 = rhs_zero_point | rhs_zero_point << 4;
                        uint8_t byte1 = rhs_zero_point | rhs_zero_point << 4;

                        if (k_idx_start0 < (k / 2)) {
                            byte0 = src_row[src_addr_byte0];
                        }

                        if (k_idx_start1 < (k / 2)) {
                            byte1 = src_row[src_addr_byte1];
                        }

                        const uint8_t src_x0_lo = (byte0 & 0x0F);
                        const uint8_t src_x1_lo = (byte0 >> 4);

                        const uint8_t src_x0_hi = (byte1 & 0x0F);
                        const uint8_t src_x1_hi = (byte1 >> 4);

                        sums[i] += (int32_t)src_x0_lo + (int32_t)src_x0_hi - 2 * (int32_t)rhs_zero_point;
                        sums[i] += (int32_t)src_x1_lo + (int32_t)src_x1_hi - 2 * (int32_t)rhs_zero_point;

                        const uint8_t dst_qs0 = src_x0_lo | (src_x0_hi << 4);
                        const uint8_t dst_qs1 = src_x1_lo | (src_x1_hi << 4);

                        *dst_row = dst_qs0 ^ 0x88;
                        dst_row += sizeof(uint8_t);
                        *dst_row = dst_qs1 ^ 0x88;
                        dst_row += sizeof(uint8_t);
                    }
                }
            }
        }

        // Adjust the reduction sums
        for (size_t i = 0; i < nr; ++i) {
            *((int32_t*)(dst_row)) = sums[i] * 16;
            dst_row += sizeof(int32_t);
        }

        // Adjust the scales
        for (size_t i = 0; i < nr; ++i) {
            *((float*)(dst_row)) = scale[y + i] * 0.0625F;
            dst_row += sizeof(float);
        }
    }
}
