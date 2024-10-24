//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0.h"

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);

inline static size_t kai_k_roundedup(size_t k, size_t kr) {
    // Since we pack a float and int32 value at the end of the row,
    // we must make sure that k is a multiple of 4 for alignment
    size_t kr_roundedup4 = kai_roundup(kr, 4);
    // Round k up to a multiple of kr
    return kai_roundup(k, kr_roundedup4);
}

inline static size_t kai_rhs_packed_stride(size_t k, size_t kr, size_t nr) {
    const size_t k_internal = kai_k_roundedup(k, kr);

    // multiple of 2 because 2 elements in a byte
    KAI_ASSERT((k_internal % 2) == 0);

    return nr * ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs);
}

size_t kai_get_n_step_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(
    size_t n_idx, size_t k, size_t nr, size_t kr) {
    KAI_ASSERT((n_idx % nr) == 0);

    return (n_idx / nr) * kai_rhs_packed_stride(k, kr, nr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(size_t n, size_t k, size_t nr, size_t kr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_rhs_packed_stride(k, kr, nr);
}

void kai_run_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, const uint8_t* rhs, const int32_t* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0_params* params) {
    KAI_ASSERT((k % 2) == 0);
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(bias == NULL);
    KAI_ASSERT(extra_bytes == 0);

    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->rhs_zero_point == 8);
    KAI_ASSERT(params->lhs_zero_point == 1);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const int32_t rhs_zero_point = params->rhs_zero_point;
    const size_t rhs_stride = k / 2;
    const size_t rhs_packed_stride = kai_rhs_packed_stride(k, kr, nr);
    const size_t k_internal = kai_k_roundedup(k, kr);
    const size_t dst_nr_block_size = nr * kr * sizeof(uint8_t) / 2;

    // Iterate over n src rows in blocks of nr rows
    for (size_t row_idx = 0; row_idx < n; row_idx += nr) {
        int8_t * const dst_row = (int8_t*)rhs_packed + (row_idx / nr) * rhs_packed_stride;

        int32_t * const sums = (int32_t*)(dst_row + nr * (k_internal / 2));
        float32_t * const scaling_factors = (float32_t*)((uint8_t *) sums + nr * sizeof(int32_t));

        // Initialize to zero the RHS reduction sums and scaling factors (sum accumulate and padding)
        memset(sums, 0, nr * (sizeof(int32_t) + sizeof(float32_t)));

        // Iterate over rows in the nr row block
        for (size_t nr_block_idx = 0; nr_block_idx < nr; ++nr_block_idx) {
            const uint8_t * const src_row = rhs + (row_idx + nr_block_idx) * rhs_stride;
            // Go to the first kr block for this row in the nr block
            int8_t *dst_kr_block = dst_row + nr_block_idx * kr / 2 ;

            int32_t sum = 0;

            // Could do memcopy for all scaling factors for the nr block instead
            scaling_factors[nr_block_idx] = row_idx + nr_block_idx < n ? scale[row_idx + nr_block_idx] : 0;

            // Iterate over k src columns in blocks of kr columns
            for (size_t col_idx = 0; col_idx < k_internal; col_idx += kr) {

                // Iterate over columns in the kr block
                // Kr checked to be multiple of 2 (because 2 values per byte)
                for (size_t kr_block_idx = 0; kr_block_idx < kr; kr_block_idx += 2) {

                    // We pad dst with 0s if the rounded k or n values have been exceeded
                    if (row_idx + nr_block_idx >= n || col_idx + kr_block_idx >= k) {
                      dst_kr_block[kr_block_idx / 2] = 0;
                      continue;
                    }

                    // Load the 2 u4 values from source
                    const uint8_t dst_byte = src_row[(col_idx + kr_block_idx) / 2];

                    // extract i8 values from the 2 u4 values
                    const int32_t first_value = (dst_byte & 0xF) - rhs_zero_point;
                    const int32_t second_value = (dst_byte >> 4) - rhs_zero_point;

                    // Add the i4 value to the row sum
                    sum += first_value + second_value;

                    // Truncate i8 to i4 and write to dst
                    dst_kr_block[kr_block_idx / 2] = (second_value << 4) | (first_value & 0xF);
                }

                // Go to the next kr block for this row in the nr rows
                dst_kr_block += dst_nr_block_size;
            }

            // save sum
            sums[nr_block_idx] = sum;
        }
    }
}
