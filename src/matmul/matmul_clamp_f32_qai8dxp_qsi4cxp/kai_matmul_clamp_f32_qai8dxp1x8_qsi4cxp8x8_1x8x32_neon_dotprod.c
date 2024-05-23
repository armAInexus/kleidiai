//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h"

#include <arm_neon.h>
#include <stdint.h>

#include "kai_common.h"

static const size_t kai_m_step = 1;
static const size_t kai_n_step = 8;
static const size_t kai_mr = 1;
static const size_t kai_nr = 8;
static const size_t kai_kr = 16;
static const size_t kai_sr = 2;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);

inline static size_t kai_k_roundedup(size_t k) {
    // Since we pack a float and int32 value at the end of the row,
    // we must make sure that k is a multiple of 4 for alignment
    size_t kr_sr_roundedup4 = kai_roundup(kai_kr * kai_sr, 4);
    return kai_roundup(k, kr_sr_roundedup4);
}

inline static size_t kai_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return kai_mr * (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

inline static size_t kai_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return kai_nr * ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed, const void* restrict rhs_packed, float* restrict dst,
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
#if defined(__ARM_FEATURE_DOTPROD)
    KAI_ASSERT(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t kai_k0 = kai_kr * kai_sr;
    const size_t num_rows = m;
    const size_t num_cols = n;

    const size_t lhs_packed_stride = kai_lhs_packed_stride(k);
    const size_t k_internal = kai_k_roundedup(k);

    const int8x16_t nibble_mask = vdupq_n_s8(0xF0);

    const uint8_t* lhs_ptr_start = lhs_packed;

    for (size_t row_idx = 0; row_idx < num_rows; row_idx += kai_mr) {
        const uint8_t* rhs_ptr = rhs_packed;
        for (size_t col_idx = 0; col_idx < num_cols; col_idx += kai_nr) {
            const uint8_t* lhs_ptr = lhs_ptr_start;

            // Main f32 accumulator
            int32x4_t iacc0011 = vdupq_n_s32(0);
            int32x4_t iacc2233 = vdupq_n_s32(0);
            int32x4_t iacc4455 = vdupq_n_s32(0);
            int32x4_t iacc6677 = vdupq_n_s32(0);

            for (size_t b = 0; b < k_internal; b += kai_k0) {
                // Set up RHS
                const int8x16_t rhs_raw_vec_0 = vld1q_s8((const int8_t*)(rhs_ptr + 0));
                const int8x16_t rhs_raw_vec_1 = vld1q_s8((const int8_t*)(rhs_ptr + 16));
                const int8x16_t rhs_raw_vec_2 = vld1q_s8((const int8_t*)(rhs_ptr + 32));
                const int8x16_t rhs_raw_vec_3 = vld1q_s8((const int8_t*)(rhs_ptr + 48));
                const int8x16_t rhs_raw_vec_4 = vld1q_s8((const int8_t*)(rhs_ptr + 64));
                const int8x16_t rhs_raw_vec_5 = vld1q_s8((const int8_t*)(rhs_ptr + 80));
                const int8x16_t rhs_raw_vec_6 = vld1q_s8((const int8_t*)(rhs_ptr + 96));
                const int8x16_t rhs_raw_vec_7 = vld1q_s8((const int8_t*)(rhs_ptr + 112));

                // Low nibble
                const int8x16_t rhs_vec_0_0 = vshlq_n_s8(rhs_raw_vec_0, 4);
                const int8x16_t rhs_vec_1_0 = vshlq_n_s8(rhs_raw_vec_1, 4);
                const int8x16_t rhs_vec_2_0 = vshlq_n_s8(rhs_raw_vec_2, 4);
                const int8x16_t rhs_vec_3_0 = vshlq_n_s8(rhs_raw_vec_3, 4);
                const int8x16_t rhs_vec_4_0 = vshlq_n_s8(rhs_raw_vec_4, 4);
                const int8x16_t rhs_vec_5_0 = vshlq_n_s8(rhs_raw_vec_5, 4);
                const int8x16_t rhs_vec_6_0 = vshlq_n_s8(rhs_raw_vec_6, 4);
                const int8x16_t rhs_vec_7_0 = vshlq_n_s8(rhs_raw_vec_7, 4);

                // High nibble
                const int8x16_t rhs_vec_0_1 = vandq_s8(rhs_raw_vec_0, nibble_mask);
                const int8x16_t rhs_vec_1_1 = vandq_s8(rhs_raw_vec_1, nibble_mask);
                const int8x16_t rhs_vec_2_1 = vandq_s8(rhs_raw_vec_2, nibble_mask);
                const int8x16_t rhs_vec_3_1 = vandq_s8(rhs_raw_vec_3, nibble_mask);
                const int8x16_t rhs_vec_4_1 = vandq_s8(rhs_raw_vec_4, nibble_mask);
                const int8x16_t rhs_vec_5_1 = vandq_s8(rhs_raw_vec_5, nibble_mask);
                const int8x16_t rhs_vec_6_1 = vandq_s8(rhs_raw_vec_6, nibble_mask);
                const int8x16_t rhs_vec_7_1 = vandq_s8(rhs_raw_vec_7, nibble_mask);

                const int8x16_t lhs_vec_0 = vld1q_s8((const int8_t*)(lhs_ptr + 0));
                const int8x16_t lhs_vec_1 = vld1q_s8((const int8_t*)(lhs_ptr + 16));

                lhs_ptr += 32;
                rhs_ptr += 128;

                int8x16_t t;

                t = vcombine_s8(vget_low_s8(lhs_vec_0), vget_low_s8(lhs_vec_0));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_0_0, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_1_0, t);
                iacc4455 = vdotq_s32(iacc4455, rhs_vec_2_0, t);
                iacc6677 = vdotq_s32(iacc6677, rhs_vec_3_0, t);
                t = vcombine_s8(vget_high_s8(lhs_vec_0), vget_high_s8(lhs_vec_0));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_4_0, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_5_0, t);
                iacc4455 = vdotq_s32(iacc4455, rhs_vec_6_0, t);
                iacc6677 = vdotq_s32(iacc6677, rhs_vec_7_0, t);

                t = vcombine_s8(vget_low_s8(lhs_vec_1), vget_low_s8(lhs_vec_1));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_0_1, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_1_1, t);
                iacc4455 = vdotq_s32(iacc4455, rhs_vec_2_1, t);
                iacc6677 = vdotq_s32(iacc6677, rhs_vec_3_1, t);
                t = vcombine_s8(vget_high_s8(lhs_vec_1), vget_high_s8(lhs_vec_1));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_4_1, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_5_1, t);
                iacc4455 = vdotq_s32(iacc4455, rhs_vec_6_1, t);
                iacc6677 = vdotq_s32(iacc6677, rhs_vec_7_1, t);
            }

            int32x4_t iacc0 = vpaddq_s32(iacc0011, iacc2233);
            int32x4_t iacc1 = vpaddq_s32(iacc4455, iacc6677);

            // LHS offset
            const int32x4_t lhs_offset = vld1q_dup_s32((const int32_t*)lhs_ptr);
            lhs_ptr += sizeof(int32_t);

            // LHS scale
            const float32x4_t lhs_scale = vld1q_dup_f32((const float*)lhs_ptr);
            lhs_ptr += sizeof(float);

            // RHS sum values
            const int32x4_t sum_n_s32_0 = vld1q_s32((const int32_t*)(rhs_ptr));
            rhs_ptr += sizeof(int32x4_t);
            const int32x4_t sum_n_s32_1 = vld1q_s32((const int32_t*)(rhs_ptr));
            rhs_ptr += sizeof(int32x4_t);

            // RHS scale
            const float32x4_t rhs_scale0 = vld1q_f32((const float*)rhs_ptr);
            rhs_ptr += sizeof(float32x4_t);
            const float32x4_t rhs_scale1 = vld1q_f32((const float*)rhs_ptr);
            rhs_ptr += sizeof(float32x4_t);

            // Add the reduction sum
            iacc0 = vmlaq_s32(iacc0, sum_n_s32_0, lhs_offset);
            iacc1 = vmlaq_s32(iacc1, sum_n_s32_1, lhs_offset);

            float32x4_t main_acc0 = vmulq_f32(vcvtq_f32_s32(iacc0), rhs_scale0);
            float32x4_t main_acc1 = vmulq_f32(vcvtq_f32_s32(iacc1), rhs_scale1);

            main_acc0 = vmulq_f32(main_acc0, lhs_scale);
            main_acc1 = vmulq_f32(main_acc1, lhs_scale);

            // clamp (min-max) operation
            const float32x4_t vmin_f32 = vdupq_n_f32(scalar_min);
            const float32x4_t vmax_f32 = vdupq_n_f32(scalar_max);

            main_acc0 = vmaxq_f32(main_acc0, vmin_f32);
            main_acc0 = vminq_f32(main_acc0, vmax_f32);

            main_acc1 = vmaxq_f32(main_acc1, vmin_f32);
            main_acc1 = vminq_f32(main_acc1, vmax_f32);

            if (col_idx + kai_nr <= n) {
                vst1q_f32(
                    (float*)((uint8_t*)dst + (col_idx + 0) * sizeof(float) + row_idx * dst_stride_row), main_acc0);
                vst1q_f32(
                    (float*)((uint8_t*)dst + (col_idx + 4) * sizeof(float) + row_idx * dst_stride_row), main_acc1);
            } else {
                size_t leftover = n % kai_nr;
                *(float*)((uint8_t*)dst + (col_idx + 0) * sizeof(float) + row_idx * dst_stride_row) =
                    vgetq_lane_f32(main_acc0, 0);
                if (leftover > 1) {
                    *(float*)((uint8_t*)dst + (col_idx + 1) * sizeof(float) + row_idx * dst_stride_row) =
                        vgetq_lane_f32(main_acc0, 1);
                }
                if (leftover > 2) {
                    *(float*)((uint8_t*)dst + (col_idx + 2) * sizeof(float) + row_idx * dst_stride_row) =
                        vgetq_lane_f32(main_acc0, 2);
                }
                if (leftover > 3) {
                    *(float*)((uint8_t*)dst + (col_idx + 3) * sizeof(float) + row_idx * dst_stride_row) =
                        vgetq_lane_f32(main_acc0, 3);
                }
                if (leftover > 4) {
                    *(float*)((uint8_t*)dst + (col_idx + 4) * sizeof(float) + row_idx * dst_stride_row) =
                        vgetq_lane_f32(main_acc1, 0);
                }
                if (leftover > 5) {
                    *(float*)((uint8_t*)dst + (col_idx + 5) * sizeof(float) + row_idx * dst_stride_row) =
                        vgetq_lane_f32(main_acc1, 1);
                }
                if (leftover > 6) {
                    *(float*)((uint8_t*)dst + (col_idx + 6) * sizeof(float) + row_idx * dst_stride_row) =
                        vgetq_lane_f32(main_acc1, 2);
                }
            }
        }
        lhs_ptr_start += lhs_packed_stride;
    }
#else
    KAI_ASSERT(false);
    KAI_UNUSED(m);
    KAI_UNUSED(n);
    KAI_UNUSED(k);
    KAI_UNUSED(lhs_packed);
    KAI_UNUSED(rhs_packed);
    KAI_UNUSED(dst);
    KAI_UNUSED(dst_stride_row);
    KAI_UNUSED(dst_stride_col);
    KAI_UNUSED(scalar_min);
    KAI_UNUSED(scalar_max);
#endif
}
