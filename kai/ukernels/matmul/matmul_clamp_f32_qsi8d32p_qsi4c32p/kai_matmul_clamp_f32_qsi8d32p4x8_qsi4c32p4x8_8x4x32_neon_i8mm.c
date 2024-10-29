
//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "i8mm extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 8;
static const size_t kai_n_step = 4;
static const size_t kai_mr = 4;
static const size_t kai_nr = 4;
static const size_t kai_kr = 16;
static const size_t kai_sr = 2;
static const size_t kai_k0 = kai_kr * kai_sr;
static const size_t kai_block_size = 32;
static const size_t kai_num_bytes_multiplier = sizeof(uint16_t);

inline static size_t kai_num_bytes_per_block_lhs(size_t bl) {
    return bl * sizeof(int8_t) + kai_num_bytes_multiplier;
}

inline static size_t kai_num_bytes_per_block_rhs(size_t bl) {
    return (bl / 2) * sizeof(int8_t) + kai_num_bytes_multiplier;
}

inline static size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((k % bl) == 0);
    return k / bl;
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t bl) {
    return kai_mr * kai_num_blocks_per_row(k, bl) * kai_num_bytes_per_block_lhs(bl);
}

inline static size_t kai_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kai_kr) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block_rhs(bl);

    return kai_nr * (num_bytes_per_block * num_blocks_per_row);
}

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    size_t m_idx, size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % kai_nr) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm(
    size_t m, size_t n, size_t k, size_t bl, const void* lhs_packed, const void* rhs_packed, float* dst,
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSUME(n % kai_nr == 0);
    KAI_ASSUME(k % kai_k0 == 0);
    KAI_ASSUME(bl == 32);
    KAI_ASSUME(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t lhs_packed_stride = kai_lhs_packed_stride(k, bl);
    const size_t num_blocks = k / kai_block_size;
    const size_t num_cols = n;
    const size_t num_rows = m;

    const int8x16_t nibble_mask = vreinterpretq_s8_u8(vdupq_n_u8(0xF0));

    const uint8_t* lhs_ptr_start = lhs_packed;

    for (size_t row_idx = 0; row_idx < num_rows; row_idx += kai_m_step) {
        const size_t step_packed_row = (int32_t)num_rows - (int32_t)row_idx <= 4 ? 0 : 1;

        const uint8_t* rhs_ptr = rhs_packed;

        for (size_t col_idx = 0; col_idx < num_cols; col_idx += kai_n_step) {
            const uint8_t* lhs_ptr = lhs_ptr_start;

            // Main f32 accumulator
            float32x4_t main_acc0 = vdupq_n_f32(0.0F);
            float32x4_t main_acc1 = vdupq_n_f32(0.0F);
            float32x4_t main_acc2 = vdupq_n_f32(0.0F);
            float32x4_t main_acc3 = vdupq_n_f32(0.0F);
            float32x4_t main_acc4 = vdupq_n_f32(0.0F);
            float32x4_t main_acc5 = vdupq_n_f32(0.0F);
            float32x4_t main_acc6 = vdupq_n_f32(0.0F);
            float32x4_t main_acc7 = vdupq_n_f32(0.0F);

            for (size_t b = 0; b < num_blocks; b++) {
                // Set up RHS
                const int8x16_t rhs_raw_mat_01_0 = vld1q_s8((const int8_t*)rhs_ptr + 0);
                const int8x16_t rhs_raw_mat_23_0 = vld1q_s8((const int8_t*)rhs_ptr + 16);
                const int8x16_t rhs_raw_mat_01_1 = vld1q_s8((const int8_t*)rhs_ptr + 32);
                const int8x16_t rhs_raw_mat_23_1 = vld1q_s8((const int8_t*)rhs_ptr + 48);

                const float16x4_t col_scale_f16 = vld1_f16((const float16_t*)((const uint8_t*)rhs_ptr + 64));
                const float32x4_t col_scale_f32 = vcvt_f32_f16(col_scale_f16);

                // Low nibble
                const int8x16_t rhs_mat_01_0 = vshlq_n_s8(rhs_raw_mat_01_0, 4);
                const int8x16_t rhs_mat_23_0 = vshlq_n_s8(rhs_raw_mat_23_0, 4);
                const int8x16_t rhs_mat_01_1 = vshlq_n_s8(rhs_raw_mat_01_1, 4);
                const int8x16_t rhs_mat_23_1 = vshlq_n_s8(rhs_raw_mat_23_1, 4);

                // High nibble
                const int8x16_t rhs_mat_01_2 = vandq_s8(rhs_raw_mat_01_0, nibble_mask);
                const int8x16_t rhs_mat_23_2 = vandq_s8(rhs_raw_mat_23_0, nibble_mask);
                const int8x16_t rhs_mat_01_3 = vandq_s8(rhs_raw_mat_01_1, nibble_mask);
                const int8x16_t rhs_mat_23_3 = vandq_s8(rhs_raw_mat_23_1, nibble_mask);

                // Process LHS in pairs of rows
                {
                    const int8x16_t lhs_mat_01_0 = vld1q_s8((const int8_t*)lhs_ptr + 0);
                    const int8x16_t lhs_mat_23_0 = vld1q_s8((const int8_t*)lhs_ptr + 16);
                    const int8x16_t lhs_mat_01_1 = vld1q_s8((const int8_t*)lhs_ptr + 32);
                    const int8x16_t lhs_mat_23_1 = vld1q_s8((const int8_t*)lhs_ptr + 48);
                    const int8x16_t lhs_mat_01_2 = vld1q_s8((const int8_t*)lhs_ptr + 64);
                    const int8x16_t lhs_mat_23_2 = vld1q_s8((const int8_t*)lhs_ptr + 80);
                    const int8x16_t lhs_mat_01_3 = vld1q_s8((const int8_t*)lhs_ptr + 96);
                    const int8x16_t lhs_mat_23_3 = vld1q_s8((const int8_t*)lhs_ptr + 112);

                    // Do the MMLAs into 2x2 matrices
                    const int32x4_t iacc_mat_00 = vmmlaq_s32(
                        vmmlaq_s32(
                            vmmlaq_s32(
                                vmmlaq_s32(vdupq_n_s32(0), lhs_mat_01_0, rhs_mat_01_0), lhs_mat_01_1, rhs_mat_01_1),
                            lhs_mat_01_2, rhs_mat_01_2),
                        lhs_mat_01_3, rhs_mat_01_3);
                    const int32x4_t iacc_mat_01 = vmmlaq_s32(
                        vmmlaq_s32(
                            vmmlaq_s32(
                                vmmlaq_s32(vdupq_n_s32(0), lhs_mat_01_0, rhs_mat_23_0), lhs_mat_01_1, rhs_mat_23_1),
                            lhs_mat_01_2, rhs_mat_23_2),
                        lhs_mat_01_3, rhs_mat_23_3);
                    const int32x4_t iacc_mat_10 = vmmlaq_s32(
                        vmmlaq_s32(
                            vmmlaq_s32(
                                vmmlaq_s32(vdupq_n_s32(0), lhs_mat_23_0, rhs_mat_01_0), lhs_mat_23_1, rhs_mat_01_1),
                            lhs_mat_23_2, rhs_mat_01_2),
                        lhs_mat_23_3, rhs_mat_01_3);
                    const int32x4_t iacc_mat_11 = vmmlaq_s32(
                        vmmlaq_s32(
                            vmmlaq_s32(
                                vmmlaq_s32(vdupq_n_s32(0), lhs_mat_23_0, rhs_mat_23_0), lhs_mat_23_1, rhs_mat_23_1),
                            lhs_mat_23_2, rhs_mat_23_2),
                        lhs_mat_23_3, rhs_mat_23_3);

                    // Straighten out to make 4 row vectors
                    const int32x4_t iacc_row_0 = vreinterpretq_s32_u64(
                        vtrn1q_u64(vreinterpretq_u64_s32(iacc_mat_00), vreinterpretq_u64_s32(iacc_mat_01)));
                    const int32x4_t iacc_row_1 = vreinterpretq_s32_u64(
                        vtrn2q_u64(vreinterpretq_u64_s32(iacc_mat_00), vreinterpretq_u64_s32(iacc_mat_01)));
                    const int32x4_t iacc_row_2 = vreinterpretq_s32_u64(
                        vtrn1q_u64(vreinterpretq_u64_s32(iacc_mat_10), vreinterpretq_u64_s32(iacc_mat_11)));
                    const int32x4_t iacc_row_3 = vreinterpretq_s32_u64(
                        vtrn2q_u64(vreinterpretq_u64_s32(iacc_mat_10), vreinterpretq_u64_s32(iacc_mat_11)));

                    const float16x4_t row_scale_f16 = vld1_f16((const float16_t*)((const uint8_t*)lhs_ptr + 128));
                    const float32x4_t row_scale_f32 = vcvt_f32_f16(row_scale_f16);

                    main_acc0 = vfmaq_f32(
                        main_acc0, vcvtq_f32_s32(iacc_row_0), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 0));
                    main_acc1 = vfmaq_f32(
                        main_acc1, vcvtq_f32_s32(iacc_row_1), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 1));
                    main_acc2 = vfmaq_f32(
                        main_acc2, vcvtq_f32_s32(iacc_row_2), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 2));
                    main_acc3 = vfmaq_f32(
                        main_acc3, vcvtq_f32_s32(iacc_row_3), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 3));
                }

                lhs_ptr += step_packed_row * lhs_packed_stride;

                {
                    const int8x16_t lhs_mat_01_0 = vld1q_s8((const int8_t*)lhs_ptr + 0);
                    const int8x16_t lhs_mat_23_0 = vld1q_s8((const int8_t*)lhs_ptr + 16);
                    const int8x16_t lhs_mat_01_1 = vld1q_s8((const int8_t*)lhs_ptr + 32);
                    const int8x16_t lhs_mat_23_1 = vld1q_s8((const int8_t*)lhs_ptr + 48);
                    const int8x16_t lhs_mat_01_2 = vld1q_s8((const int8_t*)lhs_ptr + 64);
                    const int8x16_t lhs_mat_23_2 = vld1q_s8((const int8_t*)lhs_ptr + 80);
                    const int8x16_t lhs_mat_01_3 = vld1q_s8((const int8_t*)lhs_ptr + 96);
                    const int8x16_t lhs_mat_23_3 = vld1q_s8((const int8_t*)lhs_ptr + 112);

                    // Do the MMLAs into 2x2 matrices
                    const int32x4_t iacc_mat_00 = vmmlaq_s32(
                        vmmlaq_s32(
                            vmmlaq_s32(
                                vmmlaq_s32(vdupq_n_s32(0), lhs_mat_01_0, rhs_mat_01_0), lhs_mat_01_1, rhs_mat_01_1),
                            lhs_mat_01_2, rhs_mat_01_2),
                        lhs_mat_01_3, rhs_mat_01_3);
                    const int32x4_t iacc_mat_01 = vmmlaq_s32(
                        vmmlaq_s32(
                            vmmlaq_s32(
                                vmmlaq_s32(vdupq_n_s32(0), lhs_mat_01_0, rhs_mat_23_0), lhs_mat_01_1, rhs_mat_23_1),
                            lhs_mat_01_2, rhs_mat_23_2),
                        lhs_mat_01_3, rhs_mat_23_3);
                    const int32x4_t iacc_mat_10 = vmmlaq_s32(
                        vmmlaq_s32(
                            vmmlaq_s32(
                                vmmlaq_s32(vdupq_n_s32(0), lhs_mat_23_0, rhs_mat_01_0), lhs_mat_23_1, rhs_mat_01_1),
                            lhs_mat_23_2, rhs_mat_01_2),
                        lhs_mat_23_3, rhs_mat_01_3);
                    const int32x4_t iacc_mat_11 = vmmlaq_s32(
                        vmmlaq_s32(
                            vmmlaq_s32(
                                vmmlaq_s32(vdupq_n_s32(0), lhs_mat_23_0, rhs_mat_23_0), lhs_mat_23_1, rhs_mat_23_1),
                            lhs_mat_23_2, rhs_mat_23_2),
                        lhs_mat_23_3, rhs_mat_23_3);

                    // Straighten out to make 4 row vectors
                    const int32x4_t iacc_row_0 = vreinterpretq_s32_u64(
                        vtrn1q_u64(vreinterpretq_u64_s32(iacc_mat_00), vreinterpretq_u64_s32(iacc_mat_01)));
                    const int32x4_t iacc_row_1 = vreinterpretq_s32_u64(
                        vtrn2q_u64(vreinterpretq_u64_s32(iacc_mat_00), vreinterpretq_u64_s32(iacc_mat_01)));
                    const int32x4_t iacc_row_2 = vreinterpretq_s32_u64(
                        vtrn1q_u64(vreinterpretq_u64_s32(iacc_mat_10), vreinterpretq_u64_s32(iacc_mat_11)));
                    const int32x4_t iacc_row_3 = vreinterpretq_s32_u64(
                        vtrn2q_u64(vreinterpretq_u64_s32(iacc_mat_10), vreinterpretq_u64_s32(iacc_mat_11)));

                    const float16x4_t row_scale_f16 = vld1_f16((const float16_t*)((const uint8_t*)lhs_ptr + 128));
                    const float32x4_t row_scale_f32 = vcvt_f32_f16(row_scale_f16);

                    main_acc4 = vfmaq_f32(
                        main_acc4, vcvtq_f32_s32(iacc_row_0), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 0));
                    main_acc5 = vfmaq_f32(
                        main_acc5, vcvtq_f32_s32(iacc_row_1), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 1));
                    main_acc6 = vfmaq_f32(
                        main_acc6, vcvtq_f32_s32(iacc_row_2), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 2));
                    main_acc7 = vfmaq_f32(
                        main_acc7, vcvtq_f32_s32(iacc_row_3), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 3));
                }

                lhs_ptr -= step_packed_row * lhs_packed_stride;

                lhs_ptr += 136;
                rhs_ptr += 72;
            }

            const float32x4_t vmin_f32 = vdupq_n_f32(scalar_min);
            const float32x4_t vmax_f32 = vdupq_n_f32(scalar_max);

            main_acc0 = vmaxq_f32(main_acc0, vmin_f32);
            main_acc0 = vminq_f32(main_acc0, vmax_f32);
            main_acc1 = vmaxq_f32(main_acc1, vmin_f32);
            main_acc1 = vminq_f32(main_acc1, vmax_f32);
            main_acc2 = vmaxq_f32(main_acc2, vmin_f32);
            main_acc2 = vminq_f32(main_acc2, vmax_f32);
            main_acc3 = vmaxq_f32(main_acc3, vmin_f32);
            main_acc3 = vminq_f32(main_acc3, vmax_f32);
            main_acc4 = vmaxq_f32(main_acc4, vmin_f32);
            main_acc4 = vminq_f32(main_acc4, vmax_f32);
            main_acc5 = vmaxq_f32(main_acc5, vmin_f32);
            main_acc5 = vminq_f32(main_acc5, vmax_f32);
            main_acc6 = vmaxq_f32(main_acc6, vmin_f32);
            main_acc6 = vminq_f32(main_acc6, vmax_f32);
            main_acc7 = vmaxq_f32(main_acc7, vmin_f32);
            main_acc7 = vminq_f32(main_acc7, vmax_f32);

            // Stores the rows in reverse order to avoid out-of-bound writes.
            // Override out-of-bound values with in-bound values
            vst1q_f32(
                (float*)((uint8_t*)dst + col_idx * sizeof(float) + KAI_MIN(row_idx + 7, m - 1) * dst_stride_row),
                main_acc7);
            vst1q_f32(
                (float*)((uint8_t*)dst + col_idx * sizeof(float) + KAI_MIN(row_idx + 6, m - 1) * dst_stride_row),
                main_acc6);
            vst1q_f32(
                (float*)((uint8_t*)dst + col_idx * sizeof(float) + KAI_MIN(row_idx + 5, m - 1) * dst_stride_row),
                main_acc5);
            vst1q_f32(
                (float*)((uint8_t*)dst + col_idx * sizeof(float) + KAI_MIN(row_idx + 4, m - 1) * dst_stride_row),
                main_acc4);
            vst1q_f32(
                (float*)((uint8_t*)dst + col_idx * sizeof(float) + KAI_MIN(row_idx + 3, m - 1) * dst_stride_row),
                main_acc3);
            vst1q_f32(
                (float*)((uint8_t*)dst + col_idx * sizeof(float) + KAI_MIN(row_idx + 2, m - 1) * dst_stride_row),
                main_acc2);
            vst1q_f32(
                (float*)((uint8_t*)dst + col_idx * sizeof(float) + KAI_MIN(row_idx + 1, m - 1) * dst_stride_row),
                main_acc1);
            vst1q_f32(
                (float*)((uint8_t*)dst + col_idx * sizeof(float) + KAI_MIN(row_idx + 0, m - 1) * dst_stride_row),
                main_acc0);
        }

        lhs_ptr_start += 2 * lhs_packed_stride;
    }
}

#endif  // Architectural feature check
