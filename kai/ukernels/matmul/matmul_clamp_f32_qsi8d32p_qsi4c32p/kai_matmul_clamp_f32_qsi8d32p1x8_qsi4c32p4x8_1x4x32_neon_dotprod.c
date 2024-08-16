
//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"

#include <arm_neon.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;
static const size_t kai_mr = 1;
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

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
    size_t m_idx, size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % kai_nr) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
    size_t m, size_t n, size_t k, size_t bl, const void* lhs_packed, const void* rhs_packed, float* dst,
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSUME(n % kai_nr == 0);
    KAI_ASSUME(k % kai_k0 == 0);
    KAI_ASSUME(bl == 32);
    KAI_ASSUME(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t num_blocks = k / kai_block_size;
    const size_t num_cols = n;
    const size_t num_rows = m;
    const size_t lhs_packed_stride = kai_lhs_packed_stride(k, bl);

    const int8x16_t nibble_mask = vdupq_n_s8(0xF0);

    const uint8_t* lhs_ptr_start = lhs_packed;

    for (size_t row_idx = 0; row_idx < num_rows; row_idx += kai_m_step) {
        const uint8_t* rhs_ptr = rhs_packed;
        for (size_t col_idx = 0; col_idx < num_cols; col_idx += kai_n_step) {
            // Main f32 accumulator
            float32x4_t main_acc = vdupq_n_f32(0.0F);

            const uint8_t* lhs_ptr = lhs_ptr_start;

            for (size_t b = 0; b < num_blocks; b++) {
                // Set up RHS
                const int8x16_t rhs_raw_vec_0 = vld1q_s8((const int8_t*)rhs_ptr + 0);
                const int8x16_t rhs_raw_vec_1 = vld1q_s8((const int8_t*)rhs_ptr + 16);
                const int8x16_t rhs_raw_vec_2 = vld1q_s8((const int8_t*)rhs_ptr + 32);
                const int8x16_t rhs_raw_vec_3 = vld1q_s8((const int8_t*)rhs_ptr + 48);

                // Low nibble
                const int8x16_t rhs_vec_0_0 = vshlq_n_s8(rhs_raw_vec_0, 4);
                const int8x16_t rhs_vec_1_0 = vshlq_n_s8(rhs_raw_vec_1, 4);
                const int8x16_t rhs_vec_2_0 = vshlq_n_s8(rhs_raw_vec_2, 4);
                const int8x16_t rhs_vec_3_0 = vshlq_n_s8(rhs_raw_vec_3, 4);

                // High nibble
                const int8x16_t rhs_vec_0_1 = vandq_s8(rhs_raw_vec_0, nibble_mask);
                const int8x16_t rhs_vec_1_1 = vandq_s8(rhs_raw_vec_1, nibble_mask);
                const int8x16_t rhs_vec_2_1 = vandq_s8(rhs_raw_vec_2, nibble_mask);
                const int8x16_t rhs_vec_3_1 = vandq_s8(rhs_raw_vec_3, nibble_mask);

                const int8x16_t lhs_vec_0 = vld1q_s8((const int8_t*)(lhs_ptr + 0));
                const int8x16_t lhs_vec_1 = vld1q_s8((const int8_t*)(lhs_ptr + 16));

                int32x4_t iacc0011 = vdupq_n_s32(0);
                int32x4_t iacc2233 = vdupq_n_s32(0);

                int8x16_t t;

                t = vcombine_s8(vget_low_s8(lhs_vec_0), vget_low_s8(lhs_vec_0));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_0_0, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_1_0, t);
                t = vcombine_s8(vget_high_s8(lhs_vec_0), vget_high_s8(lhs_vec_0));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_2_0, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_3_0, t);
                t = vcombine_s8(vget_low_s8(lhs_vec_1), vget_low_s8(lhs_vec_1));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_0_1, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_1_1, t);
                t = vcombine_s8(vget_high_s8(lhs_vec_1), vget_high_s8(lhs_vec_1));
                iacc0011 = vdotq_s32(iacc0011, rhs_vec_2_1, t);
                iacc2233 = vdotq_s32(iacc2233, rhs_vec_3_1, t);

                int32x4_t iacc = vpaddq_s32(iacc0011, iacc2233);

                // RHS scale values
                const float16x4_t col_scale_f16 = vld1_f16((const float16_t*)(rhs_ptr + 64));
                const float32x4_t col_scale_f32 = vcvt_f32_f16(col_scale_f16);

                // LHS scale values
                const float16x4_t row_scale_f16 = vld1_dup_f16((const float16_t*)(lhs_ptr + 32));
                const float32x4_t row_scale_f32 = vcvt_f32_f16(row_scale_f16);

                lhs_ptr += 34;
                rhs_ptr += 72;

                main_acc = vfmaq_f32(main_acc, vcvtq_f32_s32(iacc), vmulq_f32(col_scale_f32, row_scale_f32));
            }

            const float32x4_t vmin_f32 = vdupq_n_f32(scalar_min);
            const float32x4_t vmax_f32 = vdupq_n_f32(scalar_max);

            main_acc = vmaxq_f32(main_acc, vmin_f32);
            main_acc = vminq_f32(main_acc, vmax_f32);

            vst1q_f32((float*)((uint8_t*)dst + col_idx * sizeof(float) + row_idx * dst_stride_row), main_acc);
        }
        lhs_ptr_start += lhs_packed_stride;
    }
}

#endif  // Architectural feature check
