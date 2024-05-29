//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "I8mm extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h"

#include <arm_neon.h>
#include <stdint.h>

#include "kai_common.h"

static const size_t kai_m_step = 8;
static const size_t kai_n_step = 4;
static const size_t kai_mr = 4;
static const size_t kai_nr = 4;
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

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t k_internal = kai_k_roundedup(k);

    size_t num_blocks = k_internal / 32;

    float clamp_vals[2] = {scalar_min, scalar_max};

    __asm__ __volatile__(
        "mov x12, %x[m]\n"
        "mov x11, #0x80\n"
        "movi v11.16b, #0xf0\n"
        "mov x20, #0x20\n"
        "cmp x12, #0x8\n"
        "madd x11, %x[num_blocks], x11, x20\n"
        "blt 8f\n"
        "1:"  // Row loop
        "mov x10, %x[rhs_packed]\n"
        "mov x9, %x[n]\n"
        "add x28, %x[dst], %x[dst_stride_row], LSL #3\n"
        "2:"  // Column loop
        "mov x22, %x[lhs_packed]\n"
        "movi v10.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "mov x21, %x[num_blocks]\n"
        "movi v8.4s, #0x0\n"
        "movi v7.4s, #0x0\n"
        "movi v6.4s, #0x0\n"
        "movi v5.4s, #0x0\n"
        "movi v4.4s, #0x0\n"
        "movi v3.4s, #0x0\n"
        "add x20, x22, x11\n"
        "3:"  // Block loop
        "ldr q2, [x10, #0x0]\n"
        "ldr q1, [x10, #0x10]\n"
        "subs x21, x21, #0x1\n"
        "ldr q20, [x22, #0x0]\n"
        "ldr q19, [x22, #0x10]\n"
        "ldr q18, [x20, #0x0]\n"
        "ldr q0, [x20, #0x10]\n"
        "ldr q31, [x10, #0x20]\n"
        "ldr q30, [x10, #0x30]\n"
        "shl v17.16b, v2.16b, #0x4\n"
        "shl v16.16b, v1.16b, #0x4\n"
        "ldr q29, [x22, #0x20]\n"
        "ldr q28, [x22, #0x30]\n"
        "and v2.16b, v2.16b, v11.16b\n"
        "and v1.16b, v1.16b, v11.16b\n"
        "ldr q27, [x20, #0x20]\n"
        "ldr q26, [x20, #0x30]\n"
        "add x10, x10, #0x40\n"
        "ldr q25, [x22, #0x40]\n"
        "ldr q24, [x22, #0x50]\n"
        ".inst 0x4e91a68a  // smmla v10.4s, v20.16b, v17.16b\n"
        ".inst 0x4e90a689  // smmla v9.4s, v20.16b, v16.16b\n"
        "ldr q23, [x20, #0x40]\n"
        "ldr q22, [x20, #0x50]\n"
        ".inst 0x4e91a668  // smmla v8.4s, v19.16b, v17.16b\n"
        ".inst 0x4e90a667  // smmla v7.4s, v19.16b, v16.16b\n"
        "ldr q21, [x22, #0x60]\n"
        "ldr q20, [x22, #0x70]\n"
        ".inst 0x4e91a646  // smmla v6.4s, v18.16b, v17.16b\n"
        ".inst 0x4e90a645  // smmla v5.4s, v18.16b, v16.16b\n"
        "ldr q19, [x20, #0x60]\n"
        "ldr q18, [x20, #0x70]\n"
        ".inst 0x4e91a404  // smmla v4.4s, v0.16b, v17.16b\n"
        ".inst 0x4e90a403  // smmla v3.4s, v0.16b, v16.16b\n"
        "shl v17.16b, v31.16b, #0x4\n"
        "shl v16.16b, v30.16b, #0x4\n"
        "add x22, x22, #0x80\n"
        "add x20, x20, #0x80\n"
        "and v31.16b, v31.16b, v11.16b\n"
        "and v30.16b, v30.16b, v11.16b\n"
        ".inst 0x4e91a7aa  // smmla v10.4s, v29.16b, v17.16b\n"
        ".inst 0x4e90a7a9  // smmla v9.4s, v29.16b, v16.16b\n"
        ".inst 0x4e91a788  // smmla v8.4s, v28.16b, v17.16b\n"
        ".inst 0x4e90a787  // smmla v7.4s, v28.16b, v16.16b\n"
        ".inst 0x4e91a766  // smmla v6.4s, v27.16b, v17.16b\n"
        ".inst 0x4e90a765  // smmla v5.4s, v27.16b, v16.16b\n"
        ".inst 0x4e91a744  // smmla v4.4s, v26.16b, v17.16b\n"
        ".inst 0x4e90a743  // smmla v3.4s, v26.16b, v16.16b\n"
        ".inst 0x4e82a72a  // smmla v10.4s, v25.16b, v2.16b\n"
        ".inst 0x4e81a729  // smmla v9.4s, v25.16b, v1.16b\n"
        ".inst 0x4e82a708  // smmla v8.4s, v24.16b, v2.16b\n"
        ".inst 0x4e81a707  // smmla v7.4s, v24.16b, v1.16b\n"
        ".inst 0x4e82a6e6  // smmla v6.4s, v23.16b, v2.16b\n"
        ".inst 0x4e81a6e5  // smmla v5.4s, v23.16b, v1.16b\n"
        ".inst 0x4e82a6c4  // smmla v4.4s, v22.16b, v2.16b\n"
        ".inst 0x4e81a6c3  // smmla v3.4s, v22.16b, v1.16b\n"
        ".inst 0x4e9fa6aa  // smmla v10.4s, v21.16b, v31.16b\n"
        ".inst 0x4e9ea6a9  // smmla v9.4s, v21.16b, v30.16b\n"
        ".inst 0x4e9fa688  // smmla v8.4s, v20.16b, v31.16b\n"
        ".inst 0x4e9ea687  // smmla v7.4s, v20.16b, v30.16b\n"
        ".inst 0x4e9fa666  // smmla v6.4s, v19.16b, v31.16b\n"
        ".inst 0x4e9ea665  // smmla v5.4s, v19.16b, v30.16b\n"
        ".inst 0x4e9fa644  // smmla v4.4s, v18.16b, v31.16b\n"
        ".inst 0x4e9ea643  // smmla v3.4s, v18.16b, v30.16b\n"
        "bgt 3b\n"
        "ldr q20, [x10, #0x0]\n"
        "ldr q19, [x22, #0x0]\n"
        "uzp1 v2.2d, v10.2d, v9.2d\n"
        "uzp2 v1.2d, v10.2d, v9.2d\n"
        "ldr q18, [x20, #0x0]\n"
        "ldr q0, [x10, #0x10]\n"
        "uzp1 v31.2d, v8.2d, v7.2d\n"
        "uzp2 v30.2d, v8.2d, v7.2d\n"
        "ldr q17, [x22, #0x10]\n"
        "ldr q16, [x20, #0x10]\n"
        "uzp1 v29.2d, v6.2d, v5.2d\n"
        "uzp2 v28.2d, v6.2d, v5.2d\n"
        "ld1r { v27.4s }, [%x[clamp_vals]]\n"
        "uzp1 v26.2d, v4.2d, v3.2d\n"
        "uzp2 v25.2d, v4.2d, v3.2d\n"
        "mla v2.4s, v20.4s, v19.s[0]\n"
        "mla v1.4s, v20.4s, v19.s[1]\n"
        "mla v31.4s, v20.4s, v19.s[2]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x9, #0x4\n"
        "ld1r { v24.4s }, [x20]\n"
        "mla v30.4s, v20.4s, v19.s[3]\n"
        "mla v29.4s, v20.4s, v18.s[0]\n"
        "fmul v23.4s, v0.4s, v17.s[0]\n"
        "mla v28.4s, v20.4s, v18.s[1]\n"
        "mla v26.4s, v20.4s, v18.s[2]\n"
        "fmul v22.4s, v0.4s, v17.s[1]\n"
        "add x10, x10, #0x20\n"
        "mla v25.4s, v20.4s, v18.s[3]\n"
        "scvtf v2.4s, v2.4s\n"
        "scvtf v1.4s, v1.4s\n"
        "scvtf v31.4s, v31.4s\n"
        "fmul v21.4s, v0.4s, v17.s[2]\n"
        "scvtf v30.4s, v30.4s\n"
        "fmul v20.4s, v0.4s, v17.s[3]\n"
        "scvtf v29.4s, v29.4s\n"
        "fmul v19.4s, v0.4s, v16.s[0]\n"
        "scvtf v28.4s, v28.4s\n"
        "fmul v18.4s, v0.4s, v16.s[1]\n"
        "scvtf v26.4s, v26.4s\n"
        "fmul v17.4s, v0.4s, v16.s[2]\n"
        "scvtf v25.4s, v25.4s\n"
        "fmul v16.4s, v0.4s, v16.s[3]\n"
        "fmul v2.4s, v2.4s, v23.4s\n"
        "fmul v1.4s, v1.4s, v22.4s\n"
        "fmul v31.4s, v31.4s, v21.4s\n"
        "fmul v30.4s, v30.4s, v20.4s\n"
        "fmul v29.4s, v29.4s, v19.4s\n"
        "fmul v28.4s, v28.4s, v18.4s\n"
        "fmul v26.4s, v26.4s, v17.4s\n"
        "fmul v25.4s, v25.4s, v16.4s\n"
        "fmax v2.4s, v2.4s, v27.4s\n"
        "fmax v1.4s, v1.4s, v27.4s\n"
        "fmax v31.4s, v31.4s, v27.4s\n"
        "fmax v30.4s, v30.4s, v27.4s\n"
        "fmax v29.4s, v29.4s, v27.4s\n"
        "fmax v28.4s, v28.4s, v27.4s\n"
        "fmax v26.4s, v26.4s, v27.4s\n"
        "fmax v25.4s, v25.4s, v27.4s\n"
        "fmin v2.4s, v2.4s, v24.4s\n"
        "fmin v1.4s, v1.4s, v24.4s\n"
        "fmin v31.4s, v31.4s, v24.4s\n"
        "fmin v30.4s, v30.4s, v24.4s\n"
        "fmin v29.4s, v29.4s, v24.4s\n"
        "fmin v28.4s, v28.4s, v24.4s\n"
        "fmin v26.4s, v26.4s, v24.4s\n"
        "fmin v25.4s, v25.4s, v24.4s\n"
        "bge 6f\n"
        "mov x27, %x[dst]\n"
        "add x26, x27, %x[dst_stride_row], LSL #2\n"
        "add x25, x26, %x[dst_stride_row], LSL #1\n"
        "add x24, x26, %x[dst_stride_row]\n"
        "add x23, x25, %x[dst_stride_row]\n"
        "add x22, x27, %x[dst_stride_row], LSL #1\n"
        "add x21, x27, %x[dst_stride_row]\n"
        "add x20, x22, %x[dst_stride_row]\n"
        "tbz x9, #1, 4f\n"
        "str d25, [x23], #0x8\n"
        "str d26, [x25], #0x8\n"
        "str d28, [x24], #0x8\n"
        "str d29, [x26], #0x8\n"
        "str d30, [x20], #0x8\n"
        "str d31, [x22], #0x8\n"
        "str d1, [x21], #0x8\n"
        "str d2, [x27], #0x8\n"
        "tbz x9, #0, 5f\n"
        "st1 { v25.s }[2], [x23]\n"
        "st1 { v26.s }[2], [x25]\n"
        "st1 { v28.s }[2], [x24]\n"
        "st1 { v29.s }[2], [x26]\n"
        "st1 { v30.s }[2], [x20]\n"
        "st1 { v31.s }[2], [x22]\n"
        "st1 { v1.s }[2], [x21]\n"
        "st1 { v2.s }[2], [x27]\n"
        "b 5f\n"
        "4:"  // Output block 0: partial_1_0
        "str s25, [x23, #0x0]\n"
        "str s26, [x25, #0x0]\n"
        "str s28, [x24, #0x0]\n"
        "str s29, [x26, #0x0]\n"
        "str s30, [x20, #0x0]\n"
        "str s31, [x22, #0x0]\n"
        "str s1, [x21, #0x0]\n"
        "str s2, [x27, #0x0]\n"
        "5:"  // Output block 0: Done
        "b 7f\n"
        "6:"  // Full output
        "mov x20, %x[dst]\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q1, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q31, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q29, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q28, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q26, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q25, [x20, #0x0]\n"
        "7:"  // Output stage exit
        "subs x9, x9, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 2b\n"
        "mov x20, #0x2\n"
        "sub x12, x12, #0x8\n"
        "cmp x12, #0x8\n"
        "mov %x[dst], x28\n"
        "madd %x[lhs_packed], x20, x11, %x[lhs_packed]\n"
        "bge 1b\n"
        "8:"  // Row loop skip
        "cbz x12, 16f\n"
        "9:"  // Row tail: Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "10:"  // Row tail: Column loop
        "movi v10.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "mov x22, %x[lhs_packed]\n"
        "mov x20, %x[num_blocks]\n"
        "movi v8.4s, #0x0\n"
        "movi v7.4s, #0x0\n"
        "11:"  // Row tail: Block loop
        "ldr q31, [x26, #0x0]\n"
        "ldr q30, [x26, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q29, [x22, #0x0]\n"
        "ldr q28, [x22, #0x10]\n"
        "ldr q27, [x26, #0x20]\n"
        "ldr q26, [x26, #0x30]\n"
        "add x26, x26, #0x40\n"
        "ldr q25, [x22, #0x20]\n"
        "ldr q24, [x22, #0x30]\n"
        "shl v23.16b, v31.16b, #0x4\n"
        "shl v22.16b, v30.16b, #0x4\n"
        "ldr q21, [x22, #0x40]\n"
        "ldr q20, [x22, #0x50]\n"
        "and v31.16b, v31.16b, v11.16b\n"
        "and v30.16b, v30.16b, v11.16b\n"
        "ldr q19, [x22, #0x60]\n"
        "ldr q18, [x22, #0x70]\n"
        "shl v17.16b, v27.16b, #0x4\n"
        "shl v16.16b, v26.16b, #0x4\n"
        ".inst 0x4e97a7aa  // smmla v10.4s, v29.16b, v23.16b\n"
        ".inst 0x4e96a7a9  // smmla v9.4s, v29.16b, v22.16b\n"
        "and v27.16b, v27.16b, v11.16b\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4e97a788  // smmla v8.4s, v28.16b, v23.16b\n"
        ".inst 0x4e96a787  // smmla v7.4s, v28.16b, v22.16b\n"
        "and v26.16b, v26.16b, v11.16b\n"
        ".inst 0x4e91a72a  // smmla v10.4s, v25.16b, v17.16b\n"
        ".inst 0x4e90a729  // smmla v9.4s, v25.16b, v16.16b\n"
        ".inst 0x4e91a708  // smmla v8.4s, v24.16b, v17.16b\n"
        ".inst 0x4e90a707  // smmla v7.4s, v24.16b, v16.16b\n"
        ".inst 0x4e9fa6aa  // smmla v10.4s, v21.16b, v31.16b\n"
        ".inst 0x4e9ea6a9  // smmla v9.4s, v21.16b, v30.16b\n"
        ".inst 0x4e9fa688  // smmla v8.4s, v20.16b, v31.16b\n"
        ".inst 0x4e9ea687  // smmla v7.4s, v20.16b, v30.16b\n"
        ".inst 0x4e9ba66a  // smmla v10.4s, v19.16b, v27.16b\n"
        ".inst 0x4e9aa669  // smmla v9.4s, v19.16b, v26.16b\n"
        ".inst 0x4e9ba648  // smmla v8.4s, v18.16b, v27.16b\n"
        ".inst 0x4e9aa647  // smmla v7.4s, v18.16b, v26.16b\n"
        "bgt 11b\n"
        "ldr q18, [x26, #0x0]\n"
        "ldr q17, [x22, #0x0]\n"
        "uzp1 v26.2d, v10.2d, v9.2d\n"
        "uzp2 v25.2d, v10.2d, v9.2d\n"
        "ldr q24, [x26, #0x10]\n"
        "ldr q16, [x22, #0x10]\n"
        "uzp1 v23.2d, v8.2d, v7.2d\n"
        "uzp2 v22.2d, v8.2d, v7.2d\n"
        "ld1r { v21.4s }, [%x[clamp_vals]]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x25, #0x4\n"
        "ld1r { v20.4s }, [x20]\n"
        "mla v26.4s, v18.4s, v17.s[0]\n"
        "mla v25.4s, v18.4s, v17.s[1]\n"
        "add x26, x26, #0x20\n"
        "mla v23.4s, v18.4s, v17.s[2]\n"
        "mla v22.4s, v18.4s, v17.s[3]\n"
        "fmul v19.4s, v24.4s, v16.s[0]\n"
        "fmul v18.4s, v24.4s, v16.s[1]\n"
        "fmul v17.4s, v24.4s, v16.s[2]\n"
        "fmul v16.4s, v24.4s, v16.s[3]\n"
        "scvtf v26.4s, v26.4s\n"
        "scvtf v25.4s, v25.4s\n"
        "scvtf v23.4s, v23.4s\n"
        "scvtf v22.4s, v22.4s\n"
        "fmul v26.4s, v26.4s, v19.4s\n"
        "fmul v25.4s, v25.4s, v18.4s\n"
        "fmul v23.4s, v23.4s, v17.4s\n"
        "fmul v22.4s, v22.4s, v16.4s\n"
        "fmax v26.4s, v26.4s, v21.4s\n"
        "fmax v25.4s, v25.4s, v21.4s\n"
        "fmax v23.4s, v23.4s, v21.4s\n"
        "fmax v22.4s, v22.4s, v21.4s\n"
        "fmin v26.4s, v26.4s, v20.4s\n"
        "fmin v25.4s, v25.4s, v20.4s\n"
        "fmin v23.4s, v23.4s, v20.4s\n"
        "fmin v22.4s, v22.4s, v20.4s\n"
        "bge 14f\n"
        "mov x23, %x[dst]\n"
        "cmp x12, #0x1\n"
        "add x22, x23, %x[dst_stride_row]\n"
        "csel x22, x22, x23, GE\n"
        "cmp x12, #0x2\n"
        "add x21, x23, %x[dst_stride_row], LSL #1\n"
        "csel x21, x21, x22, GE\n"
        "cmp x12, #0x3\n"
        "add x20, x21, %x[dst_stride_row]\n"
        "csel x20, x20, x21, GE\n"
        "tbz x25, #1, 12f\n"
        "str d22, [x20], #0x8\n"
        "str d23, [x21], #0x8\n"
        "str d25, [x22], #0x8\n"
        "str d26, [x23], #0x8\n"
        "tbz x25, #0, 13f\n"
        "st1 { v22.s }[2], [x20]\n"
        "st1 { v23.s }[2], [x21]\n"
        "st1 { v25.s }[2], [x22]\n"
        "st1 { v26.s }[2], [x23]\n"
        "b 13f\n"
        "12:"  // Row tail: Output block 0: partial_1_0
        "str s22, [x20, #0x0]\n"
        "str s23, [x21, #0x0]\n"
        "str s25, [x22, #0x0]\n"
        "str s26, [x23, #0x0]\n"
        "13:"  // Row tail: Output block 0: Done
        "b 15f\n"
        "14:"  // Row tail: Full output
        "mov x20, %x[dst]\n"
        "cmp x12, #0x1\n"
        "str q26, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 15f\n"
        "cmp x12, #0x2\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 15f\n"
        "cmp x12, #0x3\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 15f\n"
        "str q22, [x20, #0x0]\n"
        "15:"  // Row tail: Output stage exit
        "subs x25, x25, #0x4\n"
        "add %x[dst], %x[dst], #0x10\n"
        "bgt 10b\n"
        "subs x12, x12, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x11\n"
        "mov %x[dst], x24\n"
        "bgt 9b\n"
        "16:"  // Row tail: Row loop skip
        : [lhs_packed] "+&r"(lhs_packed), [dst] "+&r"(dst)
        : [rhs_packed] "r"(rhs_packed), [clamp_vals] "r"(clamp_vals), [m] "r"(m), [num_blocks] "r"(num_blocks),
          [dst_stride_row] "r"(dst_stride_row), [n] "r"(n)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x11",
          "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}
#endif  // Architectural feature check
