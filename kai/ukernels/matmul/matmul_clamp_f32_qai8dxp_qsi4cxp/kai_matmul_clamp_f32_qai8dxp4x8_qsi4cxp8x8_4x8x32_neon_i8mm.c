//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error "I8mm extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.h"

#include <arm_neon.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 4;
static const size_t kai_n_step = 8;
static const size_t kai_mr = 4;
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

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed, const void* restrict rhs_packed, float* restrict dst,
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t k_internal = kai_k_roundedup(k);

    size_t num_blocks = k_internal / 32;

    float clamp_vals[2] = {scalar_min, scalar_max};

    __asm__ __volatile__(
        "mov x28, #0x80\n"
        "mov x20, #0x20\n"
        "movi v12.16b, #0xf0\n"
        "mov x27, %x[m]\n"
        "madd x28, %x[num_blocks], x28, x20\n"
        "cbz x27, 10f\n"
        "1:"  // Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "2:"  // Column loop
        "movi v11.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "mov x21, %x[lhs_packed]\n"
        "mov x20, %x[num_blocks]\n"
        "movi v9.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "movi v7.4s, #0x0\n"
        "movi v6.4s, #0x0\n"
        "movi v5.4s, #0x0\n"
        "movi v4.4s, #0x0\n"
        "3:"  // Block loop
        "ldr q3, [x26, #0x0]\n"
        "ldr q2, [x26, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q1, [x26, #0x20]\n"
        "ldr q0, [x26, #0x30]\n"
        "ldr q31, [x21, #0x0]\n"
        "ldr q30, [x21, #0x10]\n"
        "ldr q29, [x26, #0x40]\n"
        "ldr q28, [x26, #0x50]\n"
        "shl v19.16b, v3.16b, #0x4\n"
        "shl v18.16b, v2.16b, #0x4\n"
        "ldr q27, [x26, #0x60]\n"
        "ldr q26, [x26, #0x70]\n"
        "shl v17.16b, v1.16b, #0x4\n"
        "shl v16.16b, v0.16b, #0x4\n"
        "ldr q25, [x21, #0x20]\n"
        "ldr q24, [x21, #0x30]\n"
        "and v3.16b, v3.16b, v12.16b\n"
        "and v2.16b, v2.16b, v12.16b\n"
        "ldr q23, [x21, #0x40]\n"
        "ldr q22, [x21, #0x50]\n"
        ".inst 0x4e93a7eb  // smmla v11.4s, v31.16b, v19.16b\n"
        ".inst 0x4e92a7e9  // smmla v9.4s, v31.16b, v18.16b\n"
        "ldr q21, [x21, #0x60]\n"
        "ldr q20, [x21, #0x70]\n"
        ".inst 0x4e91a7ea  // smmla v10.4s, v31.16b, v17.16b\n"
        ".inst 0x4e90a7e8  // smmla v8.4s, v31.16b, v16.16b\n"
        ".inst 0x4e93a7c7  // smmla v7.4s, v30.16b, v19.16b\n"
        ".inst 0x4e92a7c5  // smmla v5.4s, v30.16b, v18.16b\n"
        "shl v19.16b, v29.16b, #0x4\n"
        "add x21, x21, #0x80\n"
        ".inst 0x4e91a7c6  // smmla v6.4s, v30.16b, v17.16b\n"
        ".inst 0x4e90a7c4  // smmla v4.4s, v30.16b, v16.16b\n"
        "shl v18.16b, v28.16b, #0x4\n"
        "add x26, x26, #0x80\n"
        "shl v17.16b, v27.16b, #0x4\n"
        "shl v16.16b, v26.16b, #0x4\n"
        ".inst 0x4e93a72b  // smmla v11.4s, v25.16b, v19.16b\n"
        "and v1.16b, v1.16b, v12.16b\n"
        "and v0.16b, v0.16b, v12.16b\n"
        ".inst 0x4e92a729  // smmla v9.4s, v25.16b, v18.16b\n"
        ".inst 0x4e93a707  // smmla v7.4s, v24.16b, v19.16b\n"
        ".inst 0x4e92a705  // smmla v5.4s, v24.16b, v18.16b\n"
        "and v29.16b, v29.16b, v12.16b\n"
        ".inst 0x4e91a72a  // smmla v10.4s, v25.16b, v17.16b\n"
        ".inst 0x4e90a728  // smmla v8.4s, v25.16b, v16.16b\n"
        "and v28.16b, v28.16b, v12.16b\n"
        ".inst 0x4e91a706  // smmla v6.4s, v24.16b, v17.16b\n"
        ".inst 0x4e90a704  // smmla v4.4s, v24.16b, v16.16b\n"
        "and v27.16b, v27.16b, v12.16b\n"
        ".inst 0x4e83a6eb  // smmla v11.4s, v23.16b, v3.16b\n"
        ".inst 0x4e82a6e9  // smmla v9.4s, v23.16b, v2.16b\n"
        "and v26.16b, v26.16b, v12.16b\n"
        ".inst 0x4e83a6c7  // smmla v7.4s, v22.16b, v3.16b\n"
        ".inst 0x4e82a6c5  // smmla v5.4s, v22.16b, v2.16b\n"
        ".inst 0x4e81a6ea  // smmla v10.4s, v23.16b, v1.16b\n"
        ".inst 0x4e80a6e8  // smmla v8.4s, v23.16b, v0.16b\n"
        ".inst 0x4e81a6c6  // smmla v6.4s, v22.16b, v1.16b\n"
        ".inst 0x4e80a6c4  // smmla v4.4s, v22.16b, v0.16b\n"
        ".inst 0x4e9da6ab  // smmla v11.4s, v21.16b, v29.16b\n"
        ".inst 0x4e9ca6a9  // smmla v9.4s, v21.16b, v28.16b\n"
        ".inst 0x4e9da687  // smmla v7.4s, v20.16b, v29.16b\n"
        ".inst 0x4e9ca685  // smmla v5.4s, v20.16b, v28.16b\n"
        ".inst 0x4e9ba6aa  // smmla v10.4s, v21.16b, v27.16b\n"
        ".inst 0x4e9aa6a8  // smmla v8.4s, v21.16b, v26.16b\n"
        ".inst 0x4e9ba686  // smmla v6.4s, v20.16b, v27.16b\n"
        ".inst 0x4e9aa684  // smmla v4.4s, v20.16b, v26.16b\n"
        "bgt 3b\n"
        "ldr q20, [x26, #0x0]\n"
        "ldr q19, [x26, #0x10]\n"
        "uzp1 v2.2d, v11.2d, v9.2d\n"
        "uzp1 v1.2d, v10.2d, v8.2d\n"
        "ldr q18, [x21, #0x0]\n"
        "ldr q17, [x26, #0x20]\n"
        "uzp2 v0.2d, v11.2d, v9.2d\n"
        "uzp2 v31.2d, v10.2d, v8.2d\n"
        "ldr q30, [x26, #0x30]\n"
        "ldr q16, [x21, #0x10]\n"
        "uzp1 v29.2d, v7.2d, v5.2d\n"
        "uzp1 v28.2d, v6.2d, v4.2d\n"
        "ld1r { v27.4s }, [%x[clamp_vals]]\n"
        "uzp2 v26.2d, v7.2d, v5.2d\n"
        "uzp2 v25.2d, v6.2d, v4.2d\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "ld1r { v24.4s }, [x20]\n"
        "mla v2.4s, v20.4s, v18.s[0]\n"
        "mla v1.4s, v19.4s, v18.s[0]\n"
        "cmp x25, #0x8\n"
        "mla v0.4s, v20.4s, v18.s[1]\n"
        "mla v31.4s, v19.4s, v18.s[1]\n"
        "fmul v23.4s, v17.4s, v16.s[0]\n"
        "add x26, x26, #0x40\n"
        "mla v29.4s, v20.4s, v18.s[2]\n"
        "mla v28.4s, v19.4s, v18.s[2]\n"
        "fmul v22.4s, v30.4s, v16.s[0]\n"
        "mla v26.4s, v20.4s, v18.s[3]\n"
        "mla v25.4s, v19.4s, v18.s[3]\n"
        "fmul v21.4s, v17.4s, v16.s[1]\n"
        "scvtf v2.4s, v2.4s\n"
        "scvtf v1.4s, v1.4s\n"
        "scvtf v0.4s, v0.4s\n"
        "scvtf v31.4s, v31.4s\n"
        "fmul v20.4s, v30.4s, v16.s[1]\n"
        "scvtf v29.4s, v29.4s\n"
        "fmul v19.4s, v17.4s, v16.s[2]\n"
        "scvtf v28.4s, v28.4s\n"
        "fmul v18.4s, v30.4s, v16.s[2]\n"
        "scvtf v26.4s, v26.4s\n"
        "fmul v17.4s, v17.4s, v16.s[3]\n"
        "scvtf v25.4s, v25.4s\n"
        "fmul v16.4s, v30.4s, v16.s[3]\n"
        "fmul v2.4s, v2.4s, v23.4s\n"
        "fmul v1.4s, v1.4s, v22.4s\n"
        "fmul v0.4s, v0.4s, v21.4s\n"
        "fmul v31.4s, v31.4s, v20.4s\n"
        "fmul v29.4s, v29.4s, v19.4s\n"
        "fmul v28.4s, v28.4s, v18.4s\n"
        "fmul v26.4s, v26.4s, v17.4s\n"
        "fmul v25.4s, v25.4s, v16.4s\n"
        "fmax v2.4s, v2.4s, v27.4s\n"
        "fmax v1.4s, v1.4s, v27.4s\n"
        "fmax v0.4s, v0.4s, v27.4s\n"
        "fmax v31.4s, v31.4s, v27.4s\n"
        "fmax v29.4s, v29.4s, v27.4s\n"
        "fmax v28.4s, v28.4s, v27.4s\n"
        "fmax v26.4s, v26.4s, v27.4s\n"
        "fmax v25.4s, v25.4s, v27.4s\n"
        "fmin v2.4s, v2.4s, v24.4s\n"
        "fmin v1.4s, v1.4s, v24.4s\n"
        "fmin v0.4s, v0.4s, v24.4s\n"
        "fmin v31.4s, v31.4s, v24.4s\n"
        "fmin v29.4s, v29.4s, v24.4s\n"
        "fmin v28.4s, v28.4s, v24.4s\n"
        "fmin v26.4s, v26.4s, v24.4s\n"
        "fmin v25.4s, v25.4s, v24.4s\n"
        "bge 8f\n"
        "mov x23, %x[dst]\n"
        "cmp x27, #0x1\n"
        "add x22, x23, %x[dst_stride_row]\n"
        "csel x22, x22, x23, GE\n"
        "cmp x27, #0x2\n"
        "add x21, x23, %x[dst_stride_row], LSL #1\n"
        "csel x21, x21, x22, GE\n"
        "cmp x27, #0x3\n"
        "add x20, x21, %x[dst_stride_row]\n"
        "csel x20, x20, x21, GE\n"
        "tbz x25, #2, 5f\n"
        "st1 { v26.4s }, [x20], #0x10\n"
        "st1 { v29.4s }, [x21], #0x10\n"
        "st1 { v0.4s }, [x22], #0x10\n"
        "st1 { v2.4s }, [x23], #0x10\n"
        "tbz x25, #1, 4f\n"
        "str d25, [x20], #0x8\n"
        "str d28, [x21], #0x8\n"
        "str d31, [x22], #0x8\n"
        "str d1, [x23], #0x8\n"
        "tbz x25, #0, 7f\n"
        "st1 { v25.s }[2], [x20]\n"
        "st1 { v28.s }[2], [x21]\n"
        "st1 { v31.s }[2], [x22]\n"
        "st1 { v1.s }[2], [x23]\n"
        "b 7f\n"
        "4:"  // Output block 0: partial_1_4
        "tbz x25, #0, 7f\n"
        "str s25, [x20, #0x0]\n"
        "str s28, [x21, #0x0]\n"
        "str s31, [x22, #0x0]\n"
        "str s1, [x23, #0x0]\n"
        "b 7f\n"
        "5:"  // Output block 0: partial_2_0
        "tbz x25, #1, 6f\n"
        "str d26, [x20], #0x8\n"
        "str d29, [x21], #0x8\n"
        "str d0, [x22], #0x8\n"
        "str d2, [x23], #0x8\n"
        "tbz x25, #0, 7f\n"
        "st1 { v26.s }[2], [x20]\n"
        "st1 { v29.s }[2], [x21]\n"
        "st1 { v0.s }[2], [x22]\n"
        "st1 { v2.s }[2], [x23]\n"
        "b 7f\n"
        "6:"  // Output block 0: partial_1_0
        "str s26, [x20, #0x0]\n"
        "str s29, [x21, #0x0]\n"
        "str s0, [x22, #0x0]\n"
        "str s2, [x23, #0x0]\n"
        "7:"  // Output block 0: Done
        "b 9f\n"
        "8:"  // Full output
        "mov x20, %x[dst]\n"
        "cmp x27, #0x1\n"
        "str q2, [x20, #0x0]\n"
        "str q1, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 9f\n"
        "cmp x27, #0x2\n"
        "str q0, [x20, #0x0]\n"
        "str q31, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 9f\n"
        "cmp x27, #0x3\n"
        "str q29, [x20, #0x0]\n"
        "str q28, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 9f\n"
        "str q26, [x20, #0x0]\n"
        "str q25, [x20, #0x10]\n"
        "9:"  // Output stage exit
        "subs x25, x25, #0x8\n"
        "add %x[dst], %x[dst], #0x20\n"
        "bgt 2b\n"
        "subs x27, x27, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x28\n"
        "mov %x[dst], x24\n"
        "bgt 1b\n"
        "10:"  // Row loop skip
        : [lhs_packed] "+&r"(lhs_packed), [dst] "+&r"(dst)
        : [rhs_packed] "r"(rhs_packed), [clamp_vals] "r"(clamp_vals), [m] "r"(m), [num_blocks] "r"(num_blocks),
          [dst_stride_row] "r"(dst_stride_row), [n] "r"(n)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v16", "v17",
          "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20",
          "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}
#endif  // Architectural feature check
