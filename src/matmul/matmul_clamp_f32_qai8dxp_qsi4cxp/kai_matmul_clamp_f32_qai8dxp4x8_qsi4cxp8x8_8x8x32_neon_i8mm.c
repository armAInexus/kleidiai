//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"

#include <arm_neon.h>
#include <stdint.h>

#include "kai_common.h"

static const size_t kai_m_step = 8;
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

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, float scalar_min, float scalar_max) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
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
        "movi v3.16b, #0xf0\n"
        "mov x20, #0x20\n"
        "cmp x12, #0x8\n"
        "madd x11, %x[num_blocks], x11, x20\n"
        "blt 10f\n"
        "1:"  // Row loop
        "mov x10, %x[rhs_packed]\n"
        "mov x9, %x[n]\n"
        "add x28, %x[dst], %x[dst_stride_row], LSL #3\n"
        "2:"  // Column loop
        "mov x22, %x[lhs_packed]\n"
        "movi v13.4s, #0x0\n"
        "movi v14.4s, #0x0\n"
        "mov x21, %x[num_blocks]\n"
        "movi v25.4s, #0x0\n"
        "movi v16.4s, #0x0\n"
        "movi v26.4s, #0x0\n"
        "movi v30.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "movi v19.4s, #0x0\n"
        "add x20, x22, x11\n"
        "movi v24.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "movi v28.4s, #0x0\n"
        "movi v15.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        "movi v11.4s, #0x0\n"
        "movi v31.4s, #0x0\n"
        "movi v7.4s, #0x0\n"
        "3:"  // Block loop
        "ldr q21, [x10, #0x0]\n"
        "ldr q20, [x10, #0x10]\n"
        "subs x21, x21, #0x1\n"
        "ldr q2, [x10, #0x20]\n"
        "ldr q23, [x10, #0x30]\n"
        "ldr q8, [x22, #0x0]\n"
        "ldr q1, [x22, #0x10]\n"
        "ldr q12, [x20, #0x0]\n"
        "ldr q6, [x20, #0x10]\n"
        "shl v17.16b, v21.16b, #0x4\n"
        "shl v22.16b, v20.16b, #0x4\n"
        "ldr q9, [x10, #0x40]\n"
        "ldr q18, [x10, #0x50]\n"
        "shl v4.16b, v2.16b, #0x4\n"
        "shl v5.16b, v23.16b, #0x4\n"
        "ldr q27, [x10, #0x60]\n"
        "and v21.16b, v21.16b, v3.16b\n"
        "and v20.16b, v20.16b, v3.16b\n"
        ".inst 0x4e91a50d  // smmla v13.4s, v8.16b, v17.16b\n"
        ".inst 0x4e96a519  // smmla v25.4s, v8.16b, v22.16b\n"
        ".inst 0x4e91a43a  // smmla v26.4s, v1.16b, v17.16b\n"
        "and v2.16b, v2.16b, v3.16b\n"
        ".inst 0x4e84a50e  // smmla v14.4s, v8.16b, v4.16b\n"
        ".inst 0x4e85a510  // smmla v16.4s, v8.16b, v5.16b\n"
        "ldr q8, [x10, #0x70]\n"
        "and v23.16b, v23.16b, v3.16b\n"
        ".inst 0x4e96a42a  // smmla v10.4s, v1.16b, v22.16b\n"
        ".inst 0x4e84a43e  // smmla v30.4s, v1.16b, v4.16b\n"
        "add x10, x10, #0x80\n"
        ".inst 0x4e85a433  // smmla v19.4s, v1.16b, v5.16b\n"
        "ldr q1, [x22, #0x20]\n"
        ".inst 0x4e91a598  // smmla v24.4s, v12.16b, v17.16b\n"
        ".inst 0x4e96a59c  // smmla v28.4s, v12.16b, v22.16b\n"
        ".inst 0x4e84a580  // smmla v0.4s, v12.16b, v4.16b\n"
        ".inst 0x4e85a58f  // smmla v15.4s, v12.16b, v5.16b\n"
        "ldr q12, [x22, #0x30]\n"
        ".inst 0x4e91a4dd  // smmla v29.4s, v6.16b, v17.16b\n"
        "ldr q17, [x20, #0x20]\n"
        ".inst 0x4e96a4df  // smmla v31.4s, v6.16b, v22.16b\n"
        "ldr q22, [x20, #0x30]\n"
        ".inst 0x4e84a4cb  // smmla v11.4s, v6.16b, v4.16b\n"
        "ldr q4, [x22, #0x40]\n"
        ".inst 0x4e85a4c7  // smmla v7.4s, v6.16b, v5.16b\n"
        "ldr q5, [x22, #0x50]\n"
        "shl v6.16b, v9.16b, #0x4\n"
        "and v9.16b, v9.16b, v3.16b\n"
        ".inst 0x4e86a42d  // smmla v13.4s, v1.16b, v6.16b\n"
        ".inst 0x4e86a59a  // smmla v26.4s, v12.16b, v6.16b\n"
        ".inst 0x4e86a638  // smmla v24.4s, v17.16b, v6.16b\n"
        ".inst 0x4e86a6dd  // smmla v29.4s, v22.16b, v6.16b\n"
        "shl v6.16b, v18.16b, #0x4\n"
        "and v18.16b, v18.16b, v3.16b\n"
        ".inst 0x4e86a439  // smmla v25.4s, v1.16b, v6.16b\n"
        ".inst 0x4e86a58a  // smmla v10.4s, v12.16b, v6.16b\n"
        ".inst 0x4e86a63c  // smmla v28.4s, v17.16b, v6.16b\n"
        ".inst 0x4e86a6df  // smmla v31.4s, v22.16b, v6.16b\n"
        "shl v6.16b, v27.16b, #0x4\n"
        ".inst 0x4e95a48d  // smmla v13.4s, v4.16b, v21.16b\n"
        ".inst 0x4e95a4ba  // smmla v26.4s, v5.16b, v21.16b\n"
        "and v27.16b, v27.16b, v3.16b\n"
        ".inst 0x4e86a42e  // smmla v14.4s, v1.16b, v6.16b\n"
        ".inst 0x4e86a59e  // smmla v30.4s, v12.16b, v6.16b\n"
        ".inst 0x4e86a620  // smmla v0.4s, v17.16b, v6.16b\n"
        ".inst 0x4e86a6cb  // smmla v11.4s, v22.16b, v6.16b\n"
        "shl v6.16b, v8.16b, #0x4\n"
        ".inst 0x4e94a499  // smmla v25.4s, v4.16b, v20.16b\n"
        ".inst 0x4e94a4aa  // smmla v10.4s, v5.16b, v20.16b\n"
        "and v8.16b, v8.16b, v3.16b\n"
        ".inst 0x4e86a430  // smmla v16.4s, v1.16b, v6.16b\n"
        "ldr q1, [x20, #0x40]\n"
        ".inst 0x4e86a593  // smmla v19.4s, v12.16b, v6.16b\n"
        "ldr q12, [x20, #0x50]\n"
        ".inst 0x4e86a62f  // smmla v15.4s, v17.16b, v6.16b\n"
        "ldr q17, [x22, #0x60]\n"
        ".inst 0x4e86a6c7  // smmla v7.4s, v22.16b, v6.16b\n"
        "ldr q22, [x22, #0x70]\n"
        "ldr q6, [x20, #0x60]\n"
        ".inst 0x4e82a48e  // smmla v14.4s, v4.16b, v2.16b\n"
        ".inst 0x4e82a4be  // smmla v30.4s, v5.16b, v2.16b\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4e95a438  // smmla v24.4s, v1.16b, v21.16b\n"
        ".inst 0x4e94a43c  // smmla v28.4s, v1.16b, v20.16b\n"
        ".inst 0x4e97a490  // smmla v16.4s, v4.16b, v23.16b\n"
        "ldr q4, [x20, #0x70]\n"
        ".inst 0x4e97a4b3  // smmla v19.4s, v5.16b, v23.16b\n"
        "add x20, x20, #0x80\n"
        ".inst 0x4e82a420  // smmla v0.4s, v1.16b, v2.16b\n"
        ".inst 0x4e97a42f  // smmla v15.4s, v1.16b, v23.16b\n"
        ".inst 0x4e95a59d  // smmla v29.4s, v12.16b, v21.16b\n"
        ".inst 0x4e94a59f  // smmla v31.4s, v12.16b, v20.16b\n"
        ".inst 0x4e82a58b  // smmla v11.4s, v12.16b, v2.16b\n"
        ".inst 0x4e97a587  // smmla v7.4s, v12.16b, v23.16b\n"
        ".inst 0x4e89a62d  // smmla v13.4s, v17.16b, v9.16b\n"
        ".inst 0x4e92a639  // smmla v25.4s, v17.16b, v18.16b\n"
        ".inst 0x4e9ba62e  // smmla v14.4s, v17.16b, v27.16b\n"
        ".inst 0x4e88a630  // smmla v16.4s, v17.16b, v8.16b\n"
        ".inst 0x4e89a6da  // smmla v26.4s, v22.16b, v9.16b\n"
        ".inst 0x4e92a6ca  // smmla v10.4s, v22.16b, v18.16b\n"
        ".inst 0x4e9ba6de  // smmla v30.4s, v22.16b, v27.16b\n"
        ".inst 0x4e88a6d3  // smmla v19.4s, v22.16b, v8.16b\n"
        ".inst 0x4e89a4d8  // smmla v24.4s, v6.16b, v9.16b\n"
        ".inst 0x4e92a4dc  // smmla v28.4s, v6.16b, v18.16b\n"
        ".inst 0x4e9ba4c0  // smmla v0.4s, v6.16b, v27.16b\n"
        ".inst 0x4e88a4cf  // smmla v15.4s, v6.16b, v8.16b\n"
        ".inst 0x4e89a49d  // smmla v29.4s, v4.16b, v9.16b\n"
        ".inst 0x4e92a49f  // smmla v31.4s, v4.16b, v18.16b\n"
        ".inst 0x4e9ba48b  // smmla v11.4s, v4.16b, v27.16b\n"
        ".inst 0x4e88a487  // smmla v7.4s, v4.16b, v8.16b\n"
        "bgt 3b\n"
        "ldr q18, [x10, #0x0]\n"
        "ldr q2, [x10, #0x10]\n"
        "uzp1 v4.2d, v13.2d, v25.2d\n"
        "uzp1 v5.2d, v14.2d, v16.2d\n"
        "ldr q22, [x22, #0x0]\n"
        "ldr q27, [x20, #0x0]\n"
        "uzp2 v1.2d, v13.2d, v25.2d\n"
        "uzp2 v20.2d, v14.2d, v16.2d\n"
        "ldr q17, [x10, #0x20]\n"
        "ldr q6, [x10, #0x30]\n"
        "uzp1 v9.2d, v26.2d, v10.2d\n"
        "uzp1 v13.2d, v30.2d, v19.2d\n"
        "ldr q23, [x22, #0x10]\n"
        "ldr q12, [x20, #0x10]\n"
        "uzp2 v21.2d, v26.2d, v10.2d\n"
        "uzp2 v25.2d, v30.2d, v19.2d\n"
        "ld1r { v8.4s }, [%x[clamp_vals]]\n"
        "uzp1 v16.2d, v24.2d, v28.2d\n"
        "uzp1 v10.2d, v0.2d, v15.2d\n"
        "mla v4.4s, v18.4s, v22.s[0]\n"
        "uzp2 v30.2d, v24.2d, v28.2d\n"
        "uzp2 v28.2d, v0.2d, v15.2d\n"
        "mla v5.4s, v2.4s, v22.s[0]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "ld1r { v24.4s }, [x20]\n"
        "uzp1 v14.2d, v29.2d, v31.2d\n"
        "uzp1 v26.2d, v11.2d, v7.2d\n"
        "mla v1.4s, v18.4s, v22.s[1]\n"
        "uzp2 v0.2d, v29.2d, v31.2d\n"
        "uzp2 v11.2d, v11.2d, v7.2d\n"
        "mla v20.4s, v2.4s, v22.s[1]\n"
        "cmp x9, #0x8\n"
        "mla v9.4s, v18.4s, v22.s[2]\n"
        "mla v13.4s, v2.4s, v22.s[2]\n"
        "scvtf v4.4s, v4.4s\n"
        "add x10, x10, #0x40\n"
        "mla v21.4s, v18.4s, v22.s[3]\n"
        "mla v25.4s, v2.4s, v22.s[3]\n"
        "fmul v19.4s, v17.4s, v23.s[0]\n"
        "mla v16.4s, v18.4s, v27.s[0]\n"
        "mla v10.4s, v2.4s, v27.s[0]\n"
        "scvtf v5.4s, v5.4s\n"
        "mla v30.4s, v18.4s, v27.s[1]\n"
        "mla v28.4s, v2.4s, v27.s[1]\n"
        "fmul v15.4s, v6.4s, v23.s[0]\n"
        "mla v14.4s, v18.4s, v27.s[2]\n"
        "mla v26.4s, v2.4s, v27.s[2]\n"
        "scvtf v1.4s, v1.4s\n"
        "mla v0.4s, v18.4s, v27.s[3]\n"
        "mla v11.4s, v2.4s, v27.s[3]\n"
        "fmul v22.4s, v17.4s, v23.s[1]\n"
        "scvtf v20.4s, v20.4s\n"
        "fmul v29.4s, v6.4s, v23.s[1]\n"
        "scvtf v9.4s, v9.4s\n"
        "fmul v2.4s, v17.4s, v23.s[2]\n"
        "scvtf v13.4s, v13.4s\n"
        "fmul v18.4s, v6.4s, v23.s[2]\n"
        "scvtf v21.4s, v21.4s\n"
        "fmul v31.4s, v17.4s, v23.s[3]\n"
        "scvtf v25.4s, v25.4s\n"
        "fmul v7.4s, v6.4s, v23.s[3]\n"
        "scvtf v16.4s, v16.4s\n"
        "fmul v27.4s, v17.4s, v12.s[0]\n"
        "scvtf v10.4s, v10.4s\n"
        "fmul v23.4s, v6.4s, v12.s[0]\n"
        "scvtf v30.4s, v30.4s\n"
        "scvtf v28.4s, v28.4s\n"
        "scvtf v14.4s, v14.4s\n"
        "scvtf v26.4s, v26.4s\n"
        "scvtf v0.4s, v0.4s\n"
        "scvtf v11.4s, v11.4s\n"
        "fmul v4.4s, v4.4s, v19.4s\n"
        "fmul v19.4s, v17.4s, v12.s[1]\n"
        "fmul v5.4s, v5.4s, v15.4s\n"
        "fmul v15.4s, v6.4s, v12.s[1]\n"
        "fmul v1.4s, v1.4s, v22.4s\n"
        "fmul v22.4s, v17.4s, v12.s[2]\n"
        "fmul v17.4s, v17.4s, v12.s[3]\n"
        "fmul v20.4s, v20.4s, v29.4s\n"
        "fmul v29.4s, v6.4s, v12.s[2]\n"
        "fmul v12.4s, v6.4s, v12.s[3]\n"
        "fmul v9.4s, v9.4s, v2.4s\n"
        "fmul v13.4s, v13.4s, v18.4s\n"
        "fmul v21.4s, v21.4s, v31.4s\n"
        "fmul v25.4s, v25.4s, v7.4s\n"
        "fmul v16.4s, v16.4s, v27.4s\n"
        "fmul v10.4s, v10.4s, v23.4s\n"
        "fmul v30.4s, v30.4s, v19.4s\n"
        "fmul v28.4s, v28.4s, v15.4s\n"
        "fmul v14.4s, v14.4s, v22.4s\n"
        "fmul v26.4s, v26.4s, v29.4s\n"
        "fmul v0.4s, v0.4s, v17.4s\n"
        "fmul v11.4s, v11.4s, v12.4s\n"
        "fmax v4.4s, v4.4s, v8.4s\n"
        "fmax v5.4s, v5.4s, v8.4s\n"
        "fmax v1.4s, v1.4s, v8.4s\n"
        "fmax v20.4s, v20.4s, v8.4s\n"
        "fmax v9.4s, v9.4s, v8.4s\n"
        "fmax v13.4s, v13.4s, v8.4s\n"
        "fmax v21.4s, v21.4s, v8.4s\n"
        "fmax v25.4s, v25.4s, v8.4s\n"
        "fmax v16.4s, v16.4s, v8.4s\n"
        "fmax v10.4s, v10.4s, v8.4s\n"
        "fmax v30.4s, v30.4s, v8.4s\n"
        "fmax v28.4s, v28.4s, v8.4s\n"
        "fmax v14.4s, v14.4s, v8.4s\n"
        "fmax v26.4s, v26.4s, v8.4s\n"
        "fmax v0.4s, v0.4s, v8.4s\n"
        "fmax v11.4s, v11.4s, v8.4s\n"
        "fmin v4.4s, v4.4s, v24.4s\n"
        "fmin v5.4s, v5.4s, v24.4s\n"
        "fmin v1.4s, v1.4s, v24.4s\n"
        "fmin v20.4s, v20.4s, v24.4s\n"
        "fmin v9.4s, v9.4s, v24.4s\n"
        "fmin v13.4s, v13.4s, v24.4s\n"
        "fmin v21.4s, v21.4s, v24.4s\n"
        "fmin v25.4s, v25.4s, v24.4s\n"
        "fmin v16.4s, v16.4s, v24.4s\n"
        "fmin v10.4s, v10.4s, v24.4s\n"
        "fmin v30.4s, v30.4s, v24.4s\n"
        "fmin v28.4s, v28.4s, v24.4s\n"
        "fmin v14.4s, v14.4s, v24.4s\n"
        "fmin v26.4s, v26.4s, v24.4s\n"
        "fmin v0.4s, v0.4s, v24.4s\n"
        "fmin v11.4s, v11.4s, v24.4s\n"
        "bge 8f\n"
        "mov x27, %x[dst]\n"
        "add x26, x27, %x[dst_stride_row], LSL #2\n"
        "add x25, x26, %x[dst_stride_row], LSL #1\n"
        "add x24, x26, %x[dst_stride_row]\n"
        "add x23, x25, %x[dst_stride_row]\n"
        "add x22, x27, %x[dst_stride_row], LSL #1\n"
        "add x21, x27, %x[dst_stride_row]\n"
        "add x20, x22, %x[dst_stride_row]\n"
        "tbz x9, #2, 5f\n"
        "st1 { v0.4s }, [x23], #0x10\n"
        "st1 { v14.4s }, [x25], #0x10\n"
        "st1 { v30.4s }, [x24], #0x10\n"
        "st1 { v16.4s }, [x26], #0x10\n"
        "st1 { v21.4s }, [x20], #0x10\n"
        "st1 { v9.4s }, [x22], #0x10\n"
        "st1 { v1.4s }, [x21], #0x10\n"
        "st1 { v4.4s }, [x27], #0x10\n"
        "tbz x9, #1, 4f\n"
        "str d11, [x23], #0x8\n"
        "str d26, [x25], #0x8\n"
        "str d28, [x24], #0x8\n"
        "str d10, [x26], #0x8\n"
        "str d25, [x20], #0x8\n"
        "str d13, [x22], #0x8\n"
        "str d20, [x21], #0x8\n"
        "str d5, [x27], #0x8\n"
        "tbz x9, #0, 7f\n"
        "st1 { v11.s }[2], [x23]\n"
        "st1 { v26.s }[2], [x25]\n"
        "st1 { v28.s }[2], [x24]\n"
        "st1 { v10.s }[2], [x26]\n"
        "st1 { v25.s }[2], [x20]\n"
        "st1 { v13.s }[2], [x22]\n"
        "st1 { v20.s }[2], [x21]\n"
        "st1 { v5.s }[2], [x27]\n"
        "b 7f\n"
        "4:"  // Output block 0: partial_1_4
        "tbz x9, #0, 7f\n"
        "str s11, [x23, #0x0]\n"
        "str s26, [x25, #0x0]\n"
        "str s28, [x24, #0x0]\n"
        "str s10, [x26, #0x0]\n"
        "str s25, [x20, #0x0]\n"
        "str s13, [x22, #0x0]\n"
        "str s20, [x21, #0x0]\n"
        "str s5, [x27, #0x0]\n"
        "b 7f\n"
        "5:"  // Output block 0: partial_2_0
        "tbz x9, #1, 6f\n"
        "str d0, [x23], #0x8\n"
        "str d14, [x25], #0x8\n"
        "str d30, [x24], #0x8\n"
        "str d16, [x26], #0x8\n"
        "str d21, [x20], #0x8\n"
        "str d9, [x22], #0x8\n"
        "str d1, [x21], #0x8\n"
        "str d4, [x27], #0x8\n"
        "tbz x9, #0, 7f\n"
        "st1 { v0.s }[2], [x23]\n"
        "st1 { v14.s }[2], [x25]\n"
        "st1 { v30.s }[2], [x24]\n"
        "st1 { v16.s }[2], [x26]\n"
        "st1 { v21.s }[2], [x20]\n"
        "st1 { v9.s }[2], [x22]\n"
        "st1 { v1.s }[2], [x21]\n"
        "st1 { v4.s }[2], [x27]\n"
        "b 7f\n"
        "6:"  // Output block 0: partial_1_0
        "str s0, [x23, #0x0]\n"
        "str s14, [x25, #0x0]\n"
        "str s30, [x24, #0x0]\n"
        "str s16, [x26, #0x0]\n"
        "str s21, [x20, #0x0]\n"
        "str s9, [x22, #0x0]\n"
        "str s1, [x21, #0x0]\n"
        "str s4, [x27, #0x0]\n"
        "7:"  // Output block 0: Done
        "b 9f\n"
        "8:"  // Full output
        "mov x20, %x[dst]\n"
        "str q4, [x20, #0x0]\n"
        "str q5, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q1, [x20, #0x0]\n"
        "str q20, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q9, [x20, #0x0]\n"
        "str q13, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q21, [x20, #0x0]\n"
        "str q25, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q16, [x20, #0x0]\n"
        "str q10, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q30, [x20, #0x0]\n"
        "str q28, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q14, [x20, #0x0]\n"
        "str q26, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q0, [x20, #0x0]\n"
        "str q11, [x20, #0x10]\n"
        "9:"  // Output stage exit
        "subs x9, x9, #0x8\n"
        "add %x[dst], %x[dst], #0x20\n"
        "bgt 2b\n"
        "mov x20, #0x2\n"
        "sub x12, x12, #0x8\n"
        "cmp x12, #0x8\n"
        "mov %x[dst], x28\n"
        "madd %x[lhs_packed], x20, x11, %x[lhs_packed]\n"
        "bge 1b\n"
        "10:"  // Row loop skip
        "cbz x12, 20f\n"
        "11:"  // Row tail: Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "12:"  // Row tail: Column loop
        "movi v13.4s, #0x0\n"
        "movi v14.4s, #0x0\n"
        "mov x22, %x[lhs_packed]\n"
        "mov x20, %x[num_blocks]\n"
        "movi v25.4s, #0x0\n"
        "movi v16.4s, #0x0\n"
        "movi v26.4s, #0x0\n"
        "movi v30.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "movi v19.4s, #0x0\n"
        "13:"  // Row tail: Block loop
        "ldr q4, [x26, #0x0]\n"
        "ldr q8, [x26, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q2, [x26, #0x20]\n"
        "ldr q11, [x26, #0x30]\n"
        "ldr q18, [x22, #0x0]\n"
        "ldr q15, [x22, #0x10]\n"
        "ldr q12, [x26, #0x40]\n"
        "ldr q6, [x26, #0x50]\n"
        "shl v9.16b, v4.16b, #0x4\n"
        "shl v22.16b, v8.16b, #0x4\n"
        "ldr q28, [x26, #0x60]\n"
        "ldr q27, [x26, #0x70]\n"
        "shl v17.16b, v2.16b, #0x4\n"
        "shl v23.16b, v11.16b, #0x4\n"
        "ldr q31, [x22, #0x20]\n"
        "ldr q7, [x22, #0x30]\n"
        "and v4.16b, v4.16b, v3.16b\n"
        "and v8.16b, v8.16b, v3.16b\n"
        "ldr q24, [x22, #0x40]\n"
        "ldr q1, [x22, #0x50]\n"
        ".inst 0x4e89a64d  // smmla v13.4s, v18.16b, v9.16b\n"
        ".inst 0x4e96a659  // smmla v25.4s, v18.16b, v22.16b\n"
        "ldr q21, [x22, #0x60]\n"
        "ldr q20, [x22, #0x70]\n"
        ".inst 0x4e91a64e  // smmla v14.4s, v18.16b, v17.16b\n"
        ".inst 0x4e97a650  // smmla v16.4s, v18.16b, v23.16b\n"
        ".inst 0x4e89a5fa  // smmla v26.4s, v15.16b, v9.16b\n"
        ".inst 0x4e96a5ea  // smmla v10.4s, v15.16b, v22.16b\n"
        "shl v22.16b, v12.16b, #0x4\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4e91a5fe  // smmla v30.4s, v15.16b, v17.16b\n"
        ".inst 0x4e97a5f3  // smmla v19.4s, v15.16b, v23.16b\n"
        "shl v17.16b, v6.16b, #0x4\n"
        "add x26, x26, #0x80\n"
        "shl v23.16b, v28.16b, #0x4\n"
        "shl v5.16b, v27.16b, #0x4\n"
        ".inst 0x4e96a7ed  // smmla v13.4s, v31.16b, v22.16b\n"
        "and v2.16b, v2.16b, v3.16b\n"
        "and v11.16b, v11.16b, v3.16b\n"
        ".inst 0x4e91a7f9  // smmla v25.4s, v31.16b, v17.16b\n"
        ".inst 0x4e96a4fa  // smmla v26.4s, v7.16b, v22.16b\n"
        ".inst 0x4e91a4ea  // smmla v10.4s, v7.16b, v17.16b\n"
        "and v12.16b, v12.16b, v3.16b\n"
        ".inst 0x4e97a7ee  // smmla v14.4s, v31.16b, v23.16b\n"
        ".inst 0x4e85a7f0  // smmla v16.4s, v31.16b, v5.16b\n"
        "and v6.16b, v6.16b, v3.16b\n"
        ".inst 0x4e97a4fe  // smmla v30.4s, v7.16b, v23.16b\n"
        ".inst 0x4e85a4f3  // smmla v19.4s, v7.16b, v5.16b\n"
        "and v28.16b, v28.16b, v3.16b\n"
        ".inst 0x4e84a70d  // smmla v13.4s, v24.16b, v4.16b\n"
        ".inst 0x4e88a719  // smmla v25.4s, v24.16b, v8.16b\n"
        "and v27.16b, v27.16b, v3.16b\n"
        ".inst 0x4e84a43a  // smmla v26.4s, v1.16b, v4.16b\n"
        ".inst 0x4e88a42a  // smmla v10.4s, v1.16b, v8.16b\n"
        ".inst 0x4e82a70e  // smmla v14.4s, v24.16b, v2.16b\n"
        ".inst 0x4e8ba710  // smmla v16.4s, v24.16b, v11.16b\n"
        ".inst 0x4e82a43e  // smmla v30.4s, v1.16b, v2.16b\n"
        ".inst 0x4e8ba433  // smmla v19.4s, v1.16b, v11.16b\n"
        ".inst 0x4e8ca6ad  // smmla v13.4s, v21.16b, v12.16b\n"
        ".inst 0x4e86a6b9  // smmla v25.4s, v21.16b, v6.16b\n"
        ".inst 0x4e8ca69a  // smmla v26.4s, v20.16b, v12.16b\n"
        ".inst 0x4e86a68a  // smmla v10.4s, v20.16b, v6.16b\n"
        ".inst 0x4e9ca6ae  // smmla v14.4s, v21.16b, v28.16b\n"
        ".inst 0x4e9ba6b0  // smmla v16.4s, v21.16b, v27.16b\n"
        ".inst 0x4e9ca69e  // smmla v30.4s, v20.16b, v28.16b\n"
        ".inst 0x4e9ba693  // smmla v19.4s, v20.16b, v27.16b\n"
        "bgt 13b\n"
        "ldr q5, [x26, #0x0]\n"
        "ldr q20, [x26, #0x10]\n"
        "uzp1 v2.2d, v13.2d, v25.2d\n"
        "uzp1 v21.2d, v14.2d, v16.2d\n"
        "ldr q6, [x22, #0x0]\n"
        "ldr q1, [x26, #0x20]\n"
        "uzp2 v4.2d, v13.2d, v25.2d\n"
        "uzp2 v28.2d, v14.2d, v16.2d\n"
        "ldr q7, [x26, #0x30]\n"
        "ldr q17, [x22, #0x10]\n"
        "uzp1 v29.2d, v26.2d, v10.2d\n"
        "uzp1 v15.2d, v30.2d, v19.2d\n"
        "ld1r { v27.4s }, [%x[clamp_vals]]\n"
        "uzp2 v26.2d, v26.2d, v10.2d\n"
        "uzp2 v25.2d, v30.2d, v19.2d\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "ld1r { v19.4s }, [x20]\n"
        "mla v2.4s, v5.4s, v6.s[0]\n"
        "mla v21.4s, v20.4s, v6.s[0]\n"
        "cmp x25, #0x8\n"
        "mla v4.4s, v5.4s, v6.s[1]\n"
        "mla v28.4s, v20.4s, v6.s[1]\n"
        "fmul v23.4s, v1.4s, v17.s[0]\n"
        "add x26, x26, #0x40\n"
        "mla v29.4s, v5.4s, v6.s[2]\n"
        "mla v15.4s, v20.4s, v6.s[2]\n"
        "fmul v31.4s, v7.4s, v17.s[0]\n"
        "mla v26.4s, v5.4s, v6.s[3]\n"
        "mla v25.4s, v20.4s, v6.s[3]\n"
        "fmul v22.4s, v1.4s, v17.s[1]\n"
        "scvtf v2.4s, v2.4s\n"
        "scvtf v21.4s, v21.4s\n"
        "scvtf v4.4s, v4.4s\n"
        "scvtf v28.4s, v28.4s\n"
        "fmul v20.4s, v7.4s, v17.s[1]\n"
        "scvtf v29.4s, v29.4s\n"
        "fmul v24.4s, v1.4s, v17.s[2]\n"
        "scvtf v15.4s, v15.4s\n"
        "fmul v10.4s, v7.4s, v17.s[2]\n"
        "scvtf v26.4s, v26.4s\n"
        "fmul v0.4s, v1.4s, v17.s[3]\n"
        "scvtf v25.4s, v25.4s\n"
        "fmul v8.4s, v7.4s, v17.s[3]\n"
        "fmul v2.4s, v2.4s, v23.4s\n"
        "fmul v21.4s, v21.4s, v31.4s\n"
        "fmul v4.4s, v4.4s, v22.4s\n"
        "fmul v28.4s, v28.4s, v20.4s\n"
        "fmul v29.4s, v29.4s, v24.4s\n"
        "fmul v15.4s, v15.4s, v10.4s\n"
        "fmul v26.4s, v26.4s, v0.4s\n"
        "fmul v25.4s, v25.4s, v8.4s\n"
        "fmax v2.4s, v2.4s, v27.4s\n"
        "fmax v21.4s, v21.4s, v27.4s\n"
        "fmax v4.4s, v4.4s, v27.4s\n"
        "fmax v28.4s, v28.4s, v27.4s\n"
        "fmax v29.4s, v29.4s, v27.4s\n"
        "fmax v15.4s, v15.4s, v27.4s\n"
        "fmax v26.4s, v26.4s, v27.4s\n"
        "fmax v25.4s, v25.4s, v27.4s\n"
        "fmin v2.4s, v2.4s, v19.4s\n"
        "fmin v21.4s, v21.4s, v19.4s\n"
        "fmin v4.4s, v4.4s, v19.4s\n"
        "fmin v28.4s, v28.4s, v19.4s\n"
        "fmin v29.4s, v29.4s, v19.4s\n"
        "fmin v15.4s, v15.4s, v19.4s\n"
        "fmin v26.4s, v26.4s, v19.4s\n"
        "fmin v25.4s, v25.4s, v19.4s\n"
        "bge 18f\n"
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
        "tbz x25, #2, 15f\n"
        "st1 { v26.4s }, [x20], #0x10\n"
        "st1 { v29.4s }, [x21], #0x10\n"
        "st1 { v4.4s }, [x22], #0x10\n"
        "st1 { v2.4s }, [x23], #0x10\n"
        "tbz x25, #1, 14f\n"
        "str d25, [x20], #0x8\n"
        "str d15, [x21], #0x8\n"
        "str d28, [x22], #0x8\n"
        "str d21, [x23], #0x8\n"
        "tbz x25, #0, 17f\n"
        "st1 { v25.s }[2], [x20]\n"
        "st1 { v15.s }[2], [x21]\n"
        "st1 { v28.s }[2], [x22]\n"
        "st1 { v21.s }[2], [x23]\n"
        "b 17f\n"
        "14:"  // Row tail: Output block 0: partial_1_4
        "tbz x25, #0, 17f\n"
        "str s25, [x20, #0x0]\n"
        "str s15, [x21, #0x0]\n"
        "str s28, [x22, #0x0]\n"
        "str s21, [x23, #0x0]\n"
        "b 17f\n"
        "15:"  // Row tail: Output block 0: partial_2_0
        "tbz x25, #1, 16f\n"
        "str d26, [x20], #0x8\n"
        "str d29, [x21], #0x8\n"
        "str d4, [x22], #0x8\n"
        "str d2, [x23], #0x8\n"
        "tbz x25, #0, 17f\n"
        "st1 { v26.s }[2], [x20]\n"
        "st1 { v29.s }[2], [x21]\n"
        "st1 { v4.s }[2], [x22]\n"
        "st1 { v2.s }[2], [x23]\n"
        "b 17f\n"
        "16:"  // Row tail: Output block 0: partial_1_0
        "str s26, [x20, #0x0]\n"
        "str s29, [x21, #0x0]\n"
        "str s4, [x22, #0x0]\n"
        "str s2, [x23, #0x0]\n"
        "17:"  // Row tail: Output block 0: Done
        "b 19f\n"
        "18:"  // Row tail: Full output
        "mov x20, %x[dst]\n"
        "cmp x12, #0x1\n"
        "str q2, [x20, #0x0]\n"
        "str q21, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 19f\n"
        "cmp x12, #0x2\n"
        "str q4, [x20, #0x0]\n"
        "str q28, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 19f\n"
        "cmp x12, #0x3\n"
        "str q29, [x20, #0x0]\n"
        "str q15, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 19f\n"
        "str q26, [x20, #0x0]\n"
        "str q25, [x20, #0x10]\n"
        "19:"  // Row tail: Output stage exit
        "subs x25, x25, #0x8\n"
        "add %x[dst], %x[dst], #0x20\n"
        "bgt 12b\n"
        "subs x12, x12, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x11\n"
        "mov %x[dst], x24\n"
        "bgt 11b\n"
        "20:"  // Row tail: Row loop skip
        : [lhs_packed] "+&r"(lhs_packed), [dst] "+&r"(dst)
        : [rhs_packed] "r"(rhs_packed), [clamp_vals] "r"(clamp_vals), [m] "r"(m), [num_blocks] "r"(num_blocks),
          [dst_stride_row] "r"(dst_stride_row), [n] "r"(n)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31", "x9", "x10", "x11", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
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
