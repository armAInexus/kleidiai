//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 1;
static const size_t kai_nr = 8;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(void) {
    return kai_mr;
}

size_t kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(void) {
    return kai_nr;
}

size_t kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(void) {
    return kai_sr;
}

size_t kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(size_t m_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_mr == 0);

    return m_idx * stride;
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_nr == 0);

    return n_idx / kai_nr * (kai_nr * sizeof(float) + kai_nr * k * sizeof(float));
}

size_t kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(
    size_t m_idx, size_t n_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_mr == 0);
    KAI_ASSUME(n_idx % kai_nr == 0);

    return m_idx * stride + n_idx * sizeof(float);
}

size_t kai_get_dst_size_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_1x8x4_neon_mla(
    size_t m, size_t n, size_t k,                             //
    const void* lhs, size_t lhs_stride,                       //
    const void* rhs_packed,                                   //
    void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
    float clamp_min, float clamp_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));

    typedef struct {
        float maxval;
        float minval;
        unsigned int num_strings;
        const unsigned int* string_lengths;
        size_t N;
        const void* B_ptr;
        size_t output_offset;
        size_t input_initial_col;
        size_t input_offset;
        void* output_ptr;
        const void* bias;
    } KernelArgs;

    KernelArgs ka;

    unsigned long flags = 0;

    unsigned int string_length = k;
    ka.num_strings = 1;
    ka.string_lengths = &string_length;
    ka.N = n;
    ka.B_ptr = rhs_packed;
    ka.bias = NULL;

    // Direct input.
    const void* input_ptr = lhs;
    ka.input_offset = lhs_stride / sizeof(float);
    ka.input_initial_col = 0;

    // Direct output.
    ka.output_ptr = dst;
    ka.output_offset = dst_stride_row / sizeof(float);

    // Clamping output.
    flags |= 0x2;
    ka.maxval = clamp_max;
    ka.minval = clamp_min;

    __asm__ __volatile__(
        "1:"  // Row loop
        "ldr x21, [%x[args_ptr], %[offsetof_output_offset]]\n"
        "ldr x26, [%x[args_ptr], %[offsetof_output_ptr]]\n"
        "mov x20, #0x4\n"
        "ldr x25, [%x[args_ptr], %[offsetof_N]]\n"
        "ldr x24, [%x[args_ptr], %[offsetof_B_ptr]]\n"
        "madd x20, x21, x20, x26\n"
        "str x20, [%x[args_ptr], %[offsetof_output_ptr]]\n"
        "2:"  // Height 1: Column loop
        "cbz x24, 3f\n"
        "ldr q30, [x24, #0x0]\n"
        "ldr q31, [x24, #0x10]\n"
        "add x24, x24, #0x20\n"
        "b 10f\n"
        "3:"  // Height 1: no bias
        "tbz %x[flags], #0, 9f\n"
        "cmp x25, #0x8\n"
        "bge 8f\n"
        "tbz x25, #2, 5f\n"
        "ld1 { v30.4s }, [x26], #0x10\n"
        "tbz x25, #1, 4f\n"
        "ldr d31, [x26], #0x8\n"
        "mov x20, #0x18\n"
        "tbz x25, #0, 7f\n"
        "ld1 { v31.s }[2], [x26]\n"
        "b 7f\n"
        "4:"  // Height 1: Partial accumulate: partial_1_4
        "mov x20, #0x10\n"
        "tbz x25, #0, 7f\n"
        "ldr s31, [x26, #0x0]\n"
        "b 7f\n"
        "5:"  // Height 1: Partial accumulate: partial_2_0
        "tbz x25, #1, 6f\n"
        "ldr d30, [x26], #0x8\n"
        "mov x20, #0x8\n"
        "tbz x25, #0, 7f\n"
        "ld1 { v30.s }[2], [x26]\n"
        "b 7f\n"
        "6:"  // Height 1: Partial accumulate: partial_1_0
        "ldr s30, [x26, #0x0]\n"
        "mov x20, #0x0\n"
        "7:"  // Height 1: Partial accumulate: Done
        "sub x26, x26, x20\n"
        "b 10f\n"
        "8:"  // Height 1: full accumulate
        "ldr q30, [x26, #0x0]\n"
        "ldr q31, [x26, #0x10]\n"
        "b 10f\n"
        "9:"  // Height 1: no accumulate
        "movi v30.16b, #0x0\n"
        "movi v31.16b, #0x0\n"
        "10:"  // Height 1: setup done
        "mov x23, #0x0\n"
        "11:"  // Height 1: String loop
        "ldr x20, [%x[args_ptr], %[offsetof_string_lengths]]\n"
        "ldr x21, [%x[args_ptr], %[offsetof_input_offset]]\n"
        "ldr w22, [x20, x23, LSL #0x2]\n"
        "tbz %x[flags], #3, 12f\n"
        "ldr x20, [%x[input_ptr], x23, LSL #0x3]\n"
        "add x20, x20, x21, LSL #3\n"
        "ldr x21, [x20, #0x0]\n"
        "cbnz x23, 13f\n"
        "ldr x20, [%x[args_ptr], %[offsetof_input_initial_col]]\n"
        "add x21, x21, x20, LSL #2\n"
        "b 13f\n"
        "12:"  // Height 1: setup direct input
        "mov x21, %x[input_ptr]\n"
        "13:"  // Height 1: input setup done
        "cmp x22, #0x4\n"
        "blt 16f\n"
        "ldr q0, [x21, #0x0]\n"
        "ldr q1, [x24, #0x0]\n"
        "cmp x22, #0x8\n"
        "ldr q2, [x24, #0x10]\n"
        "ldr q3, [x24, #0x20]\n"
        "ldr q4, [x24, #0x30]\n"
        "ldr q5, [x24, #0x40]\n"
        "ldr q6, [x24, #0x50]\n"
        "ldr q7, [x24, #0x60]\n"
        "ldr q8, [x24, #0x70]\n"
        "blt 15f\n"
        "14:"  // Height 1: Multiply loop: Main loop head
        "fmla v30.4s, v1.4s, v0.s[0]\n"
        "fmla v31.4s, v2.4s, v0.s[0]\n"
        "sub x22, x22, #0x4\n"
        "add x21, x21, #0x10\n"
        "cmp x22, #0x8\n"
        "add x24, x24, #0x80\n"
        "prfm pldl1keep, [x21, #0x80]\n"
        "ldr q1, [x24, #0x0]\n"
        "ldr q2, [x24, #0x10]\n"
        "fmla v30.4s, v3.4s, v0.s[1]\n"
        "ldr q3, [x24, #0x20]\n"
        "fmla v31.4s, v4.4s, v0.s[1]\n"
        "ldr q4, [x24, #0x30]\n"
        "fmla v30.4s, v5.4s, v0.s[2]\n"
        "ldr q5, [x24, #0x40]\n"
        "fmla v31.4s, v6.4s, v0.s[2]\n"
        "ldr q6, [x24, #0x50]\n"
        "fmla v30.4s, v7.4s, v0.s[3]\n"
        "ldr q7, [x24, #0x60]\n"
        "fmla v31.4s, v8.4s, v0.s[3]\n"
        "ldr q0, [x21, #0x0]\n"
        "ldr q8, [x24, #0x70]\n"
        "bge 14b\n"
        "15:"  // Height 1: Multiply loop: Single iteration only
        "fmla v30.4s, v1.4s, v0.s[0]\n"
        "fmla v31.4s, v2.4s, v0.s[0]\n"
        "add x21, x21, #0x10\n"
        "sub x22, x22, #0x4\n"
        "add x24, x24, #0x80\n"
        "prfm pldl1keep, [x21, #0x80]\n"
        "fmla v30.4s, v3.4s, v0.s[1]\n"
        "fmla v31.4s, v4.4s, v0.s[1]\n"
        "fmla v30.4s, v5.4s, v0.s[2]\n"
        "fmla v31.4s, v6.4s, v0.s[2]\n"
        "fmla v30.4s, v7.4s, v0.s[3]\n"
        "fmla v31.4s, v8.4s, v0.s[3]\n"
        "16:"  // Height 1: Multiply loop: Main loop skip
        "cbz x22, 18f\n"
        "17:"  // Height 1: Multiply loop: Odd block loop
        "ldr s18, [x21], #0x4\n"
        "ldr q17, [x24, #0x0]\n"
        "sub x22, x22, #0x1\n"
        "ldr q16, [x24, #0x10]\n"
        "add x24, x24, #0x20\n"
        "fmla v30.4s, v17.4s, v18.s[0]\n"
        "fmla v31.4s, v16.4s, v18.s[0]\n"
        "cbnz x22, 17b\n"
        "18:"  // Height 1: Multiply loop: No odd multiplies
        "ldr w20, [%x[args_ptr], %[offsetof_num_strings]]\n"
        "add x23, x23, #0x1\n"
        "cmp x23, x20\n"
        "bne 11b\n"
        "prfm pstl1keep, [x26, #0x0]\n"
        "tbz %x[flags], #1, 19f\n"
        "add x21, %x[args_ptr], %[offset_max]\n"
        "add x20, %x[args_ptr], %[offset_min]\n"
        "ld1r { v17.4s }, [x21]\n"
        "ld1r { v16.4s }, [x20]\n"
        "fmin v30.4s, v30.4s, v17.4s\n"
        "fmin v31.4s, v31.4s, v17.4s\n"
        "fmax v30.4s, v30.4s, v16.4s\n"
        "fmax v31.4s, v31.4s, v16.4s\n"
        "19:"  // Height 1: No activation
        "cmp x25, #0x8\n"
        "bge 24f\n"
        "tbz x25, #2, 21f\n"
        "st1 { v30.4s }, [x26], #0x10\n"
        "tbz x25, #1, 20f\n"
        "str d31, [x26], #0x8\n"
        "tbz x25, #0, 23f\n"
        "st1 { v31.s }[2], [x26]\n"
        "b 23f\n"
        "20:"  // Height 1: Partial direct writeback: partial_1_4
        "tbz x25, #0, 23f\n"
        "str s31, [x26, #0x0]\n"
        "b 23f\n"
        "21:"  // Height 1: Partial direct writeback: partial_2_0
        "tbz x25, #1, 22f\n"
        "str d30, [x26], #0x8\n"
        "tbz x25, #0, 23f\n"
        "st1 { v30.s }[2], [x26]\n"
        "b 23f\n"
        "22:"  // Height 1: Partial direct writeback: partial_1_0
        "str s30, [x26, #0x0]\n"
        "23:"  // Height 1: Partial direct writeback: Done
        "b 25f\n"
        "24:"  // Height 1: Full writeback
        "str q30, [x26, #0x0]\n"
        "str q31, [x26, #0x10]\n"
        "add x26, x26, #0x20\n"
        "25:"  // Height 1: Writeback done
        "subs x25, x25, #0x8\n"
        "bgt 2b\n"
        "subs %x[m], %x[m], #0x1\n"
        "beq 27f\n"
        "ldr x21, [%x[args_ptr], %[offsetof_input_offset]]\n"
        "tbz %x[flags], #3, 26f\n"
        "add x21, x21, #0x1\n"
        "str x21, [%x[args_ptr], %[offsetof_input_offset]]\n"
        "b 1b\n"
        "26:"  // Update direct input
        "mov x20, #0x4\n"
        "madd %x[input_ptr], x20, x21, %x[input_ptr]\n"
        "b 1b\n"
        "27:"  // Exit
        : [input_ptr] "+&r"(input_ptr), [m] "+&r"(m)
        : [args_ptr] "r"(&ka), [flags] "r"(flags), [offset_max] "I"(offsetof(KernelArgs, maxval)),
          [offset_min] "I"(offsetof(KernelArgs, minval)), [offsetof_B_ptr] "I"(offsetof(KernelArgs, B_ptr)),
          [offsetof_N] "I"(offsetof(KernelArgs, N)),
          [offsetof_input_initial_col] "I"(offsetof(KernelArgs, input_initial_col)),
          [offsetof_input_offset] "I"(offsetof(KernelArgs, input_offset)),
          [offsetof_num_strings] "I"(offsetof(KernelArgs, num_strings)),
          [offsetof_output_offset] "I"(offsetof(KernelArgs, output_offset)),
          [offsetof_output_ptr] "I"(offsetof(KernelArgs, output_ptr)),
          [offsetof_string_lengths] "I"(offsetof(KernelArgs, string_lengths))
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17", "v18", "v30", "v31",
          "x20", "x21", "x22", "x23", "x24", "x25", "x26");
}

#endif  // Architectural features check.
