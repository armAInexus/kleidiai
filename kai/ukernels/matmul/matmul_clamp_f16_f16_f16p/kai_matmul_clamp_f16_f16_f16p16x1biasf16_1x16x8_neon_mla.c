//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16.
#else  // Architectural features check.

#include "kai_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 1;
static const size_t kai_nr = 16;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(void) {
    return kai_mr;
}

size_t kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(void) {
    return kai_nr;
}

size_t kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(void) {
    return kai_sr;
}

size_t kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(size_t m_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_mr == 0);

    return m_idx * stride;
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_nr == 0);

    return n_idx / kai_nr * (kai_nr * sizeof(__fp16) + kai_nr * k * sizeof(__fp16));
}

size_t kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(
    size_t m_idx, size_t n_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_mr == 0);
    KAI_ASSUME(n_idx % kai_nr == 0);

    return m_idx * stride + n_idx * sizeof(__fp16);
}

size_t kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(size_t m, size_t n) {
    return m * n * sizeof(__fp16);
}

void kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla(
    size_t m, size_t n, size_t k,                             //
    const void* lhs, size_t lhs_stride,                       //
    const void* rhs_packed,                                   //
    void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
    __fp16 clamp_min, __fp16 clamp_max) {
    KAI_ASSERT(dst_stride_col == sizeof(__fp16));

    typedef struct {
        __fp16 maxval;
        __fp16 minval;
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
    ka.input_offset = lhs_stride / sizeof(__fp16);
    ka.input_initial_col = 0;

    // Direct output.
    ka.output_ptr = dst;
    ka.output_offset = dst_stride_row / sizeof(__fp16);

    // Clamping output.
    flags |= 0x2;
    ka.maxval = clamp_max;
    ka.minval = clamp_min;

    __asm__ __volatile__(
        "1:"  // Row loop
        "ldr x21, [%x[args_ptr], %[offsetof_output_offset]]\n"
        "ldr x26, [%x[args_ptr], %[offsetof_output_ptr]]\n"
        "mov x20, #0x2\n"
        "ldr x25, [%x[args_ptr], %[offsetof_N]]\n"
        "ldr x24, [%x[args_ptr], %[offsetof_B_ptr]]\n"
        "madd x20, x21, x20, x26\n"
        "str x20, [%x[args_ptr], %[offsetof_output_ptr]]\n"
        "2:"  // Height 1: Column loop
        "cbz x24, 3f\n"
        "ldr q30, [x24, #0x0]\n"
        "ldr q31, [x24, #0x10]\n"
        "add x24, x24, #0x20\n"
        "b 14f\n"
        "3:"  // Height 1: no bias
        "tbz %x[flags], #0, 13f\n"
        "cmp x25, #0x10\n"
        "bge 12f\n"
        "tbz x25, #3, 7f\n"
        "ld1 { v30.8h }, [x26], #0x10\n"
        "tbz x25, #2, 5f\n"
        "ldr d31, [x26], #0x8\n"
        "tbz x25, #1, 4f\n"
        "ld1 { v31.s }[2], [x26], #0x4\n"
        "mov x20, #0x1c\n"
        "tbz x25, #0, 11f\n"
        "ld1 { v31.h }[6], [x26]\n"
        "b 11f\n"
        "4:"  // Height 1: Partial accumulate: partial_1_12
        "mov x20, #0x18\n"
        "tbz x25, #0, 11f\n"
        "ld1 { v31.h }[4], [x26]\n"
        "b 11f\n"
        "5:"  // Height 1: Partial accumulate: partial_2_8
        "tbz x25, #1, 6f\n"
        "ldr s31, [x26], #0x4\n"
        "mov x20, #0x14\n"
        "tbz x25, #0, 11f\n"
        "ld1 { v31.h }[2], [x26]\n"
        "b 11f\n"
        "6:"  // Height 1: Partial accumulate: partial_1_8
        "mov x20, #0x10\n"
        "tbz x25, #0, 11f\n"
        "ldr h31, [x26, #0x0]\n"
        "b 11f\n"
        "7:"  // Height 1: Partial accumulate: partial_4_0
        "tbz x25, #2, 9f\n"
        "ldr d30, [x26], #0x8\n"
        "tbz x25, #1, 8f\n"
        "ld1 { v30.s }[2], [x26], #0x4\n"
        "mov x20, #0xc\n"
        "tbz x25, #0, 11f\n"
        "ld1 { v30.h }[6], [x26]\n"
        "b 11f\n"
        "8:"  // Height 1: Partial accumulate: partial_1_4
        "mov x20, #0x8\n"
        "tbz x25, #0, 11f\n"
        "ld1 { v30.h }[4], [x26]\n"
        "b 11f\n"
        "9:"  // Height 1: Partial accumulate: partial_2_0
        "tbz x25, #1, 10f\n"
        "ldr s30, [x26], #0x4\n"
        "mov x20, #0x4\n"
        "tbz x25, #0, 11f\n"
        "ld1 { v30.h }[2], [x26]\n"
        "b 11f\n"
        "10:"  // Height 1: Partial accumulate: partial_1_0
        "ldr h30, [x26, #0x0]\n"
        "mov x20, #0x0\n"
        "11:"  // Height 1: Partial accumulate: Done
        "sub x26, x26, x20\n"
        "b 14f\n"
        "12:"  // Height 1: full accumulate
        "ldr q30, [x26, #0x0]\n"
        "ldr q31, [x26, #0x10]\n"
        "b 14f\n"
        "13:"  // Height 1: no accumulate
        "movi v30.16b, #0x0\n"
        "movi v31.16b, #0x0\n"
        "14:"  // Height 1: setup done
        "mov x23, #0x0\n"
        "15:"  // Height 1: String loop
        "ldr x20, [%x[args_ptr], %[offsetof_string_lengths]]\n"
        "ldr x21, [%x[args_ptr], %[offsetof_input_offset]]\n"
        "ldr w22, [x20, x23, LSL #0x2]\n"
        "tbz %x[flags], #3, 16f\n"
        "ldr x20, [%x[input_ptr], x23, LSL #0x3]\n"
        "add x20, x20, x21, LSL #3\n"
        "ldr x21, [x20, #0x0]\n"
        "cbnz x23, 17f\n"
        "ldr x20, [%x[args_ptr], %[offsetof_input_initial_col]]\n"
        "add x21, x21, x20, LSL #1\n"
        "b 17f\n"
        "16:"  // Height 1: setup direct input
        "mov x21, %x[input_ptr]\n"
        "17:"  // Height 1: input setup done
        "cmp x22, #0x8\n"
        "blt 20f\n"
        "ldr q0, [x21, #0x0]\n"
        "ldr q1, [x24, #0x0]\n"
        "cmp x22, #0x10\n"
        "ldr q2, [x24, #0x10]\n"
        "ldr q3, [x24, #0x20]\n"
        "ldr q4, [x24, #0x30]\n"
        "ldr q5, [x24, #0x40]\n"
        "ldr q6, [x24, #0x50]\n"
        "ldr q7, [x24, #0x60]\n"
        "ldr q8, [x24, #0x70]\n"
        "ldr q9, [x24, #0x80]\n"
        "ldr q10, [x24, #0x90]\n"
        "ldr q11, [x24, #0xa0]\n"
        "ldr q12, [x24, #0xb0]\n"
        "ldr q13, [x24, #0xc0]\n"
        "ldr q14, [x24, #0xd0]\n"
        "ldr q15, [x24, #0xe0]\n"
        "ldr q16, [x24, #0xf0]\n"
        "blt 19f\n"
        "18:"  // Height 1: Multiply loop: Main loop head
        "fmla v30.8h, v1.8h, v0.h[0]\n"
        "fmla v31.8h, v2.8h, v0.h[0]\n"
        "sub x22, x22, #0x8\n"
        "add x21, x21, #0x10\n"
        "cmp x22, #0x10\n"
        "add x24, x24, #0x100\n"
        "prfm pldl1keep, [x21, #0x80]\n"
        "ldr q1, [x24, #0x0]\n"
        "ldr q2, [x24, #0x10]\n"
        "fmla v30.8h, v3.8h, v0.h[1]\n"
        "ldr q3, [x24, #0x20]\n"
        "fmla v31.8h, v4.8h, v0.h[1]\n"
        "ldr q4, [x24, #0x30]\n"
        "fmla v30.8h, v5.8h, v0.h[2]\n"
        "ldr q5, [x24, #0x40]\n"
        "fmla v31.8h, v6.8h, v0.h[2]\n"
        "ldr q6, [x24, #0x50]\n"
        "fmla v30.8h, v7.8h, v0.h[3]\n"
        "ldr q7, [x24, #0x60]\n"
        "fmla v31.8h, v8.8h, v0.h[3]\n"
        "ldr q8, [x24, #0x70]\n"
        "fmla v30.8h, v9.8h, v0.h[4]\n"
        "ldr q9, [x24, #0x80]\n"
        "fmla v31.8h, v10.8h, v0.h[4]\n"
        "ldr q10, [x24, #0x90]\n"
        "fmla v30.8h, v11.8h, v0.h[5]\n"
        "ldr q11, [x24, #0xa0]\n"
        "fmla v31.8h, v12.8h, v0.h[5]\n"
        "ldr q12, [x24, #0xb0]\n"
        "fmla v30.8h, v13.8h, v0.h[6]\n"
        "ldr q13, [x24, #0xc0]\n"
        "fmla v31.8h, v14.8h, v0.h[6]\n"
        "ldr q14, [x24, #0xd0]\n"
        "fmla v30.8h, v15.8h, v0.h[7]\n"
        "ldr q15, [x24, #0xe0]\n"
        "fmla v31.8h, v16.8h, v0.h[7]\n"
        "ldr q0, [x21, #0x0]\n"
        "ldr q16, [x24, #0xf0]\n"
        "bge 18b\n"
        "19:"  // Height 1: Multiply loop: Single iteration only
        "fmla v30.8h, v1.8h, v0.h[0]\n"
        "fmla v31.8h, v2.8h, v0.h[0]\n"
        "add x21, x21, #0x10\n"
        "sub x22, x22, #0x8\n"
        "add x24, x24, #0x100\n"
        "prfm pldl1keep, [x21, #0x80]\n"
        "fmla v30.8h, v3.8h, v0.h[1]\n"
        "fmla v31.8h, v4.8h, v0.h[1]\n"
        "fmla v30.8h, v5.8h, v0.h[2]\n"
        "fmla v31.8h, v6.8h, v0.h[2]\n"
        "fmla v30.8h, v7.8h, v0.h[3]\n"
        "fmla v31.8h, v8.8h, v0.h[3]\n"
        "fmla v30.8h, v9.8h, v0.h[4]\n"
        "fmla v31.8h, v10.8h, v0.h[4]\n"
        "fmla v30.8h, v11.8h, v0.h[5]\n"
        "fmla v31.8h, v12.8h, v0.h[5]\n"
        "fmla v30.8h, v13.8h, v0.h[6]\n"
        "fmla v31.8h, v14.8h, v0.h[6]\n"
        "fmla v30.8h, v15.8h, v0.h[7]\n"
        "fmla v31.8h, v16.8h, v0.h[7]\n"
        "20:"  // Height 1: Multiply loop: Main loop skip
        "cbz x22, 22f\n"
        "21:"  // Height 1: Multiply loop: Odd block loop
        "ldr h0, [x21], #0x2\n"
        "ldr q17, [x24, #0x0]\n"
        "sub x22, x22, #0x1\n"
        "ldr q18, [x24, #0x10]\n"
        "add x24, x24, #0x20\n"
        "fmla v30.8h, v17.8h, v0.h[0]\n"
        "fmla v31.8h, v18.8h, v0.h[0]\n"
        "cbnz x22, 21b\n"
        "22:"  // Height 1: Multiply loop: No odd multiplies
        "ldr w20, [%x[args_ptr], %[offsetof_num_strings]]\n"
        "add x23, x23, #0x1\n"
        "cmp x23, x20\n"
        "bne 15b\n"
        "prfm pstl1keep, [x26, #0x0]\n"
        "tbz %x[flags], #1, 23f\n"
        "add x21, %x[args_ptr], %[offset_max]\n"
        "add x20, %x[args_ptr], %[offset_min]\n"
        "ld1r { v17.8h }, [x21]\n"
        "ld1r { v16.8h }, [x20]\n"
        "fmin v30.8h, v30.8h, v17.8h\n"
        "fmin v31.8h, v31.8h, v17.8h\n"
        "fmax v30.8h, v30.8h, v16.8h\n"
        "fmax v31.8h, v31.8h, v16.8h\n"
        "23:"  // Height 1: No activation
        "cmp x25, #0x10\n"
        "bge 32f\n"
        "tbz x25, #3, 27f\n"
        "st1 { v30.8h }, [x26], #0x10\n"
        "tbz x25, #2, 25f\n"
        "str d31, [x26], #0x8\n"
        "tbz x25, #1, 24f\n"
        "st1 { v31.s }[2], [x26], #0x4\n"
        "tbz x25, #0, 31f\n"
        "st1 { v31.h }[6], [x26]\n"
        "b 31f\n"
        "24:"  // Height 1: Partial direct writeback: partial_1_12
        "tbz x25, #0, 31f\n"
        "st1 { v31.h }[4], [x26]\n"
        "b 31f\n"
        "25:"  // Height 1: Partial direct writeback: partial_2_8
        "tbz x25, #1, 26f\n"
        "str s31, [x26], #0x4\n"
        "tbz x25, #0, 31f\n"
        "st1 { v31.h }[2], [x26]\n"
        "b 31f\n"
        "26:"  // Height 1: Partial direct writeback: partial_1_8
        "tbz x25, #0, 31f\n"
        "str h31, [x26, #0x0]\n"
        "b 31f\n"
        "27:"  // Height 1: Partial direct writeback: partial_4_0
        "tbz x25, #2, 29f\n"
        "str d30, [x26], #0x8\n"
        "tbz x25, #1, 28f\n"
        "st1 { v30.s }[2], [x26], #0x4\n"
        "tbz x25, #0, 31f\n"
        "st1 { v30.h }[6], [x26]\n"
        "b 31f\n"
        "28:"  // Height 1: Partial direct writeback: partial_1_4
        "tbz x25, #0, 31f\n"
        "st1 { v30.h }[4], [x26]\n"
        "b 31f\n"
        "29:"  // Height 1: Partial direct writeback: partial_2_0
        "tbz x25, #1, 30f\n"
        "str s30, [x26], #0x4\n"
        "tbz x25, #0, 31f\n"
        "st1 { v30.h }[2], [x26]\n"
        "b 31f\n"
        "30:"  // Height 1: Partial direct writeback: partial_1_0
        "str h30, [x26, #0x0]\n"
        "31:"  // Height 1: Partial direct writeback: Done
        "b 33f\n"
        "32:"  // Height 1: Full writeback
        "str q30, [x26, #0x0]\n"
        "str q31, [x26, #0x10]\n"
        "add x26, x26, #0x20\n"
        "33:"  // Height 1: Writeback done
        "subs x25, x25, #0x10\n"
        "bgt 2b\n"
        "subs %x[m], %x[m], #0x1\n"
        "beq 35f\n"
        "ldr x21, [%x[args_ptr], %[offsetof_input_offset]]\n"
        "tbz %x[flags], #3, 34f\n"
        "add x21, x21, #0x1\n"
        "str x21, [%x[args_ptr], %[offsetof_input_offset]]\n"
        "b 1b\n"
        "34:"  // Update direct input
        "mov x20, #0x2\n"
        "madd %x[input_ptr], x20, x21, %x[input_ptr]\n"
        "b 1b\n"
        "35:"  // Exit
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
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26");
}

#endif  // Architectural features check.
