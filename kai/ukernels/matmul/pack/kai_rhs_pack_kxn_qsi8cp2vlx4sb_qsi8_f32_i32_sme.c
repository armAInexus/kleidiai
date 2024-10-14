//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme.h"

#include <alloca.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_nr = 2;
static const size_t kai_kr = 4;
static const size_t kai_num_bytes_input = 1;
static const size_t kai_num_bytes_output = 1;
static const size_t kai_num_bytes_bias = 4;
static const size_t kai_num_bytes_scale = 4;

size_t kai_get_n_step_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme(void) {
    return kai_nr * kai_get_sme_vector_length_u8();
}

size_t kai_get_rhs_offset_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % (kai_nr * kai_get_sme_vector_length_u8()) == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme(size_t k) {
    return kai_nr * kai_get_sme_vector_length_u8() / kai_kr *
        (kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_output + kai_num_bytes_scale);
}

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % (kai_nr * kai_get_sme_vector_length_u8() / kai_kr) == 0);

    return n_idx * (kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_output + kai_num_bytes_scale);
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme(size_t n, size_t k) {
    return kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme(
        kai_roundup(n, kai_nr * kai_get_sme_vector_length_u8() / kai_kr), k);
}

void kai_run_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_qsi8_params* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == kai_nr * kai_get_sme_vector_length_u8() / kai_kr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == 1);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params != NULL);

    size_t height = k;
    const size_t width = n;
    const void* in = rhs;
    void* out = rhs_packed;
    const size_t in_stride = rhs_stride;
    uint8_t* pad_row = (uint8_t*)alloca(width * sizeof(uint8_t));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme(height);
    const int32_t input_zero_point = params->input_zero_point;
    const float scale_multiplier = params->scale_multiplier;

    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "mov x27, %x[out]\n"
        "mov x26, %x[height]\n"
        "ptrue p2.b\n"
        "incb %x[out], ALL, MUL #2\n"
        "1:"  // Main row loop: Head
        "mov x25, %x[in]\n"
        "cmp %x[height], #0x3\n"
        "add x24, x25, %x[in_stride]\n"
        "mov x23, %x[out]\n"
        "add x22, x24, %x[in_stride]\n"
        "mov x21, %x[width]\n"
        "add x20, x22, %x[in_stride]\n"
        "csel x22, x22, %x[pad_row], GE\n"
        "add %x[in], x20, %x[in_stride]\n"
        "csel x20, x20, %x[pad_row], GT\n"
        "cmp %x[height], #0x1\n"
        "sub %x[height], %x[height], #0x4\n"
        "csel x24, x24, %x[pad_row], GT\n"
        "2:"  // Main row loop: Column loop
        "whilelt p0.b, XZR, x21\n"
        "decw x21, ALL, MUL #2\n"
        "ld1b { z18.b }, p0/Z, [x25]\n"
        "cmp x21, #0x0\n"
        "incd x25, ALL, MUL #4\n"
        "ld1b { z19.b }, p0/Z, [x24]\n"
        "incd x24, ALL, MUL #4\n"
        "ld1b { z17.b }, p0/Z, [x22]\n"
        "incd x22, ALL, MUL #4\n"
        "ld1b { z16.b }, p0/Z, [x20]\n"
        "incd x20, ALL, MUL #4\n"
        "zip1 z18.b, z18.b, z17.b\n"
        "zip1 z16.b, z19.b, z16.b\n"
        "zip1 z17.b, z18.b, z16.b\n"
        "zip2 z16.b, z18.b, z16.b\n"
        "st1b { z17.b }, p2, [x23]\n"
        "st1b { z16.b }, p2, [x23, #1, MUL VL]\n"
        "add x23, x23, %x[out_stride]\n"
        "bgt 2b\n"
        "cmp %x[height], #0x1\n"
        "addvl %x[out], %x[out], #2\n"
        "bge 1b\n"
        "mov x22, %x[out]\n"
        "mov x21, %x[width]\n"
        "dup z18.s, %w[scale_multiplier]\n"
        "cbz %x[scale], 5f\n"
        "4:"  // Scale: Full loop
        "mov x20, x21\n"
        "decw x21, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "ld1w { z17.s }, p1/Z, [%x[scale]]\n"
        "cmp x21, #0x0\n"
        "ld1w { z16.s }, p0/Z, [%x[scale], #1, MUL VL]\n"
        "incb %x[scale], ALL, MUL #2\n"
        "fmul z17.s, z17.s, z18.s\n"
        "fmul z16.s, z16.s, z18.s\n"
        "st1w { z17.s }, p2, [x22]\n"
        "st1w { z16.s }, p2, [x22, #1, MUL VL]\n"
        "add x22, x22, %x[out_stride]\n"
        "bgt 4b\n"
        "5:"  // Scale: Done
        "cbz %x[width], 8f\n"
        "cbz x26, 8f\n"
        "dup z21.s, %w[input_zero_point]\n"
        "add x25, x26, #0x3\n"
        "cntw x24, ALL, MUL #2\n"
        "mov z20.b, #0x1\n"
        "lsr x25, x25, #0x2\n"
        "mov x23, %x[width]\n"
        "addvl x22, x27, #2\n"
        "neg z21.s, p2/M, z21.s\n"
        "6:"  // Bias: N loop
        "mov x21, x22\n"
        "mov x20, x25\n"
        "mov z19.s, #0x0\n"
        "mov z18.s, #0x0\n"
        "7:"  // Bias: K loop
        "ld1b { z17.b }, p2/Z, [x21]\n"
        "subs x20, x20, #0x1\n"
        "ld1b { z16.b }, p2/Z, [x21, #1, MUL VL]\n"
        "addvl x21, x21, #2\n"
        "sdot z19.s, z17.b, z20.b\n"
        "sdot z18.s, z16.b, z20.b\n"
        "bgt 7b\n"
        "mov x20, x23\n"
        "add x22, x22, %x[out_stride]\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "ld1w { z17.s }, p1/Z, [%x[bias]]\n"
        "subs x23, x23, x24\n"
        "ld1w { z16.s }, p0/Z, [%x[bias], #1, MUL VL]\n"
        "addvl %x[bias], %x[bias], #2\n"
        "mla z17.s, p2/M, z19.s, z21.s\n"
        "mla z16.s, p2/M, z18.s, z21.s\n"
        "st1w { z17.s }, p2, [x27]\n"
        "st1w { z16.s }, p2, [x27, #1, MUL VL]\n"
        "add x27, x27, %x[out_stride]\n"
        "bgt 6b\n"
        "8:"  // Bias: Done
        ".inst 0xd503467f  // SMSTOP\n"
        : [bias] "+&r"(bias), [height] "+&r"(height), [in] "+&r"(in), [out] "+&r"(out), [scale] "+&r"(scale)
        : [in_stride] "r"(in_stride), [input_zero_point] "r"(input_zero_point), [out_stride] "r"(out_stride),
          [pad_row] "r"(pad_row), [scale_multiplier] "r"(scale_multiplier), [width] "r"(width)
        : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14",
          "p15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
}

#endif  // Architectural features check.
