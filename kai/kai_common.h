//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-pro-type-vararg,cert-err33-c)
//
//   * cppcoreguidelines-avoid-do-while: do-while is necessary for macros.
//   * cppcoreguidelines-pro-type-vararg: use of variadic arguments in fprintf is expected.
//   * cert-err33-c: checking the output of fflush and fprintf is not necessary for error reporting.
#define KAI_ERROR(msg)                                        \
    do {                                                      \
        fflush(stdout);                                       \
        fprintf(stderr, "%s:%d %s", __FILE__, __LINE__, msg); \
        exit(EXIT_FAILURE);                                   \
    } while (0)

#define KAI_ASSERT_MSG(cond, msg) \
    do {                          \
        if (!(cond)) {            \
            KAI_ERROR(msg);       \
        }                         \
    } while (0)

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-pro-type-vararg,cert-err33-c)

#define KAI_ASSERT(cond) KAI_ASSERT_MSG(cond, #cond)

#define KAI_ASSERT_IF_MSG(precond, cond, msg) KAI_ASSERT_MSG(!(precond) || (cond), msg)
#define KAI_ASSERT_IF(precond, cond) KAI_ASSERT_IF_MSG(precond, cond, #precond " |-> " #cond)

#define KAI_ASSUME_MSG KAI_ASSERT_MSG
#define KAI_ASSUME KAI_ASSERT
#define KAI_ASSUME_IF_MSG KAI_ASSERT_IF_MSG
#define KAI_ASSUME_IF KAI_ASSERT_IF

#define KAI_UNUSED(x) (void)(x)
#define KAI_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define KAI_MAX(a, b) (((a) > (b)) ? (a) : (b))

inline static size_t kai_roundup(size_t a, size_t b) {
    return ((a + b - 1) / b) * b;
}

#ifdef __ARM_FEATURE_SVE

/// Gets the SME vector length for 8-bit elements.
inline static uint64_t kai_get_sme_vector_length_u8(void) {
    uint64_t res = 0;

    __asm __volatile(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "cntb %0\n"
        ".inst 0xd503467f  // SMSTOP\n"
        : "=r"(res)
        :
        : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");

    return res;
}

/// Gets the SME vector length for 16-bit elements.
inline static uint64_t kai_get_sme_vector_length_u16(void) {
    uint64_t res = 0;

    __asm __volatile(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "cnth %0\n"
        ".inst 0xd503467f  // SMSTOP\n"
        : "=r"(res)
        :
        : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");

    return res;
}

/// Gets the SME vector length for 32-bit elements.
inline static uint64_t kai_get_sme_vector_length_u32(void) {
    uint64_t res = 0;

    __asm __volatile(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "cntw %0\n"
        ".inst 0xd503467f  // SMSTOP\n"
        : "=r"(res)
        :
        : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");

    return res;
}

#endif  // __ARM_FEATURE_SVE

#ifdef __cplusplus
}
#endif
