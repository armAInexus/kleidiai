#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.18)

project(KleidiAI)

set(CMAKE_CXX_STANDARD 17)
set(KLEIDIAI_PATH ../../)
set(MATMUL_PACK_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/pack/)
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/)

# KleidiAI include directories
include_directories(
    ${KLEIDIAI_PATH}
    ${MATMUL_PACK_PATH}
    ${MATMUL_PATH})

# Files requires to build the executable
add_executable(matmul_clamp_f32_bf16p_bf16p
    matmul_clamp_f32_bf16p_bf16p.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f32_bf16p_bf16p12x1biasf32_8x12x4_neon_mmla.c
    ${MATMUL_PACK_PATH}/kai_lhs_pack_f32p8x4_bf16_neon.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_f32p4x12biasf32_f32_bf16_neon.c
)

target_compile_options(matmul_clamp_f32_bf16p_bf16p
    PRIVATE -march=armv8.2-a+bf16
)

target_compile_definitions(matmul_clamp_f32_bf16p_bf16p
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
