#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.18)

project(KleidiAI
    VERSION 0.0.1
    LANGUAGES C CXX ASM
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(FetchGTest)

option(KLEIDIAI_BUILD_TESTS "Build unit tests." ON)
option(KLEIDIAI_ENABLE_CLANG_TIDY "Build with Clang-Tidy checks." OFF)
option(KLEIDIAI_BUILD_BENCHMARK "Build the benchmark tool." OFF)

if(KLEIDIAI_ENABLE_CLANG_TIDY)
    set(CMAKE_C_CLANG_TIDY "clang-tidy")
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
endif()

set(KLEIDIAI_WARNING_FLAGS_C
    "-Wall"
    "-Wdisabled-optimization"
    "-Werror"
    "-Wextra"
    "-Wformat-security"
    "-Wformat=2"
    "-Winit-self"
    "-Wno-ignored-attributes"
    "-Wno-misleading-indentation"
    "-Wno-overlength-strings"
    "-Wstrict-overflow=2"
    "-Wswitch-default"
)

set(KLEIDIAI_WARNING_FLAGS_CXX
    "-Wctor-dtor-privacy"
    "-Weffc++"
    "-Woverloaded-virtual"
    "-Wsign-promo"
)

set(KLEIDIAI_WARNING_FLAGS
    ${KLEIDIAI_WARNING_FLAGS_C}
    $<$<COMPILE_LANGUAGE:CXX>:${KLEIDIAI_WARNING_FLAGS_CXX}>
)

set(KLEIDIAI_FILES_NEON
    kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c
    kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0.c
)

set(KLEIDIAI_FILES_NEON_FP16
    kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c
    kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c
)

set(KLEIDIAI_FILES_NEON_DOTPROD
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.c
)

set(KLEIDIAI_FILES_NEON_I8MM
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.c
)

add_library(kleidiai
    ${KLEIDIAI_FILES_NEON}
    ${KLEIDIAI_FILES_NEON_FP16}
    ${KLEIDIAI_FILES_NEON_DOTPROD}
    ${KLEIDIAI_FILES_NEON_I8MM}
)

target_include_directories(kleidiai
    PUBLIC  .
)

target_compile_options(kleidiai
    PRIVATE ${KLEIDIAI_WARNING_FLAGS}
)

foreach(KLEIDIAI_SOURCE_FILE IN LISTS KLEIDIAI_FILES_NEON)
    set_property(SOURCE ${KLEIDIAI_SOURCE_FILE} PROPERTY COMPILE_OPTIONS -march=armv8-a)
endforeach()

foreach(KLEIDIAI_SOURCE_FILE IN LISTS KLEIDIAI_FILES_NEON_FP16)
    set_property(SOURCE ${KLEIDIAI_SOURCE_FILE} PROPERTY COMPILE_OPTIONS -march=armv8.2-a+fp16)
endforeach()

foreach(KLEIDIAI_SOURCE_FILE IN LISTS KLEIDIAI_FILES_NEON_DOTPROD)
    set_property(SOURCE ${KLEIDIAI_SOURCE_FILE} PROPERTY COMPILE_OPTIONS -march=armv8.2-a+dotprod)
endforeach()

foreach(KLEIDIAI_SOURCE_FILE IN LISTS KLEIDIAI_FILES_NEON_I8MM)
    set_property(SOURCE ${KLEIDIAI_SOURCE_FILE} PROPERTY COMPILE_OPTIONS -march=armv8.2-a+i8mm)
endforeach()

if(KLEIDIAI_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)

    add_executable(kleidiai_test
        test/common/data_type.cpp
        test/common/data_format.cpp
        test/common/printer.cpp
        test/common/int4.cpp
        test/common/compare.cpp
        test/common/matrix_portion.cpp
        test/common/rect.cpp
        test/common/bfloat16.cpp
        test/common/float16.cpp

        test/reference/binary_elementwise.cpp
        test/reference/matmul.cpp
        test/reference/fill.cpp
        test/reference/pack.cpp
        test/reference/quantize.cpp
        test/reference/reduce.cpp
        test/reference/round.cpp
        test/reference/transpose.cpp
        test/reference/cast.cpp

        test/tests/matmul_test.cpp
    )

    target_compile_options(kleidiai_test
        PRIVATE ${KLEIDIAI_WARNING_FLAGS}
        PRIVATE -march=armv8.2-a+fp16+bf16
    )

    target_link_libraries(kleidiai_test
        PRIVATE GTest::gtest_main
        PRIVATE kleidiai
    )

    # Cross-compiling is a common use case which creates a conflict if DISCOVERY_MODE is set to POST_BUILD (by default)
    # since the host platform does not match the target. Setting the mode to PRE_TEST avoids this conflict.
    gtest_discover_tests(kleidiai_test DISCOVERY_MODE PRE_TEST)
endif()

if(KLEIDIAI_BUILD_BENCHMARK)
    # https://github.com/google/benchmark/issues/351
    if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM"))
        set(CMAKE_CROSSCOMPILING TRUE CACHE INTERNAL "")
        set(HAVE_POSIX_REGEX FALSE CACHE INTERNAL "")
    endif()
    include(FetchGBench)

    add_executable(kleidiai_benchmark
        benchmark/main.cpp)

    target_link_libraries(
        kleidiai_benchmark
        kleidiai
        benchmark::benchmark)

    set(KLEIDIAI_BENCHMARK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(kleidiai_benchmark
        PRIVATE KLEIDIAI_BENCHMARK_INCLUDE_DIR)
endif()
