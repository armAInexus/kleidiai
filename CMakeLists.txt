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

        test/reference/binary_elementwise.cpp
        test/reference/matmul.cpp
        test/reference/fill.cpp
        test/reference/pack.cpp
        test/reference/quantize.cpp
        test/reference/reduce.cpp
        test/reference/round.cpp

        test/tests/matmul_test.cpp
    )

    target_include_directories(kleidiai_test
        PRIVATE .
    )

    target_compile_options(kleidiai_test
        PRIVATE ${KLEIDIAI_WARNING_FLAGS}
    )

    target_link_libraries(kleidiai_test
        PRIVATE GTest::gtest_main
    )

    # Cross-compiling is a common use case which creates a conflict if DISCOVERY_MODE is set to POST_BUILD (by default)
    # since the host platform does not match the target. Setting the mode to PRE_TEST avoids this conflict.
    gtest_discover_tests(kleidiai_test DISCOVERY_MODE PRE_TEST)
endif()
