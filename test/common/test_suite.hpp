//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <tuple>

#define UKERNEL_MATMUL_VARIANT(name)              \
    {                                             \
        {kai_get_m_step_matmul_##name,            \
         kai_get_n_step_matmul_##name,            \
         kai_get_mr_matmul_##name,                \
         kai_get_nr_matmul_##name,                \
         kai_get_kr_matmul_##name,                \
         kai_get_sr_matmul_##name,                \
         kai_get_lhs_packed_offset_matmul_##name, \
         kai_get_rhs_packed_offset_matmul_##name, \
         kai_get_dst_offset_matmul_##name,        \
         kai_get_dst_size_matmul_##name,          \
         kai_run_matmul_##name},                  \
            "kai_matmul_" #name                   \
    }

namespace kai::test {

template <typename T>
struct UkernelVariant {
    T interface;
    std::string name{};
};

/// Matrix multiplication shape.
struct MatMulShape {
    size_t m{};  ///< LHS height.
    size_t n{};  ///< RHS width.
    size_t k{};  ///< LHS width and RHS height.
};

/// Matrix multiplication test information.
using MatMulTestParams = std::tuple<size_t, MatMulShape>;

class UkernelVariantTest : public ::testing::TestWithParam<MatMulTestParams> {};

}  // namespace kai::test
