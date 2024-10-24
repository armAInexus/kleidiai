//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <vector>

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32pb_f32_f32_16vlx1_sme.h"
#include "test/common/cpu_info.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"

namespace kai::test {

class kai_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla : public ::testing::TestWithParam<MatMulShape> {};

TEST_P(kai_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla, EndToEnd) {
    if (!cpu_has_sme2()) {
        GTEST_SKIP();
    }

    const std::uint64_t seed = 0;

    const auto& [m, n, k] = GetParam();

    GTEST_ASSERT_EQ(m, 1);

    const auto nr = kai_get_nr_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla();
    const auto kr = kai_get_kr_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla();
    const auto sr = kai_get_sr_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(m * k, seed + 0);
    const auto ref_rhs = fill_random<float>(n * k, seed + 1);
    const auto ref_bias = fill_random<float>(n, seed + 2);

    // Runs the reference implementation
    const auto ref_dst = matmul(
        ref_lhs.data(), nullptr, nullptr, DataType::FP32, ref_rhs.data(), nullptr, nullptr, DataType::FP32,
        ref_bias.data(), nullptr, nullptr, DataType::FP32, DataType::FP32, m, n, k, false, false);

    // Run the RHS packing micro-kernel.
    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32pb_f32_f32_16vlx1_sme(n, k);
    std::vector<float> imp_packed_rhs(imp_packed_rhs_size);
    kai_run_rhs_pack_kxn_f32pb_f32_f32_16vlx1_sme(
        1, n, k, nr, kr, sr, n * sizeof(float), ref_rhs.data(), ref_bias.data(), nullptr, imp_packed_rhs.data(), 0,
        nullptr);

    // Runs the GEMV micro-kernel.
    const auto imp_dst_size = kai_get_dst_size_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla(m, n);
    ASSERT_EQ(imp_dst_size, ref_dst.size());

    std::vector<uint8_t> imp_dst(imp_dst_size);
    kai_run_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla(
        m, n, k, ref_lhs.data(), 1, imp_packed_rhs.data(), reinterpret_cast<float*>(imp_dst.data()), 1, 1,
        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation.
    for (size_t y = 0; y < m; ++y) {
        for (size_t x = 0; x < n; ++x) {
            const auto imp_value = read_array<float>(imp_dst.data(), (y * n) + x);
            const auto ref_value = read_array<float>(ref_dst.data(), (y * n) + x);
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, kai_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla,
    testing::Values(
        MatMulShape{1, 1, 1},     //
        MatMulShape{1, 16, 1},    //
        MatMulShape{1, 32, 64},   //
        MatMulShape{1, 7, 74},    //
        MatMulShape{1, 800, 64},  //
        MatMulShape{1, 512, 130}),
    [](const testing::TestParamInfo<kai_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla::ParamType>& info) {
        std::stringstream sstream;
        sstream << "kai_matmul_clamp_f32_f32_f32pb_1x16vl_sme2_mla_"
                << "_m_" << info.param.m << "_n_" << info.param.n << "_k_" << info.param.k;
        return sstream.str();
    });

}  // namespace kai::test
