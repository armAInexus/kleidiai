//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme.h"
#include "test/common/sme.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul_pack.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

namespace {

struct GemmVariant {
    size_t acc_height;
    size_t acc_width;
    size_t acc_fanin;

    size_t (*fn_pack_rhs_get_packed_rhs_offset)(size_t n, size_t k);
    void (*fn_pack_rhs_run)(
        size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
        const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes,
        const struct kai_rhs_pack_qsi8_params* params);
};

struct GemmShape {
    size_t m;
    size_t n;
    size_t k;
};

const std::array gemm_variants = {
    GemmVariant{
        .acc_height = 2 * get_sme_vector_length<int32_t>(),
        .acc_width = 2 * get_sme_vector_length<int32_t>(),
        .acc_fanin = sizeof(int32_t) / sizeof(int8_t),

        .fn_pack_rhs_get_packed_rhs_offset = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme,
        .fn_pack_rhs_run = kai_run_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme,
    },
};

const std::array gemm_shapes = {
    GemmShape{1, 1, 1},  //
    GemmShape{
        2 * get_sme_vector_length<int32_t>(), 2 * get_sme_vector_length<int32_t>(),
        sizeof(int32_t) / sizeof(int8_t)},  //
    GemmShape{20, 30, 40},                  //
    GemmShape{1, 49, 21},                   //
    GemmShape{23, 1, 43},                   //
    GemmShape{32, 14, 1},                   //
    GemmShape{123, 85, 45},                 //
};

void run_test(const GemmShape& shape, const GemmVariant& variant) {
    const uint64_t seed = 0;

    // Generates the input data.
    const auto rhs_qsi8 = fill_random<int8_t>(shape.k * shape.n, seed + 0);
    const auto rhs_qsi8_scales = fill_random<float>(shape.n, seed + 1);
    const auto biases_qsi32 = fill_random<int32_t>(shape.n, seed + 2, -1000, 1000);

    const auto lhs_zero_point = get_random<int32_t>(seed + 3, -1000, 1000);
    const auto lhs_scale = get_random<float>(seed + 4) * 0;
    const auto dst_scale = get_random<float>(seed + 5);

    // Runs the reference implementation.
    const auto ref_rhs_qsi8_t = transpose<int8_t>(rhs_qsi8.data(), shape.k, shape.n);
    const auto ref_packed_rhs = matmul_pack_rhs_nxk_static_quantized<int8_t, float, int32_t>(
        ref_rhs_qsi8_t.data(), rhs_qsi8_scales.data(), lhs_scale, dst_scale, biases_qsi32.data(), lhs_zero_point,
        shape.n, shape.k, variant.acc_width, variant.acc_fanin);

    // Runs the implementation under test.
    const auto imp_packed_rhs_size = variant.fn_pack_rhs_get_packed_rhs_offset(shape.n, shape.k);
    ASSERT_EQ(imp_packed_rhs_size, ref_packed_rhs.size());
    std::vector<uint8_t> imp_packed_rhs(imp_packed_rhs_size);

    kai_rhs_pack_qsi8_params imp_params{
        .input_zero_point = lhs_zero_point,
        .scale_multiplier = lhs_scale * dst_scale,
    };

    variant.fn_pack_rhs_run(
        1, shape.n, shape.k, variant.acc_width, variant.acc_fanin, 1, shape.n * sizeof(int8_t), rhs_qsi8.data(),
        biases_qsi32.data(), rhs_qsi8_scales.data(), imp_packed_rhs.data(), 0, &imp_params);

    for (size_t i = 0; i < ref_packed_rhs.size(); ++i) {
        ASSERT_EQ(imp_packed_rhs[i], ref_packed_rhs[i]);
    }
}

using ThisTest = testing::TestWithParam<std::tuple<GemmVariant, GemmShape>>;

TEST_P(ThisTest, EndToEnd) {
    const auto& [variant, shape] = GetParam();

    run_test(shape, variant);
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    matmul_clamp_qai8_qai8p_qsi8cp, ThisTest,
    testing::Combine(testing::ValuesIn(gemm_variants), testing::ValuesIn(gemm_shapes)));

}  // namespace kai::test
