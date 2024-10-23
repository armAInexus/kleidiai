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
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cp/kai_matmul_clamp_qai8_qai8p_qsi8cpsb_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme.h"
#include "test/common/memory.hpp"
#include "test/common/sme.hpp"
#include "test/reference/binary_elementwise.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/matmul_pack.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/reduce.hpp"
#include "test/reference/reorder.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

namespace {

struct GemmVariant {
    size_t acc_height;
    size_t acc_width;
    size_t acc_fanin;

    size_t (*fn_pack_lhs_get_packed_lhs_size)(size_t m, size_t k, size_t mr, size_t kr, size_t sr);
    void (*fn_pack_lhs_run)(
        size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
        void* lhs_packed);

    size_t (*fn_pack_rhs_get_packed_rhs_size)(size_t n, size_t k);
    void (*fn_pack_rhs_run)(
        size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
        const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes,
        const struct kai_rhs_pack_qsi8_params* params);

    size_t (*fn_main_get_dst_size)(size_t m, size_t n);
    void (*fn_main_run)(
        size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
        size_t dst_stride_col, const kai_matmul_requantize32_params* params);
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

        .fn_pack_lhs_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme,
        .fn_pack_lhs_run = kai_run_lhs_pack_x8p2vlx4_x8_sme,

        .fn_pack_rhs_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme,
        .fn_pack_rhs_run = kai_run_rhs_pack_kxn_qsi8cp2vlx4sb_qsi8_f32_i32_sme,

        .fn_main_get_dst_size = kai_get_dst_size_matmul_clamp_qai8_qai8p_qsi8cpsb_2vlx2vl_sme2_mopa,
        .fn_main_run = kai_run_matmul_clamp_qai8_qai8p_qsi8cpsb_2vlx2vl_sme2_mopa,
    },
};

constexpr float output_clamp_rate = 0.1F;  // Clamping 10% the range of the output.

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

    // ============================================================
    // Generates input and reference output data
    // ============================================================

    // Generates the input data in floating-point.
    const auto lhs_f32 = fill_random<float>(shape.m * shape.k, seed + 0);
    const auto rhs_f32 = fill_random<float>(shape.k * shape.n, seed + 1);
    const auto bias_f32 = fill_random<float>(shape.n, seed + 2);

    // Quantizes the input data.
    //   * LHS: 8-bit asymmetric per-matrix quantization.
    //   * RHS: 8-bit symmetric per-channel quantization.
    //   * Bias: 32-bit symmetric per-channel quantization.
    const auto [lhs_qai8, lhs_qai8_scales, lhs_qai8_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(
            lhs_f32.data(), 1, shape.m * shape.k, shape.m * shape.k);
    const auto lhs_scale = read_array<float>(lhs_qai8_scales.data(), 0);
    const auto lhs_zero_point = read_array<int32_t>(lhs_qai8_zero_points.data(), 0);

    const auto rhs_f32_t = transpose<float>(rhs_f32.data(), shape.k, shape.n);
    const auto [rhs_qsi8_t, rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, int8_t, float>(rhs_f32_t.data(), shape.n, shape.k, shape.k);
    const auto rhs_qsi8 = transpose<int8_t>(rhs_qsi8_t.data(), shape.n, shape.k);

    const auto bias_scale = mul<float>(&lhs_scale, 1, 1, rhs_scales.data(), 1, shape.n);
    const auto bias_qsi32 =
        quantize_symmetric_per_block<float, int32_t, float>(bias_f32.data(), bias_scale.data(), shape.n, 1, 1);

    // Runs the reference implementation of matmul to produce floating-point result.
    const auto ref_dst_f32 =
        matmul_nt_t_quantized<int8_t, float, int32_t, int8_t, float, int32_t, int32_t, float, int32_t, float>(
            shape.m, shape.n, shape.k, lhs_qai8.data(), &lhs_scale, &lhs_zero_point, shape.m, shape.k,
            rhs_qsi8_t.data(), rhs_scales.data(), nullptr, 1, shape.k, bias_qsi32.data(), bias_scale.data(), nullptr,
            1);

    // Computes the output quantization information and clamping limits.
    //
    // To get a realistic value for the output quantization information and clamping limits
    // and avoid uncontrolled saturation problem, these information will be calculated
    // based on the reference floating-point output.
    //
    // The clamping limits will be slightly narrower than the actual range of the output
    // so that a portion of the output will be clampped.
    const auto [dst_scales, dst_zero_points] =
        compute_asymmetric_per_block_quantization_info<float, int8_t, float, int32_t>(
            ref_dst_f32.data(), 1, shape.m * shape.n, shape.m * shape.n);
    const auto dst_scale = read_array<float>(dst_scales.data(), 0);
    const auto dst_zero_point = read_array<int32_t>(dst_zero_points.data(), 0);

    const auto ref_dst_f32_min = reduce_min<float>(ref_dst_f32.data(), shape.m * shape.n);
    const auto ref_dst_f32_max = reduce_max<float>(ref_dst_f32.data(), shape.m * shape.n);
    const auto ref_dst_f32_range = ref_dst_f32_max - ref_dst_f32_min;

    const auto ref_dst_f32_clamp_min = ref_dst_f32_min + ref_dst_f32_range * output_clamp_rate / 2;
    const auto ref_dst_f32_clamp_max = ref_dst_f32_max - ref_dst_f32_range * output_clamp_rate / 2;
    const auto dst_qai8_clamp_min =
        quantize_asymmetric<float, int8_t, int32_t>(ref_dst_f32_clamp_min, dst_scale, dst_zero_point);
    const auto dst_qai8_clamp_max =
        quantize_asymmetric<float, int8_t, int32_t>(ref_dst_f32_clamp_max, dst_scale, dst_zero_point);

    // Clamps and quantizes the reference output matrix.
    const auto ref_dst_f32_clamped =
        clamp<float>(ref_dst_f32.data(), shape.m * shape.n, ref_dst_f32_clamp_min, ref_dst_f32_clamp_max);
    const auto ref_dst_qsi8_clamped = quantize_asymmetric_per_block<float, int8_t, float, int32_t>(
        ref_dst_f32_clamped.data(), &dst_scale, &dst_zero_point, 1, shape.m * shape.n, shape.m * shape.n);

    // Runs the reference implementation of the packing functions.
    //
    // The reference packing functions cannot be executed earlier
    // because we need the reference floating-point output first to have
    // the quantization information.
    const auto ref_packed_lhs =
        reorder_block<int8_t>(lhs_qai8.data(), shape.m, shape.k, variant.acc_height, variant.acc_fanin);

    const auto ref_packed_rhs = matmul_pack_rhs_nxk_static_quantized<int8_t, float, int32_t>(
        rhs_qsi8_t.data(), rhs_scales.data(), lhs_scale, dst_scale, bias_qsi32.data(), lhs_zero_point, shape.n, shape.k,
        variant.acc_width, variant.acc_fanin);

    // ============================================================
    // Runs the optimized implementation and checks for correctness
    // ============================================================

    // Runs the optimized implementation of LHS packing.
    const auto imp_packed_lhs_size =
        variant.fn_pack_lhs_get_packed_lhs_size(shape.m, shape.k, variant.acc_height, variant.acc_fanin, 1);
    ASSERT_EQ(imp_packed_lhs_size, ref_packed_lhs.size());
    std::vector<uint8_t> imp_packed_lhs(imp_packed_lhs_size);

    variant.fn_pack_lhs_run(
        shape.m, shape.k, variant.acc_height, variant.acc_fanin, 1, 0, lhs_qai8.data(), shape.k * sizeof(int8_t),
        imp_packed_lhs.data());

    for (size_t i = 0; i < ref_packed_lhs.size(); ++i) {
        ASSERT_EQ(imp_packed_lhs[i], ref_packed_lhs[i]);
    }

    // Runs the optimized implementation of RHS packing.
    const auto imp_packed_rhs_size = variant.fn_pack_rhs_get_packed_rhs_size(shape.n, shape.k);
    ASSERT_EQ(imp_packed_rhs_size, ref_packed_rhs.size());
    std::vector<uint8_t> imp_packed_rhs(imp_packed_rhs_size);

    const kai_rhs_pack_qsi8_params imp_pack_rhs_params{
        .input_zero_point = lhs_zero_point,
        .scale_multiplier = lhs_scale / dst_scale,
    };

    variant.fn_pack_rhs_run(
        1, shape.n, shape.k, variant.acc_width, variant.acc_fanin, 1, shape.n * sizeof(int8_t), rhs_qsi8.data(),
        bias_qsi32.data(), rhs_scales.data(), imp_packed_rhs.data(), 0, &imp_pack_rhs_params);

    for (size_t i = 0; i < ref_packed_rhs.size(); ++i) {
        ASSERT_EQ(imp_packed_rhs[i], ref_packed_rhs[i]);
    }

    // Runs the optimized implementation of GEMM kernel.
    const auto imp_dst_size = variant.fn_main_get_dst_size(shape.m, shape.n);
    ASSERT_EQ(imp_dst_size, ref_dst_qsi8_clamped.size());

    std::vector<uint8_t> imp_dst(imp_dst_size);

    const kai_matmul_requantize32_params imp_main_params{
        .min_value = dst_qai8_clamp_min,
        .max_value = dst_qai8_clamp_max,
        .output_zero_point = dst_zero_point,
    };

    variant.fn_main_run(
        shape.m, shape.n, shape.k, imp_packed_lhs.data(), imp_packed_rhs.data(), imp_dst.data(),
        shape.n * sizeof(int8_t), sizeof(int8_t), &imp_main_params);

    for (size_t i = 0; i < ref_dst_qsi8_clamped.size(); ++i) {
        const int32_t imp_value = read_array<int8_t>(imp_dst.data(), i);
        const int32_t ref_value = read_array<int8_t>(ref_dst_qsi8_clamped.data(), i);
        const auto error = std::abs(imp_value - ref_value);

        if (error > 1) {
            ASSERT_EQ(imp_dst[i], ref_dst_qsi8_clamped[i]);
        }
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
