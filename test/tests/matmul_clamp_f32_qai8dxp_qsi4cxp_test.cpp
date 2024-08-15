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

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0.h"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {

TEST(matmul_clamp_f32_qai8dxp_qai4cxp, EndToEnd) {
    const uint64_t seed = 0;

    const size_t M = 16;
    const size_t N = 32;
    const size_t K = 64;

    const auto mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    const auto nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    const auto kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();
    const auto sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    const auto ref_biases = fill_random<float>(N, seed + 2);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 4-bit symmetric quantization.
    //   * Performs GEMM.
    const auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    const auto [ref_rhs_qsi4, ref_rhs_scales] =
        quantize_symmetric_per_block<float, Int4, float>(ref_rhs.data(), N, K, K);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, float, int32_t, Int4, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), K, ref_rhs_qsi4.data(),
        ref_rhs_scales.data(), nullptr, K, ref_biases.data(), std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    std::vector<uint8_t> imp_packed_lhs(imp_packed_lhs_size);
    kai_run_lhs_quant_pack_qai8dxp_f32(
        M, K, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data()), K * sizeof(float), imp_packed_lhs.data());

    // Runs the RHS packing micro-kernel.
    //   * Generates the 4-bit unsigned symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_qsi4.data(), N * K);
    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(N, K, nr, kr, sr);
    std::vector<uint8_t> imp_packed_rhs(imp_packed_rhs_size);
    const kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0_params params{.lhs_zero_point = 1, .rhs_zero_point = 8};
    kai_run_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0(
        1, N, K, nr, kr, sr, ref_rhs_qsu4.data(), reinterpret_cast<const float*>(ref_biases.data()),
        reinterpret_cast<const float*>(ref_rhs_scales.data()), imp_packed_rhs.data(), 0, &params);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    std::vector<uint8_t> imp_dst(imp_dst_size);
    kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm(
        M, N, K, imp_packed_lhs.data(), imp_packed_rhs.data(), reinterpret_cast<float*>(imp_dst.data()),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation.
    for (size_t y = 0; y < M; ++y) {
        for (size_t x = 0; x < N; ++x) {
            const auto imp_value = read_array<float>(imp_dst.data(), y * N + x);
            const auto ref_value = read_array<float>(ref_dst.data(), y * N + x);
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

}  // namespace kai::test
