//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "test/common/cpu_info.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {

static const std::array<UkernelVariant<kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel>, 2>
    variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p = {{
        UKERNEL_MATMUL_VARIANT(clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm, cpu_has_i8mm),
        UKERNEL_MATMUL_VARIANT(clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod, cpu_has_dotprod),
    }};

class MatMulTest_f32_qsi8d32p_qsi4c32p : public UkernelVariantTest {};

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, EndToEnd) {
    const auto& [variant_index, matmul_shape] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_index);

    if (!ukernel_variant.fn_is_supported()) {
        GTEST_SKIP();
    }

    const std::uint64_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;
    const size_t bl = 32;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);

    // LHS dimensions
    const size_t ref_lhs_stride = round_up_multiple(K, 2);
    const size_t ref_lhs_size = M * ref_lhs_stride;
    const size_t ref_lhs_size_bytes = ref_lhs_size;
    // Transposed(nxk) RHS dimensions
    const size_t ref_rhs_qsi4_nxk_stride = round_up_multiple(K, 2);
    const size_t ref_rhs_qsi4_nxk_size = N * ref_rhs_qsi4_nxk_stride;
    const size_t ref_rhs_qsi4_nxk_size_bytes = round_up_division(ref_rhs_qsi4_nxk_size, 2);

    // Runs the reference implementation.
    const auto [ref_lhs_qvalues, ref_lhs_scales] = quantize_symmetric_per_block_dynamic<float, int8_t, Float16>(
        ref_lhs.data(), M, K, bl, ref_lhs_stride, ref_lhs_size_bytes);
    const auto [ref_rhs_qsi4, ref_rhs_scales] = quantize_symmetric_per_block_dynamic<float, Int4, Float16>(
        ref_rhs.data(), N, K, bl, ref_rhs_qsi4_nxk_stride, ref_rhs_qsi4_nxk_size_bytes);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, Float16, int32_t, Int4, Float16, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), nullptr, bl, ref_rhs_qsi4.data(), ref_rhs_scales.data(),
        nullptr, bl, nullptr, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(M, K, bl, mr, kr, sr);
    std::vector<uint8_t> imp_packed_lhs(imp_packed_lhs_size);
    kai_run_lhs_quant_pack_qsi8d32p_f32(
        M, K, bl, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data()), K * sizeof(float),
        imp_packed_lhs.data());

    // Runs the RHS packing micro-kernel.
    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_qsi4.data(), ref_rhs_qsi4_nxk_size);
    const auto ref_rhs_qsu4_scale_f16 =
        pack_data_scales_interleave_block<UInt4, Float16>(ref_rhs_qsu4.data(), ref_rhs_scales.data(), N, K, bl);

    const auto imp_packed_rhs_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(N, K, nr, kr, bl);
    std::vector<uint8_t> imp_packed_rhs(imp_packed_rhs_size);
    const kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0_params params{.lhs_zero_point = 1, .rhs_zero_point = 8};
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
        1, N, K, nr, kr, sr, bl, ref_rhs_qsu4_scale_f16.data(), nullptr, imp_packed_rhs.data(), 0, &params);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    std::vector<uint8_t> imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        M, N, K, bl, imp_packed_lhs.data(), imp_packed_rhs.data(), reinterpret_cast<float*>(imp_dst.data()),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation.
    for (size_t y = 0; y < M; ++y) {
        for (size_t x = 0; x < N; ++x) {
            const auto imp_value = read_array<float>(imp_dst.data(), (y * N) + x);
            const auto ref_value = read_array<float>(ref_dst.data(), (y * N) + x);
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.size()),
        testing::Values(
            MatMulShape{32, 64, 64},  //
            MatMulShape{16, 32, 64},  //
            MatMulShape{8, 32, 64},   //
            MatMulShape{15, 32, 32})),
    [](const auto& info) {
        const std::string name{variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(std::get<size_t>(info.param)).name};
        const auto shape = std::get<MatMulShape>(info.param);
        return name + "__M_" + std::to_string(shape.m) + "__N_" + std::to_string(shape.n) + "__K_" +
            std::to_string(shape.k);
    });

}  // namespace kai::test
