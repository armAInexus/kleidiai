//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <arm_neon.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <limits>
#include <map>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/printer.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pack.hpp"

// matmul_clamp_f32_bf16p_bf16p
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon.h"
namespace kai::test {

/// List of supported matrix multiplication methods.
namespace {

/// Adapters for using packing and matmul functions with the unified interface of the test framework
inline void kai_run_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon_adapter(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_UNUSED(num_groups);
    KAI_UNUSED(scale);
    KAI_UNUSED(extra_bytes);
    KAI_UNUSED(params);

    kai_run_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon(
        n, k, nr, kr, sr, rhs_stride, reinterpret_cast<const float*>(rhs), reinterpret_cast<const float*>(bias),
        rhs_packed);
}

const std::array matmul_methods = {
    MatMulMethod{
        .name = "matmul_nt_nt_f32_bf16p_bf16p_8x12_neon_mla",

        .m0 = 8,
        .n0 = 12,
        .k0 = 4,

        .lhs_transposed = false,
        .rhs_transposed = false,

        .dst_format = DataFormat(DataType::FP32),
        .lhs_format = DataFormat(DataType::FP32),
        .packed_lhs_format =
            DataFormat(DataType::BF16, 8, 4, DataFormat::PackFormat::NONE, DataType::FP32, DataType::UNKNOWN, 8, 4),
        .rhs_format = DataFormat(DataType::FP32),
        .packed_rhs_format = DataFormat(
            DataType::BF16, 12, 4, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 12, 4),
        .bias_format = DataFormat(DataType::FP32),
        .fn_is_supported = cpu_has_bf16,

        .fn_get_mr = kai_get_mr_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_nr = kai_get_nr_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_kr = kai_get_kr_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_sr = kai_get_sr_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,

        .fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_pack_rhs_n_step = kai_get_n_step_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon,
        .fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,

        .fn_get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_bf16p_f32_neon,
        .fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p_f32_neon,
        .fn_get_packed_lhs_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_pack_lhs = kai_run_lhs_quant_pack_bf16p_f32_neon,

        .fn_get_rhs_offset = kai_get_rhs_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon,
        .fn_get_packed_rhs_size = nullptr,
        .fn_get_packed_rhs_size_generic_block_size = kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon,
        .fn_get_pack_rhs_packed_rhs_offset = nullptr,
        .fn_get_main_packed_rhs_offset =
            kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_pack_rhs = kai_run_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon_adapter,

        .fn_get_bias_offset = kai_get_bias_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon,

        .fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,

        .fn_matmul_f32_bf16p_bf16p = kai_run_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
    },
    MatMulMethod{
        .name = "matmul_nt_nt_f32_bf16p_bf16p_8x12_neon_mla_opt_bias",

        .m0 = 8,
        .n0 = 12,
        .k0 = 4,

        .lhs_transposed = false,
        .rhs_transposed = false,

        .dst_format = DataFormat(DataType::FP32),
        .lhs_format = DataFormat(DataType::FP32),
        .packed_lhs_format =
            DataFormat(DataType::BF16, 8, 4, DataFormat::PackFormat::NONE, DataType::FP32, DataType::UNKNOWN, 8, 4),
        .rhs_format = DataFormat(DataType::FP32),
        .packed_rhs_format = DataFormat(
            DataType::BF16, 12, 4, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32, DataType::UNKNOWN, 12, 4),
        .bias_format = DataFormat(DataType::UNKNOWN),
        .fn_is_supported = cpu_has_bf16,

        .fn_get_mr = kai_get_mr_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_nr = kai_get_nr_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_kr = kai_get_kr_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_sr = kai_get_sr_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,

        .fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_pack_rhs_n_step = kai_get_n_step_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon,
        .fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,

        .fn_get_lhs_offset = kai_get_lhs_offset_lhs_quant_pack_bf16p_f32_neon,
        .fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p_f32_neon,
        .fn_get_packed_lhs_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_pack_lhs = kai_run_lhs_quant_pack_bf16p_f32_neon,

        .fn_get_rhs_offset = kai_get_rhs_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon,
        .fn_get_packed_rhs_size = nullptr,
        .fn_get_packed_rhs_size_generic_block_size = kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon,
        .fn_get_pack_rhs_packed_rhs_offset = nullptr,
        .fn_get_main_packed_rhs_offset =
            kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_pack_rhs = kai_run_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon_adapter,

        .fn_get_bias_offset = kai_get_bias_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon,

        .fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
        .fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,

        .fn_matmul_f32_bf16p_bf16p = kai_run_matmul_clamp_f32_bf16p_bf16p12x4biasf32_8x12x4_neon_mmla,
    }};
}  // namespace

/// Matrix multiplication test fixture.
class MatMulTestBf16 : public testing::TestWithParam<MatMulTestParams> {
private:
    /// Unique ID: m, n, k
    using TestDataId = std::tuple<size_t, size_t, size_t, std::string_view>;

protected:
    /// Cached test data that is shared between multiple test case.
    struct TestData {
        std::vector<uint8_t> lhs{};             ///< LHS operand.
        std::vector<uint8_t> ref_packed_lhs{};  ///< Reference packed LHS.
        std::vector<uint8_t> rhs{};             ///< RHS operand.
        std::vector<uint8_t> rhs_scales{};      ///< RHS per-row quantization scales.
        std::vector<uint8_t> bias{};            ///< Bias.
        std::vector<uint8_t> ref_packed_rhs{};  ///< Reference packed RHS.
        std::vector<uint8_t> ref_dst{};         ///< Reference output.
    };

    /// Gets the test data for the current test case.
    static const TestData& test_data() {
        const auto& [method, info, portion] = GetParam();
        const TestDataId data_id{info.m, info.n, info.k, method.name};

        // If the test data is already available, returns it.
        const auto data_it = _data.find(data_id);

        if (data_it != _data.end()) {
            return data_it->second;
        }

        // Generates the test data.
        const auto has_lhs_pack = method.packed_lhs_format.data_type() != DataType::UNKNOWN;
        const auto has_rhs_pack = method.packed_rhs_format.data_type() != DataType::UNKNOWN;
        const auto has_bias = method.bias_format.data_type() != DataType::UNKNOWN;

        const auto lhs_h = method.lhs_transposed ? info.k : info.m;
        const auto lhs_w = method.lhs_transposed ? info.m : info.k;
        auto lhs = fill_matrix_random(lhs_h, lhs_w, method.lhs_format, 0);
        std::vector<uint8_t> ref_packed_lhs;

        if (has_lhs_pack) {
            ref_packed_lhs =
                pack(method.packed_lhs_format, lhs.data(), nullptr, nullptr, method.lhs_format, lhs_h, lhs_w);
        }

        const auto rhs_h = method.rhs_transposed ? info.n : info.k;
        const auto rhs_w = method.rhs_transposed ? info.k : info.n;
        auto rhs = fill_matrix_random(rhs_h, rhs_w, method.rhs_format, 1);

        std::vector<uint8_t> rhs_scales;
        if (data_type_is_quantized(method.rhs_format.data_type()) &&
            method.rhs_format.pack_format() == DataFormat::PackFormat::NONE) {
            rhs_scales = fill_matrix_random(rhs_h, 1, DataFormat(DataType::FP32), 2);
        }

        const auto bias_h = 1;
        const auto bias_w = info.n;
        std::vector<uint8_t> bias;

        if (has_bias) {
            bias = fill_matrix_random(bias_h, bias_w, method.bias_format, 3);
        }

        constexpr size_t nr = 12;
        constexpr size_t kr = 4;

        std::vector<uint8_t> packed_rhs;
        packed_rhs.resize(method.fn_get_packed_rhs_size_generic_block_size(rhs_w, rhs_h, nr, kr));

        if (has_rhs_pack) {
            const auto ref_rhs_row_stride = method.rhs_format.default_row_stride(rhs_w);
            method.pack_rhs(
                info.n, info.k, rhs.data(), ref_rhs_row_stride, has_bias ? bias.data() : nullptr, nullptr,
                packed_rhs.data());
        }

        KAI_ASSUME(method.lhs_format.is_raw());
        KAI_ASSUME(method.rhs_format.is_raw());
        KAI_ASSUME(method.dst_format.is_raw());

        auto ref_dst = matmul(
            lhs.data(), nullptr, nullptr, method.lhs_format.data_type(),                         //
            rhs.data(), rhs_scales.data(), nullptr, method.rhs_format.data_type(),               //
            has_bias ? bias.data() : nullptr, nullptr, nullptr, method.bias_format.data_type(),  //
            method.dst_format.data_type(),                                                       //
            info.m, info.n, info.k, method.lhs_transposed, method.rhs_transposed);

        const auto& data = _data[data_id] = {
            .lhs = std::move(lhs),
            .ref_packed_lhs = std::move(ref_packed_lhs),
            .rhs = std::move(rhs),
            .rhs_scales = std::move(rhs_scales),
            .bias = std::move(bias),
            .ref_packed_rhs = std::move(packed_rhs),
            .ref_dst = std::move(ref_dst),
        };

        return data;
    }

private:
    // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
    static std::map<TestDataId, TestData> _data;
    // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::map<MatMulTestBf16::TestDataId, MatMulTestBf16::TestData> MatMulTestBf16::_data;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/// Tests the output.
TEST_P(MatMulTestBf16, Output) {
    const auto& [method, info, portion] = GetParam();
    const auto& data = test_data();

    if (method.fn_is_supported && !method.fn_is_supported()) {
        GTEST_SKIP();
    }

    if (!method.has_main_kernel()) {
        GTEST_SKIP();
    }

    const auto m_step = method.fn_get_main_m_step();
    ASSERT_EQ(m_step, method.m0);

    const auto n_step = method.fn_get_main_n_step();
    ASSERT_EQ(n_step, method.n0);

    const auto rect = portion.compute_portion(info.m, info.n, method.m0, method.n0);

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP();
    }

    const size_t lhs_w = info.k;
    const size_t rhs_w = rect.width();
    const size_t bias_w = info.n;
    const size_t dst_w = info.n;
    const bool has_bias = (data.bias.size() > 0);

    const auto lhs_start_row = method.lhs_transposed ? 0 : rect.start_row();
    const auto lhs_stride = method.lhs_format.default_row_stride(lhs_w);

    std::vector<uint8_t> lhs_data;
    const size_t lhs_packed_size = method.fn_get_packed_lhs_size(info.m, info.k, method.m0, method.k0, 1 /* sr */);
    lhs_data.resize(lhs_packed_size);

    uintptr_t lhs_offset = method.fn_get_lhs_offset(lhs_start_row, lhs_stride);
    uintptr_t lhs_packed_offset = method.fn_get_packed_lhs_offset(lhs_start_row, info.k);

    KAI_UNUSED(lhs_offset);
    method.fn_pack_lhs(
        rect.height(), info.k, method.m0, method.k0, 1 /* sr */, 0 /* m_idx_start */, data.lhs.data() + lhs_offset,
        lhs_stride, lhs_data.data() + lhs_packed_offset);

    const auto rhs_stride = method.rhs_format.default_row_stride(info.n);

    std::vector<uint8_t> rhs_data;
    const size_t rhs_packed_size =
        method.fn_get_packed_rhs_size_generic_block_size(info.n, info.k, method.n0, method.k0);
    rhs_data.resize(rhs_packed_size);

    const auto packed_rhs_start_row = rect.start_col();
    const auto packed_rhs_start_col = 0;

    uintptr_t rhs_offset = method.fn_get_rhs_offset(rect.start_col());
    uintptr_t rhs_packed_offset = method.fn_get_main_packed_rhs_offset(packed_rhs_start_row, info.k);
    const auto ref_rhs_packed_offset =
        method.packed_rhs_format.default_offset_in_bytes(packed_rhs_start_row, packed_rhs_start_col, info.k);

    ASSERT_EQ(rhs_packed_offset, ref_rhs_packed_offset);

    uintptr_t bias_offset = sizeof(float) * rect.start_col();

    method.fn_pack_rhs(
        1,  // num_groups
        rhs_w, info.k, method.n0, method.k0,
        1,  // sr
        rhs_stride, data.rhs.data() + rhs_offset, has_bias ? data.bias.data() + bias_offset : nullptr,
        NULL,  // Scale
        rhs_data.data() + rhs_packed_offset, 0, NULL);

    if (has_bias) {
        const auto ref_bias_offset = method.bias_format.default_offset_in_bytes(0, rect.start_col(), bias_w);
        ASSERT_EQ(ref_bias_offset, bias_offset);
    }

    const auto dst_stride = method.dst_format.default_row_stride(dst_w);
    const auto dst_offset = method.fn_get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = method.dst_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), dst_w);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    const auto dst_size = method.fn_get_dst_size(info.m, info.n);
    const auto ref_dst_size = method.dst_format.default_size_in_bytes(info.m, info.n);
    ASSERT_EQ(dst_size, ref_dst_size);

    std::vector<uint8_t> dst;
    dst.resize(dst_size);
    method.main_kernel(
        rect.height(), rect.width(), info.k, lhs_data.data() + lhs_packed_offset, rhs_data.data() + rhs_packed_offset,
        NULL, dst.data() + dst_offset, lhs_stride, rhs_stride, dst_stride, -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());

    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    const auto success = compare(dst.data(), data.ref_dst.data(), method.dst_format, info.m, info.n, rect, handler);

    ASSERT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTestBf16,
    testing::Combine(
        testing::ValuesIn(matmul_methods),
        testing::Values(
            MatMulShape{3, 7, 3},        // Smaller than block size
            MatMulShape{12, 8, 4},       // Same block size
            MatMulShape{1, 1, 1023},     // Long K
            MatMulShape{1013, 1, 5},     // Long M
            MatMulShape{2, 1013, 6},     // Long N
            MatMulShape{13, 33, 23},     //
            MatMulShape{93, 57, 89},     //
            MatMulShape{256, 256, 256},  // Nice shapes
            MatMulShape{257, 113, 373}   // Prime numbers
            ),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),         // Full matrix.
            MatrixPortion(0, 0, 0.25, 0.25),   // Top-left corner.
            MatrixPortion(0.75, 0.75, 1, 1),   // Bottom-right corner.
            MatrixPortion(0.75, 0, 1, 1),      // Partial rows
            MatrixPortion(0.4, 0.5, 0.6, 0.8)  // Somewhere Middle
            )),
    testing::PrintToStringParamName());
}  // namespace kai::test
