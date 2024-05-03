//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/matmul.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "src/kai_common.h"
#include "test/common/compare.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/printer.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/pack.hpp"

namespace kai::test {

// NOLINTBEGIN(misc-non-private-member-variables-in-classes)

/// Matrix multiplication method.
struct MatMulMethod {
    size_t m0;  ///< Block size in M dimension.
    size_t n0;  ///< Block size in N dimension.
    size_t k0;  ///< Block size in K dimension.

    bool lhs_transposed;  ///< LHS matrix is transposed.
    bool rhs_transposed;  ///< RHS matrix is transposed.

    DataFormat dst_format;         ///< Data format of the destination matrix.
    DataFormat lhs_format;         ///< Data format of the LHS matrix.
    DataFormat packed_lhs_format;  ///< Data format of the packed LHS matrix.
    DataFormat rhs_format;         ///< Data format of the RHS matrix.
    DataFormat packed_rhs_format;  ///< Data for mat of the packed RHS matrix.

    /// Gets the offset in bytes of the LHS matrix.
    ///
    /// @param[in] m_idx Coordinate of the matrix in M dimension.
    /// @param[in] stride Row stride in bytes.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t m_idx, size_t stride)> fn_get_lhs_offset;

    /// Gets the size in bytes of the packed LHS matrix.
    ///
    /// @param[in] m Size of the matrix in M dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The size in bytes.
    std::function<size_t(size_t m, size_t k)> fn_get_packed_lhs_size;

    /// Gets the offset in bytes of the packed LHS matrix.
    ///
    /// @param[in] m_idx Coordinate of the matrix in M dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t m_idx, size_t k)> fn_get_packed_lhs_offset;

    /// Preprocesses the LHS matrix.
    ///
    /// @param[in] m Size of the matrix in M dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] lhs LHS matrix data buffer.
    /// @param[in] lhs_row_stride Row stride in bytes of the LHS matrix.
    /// @param[out] packed_lhs Packed LHS matrix data buffer.
    std::function<void(size_t m, size_t k, const void* lhs, size_t lhs_row_stride, void* packed_lhs)> fn_pack_lhs;

    /// Gets the offset in bytes of the RHS matrix.
    ///
    /// @param[in] n_idx Coordinate of the matrix in N dimension.
    /// @param[in] stride Row stride in bytes.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t n_idx, size_t stride)> fn_get_rhs_offset;

    /// Gets the size in bytes of the packed RHS matrix.
    ///
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] block_height Block height.
    /// @param[in] block_width Block width.
    ///
    /// @return The size in bytes.
    std::function<size_t(size_t n, size_t k, size_t block_height, size_t block_width)> fn_get_packed_rhs_size;

    /// Gets the offset in bytes of the packed RHS matrix.
    ///
    /// @param[in] n_idx Coordinate of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] block_height Block height.
    /// @param[in] block_width Block width.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t n_idx, size_t k, size_t block_height, size_t block_width)> fn_get_packed_rhs_offset;

    /// Performs matrix multiplication.
    ///
    /// @param[in] m Size of the matrix in M dimension.
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] lhs_p Packed LHS data buffer.
    /// @param[in] rhs_p Packed RHS data buffer.
    /// @param[out] dst Output data buffer.
    /// @param[in] dst_stride_row Output row stride.
    /// @param[in] dst_stride_col Output column stride.
    /// @param[in] scalar_min Lower bound of the output data.
    /// @param[in] scalar_max Upper bound of the output data.
    std::function<void(
        size_t m, size_t n, size_t k, const void* lhs_p, const void* rhs_p, float* dst, size_t dst_stride_row,
        size_t dst_stride_col, float scalar_min, float scalar_max)>
        fn_main;

    /// Gets a value indicating whether pre-processing the RHS matrix is needed.
    [[nodiscard]] bool is_pack_rhs_needed() const {
        return false;
    }

    /// Preprocesses the RHS matrix.
    ///
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] rhs RHS data buffer.
    /// @param[in] rhs_row_stride RHS row stride.
    /// @param[in] bias Bias data buffer.
    /// @param[in] scale Quantization scales data buffer.
    /// @param[out] packed_rhs Packed RHS data buffer.
    void pack_rhs(
        size_t n, size_t k, const void* rhs, size_t rhs_row_stride, const void* bias, const void* scale,
        void* packed_rhs) const {
        KAI_UNUSED(n);
        KAI_UNUSED(k);
        KAI_UNUSED(rhs);
        KAI_UNUSED(rhs_row_stride);
        KAI_UNUSED(bias);
        KAI_UNUSED(scale);
        KAI_UNUSED(packed_rhs);

        KAI_ERROR("RHS pre-processing is not supported!");
    }
};

// NOLINTEND(misc-non-private-member-variables-in-classes)

/// List of supported matrix multiplication methods.
static const std::array matmul_methods = {
    MatMulMethod{
        .m0 = 4,
        .n0 = 4,
        .k0 = 32,

        .lhs_transposed = false,
        .rhs_transposed = true,

        .dst_format = DataFormat(DataType::FP32),
        .lhs_format = DataFormat(DataType::FP32),
        .packed_lhs_format =
            DataFormat(DataType::QAI8, 4, 8, DataFormat::QuantizationFormat::PER_ROW, DataType::FP32, DataType::I32),
        .rhs_format = DataFormat(DataType::QSU4),
        .packed_rhs_format = DataFormat(
            DataType::QSI4, 4, 32, DataFormat::QuantizationFormat::PER_ROW, DataType::FP32, DataType::I32, 1, 16),

        .fn_get_lhs_offset = nullptr,
        .fn_get_packed_lhs_size = nullptr,
        .fn_get_packed_lhs_offset = nullptr,
        .fn_pack_lhs = nullptr,

        .fn_get_rhs_offset = nullptr,
        .fn_get_packed_rhs_size = nullptr,
        .fn_get_packed_rhs_offset = nullptr,

        .fn_main = nullptr,
    },
};

/// Matrix multiplication shape.
struct MatMulShape {
    size_t m;  ///< LHS height.
    size_t n;  ///< RHS width.
    size_t k;  ///< LHS width and RHS height.
};

/// Matrix multiplication test information.
using MatMulTestParams = std::tuple<MatMulShape, size_t, MatrixPortion>;

/// Prints the test information.
void PrintTo(const MatMulTestParams& param, std::ostream* os) {
    const auto& [shape, method_no, portion] = param;

    *os << "m: " << shape.m << ", n: " << shape.n << ", k: " << shape.k << ", method_no: " << method_no
        << ", portion: { start_row: " << portion.start_row() << ", start_col: " << portion.start_col()
        << ", height: " << portion.height() << ", width: " << portion.width() << "}";
}

/// Matrix multiplication test fixture.
class MatMulTest : public testing::TestWithParam<MatMulTestParams> {
private:
    /// Unique ID: m, n, k, method_id.
    using TestDataId = std::tuple<size_t, size_t, size_t, size_t>;

protected:
    /// Cached test data that is shared between multiple test case.
    struct TestData {
        std::vector<uint8_t> lhs{};             ///< LHS operand.
        std::vector<uint8_t> ref_packed_lhs{};  ///< Reference packed LHS.
        std::vector<uint8_t> rhs{};             ///< RHS operand.
        std::vector<uint8_t> rhs_scales{};      ///< RHS per-row quantization scales.
        std::vector<uint8_t> ref_packed_rhs{};  ///< Reference packed RHS.
        std::vector<uint8_t> ref_dst{};         ///< Reference output.
    };

    /// Gets the test data for the current test case.
    static const TestData& test_data() {
        const auto& [info, method_no, portion] = GetParam();
        const TestDataId data_id{info.m, info.n, info.k, method_no};

        // If the test data is already available, returns it.
        const auto data_it = _data.find(data_id);

        if (data_it != _data.end()) {
            return data_it->second;
        }

        // Generates the test data.
        const auto& method = matmul_methods.at(method_no);

        const auto lhs_h = method.lhs_transposed ? info.k : info.m;
        const auto lhs_w = method.lhs_transposed ? info.m : info.k;
        auto lhs = fill_matrix_random(lhs_h, lhs_w, method.lhs_format, 0);
        auto ref_packed_lhs =
            pack(method.packed_lhs_format, lhs.data(), nullptr, nullptr, method.lhs_format, lhs_h, lhs_w);

        const auto rhs_h = method.rhs_transposed ? info.n : info.k;
        const auto rhs_w = method.rhs_transposed ? info.k : info.n;
        auto rhs = fill_matrix_random(rhs_h, rhs_w, method.rhs_format, 1);

        std::vector<uint8_t> rhs_scales;
        if (data_type_is_quantized(method.rhs_format.data_type()) &&
            method.rhs_format.quantization_format() == DataFormat::QuantizationFormat::NONE) {
            rhs_scales = fill_matrix_random(rhs_h, 1, DataFormat(DataType::FP32), 2);
        }

        auto packed_rhs = matmul_pack_rhs(
            rhs.data(), !rhs_scales.empty() ? rhs_scales.data() : nullptr, nullptr, method.rhs_format,
            method.packed_rhs_format, rhs_h, rhs_w);

        KAI_ASSUME(method.lhs_format.is_raw());
        KAI_ASSUME(method.rhs_format.is_raw());
        KAI_ASSUME(method.dst_format.is_raw());
        auto ref_dst = matmul(
            lhs.data(), nullptr, nullptr, method.lhs_format.data_type(),            //
            rhs.data(), rhs_scales.data(), nullptr, method.rhs_format.data_type(),  //
            method.dst_format.data_type(),                                          //
            info.m, info.n, info.k, method.lhs_transposed, method.rhs_transposed);

        const auto& data = _data[data_id] = {
            .lhs = std::move(lhs),
            .ref_packed_lhs = std::move(ref_packed_lhs),
            .rhs = std::move(rhs),
            .rhs_scales = std::move(rhs_scales),
            .ref_packed_rhs = std::move(packed_rhs),
            .ref_dst = std::move(ref_dst),
        };

        return data;
    }

private:
    static std::map<TestDataId, TestData> _data;
};

std::map<MatMulTest::TestDataId, MatMulTest::TestData> MatMulTest::_data;

/// Tests the LHS packing kernel.
TEST_P(MatMulTest, PackedLhs) {
    const auto& [info, method_no, portion] = GetParam();
    const auto& data = test_data();
    const auto& method = matmul_methods.at(method_no);

    if (method.fn_pack_lhs == nullptr) {
        GTEST_SKIP();
    }

    const auto lhs_h = method.lhs_transposed ? info.k : info.m;
    const auto lhs_w = method.lhs_transposed ? info.m : info.k;

    const auto rect = portion.compute_portion(
        lhs_h, lhs_w, method.packed_lhs_format.scheduler_block_height(lhs_h),
        method.packed_lhs_format.scheduler_block_width(lhs_w));

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP();
    }

    const auto ref_lhs_row_stride = method.lhs_format.default_row_stride(lhs_w);

    const auto packed_lhs_size = method.fn_get_packed_lhs_size(info.m, info.k);
    const auto ref_packed_lhs_size = method.packed_lhs_format.default_size_in_bytes(lhs_h, lhs_w);
    ASSERT_EQ(packed_lhs_size, ref_packed_lhs_size);

    const auto lhs_offset = method.fn_get_lhs_offset(rect.start_row(), ref_lhs_row_stride);
    const auto ref_lhs_offset = method.lhs_format.default_offset_in_bytes(rect.start_row(), 0, lhs_w);
    ASSERT_EQ(lhs_offset, ref_lhs_offset);

    const auto packed_lhs_offset = method.fn_get_packed_lhs_offset(rect.start_row(), info.k);
    const auto ref_packed_lhs_offset = method.packed_lhs_format.default_offset_in_bytes(rect.start_row(), 0, lhs_w);
    ASSERT_EQ(packed_lhs_offset, ref_packed_lhs_offset);

    std::vector<uint8_t> packed_lhs;
    packed_lhs.resize(packed_lhs_size);
    method.fn_pack_lhs(
        rect.height(), rect.width(), data.lhs.data() + lhs_offset, ref_lhs_row_stride,
        packed_lhs.data() + packed_lhs_offset);

    DefaultMismatchHandler handler(0, 0.0001, 0, 0.001);
    const auto success =
        compare(packed_lhs.data(), data.ref_packed_lhs.data(), method.packed_lhs_format, lhs_h, lhs_w, rect, handler);
    ASSERT_TRUE(success);
}

/// Tests the RHS packing kernel.
TEST_P(MatMulTest, PackedRhs) {
    const auto& [info, method_no, portion] = GetParam();
    const auto& data = test_data();
    const auto& method = matmul_methods.at(method_no);

    if (!method.is_pack_rhs_needed()) {
        GTEST_SKIP();
    }

    const auto rhs_h = method.rhs_transposed ? info.n : info.k;
    const auto rhs_w = method.rhs_transposed ? info.k : info.n;

    const auto rect = portion.compute_portion(
        rhs_h, rhs_w, method.packed_rhs_format.scheduler_block_height(rhs_h),
        method.packed_rhs_format.scheduler_block_width(rhs_w));

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP();
    }

    const auto ref_rhs_row_stride = method.rhs_format.default_row_stride(rhs_w);

    const auto rhs_offset = method.fn_get_rhs_offset(rect.start_row(), ref_rhs_row_stride);
    const auto ref_rhs_offset = method.rhs_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), rhs_w);
    ASSERT_EQ(rhs_offset, ref_rhs_offset);

    const auto packed_rhs_size = method.fn_get_packed_rhs_size(
        rhs_h, rhs_w, method.packed_rhs_format.block_height(), method.packed_rhs_format.block_width());
    const auto ref_packed_rhs_size = method.packed_rhs_format.default_size_in_bytes(rhs_h, rhs_w);
    ASSERT_EQ(packed_rhs_size, ref_packed_rhs_size);

    const auto packed_rhs_offset = method.fn_get_packed_rhs_offset(
        rect.start_row(), rhs_w, method.packed_rhs_format.block_height(), method.packed_rhs_format.block_width());
    const auto ref_packed_rhs_offset =
        method.packed_rhs_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), rhs_w);
    ASSERT_EQ(packed_rhs_offset, ref_packed_rhs_offset);

    const auto ref_rhs_scales_offset =
        rect.start_row() * data_type_size_in_bits(method.packed_rhs_format.scale_data_type()) / 8;

    std::vector<uint8_t> packed_rhs;
    packed_rhs.resize(packed_rhs_size);

    method.pack_rhs(
        rect.height(), rect.width(), data.rhs.data() + rhs_offset, ref_rhs_row_stride, nullptr,
        !data.rhs_scales.empty() ? data.rhs_scales.data() + ref_rhs_scales_offset : nullptr,
        packed_rhs.data() + packed_rhs_offset);

    DefaultMismatchHandler handler(0, 0.0001, 0, 0.001);
    const auto success =
        compare(packed_rhs.data(), data.ref_packed_rhs.data(), method.packed_rhs_format, rhs_h, rhs_w, rect, handler);
    ASSERT_TRUE(success);
}

/// Tests the output.
TEST_P(MatMulTest, Output) {
    const auto& [info, method_no, portion] = GetParam();
    const auto& data = test_data();
    const auto& method = matmul_methods.at(method_no);

    if (method.fn_main == nullptr) {
        GTEST_SKIP();
    }

    const auto rect = portion.compute_portion(info.m, info.n, method.m0, method.n0);

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP();
    }

    const auto ref_dst_row_stride = method.dst_format.default_row_stride(info.n);
    const auto ref_dst_col_stride = data_type_size_in_bits(method.dst_format.data_type()) / 8;

    const auto ref_packed_lhs_offset = method.packed_lhs_format.default_offset_in_bytes(
        method.lhs_transposed ? 0 : rect.start_row(), method.lhs_transposed ? rect.start_row() : 0,
        method.lhs_transposed ? info.m : info.k);
    const auto ref_packed_rhs_offset = method.packed_rhs_format.default_offset_in_bytes(
        method.rhs_transposed ? rect.start_col() : 0, method.rhs_transposed ? 0 : rect.start_col(),
        method.rhs_transposed ? info.k : info.n);
    const auto ref_dst_offset = method.dst_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), info.n);

    std::vector<uint8_t> dst;
    dst.resize(method.dst_format.default_size_in_bytes(info.m, info.n));

    method.fn_main(
        rect.height(), rect.width(), info.k, data.ref_packed_lhs.data() + ref_packed_lhs_offset,
        data.ref_packed_rhs.data() + ref_packed_rhs_offset, reinterpret_cast<float*>(dst.data() + ref_dst_offset),
        ref_dst_row_stride, ref_dst_col_stride, std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());

    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    const auto success = compare(dst.data(), data.ref_dst.data(), method.dst_format, info.m, info.n, rect, handler);
    ASSERT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest,
    testing::Combine(
        testing::Values(
            MatMulShape{4, 4, 32},  //
            MatMulShape{12, 16, 64}),
        testing::Range<size_t>(0, matmul_methods.size()),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),        // Full matrix.
            MatrixPortion(0, 0, 0.25, 0.25),  // Top-left corner.
            MatrixPortion(0.75, 0.75, 1, 1)   // Bottom-right corner.
            )));

}  // namespace kai::test
