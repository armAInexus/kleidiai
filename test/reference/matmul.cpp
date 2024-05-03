//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/matmul.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/kai_common.h"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/printer.hpp"
#include "test/reference/binary_elementwise.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/reduce.hpp"

namespace kai::test {

namespace {

/// Matrix multiplication.
///
/// @tparam T Data type.
///
/// @param[in] lhs LHS operand data buffer.
/// @param[in] rhs RHS operand data buffer.
/// @param[in] m Output height.
/// @param[in] n Output width.
/// @param[in] k Non-transposed LHS width and non-transposed RHS height.
/// @param[in] lhs_transposed `true` if LHS operand is transposed.
/// @param[in] rhs_transposed `true` if RHS operand is transposed.
///
/// @return The result data buffer.
template <typename T>
std::vector<uint8_t> matmul_any_type(
    const void* lhs, const void* rhs,  //
    size_t m, size_t n, size_t k,      //
    bool lhs_transposed, bool rhs_transposed) {
    const auto lhs_m_stride = lhs_transposed ? 1 : k;
    const auto lhs_k_stride = lhs_transposed ? m : 1;

    const auto rhs_n_stride = rhs_transposed ? k : 1;
    const auto rhs_k_stride = rhs_transposed ? 1 : n;

    std::vector<uint8_t> dst;
    dst.resize(m * n * size_in_bits<T> / 8);
    KAI_ASSUME(n * size_in_bits<T> % 8 == 0);

    for (size_t im = 0; im < m; ++im) {
        for (size_t in = 0; in < n; ++in) {
            T acc{0};

            for (size_t ik = 0; ik < k; ++ik) {
                const auto lhs_value = read_array<T>(lhs, im * lhs_m_stride + ik * lhs_k_stride);
                const auto rhs_value = read_array<T>(rhs, in * rhs_n_stride + ik * rhs_k_stride);
                acc += lhs_value * rhs_value;
            }

            write_array<T>(dst.data(), im * n + in, acc);
        }
    }

    return dst;
}

}  // namespace

std::vector<uint8_t> matmul_pack_rhs(
    const void* data, const void* scales, const void* zero_points, const DataFormat& src_format,
    const DataFormat& dst_format, size_t height, size_t width) {
    const auto src_dt = src_format.data_type();
    const auto src_qf = src_format.quantization_format();

    const auto dst_dt = dst_format.data_type();
    const auto dst_qf = dst_format.quantization_format();

    std::vector<uint8_t> tmp_data;
    std::vector<uint8_t> tmp_scales;
    std::vector<uint8_t> tmp_zero_points;

    if (src_dt == DataType::QSU4 && src_qf == DataFormat::QuantizationFormat::NONE &&  //
        dst_dt == DataType::QSI4 && dst_qf == DataFormat::QuantizationFormat::PER_ROW) {
        // For this specific RHS format conversion:
        //
        //   * 4-bit data is added by 8.
        //   * Scale is divided by 16.
        //   * Zero point is accumulation of all values in the same row.

        KAI_ASSUME(zero_points == nullptr);
        const int32_t zero_point = 8;
        const uint8_t zero_point_i4 = UInt4::pack_u8(UInt4(zero_point), UInt4(zero_point));
        const int32_t row_zero_point = zero_point * static_cast<int32_t>(width);

        KAI_ASSUME(dst_format.subblock_width() > 0);
        const auto subblock_width_i32 = static_cast<int32_t>(dst_format.subblock_width());
        const auto subblock_width_f = static_cast<float>(dst_format.subblock_width());

        tmp_zero_points = reduce_add(data, src_format, height, width, DataFormat(DataType::I32), 0);
        tmp_zero_points = sub(tmp_zero_points.data(), DataType::I32, height, 1, &row_zero_point, DataType::I32, 1, 1);
        tmp_zero_points =
            mul(tmp_zero_points.data(), DataType::I32, height, 1, &subblock_width_i32, DataType::I32, 1, 1);
        zero_points = tmp_zero_points.data();

        tmp_data = add(data, DataType::QSU4, height, width, &zero_point_i4, DataType::QSU4, 1, 1);
        data = tmp_data.data();

        tmp_scales = div(scales, DataType::FP32, height, 1, &subblock_width_f, DataType::FP32, 1, 1);
        scales = tmp_scales.data();
    }

    return pack(dst_format, data, scales, zero_points, src_format, height, width);
}

std::vector<uint8_t> matmul(
    const void* lhs, const void* lhs_scales, const void* lhs_zero_points, DataType lhs_dt,  //
    const void* rhs, const void* rhs_scales, const void* rhs_zero_points, DataType rhs_dt,  //
    DataType dst_dt,                                                                        //
    size_t m, size_t n, size_t k,                                                           //
    bool lhs_transposed, bool rhs_transposed) {
    const auto lhs_h = lhs_transposed ? k : m;
    const auto lhs_w = lhs_transposed ? m : k;

    const auto rhs_h = rhs_transposed ? n : k;
    const auto rhs_w = rhs_transposed ? k : n;

    std::vector<uint8_t> tmp_lhs;
    std::vector<uint8_t> tmp_rhs;

    if (data_type_is_quantized(lhs_dt)) {
        tmp_lhs = dequantize(
            lhs, lhs_scales, lhs_zero_points, lhs_dt, DataType::FP32, QuantizationMethod::PER_MATRIX, lhs_h, lhs_w);
        lhs = tmp_lhs.data();
    }

    if (data_type_is_quantized(rhs_dt)) {
        tmp_rhs = dequantize(
            rhs, rhs_scales, rhs_zero_points, rhs_dt, DataType::FP32, QuantizationMethod::PER_ROW, rhs_h, rhs_w);
        rhs = tmp_rhs.data();
    }

    KAI_ASSUME(dst_dt == DataType::FP32);
    const auto tmp_dst = matmul_any_type<float>(lhs, rhs, m, n, k, lhs_transposed, rhs_transposed);

    return tmp_dst;
}

}  // namespace kai::test
