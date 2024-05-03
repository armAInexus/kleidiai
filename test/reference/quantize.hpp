//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "test/common/data_type.hpp"

namespace kai::test {

/// Quantization method.
enum class QuantizationMethod : uint32_t {
    PER_MATRIX,  ///< Per-matrix, i.e. one quantization scale and zero point for the entire matrix.
    PER_ROW,     ///< Per-row, i.e. one quantization scale and zero point for each row.
};

/// Calculates the quantization information for 8-bit signed asymmetric type from the value range.
///
/// @param[in] min_value Minimum value.
/// @param[in] max_value Maximum value.
///
/// @return The scale and zero point.
std::tuple<float, int32_t> get_qai8_scale_zero_point_from_range(float min_value, float max_value);

/// Quantizes the single-precision floating-point value using 8-bit asymmetric quantization.
///
/// Formula: `q = f / scale + zero_point` where `q` is quantized value and `f` is floating-point value.
///
/// @param[in] value Value to be quantized.
/// @param[in] scale Scale.
/// @param[in] zero_point Zero point.
///
/// @return The quantized value.
int8_t quantize_i8_fp32(float value, float scale, int32_t zero_point);

/// Dequantizes the matrix to floating-point.
///
/// @param[in] data Quantized data buffer.
/// @param[in] scales Quantization scales.
/// @param[in] zero_points (Optional) Quantization zero points.
/// @param[in] src_dt Quantized data type.
/// @param[in] dst_dt Dequantized data type.
/// @param[in] method Quantization method.
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
///
/// @return The dequantized data buffer.
std::vector<uint8_t> dequantize(
    const void* data, const void* scales, const void* zero_points,  //
    DataType src_dt, DataType dst_dt, QuantizationMethod method,    //
    size_t height, size_t width);

}  // namespace kai::test
