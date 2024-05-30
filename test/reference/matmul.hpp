//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "test/common/data_type.hpp"

namespace kai::test {

class DataFormat;

/// Packs the RHS operand of matrix multiplication.
///
/// @param[in] data Data buffer.
/// @param[in] scales (Optional) Quantization scales.
/// @param[in] zero_points (Optional) Quantization zero points.
/// @param[in] src_format Data format of the RHS matrix.
/// @param[in] dst_format Data format of the packed RHS matrix.
/// @param[in] n Number of non-transposed columns.
/// @param[in] k Number of non-transposed rows.
/// @param[in] transposing Perform transpose then pack.
///
/// @return The packed RHS matrix.
std::vector<uint8_t> matmul_pack_rhs(
    const void* data, const void* scales, const void* zero_points, const DataFormat& src_format,
    const DataFormat& dst_format, size_t n, size_t k, bool transposing);

/// Matrix multiplication.
///
/// @param[in] lhs LHS operand data.
/// @param[in] lhs_scales (Optional) LHS operand quantization scales.
/// @param[in] lhs_zero_points (Optional) LHS operand quantization zero point.
/// @param[in] lhs_dt LHS operand data type.
/// @param[in] rhs RHS operand data.
/// @param[in] rhs_scales (Optional) RHS operand quantization scales.
/// @param[in] rhs_zero_points (Optional) RHS operand quantization zero point.
/// @param[in] rhs_dt RHS operand data type.
/// @param[in] bias Bias operand data.
/// @param[in] bias_scales (Optional) Bias operand quantization scales.
/// @param[in] bias_zero_points (Optional) Bias operand quantization zero point.
/// @param[in] bias_dt Bias operand data type.
/// @param[in] dst Output data.
/// @param[in] dst_scales (Optional) Output quantization scales.
/// @param[in] dst_zero_points (Optional) Output quantization zero point.
/// @param[in] dst_dt Output data type.
/// @param[in] m Output height.
/// @param[in] n Output width.
/// @param[in] k Non-transposed LHS width and non-transposed RHS height.
/// @param[in] lhs_transposed `true` if LHS operand is transposed.
/// @param[in] rhs_transposed `true` if RHS operand is transposed.
///
/// @return The result data buffer.
std::vector<uint8_t> matmul(
    const void* lhs, const void* lhs_scales, const void* lhs_zero_points, DataType lhs_dt,      //
    const void* rhs, const void* rhs_scales, const void* rhs_zero_points, DataType rhs_dt,      //
    const void* bias, const void* bias_scales, const void* bias_zero_points, DataType bias_dt,  //
    DataType dst_dt,                                                                            //
    size_t m, size_t n, size_t k,                                                               //
    bool lhs_transposed, bool rhs_transposed);

}  // namespace kai::test
