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

namespace kai::test {

/// Quantization method.
enum class QuantizationMethod : uint32_t {
    PER_MATRIX,  ///< Per-matrix, i.e. one quantization scale and zero point for the entire matrix.
    PER_ROW,     ///< Per-row, i.e. one quantization scale and zero point for each row.
};

/// Quantizes each subblock of the matrix using symmetric quantization method.
///
/// The input matrix is divided into quantization blocks of the same size.
///
/// The height of the block does not effect the behavior of this function hence it is omitted
/// from the function arguments and the figures below.
///
/// ```
/// Quantization blocks -------+
///          |                 |
///          |                 |
///          v                 v
/// +-----------------+-----------------+----- ...
/// | f00 f01 f02 f03 | f04 f05 f06 f07 | ........
/// | f10 f11 f12 f13 | f14 f15 f16 f17 | ........
/// | f20 f21 f22 f23 | f24 f25 f26 f27 | ........
/// | f30 f31 f32 f33 | f34 f35 f36 f37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
/// ```
///
/// Each row of the quantization block is quantized individually.
///
/// ```
/// Floating-point data           Scale          Quantized data
/// +-----------------+          +-----+       +-----------------+
/// | f00 f01 f02 f03 | -------> | s00 |       | q00 q01 q02 q03 |
/// | f10 f11 f12 f13 | -------> | s10 |       | q10 q11 q12 q13 |
/// | f20 f21 f22 f23 | -------> | s20 |       | q20 q21 q22 q23 |
/// | f30 f31 f32 f33 | -------> | s30 |       | q30 q31 q32 q33 |
/// | ............... |          | ... |       | ............... |
/// : ............... :          : ... :       : ............... :
/// ```
///
/// The quantization scale and quantized data are stored in separate buffers.
///
/// ```
/// Quantized data matrix:
///
/// +-----------------+-----------------+----- ...
/// | q00 q01 q02 q03 | q04 q05 q06 q07 | ........
/// | q10 q11 q12 q13 | q14 q15 q16 q17 | ........
/// | q20 q21 q22 q23 | q24 q25 q26 q27 | ........
/// | q30 q31 q32 q33 | q34 q35 q36 q37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
///
/// Quantization scale matrix:
///
/// +-----+-----+-- ...
/// | s00 | s01 | .....
/// | s10 | s11 | .....
/// | s20 | s21 | .....
/// | s30 | s31 | .....
/// | ... | ... | .....
/// : ... : ... : .....
/// ```
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
///
/// @param[in] src The input matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantized data matrix and the quantization scale matrix.
template <typename SrcType, typename DstType, typename ScaleType>
std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> quantize_symmetric_per_block(
    const void* src, size_t height, size_t width, size_t quant_width);

/// Quantizes each subblock of the matrix using asymmetric quantization method.
///
/// The input matrix is divided into quantization blocks of the same size.
///
/// The height of the block does not effect the behavior of this function hence it is omitted
/// from the function arguments and the figures below.
///
/// ```
/// Quantization blocks -------+
///          |                 |
///          |                 |
///          v                 v
/// +-----------------+-----------------+----- ...
/// | f00 f01 f02 f03 | f04 f05 f06 f07 | ........
/// | f10 f11 f12 f13 | f14 f15 f16 f17 | ........
/// | f20 f21 f22 f23 | f24 f25 f26 f27 | ........
/// | f30 f31 f32 f33 | f34 f35 f36 f37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
/// ```
///
/// Each row of the quantization block is quantized individually.
///
/// ```
/// Floating-point data           Scale       Zero point       Quantized data
/// +-----------------+          +-----+       +-----+       +-----------------+
/// | f00 f01 f02 f03 | -------> | s00 |       | z00 |       | q00 q01 q02 q03 |
/// | f10 f11 f12 f13 | -------> | s10 |       | z10 |       | q10 q11 q12 q13 |
/// | f20 f21 f22 f23 | -------> | s20 |       | z20 |       | q20 q21 q22 q23 |
/// | f30 f31 f32 f33 | -------> | s30 |       | z30 |       | q30 q31 q32 q33 |
/// | ............... |          | ... |       | ... |       | ............... |
/// : ............... :          : ... :       : ... :       : ............... :
/// ```
///
/// The quantization scales, zero points quantized data are stored in separate buffers.
///
/// ```
/// Quantized data matrix:
///
/// +-----------------+-----------------+----- ...
/// | q00 q01 q02 q03 | q04 q05 q06 q07 | ........
/// | q10 q11 q12 q13 | q14 q15 q16 q17 | ........
/// | q20 q21 q22 q23 | q24 q25 q26 q27 | ........
/// | q30 q31 q32 q33 | q34 q35 q36 q37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
///
/// Quantization scale matrix:
///
/// +-----+-----+-- ...
/// | s00 | s01 | .....
/// | s10 | s11 | .....
/// | s20 | s21 | .....
/// | s30 | s31 | .....
/// | ... | ... | .....
/// : ... : ... : .....
/// ```
///
/// Quantization zero point matrix:
///
/// +-----+-----+-- ...
/// | z00 | z01 | .....
/// | z10 | z11 | .....
/// | z20 | z21 | .....
/// | z30 | z31 | .....
/// | ... | ... | .....
/// : ... : ... : .....
/// ```
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
/// @tparam ZeroPointType The data type of the quantization zero points (must be integer).
///
/// @param[in] src The input matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantized data matrix, the scale matrix and the zero point matrix.
template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, std::vector<uint8_t>> quantize_asymmetric_per_block(
    const void* src, size_t height, size_t width, size_t quant_width);

}  // namespace kai::test
