//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Gets n step value.
///
/// The starting row index must be divisible by `n_step`.
///
/// @return The n step value.
size_t kai_get_n_step_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon(void);

/// Gets the offset in bytes to the data element in the RHS matrix buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon(size_t n_idx);

/// Gets the offset in bytes to the data element in the bias buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the data element.
size_t kai_get_bias_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon(size_t n_idx);

/// Gets the offset in bytes to the data element in the packed RHS buffer.
///
/// @param[in] n_idx Row index.
/// @param[in] k Number of columns.
/// @param[in] nr Block size in N dimension.
/// @param[in] kr Block size in K dimension.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_packed_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon(size_t n_idx, size_t k, size_t nr, size_t kr);

/// Gets the size in bytes of the packed RHS buffer.
///
/// @param[in] n Number of rows.
/// @param[in] k Number of columns.
/// @param[in] nr Block size in N dimension.
/// @param[in] kr Block size in K dimension.
///
/// @return The size in bytes of the packed RHS buffer.
size_t kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon(size_t n, size_t k, size_t nr, size_t kr);

/// Runs the RHS packing function for matrix multiplication.
///
/// The pointer of each buffers (RHS, bias and packed RHS) needs to be added with offset
/// calculated using the following functions:
///
///   * RHS: @ref kai_get_rhs_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon.
///   * Bias: @ref kai_get_bias_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon.
///   * Output: @ref kai_get_rhs_packed_offset_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon.
///
/// @param[in] n Number of columns of the output matrix.
/// @param[in] k Common dimension between the LHS and RHS matrix.
/// @param[in] nr Block size in N dimension.
/// @param[in] kr Block size in K dimension.
/// @param[in] sr Number of kr splits. It must be 1.
/// @param[in] rhs_stride Row stride in bytes of the RHS matrix.
/// @param[in] rhs RHS matrix data buffer.
/// @param[in] bias Bias matrix data buffer.
/// @param[out] rhs_packed Packed RHS matrix.
void kai_run_rhs_quant_pack_kxn_bf16pbiasf32_f32_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const float* rhs, const float* bias,
    void* rhs_packed);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
