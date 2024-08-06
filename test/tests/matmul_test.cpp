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
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/printer.hpp"
#include "test/common/sme.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/pack.hpp"

// matmul_nt_nt_fp16_fp16_fp16_1x16_neon_mla
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla.h"

// matmul_nt_nt_fp16_fp16_fp16_6x16_neon_mla
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

// matmul_nt_nt_fp32_fp32_fp32_2vlx2vl_sme2_mopa
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"

namespace kai::test {

// NOLINTBEGIN(misc-non-private-member-variables-in-classes)

/// Matrix multiplication method.
struct MatMulMethod {
    std::string_view name;  ///< Name of matmul method.

    size_t m0;  ///< Block size in M dimension.
    size_t n0;  ///< Block size in N dimension.

    bool lhs_transposed;  ///< LHS matrix is transposed.
    bool rhs_transposed;  ///< RHS matrix is transposed.

    bool is_sme2;  ///< Test is a sme2 test

    DataFormat dst_format;         ///< Data format of the destination matrix.
    DataFormat lhs_format;         ///< Data format of the LHS matrix.
    DataFormat packed_lhs_format;  ///< Data format of the packed LHS matrix.
    DataFormat rhs_format;         ///< Data format of the RHS matrix.
    DataFormat packed_rhs_format;  ///< Data format of the packed RHS matrix.
    DataFormat bias_format;        ///< Data format of the bias vector.

    /// Gets mr value.
    ///
    /// This is the packing parameter which must be used to pack the LHS matrix (if necessary).
    ///
    /// @return The mr value.
    std::function<size_t(void)> fn_get_mr;

    /// Gets nr value.
    ///
    /// This is the packing parameter which must be used to pack the RHS matrix (if necessary).
    ///
    /// @return The nr value.
    std::function<size_t(void)> fn_get_nr;

    /// Gets kr value.
    ///
    /// This is the packing parameter which must be used to pack the LHS and RHS matrix (if necessary).
    ///
    /// @return The kr value.
    std::function<size_t(void)> fn_get_kr;

    /// Gets sr value.
    ///
    /// This is the packing parameter which must be used to pack the RHS matrix.
    ///
    /// @return The sr value.
    std::function<size_t(void)> fn_get_sr;

    /// Gets m step value for main kernel.
    ///
    /// The starting row index must be divisible by `m_step`.
    ///
    /// @return The m step value.
    std::function<size_t(void)> fn_get_main_m_step;

    /// Gets n step value for RHS packing kernel.
    ///
    /// The starting row index must be divisible by `n_step`.
    ///
    /// @return The n step value.
    std::function<size_t(void)> fn_get_pack_rhs_n_step;

    /// Gets n step value for main kernel.
    ///
    /// The starting column index must be divisible by `n_step`.
    ///
    /// @return The n step value.
    std::function<size_t(void)> fn_get_main_n_step;

    /// Gets the offset in bytes of the LHS matrix.
    ///
    /// @param[in] m_idx Coordinate of the matrix in M dimension.
    /// @param[in] stride Row stride in bytes.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t m_idx, size_t stride)> fn_get_lhs_offset;

    /// Gets the size in bytes of the packed LHS matrix.
    ///
    /// @param[in] m Number of rows in the unpacked LHS matrix.
    /// @param[in] k Number of columns in the unpacked LHS matrix.
    /// @param[in] mr Number of rows to be interleaved.
    /// @param[in] kr Unused. Must be 1.
    /// @param[in] sr Unused. Must be 1.
    ///
    /// @return The size in bytes.
    std::function<size_t(size_t m, size_t k, size_t mr, size_t kr, size_t sr)> fn_get_packed_lhs_size;

    /// Gets the offset in bytes of the packed LHS matrix.
    ///
    /// @param[in] m_idx Coordinate of the matrix in M dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t m_idx, size_t k)> fn_get_packed_lhs_offset;

    /// Preprocesses the LHS matrix.
    ///
    /// @param[in] m Number of rows of the unpacked LHS matrix.
    /// @param[in] k Common dimension between the LHS and RHS matrix.
    /// @param[in] mr Block size in M dimension. It must be {{ kernel.interleave_by }}VL.
    /// @param[in] kr Block size in K dimension. It must be {{ kernel.block_by }}.
    /// @param[in] sr Number of kr splits. It must be 1.
    /// @param[in] m_idx_start Unused. Must be 0.
    /// @param[in] lhs LHS matrix data buffer.
    /// @param[in] lhs_stride Row stride in bytes of the LHS matrix.
    /// @param[out] lhs_packed Packed RHS matrix.
    std::function<void(
        size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
        void* lhs_packed)>
        fn_pack_lhs;

    /// Gets a value indicating whether LHS packing is needed.
    [[nodiscard]] bool is_pack_lhs_needed() const {
        return fn_pack_lhs != nullptr;
    }

    /// Gets the offset in bytes of the RHS matrix.
    ///
    /// @param[in] n_idx Coordinate of the matrix in N dimension.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t n_idx)> fn_get_rhs_offset;

    /// Gets the size in bytes of the packed RHS matrix.
    ///
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The size in bytes.
    std::function<size_t(size_t n, size_t k)> fn_get_packed_rhs_size;

    /// Gets the offset in bytes of the packed RHS matrix in the RHS packing kernel
    ///
    /// @param[in] n_idx Coordinate of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t n_idx, size_t k)> fn_get_pack_rhs_packed_rhs_offset;

    /// Gets the offset in bytes of the packed RHS matrix in the main kernel.
    ///
    /// @param[in] n_idx Coordinate of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t n_idx, size_t k)> fn_get_main_packed_rhs_offset;

    std::function<void(
        size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
        const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params)>
        fn_pack_rhs;

    /// Gets the offset in bytes to the data element in the bias buffer.
    ///
    /// @param[in] n_idx Column index.
    ///
    /// @return The offset in bytes to the data element.
    std::function<size_t(size_t n_idx)> fn_get_bias_offset;

    /// Gets the offset in bytes to the data element in the destination matrix buffer.
    ///
    /// @param[in] m_idx Row index.
    /// @param[in] n_idx Column index.
    /// @param[in] stride Row stride in bytes.
    ///
    /// @return The offset in bytes to the data element.
    std::function<size_t(size_t m_idx, size_t n_idx, size_t stride)> fn_get_dst_offset;

    /// Gets the size in bytes of the destination matrix buffer.
    ///
    /// @param[in] m Number of rows.
    /// @param[in] n Number of columns.
    ///
    /// @return The size in bytes of the destination matrix buffer.
    std::function<size_t(size_t m, size_t n)> fn_get_dst_size;

    /// Performs F16 matrix multiplication with RHS packing followed by
    /// clamp operation.
    ///
    /// @param[in] m Size of the matrix in M dimension.
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] lhs LHS data buffer.
    /// @param[in] packed_rhs Packed RHS data buffer.
    /// @param[out] dst Output data buffer.
    /// @param[in] lhs_stride LHS row stride.
    /// @param[in] dst_stride Output row stride.
    /// @param[in] clamp_min Lower bound of the output data.
    /// @param[in] clamp_max Upper bound of the output data.
    std::function<void(
        size_t m, size_t n, size_t k,                             //
        const void* lhs, size_t lhs_stride,                       //
        const void* packed_rhs,                                   //
        void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
        Float16 clamp_min, Float16 clamp_max)>
        fn_matmul_f16_f16_f16p;

    /// Performs F32 matrix multiplication with LHS & RHS packing
    /// followed by clamp operation.
    ///
    /// @param[in] m Number of output rows to be computed.
    /// @param[in] n Number of output columns to be computed.
    /// @param[in] k Common dimension of the LHS and RHS operands.
    /// @param[in] packed_lhs Packed LHS matrix buffer.
    /// @param[in] packed_rhs Packed RHS matrix buffer.
    /// @param[out] dst Output matrix buffer.
    /// @param[in] dst_stride_row Row stride in bytes of the output matrix.
    /// @param[in] dst_stride_col Column stride in bytes of the output matrix.
    /// @param[in] clamp_min Minimum value to clamp the final result.
    /// @param[in] clamp_max Maximum value to clamp the final result.
    std::function<void(
        size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
        size_t dst_stride_col, float clamp_min, float clamp_max)>
        fn_matmul_f32_f32p_f32p;

    /// Gets a value indicating whether pre-processing the RHS matrix is needed.
    [[nodiscard]] bool is_pack_rhs_needed() const {
        return fn_pack_rhs != nullptr;
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

        if (fn_pack_rhs != nullptr) {
            fn_pack_rhs(
                1, n, k, fn_get_nr(), fn_get_kr(), fn_get_sr(), rhs_row_stride, rhs, bias, nullptr, packed_rhs, 0,
                nullptr);
        } else {
            KAI_ERROR("RHS pre-processing is not supported!");
        }
    }

    [[nodiscard]] bool has_main_kernel() const {
        return fn_matmul_f16_f16_f16p != nullptr || fn_matmul_f32_f32p_f32p != nullptr;
    }

    void main_kernel(
        size_t m, size_t n, size_t k, const void* lhs, const void* rhs, const void* bias, void* dst, size_t lhs_stride,
        size_t rhs_stride, size_t dst_stride, float clamp_min, float clamp_max) const {
        KAI_UNUSED(bias);
        KAI_UNUSED(rhs_stride);

        if (fn_matmul_f16_f16_f16p) {
            fn_matmul_f16_f16_f16p(
                m, n, k, lhs, lhs_stride, rhs, dst, dst_stride, sizeof(Float16), static_cast<Float16>(clamp_min),
                static_cast<Float16>(clamp_max));
        } else if (fn_matmul_f32_f32p_f32p) {
            fn_matmul_f32_f32p_f32p(m, n, k, lhs, rhs, dst, dst_stride, sizeof(float), clamp_min, clamp_max);
        } else {
            KAI_ERROR("Main kernel is not available!");
        }
    }
};

// NOLINTEND(misc-non-private-member-variables-in-classes)

/// List of supported matrix multiplication methods.
static const std::array matmul_methods = {
    MatMulMethod{
        .name = "matmul_nt_nt_fp16_fp16_fp16_1x16_neon_mla",

        .m0 = 1,
        .n0 = 16,

        .lhs_transposed = false,
        .rhs_transposed = false,

        .is_sme2 = false,

        .dst_format = DataFormat(DataType::FP16),
        .lhs_format = DataFormat(DataType::FP16),
        .packed_lhs_format = DataFormat(DataType::UNKNOWN),
        .rhs_format = DataFormat(DataType::FP16),
        .packed_rhs_format = DataFormat(
            DataType::FP16, 16, 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP16, DataType::UNKNOWN, 16, 1),
        .bias_format = DataFormat(DataType::FP16),

        .fn_get_mr = nullptr,
        .fn_get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,
        .fn_get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,
        .fn_get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,

        .fn_get_main_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,
        .fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
        .fn_get_main_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,

        .fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,
        .fn_get_packed_lhs_size = nullptr,
        .fn_get_packed_lhs_offset = nullptr,
        .fn_pack_lhs = nullptr,

        .fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
        .fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
        .fn_get_pack_rhs_packed_rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
        .fn_get_main_packed_rhs_offset = kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,
        .fn_pack_rhs = kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,

        .fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,

        .fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,
        .fn_get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,

        .fn_matmul_f16_f16_f16p = kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_1x16x8_neon_mla,
        .fn_matmul_f32_f32p_f32p = nullptr,
    },

    MatMulMethod{
        .name = "matmul_nt_nt_fp16_fp16_fp16_6x16_neon_mla",

        .m0 = 6,
        .n0 = 16,

        .lhs_transposed = false,
        .rhs_transposed = false,

        .is_sme2 = false,

        .dst_format = DataFormat(DataType::FP16),
        .lhs_format = DataFormat(DataType::FP16),
        .packed_lhs_format = DataFormat(DataType::UNKNOWN),
        .rhs_format = DataFormat(DataType::FP16),
        .packed_rhs_format = DataFormat(
            DataType::FP16, 16, 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP16, DataType::UNKNOWN, 16, 1),
        .bias_format = DataFormat(DataType::FP16),

        .fn_get_mr = nullptr,
        .fn_get_nr = kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        .fn_get_kr = kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        .fn_get_sr = kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,

        .fn_get_main_m_step = kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        .fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
        .fn_get_main_n_step = kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,

        .fn_get_lhs_offset = kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        .fn_get_packed_lhs_size = nullptr,
        .fn_get_packed_lhs_offset = nullptr,
        .fn_pack_lhs = nullptr,

        .fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
        .fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
        .fn_get_pack_rhs_packed_rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,
        .fn_get_main_packed_rhs_offset = kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        .fn_pack_rhs = kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,

        .fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon,

        .fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        .fn_get_dst_size = kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,

        .fn_matmul_f16_f16_f16p = kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        .fn_matmul_f32_f32p_f32p = nullptr,
    },

    MatMulMethod{
        .name = "matmul_nt_nt_fp32_fp32_fp32_2vlx2vl_sme2_mopa",

        .m0 = 2 * get_sme_vector_length<float>(),
        .n0 = 2 * get_sme_vector_length<float>(),

        .lhs_transposed = false,
        .rhs_transposed = false,

        .is_sme2 = true,

        .dst_format = DataFormat(DataType::FP32),
        .lhs_format = DataFormat(DataType::FP32),
        .packed_lhs_format = DataFormat(DataType::FP32, 2 * get_sme_vector_length<float>(), 1),
        .rhs_format = DataFormat(DataType::FP32),
        .packed_rhs_format = DataFormat(
            DataType::FP32, 2 * get_sme_vector_length<float>(), 0, DataFormat::PackFormat::BIAS_PER_ROW, DataType::FP32,
            DataType::UNKNOWN, 2 * get_sme_vector_length<float>(), 1),
        .bias_format = DataFormat(DataType::FP32),

        .fn_get_mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
        .fn_get_nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
        .fn_get_kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
        .fn_get_sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,

        .fn_get_main_m_step = kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
        .fn_get_pack_rhs_n_step = kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        .fn_get_main_n_step = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,

        .fn_get_lhs_offset = kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
        .fn_get_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme,
        .fn_get_packed_lhs_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
        .fn_pack_lhs = kai_run_lhs_pack_f32p2vlx1_f32_sme,

        .fn_get_rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        .fn_get_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        .fn_get_pack_rhs_packed_rhs_offset =
            kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
        .fn_get_main_packed_rhs_offset =
            kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
        .fn_pack_rhs = kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,

        .fn_get_bias_offset = kai_get_bias_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,

        .fn_get_dst_offset = kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
        .fn_get_dst_size = kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,

        .fn_matmul_f16_f16_f16p = nullptr,
        .fn_matmul_f32_f32p_f32p = kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    },
};

/// Matrix multiplication shape.
struct MatMulShape {
    size_t m;  ///< LHS height.
    size_t n;  ///< RHS width.
    size_t k;  ///< LHS width and RHS height.
};

/// Matrix multiplication test information.
using MatMulTestParams = std::tuple<size_t, MatMulShape, MatrixPortion>;

/// Prints the test information.
void PrintTo(const MatMulTestParams& param, std::ostream* os) {
    const auto& [method_no, shape, portion] = param;

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
    *os << "Method_" << matmul_methods[method_no].name                           //
        << "__M_" << shape.m << "__N_" << shape.n << "__K_" << shape.k           //
        << "__PortionStartRow_" << static_cast<int>(portion.start_row() * 1000)  //
        << "__PortionStartCol_" << static_cast<int>(portion.start_col() * 1000)  //
        << "__PortionHeight_" << static_cast<int>(portion.height() * 1000)       //
        << "__PortionWidth_" << static_cast<int>(portion.width() * 1000);
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
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
        std::vector<uint8_t> bias{};            ///< Bias.
        std::vector<uint8_t> ref_packed_rhs{};  ///< Reference packed RHS.
        std::vector<uint8_t> ref_dst{};         ///< Reference output.
    };

    /// Gets the test data for the current test case.
    static const TestData& test_data() {
        const auto& [method_no, info, portion] = GetParam();
        const TestDataId data_id{info.m, info.n, info.k, method_no};

        // If the test data is already available, returns it.
        const auto data_it = _data.find(data_id);

        if (data_it != _data.end()) {
            return data_it->second;
        }

        // Generates the test data.
        const auto& method = matmul_methods.at(method_no);

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

        std::vector<uint8_t> packed_rhs;
        if (has_rhs_pack) {
            packed_rhs = matmul_pack_rhs(
                rhs.data(), !rhs_scales.empty() ? rhs_scales.data() : nullptr, bias.data(), method.rhs_format,
                method.packed_rhs_format, info.n, info.k, !method.rhs_transposed);
        }

        KAI_ASSUME(method.lhs_format.is_raw());
        KAI_ASSUME(method.rhs_format.is_raw());
        KAI_ASSUME(method.dst_format.is_raw());
        auto ref_dst = matmul(
            lhs.data(), nullptr, nullptr, method.lhs_format.data_type(),            //
            rhs.data(), rhs_scales.data(), nullptr, method.rhs_format.data_type(),  //
            bias.data(), nullptr, nullptr, method.bias_format.data_type(),          //
            method.dst_format.data_type(),                                          //
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
std::map<MatMulTest::TestDataId, MatMulTest::TestData> MatMulTest::_data;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/// Tests the LHS packing kernel.
TEST_P(MatMulTest, PackedLhs) {
    const auto& [method_no, info, portion] = GetParam();
    const auto& data = test_data();
    const auto& method = matmul_methods.at(method_no);

    if (method.is_sme2 && !cpu_has_sme2()) {
        GTEST_SKIP();
    }

    if (!method.is_pack_lhs_needed()) {
        GTEST_SKIP();
    }

    const auto lhs_h = method.lhs_transposed ? info.k : info.m;
    const auto lhs_w = method.lhs_transposed ? info.m : info.k;

    const auto rect = portion.compute_portion(
        lhs_h, lhs_w, method.packed_lhs_format.scheduler_block_height(lhs_h),
        lhs_w);  // LHS packing micro-kernel API doesn't support scheduling over K dimension.

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP();
    }

    const auto ref_lhs_row_stride = method.lhs_format.default_row_stride(lhs_w);

    const auto packed_lhs_size = method.fn_get_packed_lhs_size(info.m, info.k, method.m0, 1, 1);
    const auto ref_packed_lhs_size = method.packed_lhs_format.default_size_in_bytes(lhs_h, lhs_w);
    ASSERT_EQ(packed_lhs_size, ref_packed_lhs_size);

    const auto lhs_offset = method.fn_get_lhs_offset(rect.start_row(), ref_lhs_row_stride);
    const auto ref_lhs_offset = method.lhs_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), lhs_w);
    ASSERT_EQ(lhs_offset, ref_lhs_offset);

    const auto packed_lhs_offset = method.fn_get_packed_lhs_offset(rect.start_row(), info.k);
    const auto ref_packed_lhs_offset = method.packed_lhs_format.default_offset_in_bytes(rect.start_row(), 0, lhs_w);
    ASSERT_EQ(packed_lhs_offset, ref_packed_lhs_offset);

    std::vector<uint8_t> packed_lhs;
    packed_lhs.resize(packed_lhs_size);
    method.fn_pack_lhs(
        rect.height(), rect.width(), method.m0, 1, 1, 0, data.lhs.data() + lhs_offset, ref_lhs_row_stride,
        packed_lhs.data() + packed_lhs_offset);

    DefaultMismatchHandler handler(0, 0.0001, 0, 0.001);
    const auto success =
        compare(packed_lhs.data(), data.ref_packed_lhs.data(), method.packed_lhs_format, lhs_h, lhs_w, rect, handler);
    ASSERT_TRUE(success);
}

/// Tests the RHS packing kernel.
TEST_P(MatMulTest, PackedRhs) {
    const auto& [method_no, info, portion] = GetParam();
    const auto& data = test_data();
    const auto& method = matmul_methods.at(method_no);

    if (method.is_sme2 && !cpu_has_sme2()) {
        GTEST_SKIP();
    }

    if (!method.is_pack_rhs_needed()) {
        GTEST_SKIP();
    }

    const auto rhs_w = method.rhs_transposed ? info.k : info.n;
    const auto packed_rhs_h = info.n;
    const auto packed_rhs_w = info.k;

    const auto n_step = method.fn_get_pack_rhs_n_step();
    const auto ref_n_step = method.packed_rhs_format.scheduler_block_height(packed_rhs_h);
    ASSERT_EQ(n_step, ref_n_step);

    const auto rect = portion.compute_portion(
        packed_rhs_h, packed_rhs_w, method.packed_rhs_format.scheduler_block_height(packed_rhs_h),
        method.packed_rhs_format.scheduler_block_width(packed_rhs_w));

    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP();
    }

    const auto rhs_start_row = method.rhs_transposed ? rect.start_row() : rect.start_col();
    const auto rhs_start_col = method.rhs_transposed ? rect.start_col() : rect.start_row();

    const auto ref_rhs_row_stride = method.rhs_format.default_row_stride(rhs_w);

    const auto rhs_offset = method.fn_get_rhs_offset(rect.start_row());
    const auto ref_rhs_offset = method.rhs_format.default_offset_in_bytes(rhs_start_row, rhs_start_col, rhs_w);
    ASSERT_EQ(rhs_offset, ref_rhs_offset);

    const auto packed_rhs_size = method.fn_get_packed_rhs_size(packed_rhs_h, packed_rhs_w);
    const auto ref_packed_rhs_size = method.packed_rhs_format.default_size_in_bytes(packed_rhs_h, packed_rhs_w);
    ASSERT_EQ(packed_rhs_size, ref_packed_rhs_size);

    const auto packed_rhs_offset = method.fn_get_pack_rhs_packed_rhs_offset(rect.start_row(), info.k);
    const auto ref_packed_rhs_offset =
        method.packed_rhs_format.default_offset_in_bytes(rect.start_row(), rect.start_col(), packed_rhs_w);
    ASSERT_EQ(packed_rhs_offset, ref_packed_rhs_offset);

    const auto ref_rhs_scales_offset =
        rect.start_row() * data_type_size_in_bits(method.packed_rhs_format.scale_data_type()) / 8;

    const auto bias_offset = method.fn_get_bias_offset(rect.start_row());
    const auto ref_bias_offset = method.bias_format.default_offset_in_bytes(0, rect.start_row(), info.n);
    ASSERT_EQ(bias_offset, ref_bias_offset);

    std::vector<uint8_t> packed_rhs;
    packed_rhs.resize(packed_rhs_size);

    method.pack_rhs(
        rect.height(), rect.width(), data.rhs.data() + rhs_offset, ref_rhs_row_stride, data.bias.data() + bias_offset,
        !data.rhs_scales.empty() ? data.rhs_scales.data() + ref_rhs_scales_offset : nullptr,
        packed_rhs.data() + packed_rhs_offset);

    const auto exact = method.packed_rhs_format.pack_format() != DataFormat::PackFormat::QUANTIZE_PER_ROW;
    DefaultMismatchHandler handler(0, exact ? 0 : 0.0001, 0, exact ? 0 : 0.001);
    const auto success = compare(
        packed_rhs.data(), data.ref_packed_rhs.data(), method.packed_rhs_format, packed_rhs_h, packed_rhs_w, rect,
        handler);
    ASSERT_TRUE(success);
}

/// Tests the output.
TEST_P(MatMulTest, Output) {
    const auto& [method_no, info, portion] = GetParam();
    const auto& data = test_data();
    const auto& method = matmul_methods.at(method_no);

    if (method.is_sme2 && !cpu_has_sme2()) {
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

    const auto lhs_w = method.lhs_transposed ? info.m : info.k;
    const auto rhs_w = method.rhs_transposed ? info.k : info.n;
    const auto bias_w = info.n;
    const auto dst_w = info.n;

    const auto lhs_start_row = method.lhs_transposed ? 0 : rect.start_row();
    const auto lhs_start_col = method.lhs_transposed ? rect.start_row() : 0;
    const auto lhs_stride = method.lhs_format.default_row_stride(lhs_w);

    const uint8_t* lhs_data = nullptr;
    uintptr_t lhs_offset = 0;

    if (method.is_pack_lhs_needed()) {
        lhs_data = data.ref_packed_lhs.data();
        lhs_offset = method.packed_lhs_format.default_offset_in_bytes(lhs_start_row, lhs_start_col, info.k);
    } else {
        lhs_data = data.lhs.data();

        lhs_offset = method.fn_get_lhs_offset(lhs_start_row, lhs_stride);
        const auto ref_lhs_offset = method.lhs_format.default_offset_in_bytes(lhs_start_row, lhs_start_col, lhs_w);
        ASSERT_EQ(lhs_offset, ref_lhs_offset);
    }

    const auto rhs_stride = method.rhs_format.default_row_stride(rhs_w);

    const uint8_t* rhs_data = nullptr;
    uintptr_t rhs_offset = 0;

    if (method.is_pack_rhs_needed()) {
        const auto packed_rhs_start_row = rect.start_col();
        const auto packed_rhs_start_col = 0;

        rhs_data = data.ref_packed_rhs.data();

        rhs_offset = method.fn_get_main_packed_rhs_offset(packed_rhs_start_row, info.k);
        const auto ref_rhs_offset =
            method.packed_rhs_format.default_offset_in_bytes(packed_rhs_start_row, packed_rhs_start_col, info.k);
        ASSERT_EQ(rhs_offset, ref_rhs_offset);
    } else {
        const auto rhs_start_row = method.rhs_transposed ? rect.start_col() : 0;
        const auto rhs_start_col = method.rhs_transposed ? 0 : rect.start_col();

        rhs_data = data.rhs.data();
        rhs_offset = method.rhs_format.default_offset_in_bytes(rhs_start_row, rhs_start_col, rhs_w);
    }

    const auto* bias_data = data.bias.data();
    const auto bias_offset = method.bias_format.default_offset_in_bytes(0, rect.start_row(), bias_w);

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
        rect.height(), rect.width(), info.k, lhs_data + lhs_offset, rhs_data + rhs_offset, bias_data + bias_offset,
        dst.data() + dst_offset, lhs_stride, rhs_stride, dst_stride, -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());

    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    const auto success = compare(dst.data(), data.ref_dst.data(), method.dst_format, info.m, info.n, rect, handler);
    ASSERT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest,
    testing::Combine(
        testing::Range<size_t>(0, matmul_methods.size()),
        testing::Values(
            MatMulShape{6, 16, 32},   //
            MatMulShape{12, 32, 17},  //
            MatMulShape{13, 33, 23}   //
            ),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),        // Full matrix.
            MatrixPortion(0, 0, 0.25, 0.25),  // Top-left corner.
            MatrixPortion(0.75, 0.75, 1, 1)   // Bottom-right corner.
            )),
    testing::PrintToStringParamName());

}  // namespace kai::test
