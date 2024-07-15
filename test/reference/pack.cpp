//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/pack.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/round.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {

namespace {

std::vector<uint8_t> pack_block(
    const void* src, size_t data_esize, size_t full_height, size_t full_width, size_t block_height, size_t block_width,
    size_t subblock_height, size_t subblock_width) {
    const auto dst_bytes =
        round_up_multiple(full_height, block_height) * round_up_multiple(full_width, block_width) * data_esize;

    std::vector<uint8_t> dst;
    dst.resize(dst_bytes);

    const auto* src_ptr = reinterpret_cast<const uint8_t*>(src);
    auto* dst_ptr = dst.data();

    for (size_t y_block = 0; y_block < full_height; y_block += block_height) {
        for (size_t x_block = 0; x_block < full_width; x_block += block_width) {
            for (size_t y_subblock = 0; y_subblock < block_height; y_subblock += subblock_height) {
                for (size_t x_subblock = 0; x_subblock < block_width; x_subblock += subblock_width) {
                    for (size_t y_element = 0; y_element < subblock_height; ++y_element) {
                        if (y_block + y_subblock + y_element < full_height) {
                            const auto len = std::min(subblock_width, full_width - x_block - x_subblock);

                            memcpy(
                                dst_ptr,
                                src_ptr +
                                    ((y_block + y_subblock + y_element) * full_width + x_block + x_subblock) *
                                        data_esize,
                                len * data_esize);
                        }

                        dst_ptr += subblock_width * data_esize;
                    }
                }
            }
        }
    }

    KAI_ASSERT(reinterpret_cast<uintptr_t>(dst_ptr) - reinterpret_cast<uintptr_t>(dst.data()) == dst_bytes);

    return dst;
}

/// Packs the matrix from raw to per-row bias format.
std::vector<uint8_t> pack_bias_per_row(
    size_t data_esize, size_t zero_point_esize, const void* src, const void* bias, size_t height, size_t width,
    size_t block_height, size_t block_width, size_t subblock_height, size_t subblock_width) {
    const auto num_groups = (height + block_height - 1) / block_height;
    const auto group_num_blocks = (width + block_width - 1) / block_width;

    const auto group_zero_points_bytes = block_height * zero_point_esize;
    const auto block_data_bytes = block_height * block_width * data_esize;
    const auto group_bytes = group_zero_points_bytes + group_num_blocks * block_data_bytes;
    const auto dst_bytes = num_groups * group_bytes;

    std::vector<uint8_t> dst;
    dst.resize(dst_bytes);

    const auto* src_ptr = reinterpret_cast<const uint8_t*>(src);
    const auto* bias_ptr = reinterpret_cast<const uint8_t*>(bias);
    auto* dst_ptr = dst.data();

    for (size_t y_block = 0; y_block < height; y_block += block_height) {
        // Packs the zero points.
        const auto bias_len = std::min(block_height, height - y_block);
        memcpy(dst_ptr, bias_ptr, bias_len * zero_point_esize);
        bias_ptr += block_height * zero_point_esize;
        dst_ptr += block_height * zero_point_esize;

        for (size_t x_block = 0; x_block < width; x_block += block_width) {
            for (size_t y_subblock = 0; y_subblock < block_height; y_subblock += subblock_height) {
                for (size_t x_subblock = 0; x_subblock < block_width; x_subblock += subblock_width) {
                    for (size_t y_element = 0; y_element < subblock_height; ++y_element) {
                        if (y_block + y_subblock + y_element < height) {
                            const auto len = std::min(subblock_width, width - x_block - x_subblock);
                            memcpy(
                                dst_ptr,
                                src_ptr +
                                    ((y_block + y_subblock + y_element) * width + x_block + x_subblock) * data_esize,
                                len * data_esize);
                        }
                        dst_ptr += subblock_width * data_esize;
                    }
                }
            }
        }
    }

    KAI_ASSERT(reinterpret_cast<uintptr_t>(dst_ptr) - reinterpret_cast<uintptr_t>(dst.data()) == dst_bytes);

    return dst;
}

/// Packs the matrix from raw to quantized format.
template <typename Output, typename Input, typename Scale, typename ZeroPoint>
std::vector<uint8_t> pack_quant_per_row(
    const void* src, size_t height, size_t width, size_t block_height, size_t block_width) {
    const auto num_groups = (height + block_height - 1) / block_height;
    const auto group_num_blocks = (width + block_width - 1) / block_width;

    const auto group_zero_points_bytes = block_height * sizeof(ZeroPoint);
    const auto group_scales_bytes = block_height * sizeof(Scale);
    const auto block_data_bytes = block_height * block_width * sizeof(Output);
    const auto group_bytes = group_zero_points_bytes + group_num_blocks * block_data_bytes + group_scales_bytes;
    const auto dst_bytes = num_groups * group_bytes;

    std::vector<uint8_t> dst;
    dst.resize(dst_bytes);

    const auto* src_ptr = reinterpret_cast<const Input*>(src);
    auto* dst_ptr = dst.data();

    std::vector<Scale> scales;
    scales.resize(block_height);

    std::vector<ZeroPoint> zero_points;
    zero_points.resize(block_height);

    for (size_t group_no = 0; group_no < num_groups; ++group_no) {
        // Finds the range of values and calculates the quantization information.
        for (size_t y = 0; y < block_height; ++y) {
            auto min_value = std::numeric_limits<Input>::max();
            auto max_value = std::numeric_limits<Input>::lowest();

            for (size_t x = 0; x < width; ++x) {
                const auto value = src_ptr[(group_no * block_height + y) * width + x];

                if (value < min_value) {
                    min_value = value;
                }

                if (value > max_value) {
                    max_value = value;
                }
            }

            std::tie(scales[y], zero_points[y]) = get_qai8_scale_zero_point_from_range(min_value, max_value);
        }

        // Packs the zero points.
        memcpy(dst_ptr, zero_points.data(), group_zero_points_bytes);
        dst_ptr += group_zero_points_bytes;

        // Quantizes and packs the data.
        for (size_t x_block = 0; x_block < group_num_blocks; ++x_block) {
            for (size_t block_y = 0; block_y < block_height; ++block_y) {
                for (size_t block_x = 0; block_x < block_width; ++block_x) {
                    const auto value =
                        src_ptr[(group_no * block_height + block_y) * width + x_block * block_width + block_x];
                    const auto qvalue = quantize_i8_fp32(value, scales[block_y], zero_points[block_y]);
                    *reinterpret_cast<int8_t*>(dst_ptr) = qvalue;
                    ++dst_ptr;
                }
            }
        }

        // Packs the scales.
        memcpy(dst_ptr, scales.data(), group_scales_bytes);
        dst_ptr += group_scales_bytes;
    }

    KAI_ASSERT(reinterpret_cast<uintptr_t>(dst_ptr) - reinterpret_cast<uintptr_t>(dst.data()) == dst_bytes);

    return dst;
}

/// Packs the matrix with per-row quantized format.
///
/// The source matrix is per-row quantized with separate quantization scale and zero-points data buffer.
/// The destination data is per-row quantized with blocking and embedded quantization information.
std::vector<uint8_t> pack_per_row_qs4(
    const void* src, const void* scales, const void* zero_points, size_t height, size_t width, size_t block_height,
    size_t block_width, size_t subblock_height, size_t subblock_width) {
    // Number of elements in a sub-block in vertical and horizontal axes.
    const auto num_element_rows = subblock_height;
    const auto num_element_cols = subblock_width;
    const auto src_element_row_stride = width / 2;

    // Number of sub-blocks in a block in vertical and horizontal axes.
    const auto num_subblock_rows = block_height / subblock_height;
    const auto num_subblock_cols = block_width / subblock_width;
    const auto src_subblock_col_stride = subblock_width / 4;
    const auto src_subblock_row_stride = subblock_height * width / 2;

    // Number of blocks in the matrix in vertical and horizontal axes.
    const auto num_block_rows = (height + block_height - 1) / block_height;
    const auto num_block_cols = (width + block_width - 1) / block_width;
    const auto src_block_col_stride = block_width / 2;
    const auto src_block_row_stride = block_height * width / 2;

    const auto dst_block_row_scales_bytes = block_height * sizeof(float);
    const auto dst_block_row_zero_points_bytes = block_height * sizeof(int32_t);
    const auto dst_block_row_data_bytes = num_block_cols * block_height * block_width / 2;
    const auto dst_bytes =
        num_block_rows * (dst_block_row_zero_points_bytes + dst_block_row_data_bytes + dst_block_row_scales_bytes);

    std::vector<uint8_t> dst;
    dst.resize(dst_bytes);

    const auto* src_ptr = reinterpret_cast<const uint8_t*>(src);
    const auto* scales_ptr = reinterpret_cast<const float*>(scales);
    const auto* zero_points_ptr = reinterpret_cast<const int32_t*>(zero_points);
    auto* dst_ptr = dst.data();

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        if (zero_points_ptr != nullptr) {
            memcpy(dst_ptr, zero_points_ptr + block_row * block_height, dst_block_row_zero_points_bytes);
        }

        dst_ptr += dst_block_row_zero_points_bytes;

        for (size_t block_col = 0; block_col < num_block_cols; ++block_col) {
            for (size_t subblock_col = 0; subblock_col < num_subblock_cols; ++subblock_col) {
                for (size_t subblock_row = 0; subblock_row < num_subblock_rows; ++subblock_row) {
                    for (size_t element_col = 0; element_col < num_element_cols / 4; ++element_col) {
                        for (size_t element_row = 0; element_row < num_element_rows; ++element_row) {
                            const auto byte_lo = src_ptr[  //
                                block_row * src_block_row_stride + block_col * src_block_col_stride +
                                subblock_row * src_subblock_row_stride + subblock_col * src_subblock_col_stride +
                                element_row * src_element_row_stride + element_col];
                            const auto byte_hi = src_ptr[  //
                                block_row * src_block_row_stride + block_col * src_block_col_stride +
                                subblock_row * src_subblock_row_stride + subblock_col * src_subblock_col_stride +
                                element_row * src_element_row_stride + element_col + block_width / 4];

                            const auto packed_byte0 = (byte_lo & 0x0F) | (byte_hi << 4);
                            const auto packed_byte1 = (byte_lo >> 4) | (byte_hi & 0xF0);

                            dst_ptr[0] = packed_byte0;  // ^ 0x88;
                            dst_ptr[1] = packed_byte1;  // ^ 0x88;
                            dst_ptr += 2;
                        }
                    }
                }
            }
        }

        if (scales_ptr != nullptr) {
            memcpy(dst_ptr, scales_ptr + block_row * block_height, dst_block_row_scales_bytes);
        }
        dst_ptr += dst_block_row_scales_bytes;
    }

    KAI_ASSERT(reinterpret_cast<uintptr_t>(dst_ptr) - reinterpret_cast<uintptr_t>(dst.data()) == dst_bytes);

    return dst;
}

}  // namespace

std::vector<uint8_t> pack(
    const DataFormat& dst_format, const void* src, const void* scales, const void* zero_points,
    const DataFormat& src_format, size_t height, size_t width) {
    const auto dst_dt = dst_format.data_type();
    const auto dst_qf = dst_format.pack_format();
    const auto src_dt = src_format.data_type();
    const auto src_qf = src_format.pack_format();

    const auto block_height = dst_format.actual_block_height(height);
    const auto block_width = dst_format.actual_block_width(width);
    const auto subblock_height = dst_format.actual_subblock_height(height);
    const auto subblock_width = dst_format.actual_subblock_width(width);

    if (src_qf == DataFormat::PackFormat::NONE && dst_qf == DataFormat::PackFormat::QUANTIZE_PER_ROW) {
        if (dst_dt == DataType::QAI8 && src_dt == DataType::FP32 && dst_format.scale_data_type() == DataType::FP32 &&
            dst_format.zero_point_data_type() == DataType::I32) {
            return pack_quant_per_row<int8_t, float, float, int32_t>(src, height, width, block_height, block_width);
        } else if (
            dst_dt == DataType::QSI4 && src_dt == DataType::QSU4 && dst_format.scale_data_type() == DataType::FP32 &&
            dst_format.zero_point_data_type() == DataType::I32) {
            return pack_per_row_qs4(
                src, scales, zero_points, height, width, block_height, block_width, subblock_height, subblock_width);
        }
    }

    if (src_qf == DataFormat::PackFormat::NONE && dst_qf == DataFormat::PackFormat::BIAS_PER_ROW) {
        KAI_ASSUME(src_dt == dst_dt);

        const auto data_esize = data_type_size_in_bits(dst_dt);
        const auto zero_point_esize = data_type_size_in_bits(dst_format.zero_point_data_type());

        if (data_esize % 8 == 0 && zero_point_esize % 8 == 0) {
            return pack_bias_per_row(
                data_esize / 8, zero_point_esize / 8, src, zero_points, height, width, block_height, block_width,
                subblock_height, subblock_width);
        }
    }

    if (src_qf == DataFormat::PackFormat::NONE && dst_qf == DataFormat::PackFormat::NONE) {
        KAI_ASSUME(src_dt == dst_dt);

        const auto data_esize = data_type_size_in_bits(dst_dt);

        if (data_esize % 8 == 0) {
            return pack_block(
                src, data_esize / 8, height, width, block_height, block_width, subblock_height, subblock_width);
        }
    }

    KAI_ERROR("Unsupported operation!");
}

}  // namespace kai::test
