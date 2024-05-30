//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"

namespace kai::test {

namespace {

inline void print_data(std::ostream& os, const uint8_t* data, size_t len, DataType data_type) {
    if (data_type == DataType::QSU4) {
        for (size_t i = 0; i < len / 2; ++i) {
            const auto [low, high] = UInt4::unpack_u8(data[i]);
            os << static_cast<int32_t>(low) << ", " << static_cast<int32_t>(high) << ", ";
        }
    } else if (data_type == DataType::QSI4) {
        for (size_t i = 0; i < len / 2; ++i) {
            const auto [low, high] = Int4::unpack_u8(data[i]);
            os << static_cast<int32_t>(low) << ", " << static_cast<int32_t>(high) << ", ";
        }
    } else {
        for (size_t i = 0; i < len; ++i) {
            switch (data_type) {
                case DataType::FP32:
                    os << reinterpret_cast<const float*>(data)[i];
                    break;

                case DataType::FP16:
                    os << reinterpret_cast<const Float16*>(data)[i];
                    break;

                case DataType::BF16:
                    os << reinterpret_cast<const BFloat16*>(data)[i];
                    break;

                case DataType::I32:
                    os << reinterpret_cast<const int32_t*>(data)[i];
                    break;

                case DataType::QAI8:
                    os << static_cast<int32_t>(reinterpret_cast<const int8_t*>(data)[i]);
                    break;

                default:
                    KAI_ERROR("Unsupported data type!");
            }

            os << ", ";
        }
    }
}

void print_matrix_raw(std::ostream& os, const uint8_t* data, DataType data_type, size_t height, size_t width) {
    const auto row_stride = width * data_type_size_in_bits(data_type) / 8;

    os << "[\n";
    for (size_t y = 0; y < height; ++y) {
        os << "    [";
        print_data(os, data + y * row_stride, width, data_type);
        os << "],\n";
    }
    os << "]\n";
}

void print_matrix_per_row(
    std::ostream& os, const uint8_t* data, const DataFormat& format, size_t height, size_t width) {
    const auto has_scale = format.pack_format() == DataFormat::PackFormat::QUANTIZE_PER_ROW;

    const auto block_height = format.actual_block_height(height);

    const auto num_blocks = (height + block_height - 1) / block_height;

    KAI_ASSUME(format.default_size_in_bytes(height, width) % num_blocks == 0);
    const auto block_data_bytes = format.default_size_in_bytes(height, width) / num_blocks;
    const auto block_offsets_bytes = block_height * data_type_size_in_bits(format.zero_point_data_type()) / 8;
    const auto block_scales_bytes = has_scale ? block_height * data_type_size_in_bits(format.scale_data_type()) / 8 : 0;

    os << "[\n";
    for (size_t y = 0; y < num_blocks; ++y) {
        os << "    {\"offsets\": [";
        print_data(os, data, block_height, format.zero_point_data_type());
        os << "], \"data\": [";
        print_data(os, data + block_offsets_bytes, block_height * width, format.data_type());

        if (has_scale) {
            os << "], \"scales\": [";
            print_data(os, data + block_offsets_bytes + block_data_bytes, block_height, format.scale_data_type());
        }

        os << "]},\n";

        data += block_offsets_bytes + block_data_bytes + block_scales_bytes;
    }
    os << "]\n";
}

}  // namespace

void print_matrix(
    std::ostream& os, std::string_view name, const void* data, const DataFormat& format, size_t height, size_t width) {
    os << name << " = ";

    switch (format.pack_format()) {
        case DataFormat::PackFormat::NONE:
            print_matrix_raw(os, reinterpret_cast<const uint8_t*>(data), format.data_type(), height, width);
            break;

        case DataFormat::PackFormat::BIAS_PER_ROW:
        case DataFormat::PackFormat::QUANTIZE_PER_ROW:
            print_matrix_per_row(os, reinterpret_cast<const uint8_t*>(data), format, height, width);
            break;

        default:
            KAI_ERROR("Unsupported quantization packing format!");
    }
}

}  // namespace kai::test
