//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/data_format.hpp"

#include <cstddef>
#include <cstdint>

#include "src/kai_common.h"
#include "test/common/data_type.hpp"
#include "test/reference/round.hpp"

namespace kai::test {

DataFormat::DataFormat(
    DataType data_type, size_t block_height, size_t block_width, QuantizationFormat quant_format, DataType scale_dt,
    DataType zero_point_dt, size_t subblock_height, size_t subblock_width) noexcept :
    _data_type(data_type),
    _quant_format(quant_format),
    _scale_dt(scale_dt),
    _zero_point_dt(zero_point_dt),
    _block_height(block_height),
    _block_width(block_width),
    _subblock_height(subblock_height),
    _subblock_width(subblock_width) {
}

bool DataFormat::operator==(const DataFormat& rhs) const {
    return _data_type == rhs._data_type && _quant_format == rhs._quant_format && _scale_dt == rhs._scale_dt &&
        _zero_point_dt == rhs._zero_point_dt && _block_height == rhs._block_height && _block_width == rhs._block_width;
}

bool DataFormat::operator!=(const DataFormat& rhs) const {
    return !(*this == rhs);
}

DataType DataFormat::data_type() const {
    return _data_type;
}

DataFormat::QuantizationFormat DataFormat::quantization_format() const {
    return _quant_format;
}

DataType DataFormat::scale_data_type() const {
    return _scale_dt;
}

DataType DataFormat::zero_point_data_type() const {
    return _zero_point_dt;
}

bool DataFormat::is_raw() const {
    return _quant_format == QuantizationFormat::NONE &&  //
        _block_height == 0 && _block_width == 0 && _subblock_height == 0 && _subblock_width == 0;
}

size_t DataFormat::block_height() const {
    return _block_height;
}

size_t DataFormat::block_width() const {
    return _block_width;
}

size_t DataFormat::subblock_height() const {
    return _subblock_height;
}

size_t DataFormat::subblock_width() const {
    return _subblock_width;
}

size_t DataFormat::scheduler_block_height([[maybe_unused]] size_t full_height) const {
    switch (_quant_format) {
        case QuantizationFormat::NONE:
            return _block_height > 0 ? _block_height : 1;

        case QuantizationFormat::PER_ROW:
            return _block_height;

        default:
            KAI_ERROR("Unsupported quantization packing format!");
    }
}

size_t DataFormat::scheduler_block_width(size_t full_width) const {
    switch (_quant_format) {
        case QuantizationFormat::NONE:
            return _block_width > 0 ? _block_width : 1;

        case QuantizationFormat::PER_ROW:
            return full_width;

        default:
            KAI_ERROR("Unsupported quantization packing format!");
    }
}

uintptr_t DataFormat::default_row_stride(size_t width) const {
    const auto padded_width = round_up_multiple(width, _block_width > 0 ? _block_width : 1);

    switch (_quant_format) {
        case QuantizationFormat::NONE:
            return padded_width * data_type_size_in_bits(_data_type) / 8;

        case QuantizationFormat::PER_ROW:
            return _block_height * data_type_size_in_bits(_zero_point_dt) / 8 +          //
                _block_height * padded_width * data_type_size_in_bits(_data_type) / 8 +  //
                _block_height * data_type_size_in_bits(_scale_dt) / 8;

        default:
            KAI_ERROR("Unsupported quantization packing format!");
    }
}

uintptr_t DataFormat::default_offset_in_bytes(size_t row, size_t col, size_t width) const {
    const auto row_stride = default_row_stride(width);

    KAI_ASSERT(col % scheduler_block_width(width) == 0);

    switch (_quant_format) {
        case QuantizationFormat::NONE:
            return row * row_stride + col * data_type_size_in_bits(_data_type) / 8;

        case QuantizationFormat::PER_ROW:
            KAI_ASSERT(row % _block_height == 0);
            KAI_ASSERT(col == 0);
            return (row / _block_height) * row_stride + col * data_type_size_in_bits(_data_type) / 8;

        default:
            KAI_ERROR("Unsupported quantization packing format!");
    }
}

size_t DataFormat::default_size_in_bytes(size_t height, size_t width) const {
    const auto num_rows = _block_height > 0 ? (height + _block_height - 1) / _block_height : height;
    return num_rows * default_row_stride(width);
}

}  // namespace kai::test
