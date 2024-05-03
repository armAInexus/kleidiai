//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/compare.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <tuple>

#include "src/kai_common.h"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/logging.hpp"
#include "test/common/memory.hpp"
#include "test/common/printer.hpp"
#include "test/common/rect.hpp"

namespace kai::test {

namespace {

/// Calculates the absolute and relative errors.
///
/// @param[in] imp Value under test.
/// @param[in] ref Reference value.
///
/// @return The absolute error and relative error.
template <typename T>
std::tuple<float, float> calculate_error(T imp, T ref) {
    const auto imp_f = static_cast<float>(imp);
    const auto ref_f = static_cast<float>(ref);

    const auto abs_error = std::abs(imp_f - ref_f);
    const auto rel_error = ref_f != 0 ? abs_error / std::abs(ref_f) : 0.0F;

    return {abs_error, rel_error};
}

/// Compares matrices with per-row quantization.
template <typename Data>
bool compare_raw(
    const void* imp_data, const void* ref_data, size_t full_height, size_t full_width, const Rect& rect,
    MismatchHandler& handler) {
    for (size_t y = 0; y < full_height; ++y) {
        for (size_t x = 0; x < full_width; ++x) {
            const auto in_roi =
                y >= rect.start_row() && y < rect.end_row() && x >= rect.start_col() && x < rect.end_col();

            const auto imp_value = read_array<Data>(imp_data, y * full_width + x);
            const auto ref_value = in_roi ? read_array<Data>(ref_data, y * full_width + x) : 0;

            const auto [abs_err, rel_err] = calculate_error(imp_value, ref_value);

            if (abs_err != 0 || rel_err != 0) {
                const auto notifying = !in_roi || handler.handle_data(abs_err, rel_err);

                if (notifying) {
                    KAI_LOGE("Mismatched data at (", y, ", ", x, "): actual = ", imp_value, ", expected: ", ref_value);
                }
            }
        }
    }

    return handler.success(full_height * full_width);
}

/// Compares matrices with per-row quantization.
template <typename Data, typename Scale, typename Offset>
bool compare_per_row(
    const void* imp_data, const void* ref_data, const DataFormat& format, size_t full_height, size_t full_width,
    const Rect& rect, MismatchHandler& handler) {
    const auto block_height = format.block_height();
    const auto block_width = format.block_width();

    KAI_ASSUME(format.scheduler_block_height(full_height) == block_height);
    KAI_ASSUME(format.scheduler_block_width(full_width) == full_width);
    KAI_ASSUME(rect.start_col() == 0);
    KAI_ASSUME(rect.width() == full_width);

    const auto data_bits = size_in_bits<Data>;

    const auto num_groups = (full_height + block_height - 1) / block_height;
    const auto group_num_blocks = (full_width + block_width - 1) / block_width;

    const auto group_offsets_bytes = block_height * sizeof(Offset);
    const auto group_scales_bytes = block_height * sizeof(Scale);
    const auto block_data_bytes = block_height * block_width * data_bits / 8;

    const auto begin_group = rect.start_row() / block_height;
    const auto end_group = rect.end_row() / block_height;

    const auto* imp_ptr = reinterpret_cast<const uint8_t*>(imp_data);
    const auto* ref_ptr = reinterpret_cast<const uint8_t*>(ref_data);

    for (size_t group_no = 0; group_no < num_groups; ++group_no) {
        const auto in_roi = group_no >= begin_group && group_no < end_group;

        // Checks the quantization offsets.
        for (size_t i = 0; i < block_height; ++i) {
            const auto imp_offset = reinterpret_cast<const Offset*>(imp_ptr)[i];
            const Offset ref_offset = in_roi ? reinterpret_cast<const Offset*>(ref_ptr)[i] : 0;
            const auto [abs_err, rel_err] = calculate_error(imp_offset, ref_offset);

            if (abs_err != 0 || rel_err != 0) {
                handler.mark_as_failed();

                const auto raw_row = group_no * block_height + i;
                KAI_LOGE(
                    "Mismatched quantization offset ", raw_row, ": actual = ", imp_offset, ", expected: ", ref_offset);
            }
        }

        imp_ptr += group_offsets_bytes;
        ref_ptr += group_offsets_bytes;

        // Checks the data.
        for (size_t block_no = 0; block_no < group_num_blocks; ++block_no) {
            for (size_t y = 0; y < block_height; ++y) {
                for (size_t x = 0; x < block_width; ++x) {
                    const auto imp_data = read_array<Data>(imp_ptr, y * block_width + x);
                    const Data ref_data = in_roi ? read_array<Data>(ref_ptr, y * block_width + x) : Data(0);
                    const auto [abs_err, rel_err] = calculate_error(imp_data, ref_data);

                    if (abs_err != 0 || rel_err != 0) {
                        const auto notifying = !in_roi || handler.handle_data(abs_err, rel_err);

                        if (notifying) {
                            const auto raw_row = group_no * block_height + y;
                            const auto raw_col = block_no * block_width + x;

                            KAI_LOGE(
                                "Mismatched data at (", raw_row, ", ", raw_col, "): actual = ", imp_data,
                                ", expected: ", ref_data);
                        }
                    }
                }
            }

            imp_ptr += block_data_bytes;
            ref_ptr += block_data_bytes;
        }

        // Checks the quantization scales.
        for (size_t i = 0; i < block_height; ++i) {
            const auto imp_scale = reinterpret_cast<const Scale*>(imp_ptr)[i];
            const Scale ref_scale = in_roi ? reinterpret_cast<const Scale*>(ref_ptr)[i] : 0;
            const auto [abs_err, rel_err] = calculate_error(imp_scale, ref_scale);

            if (abs_err != 0 || rel_err != 0) {
                handler.mark_as_failed();

                const auto raw_row = group_no * block_height + i;
                KAI_LOGE(
                    "Mismatched quantization scale ", raw_row, ": actual = ", imp_scale, ", expected: ", ref_scale);
            }
        }

        imp_ptr += group_scales_bytes;
        ref_ptr += group_scales_bytes;
    }

    return handler.success(rect.height() * full_width);
}

}  // namespace

bool compare(
    const void* imp_data, const void* ref_data, const DataFormat& format, size_t full_height, size_t full_width,
    const Rect& rect, MismatchHandler& handler) {
    const auto data_type = format.data_type();
    const auto scale_dt = format.scale_data_type();
    const auto offset_dt = format.zero_point_data_type();

    switch (format.quantization_format()) {
        case DataFormat::QuantizationFormat::NONE:
            switch (data_type) {
                case DataType::FP32:
                    return compare_raw<float>(imp_data, ref_data, full_height, full_width, rect, handler);

                default:
                    break;
            }

            break;

        case DataFormat::QuantizationFormat::PER_ROW:
            if (data_type == DataType::QAI8 && scale_dt == DataType::FP32 && offset_dt == DataType::I32) {
                return compare_per_row<int8_t, float, int32_t>(
                    imp_data, ref_data, format, full_height, full_width, rect, handler);
            } else if (data_type == DataType::QSI4 && scale_dt == DataType::FP32 && offset_dt == DataType::I32) {
                return compare_per_row<Int4, float, int32_t>(
                    imp_data, ref_data, format, full_height, full_width, rect, handler);
            }

            break;

        default:
            break;
    }

    KAI_ERROR("Unsupported format!");
}

// =====================================================================================================================

DefaultMismatchHandler::DefaultMismatchHandler(
    float abs_error_threshold, float rel_error_threshold, size_t abs_mismatched_threshold,
    float rel_mismatched_threshold) :
    _abs_error_threshold(abs_error_threshold),
    _rel_error_threshold(rel_error_threshold),
    _abs_mismatched_threshold(abs_mismatched_threshold),
    _rel_mismatched_threshold(rel_mismatched_threshold),
    _num_mismatches(0),
    _failed(false) {
}

DefaultMismatchHandler::DefaultMismatchHandler(const DefaultMismatchHandler& rhs) :
    _abs_error_threshold(rhs._abs_error_threshold),
    _rel_error_threshold(rhs._rel_error_threshold),
    _abs_mismatched_threshold(rhs._abs_mismatched_threshold),
    _rel_mismatched_threshold(rhs._rel_mismatched_threshold),
    _num_mismatches(0),
    _failed(false) {
    // Cannot copy mismatch handler that is already in use.
    KAI_ASSUME(rhs._num_mismatches == 0);
    KAI_ASSUME(!rhs._failed);
}

DefaultMismatchHandler& DefaultMismatchHandler::operator=(const DefaultMismatchHandler& rhs) {
    if (this != &rhs) {
        // Cannot copy mismatch handler that is already in use.
        KAI_ASSUME(rhs._num_mismatches == 0);
        KAI_ASSUME(!rhs._failed);

        _abs_error_threshold = rhs._abs_error_threshold;
        _rel_error_threshold = rhs._rel_error_threshold;
        _abs_mismatched_threshold = rhs._abs_mismatched_threshold;
        _rel_mismatched_threshold = rhs._rel_mismatched_threshold;
    }

    return *this;
}

bool DefaultMismatchHandler::handle_data(float absolute_error, float relative_error) {
    const auto mismatched = absolute_error > _abs_error_threshold && relative_error > _rel_error_threshold;

    if (mismatched) {
        ++_num_mismatches;
    }

    return mismatched;
}

void DefaultMismatchHandler::mark_as_failed() {
    _failed = true;
}

bool DefaultMismatchHandler::success(size_t num_checks) const {
    if (_failed) {
        return false;
    }

    const auto mismatched_rate = static_cast<float>(_num_mismatches) / static_cast<float>(num_checks);
    return _num_mismatches <= _abs_mismatched_threshold || mismatched_rate <= _rel_mismatched_threshold;
}

}  // namespace kai::test
