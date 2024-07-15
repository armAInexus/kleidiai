//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/quantize.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/numeric_limits.hpp"
#include "test/common/round.hpp"
#include "test/common/type_traits.hpp"

namespace kai::test {

std::tuple<float, int32_t> get_qai8_scale_zero_point_from_range(float min_value, float max_value) {
    constexpr float q_min = std::numeric_limits<int8_t>::min();
    constexpr float q_max = std::numeric_limits<int8_t>::max();

    if (min_value > 0) {
        min_value = 0;
    }

    if (max_value < 0) {
        max_value = 0;
    }

    // The reason for computing the inverted scale first is to make it bit-perfect with quantized packing kernels.
    // If those kernels don't do it this way anymore, it makes more sense to calculate the scale directly.
    const float inv_scale = max_value != min_value ? (q_max - q_min) / (max_value - min_value) : 1.0F;
    const float scale = 1.0F / inv_scale;

    const float scaled_min = min_value / scale;
    const float scaled_max = max_value / scale;

    const float zero_point_f = -(scaled_min + q_min) < scaled_max + q_max ? scaled_min - q_min : scaled_max - q_max;
    const int32_t zero_point = round_to_nearest_even_i32(zero_point_f);

    return {scale, zero_point};
}

int8_t quantize_i8_fp32(float value, float scale, int32_t zero_point) {
    return static_cast<int8_t>(std::clamp<int32_t>(
        round_to_nearest_even_i32(value / scale) - zero_point, std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max()));
}

namespace {

template <typename Input, typename Scale, typename ZeroPoint, typename Output>
std::vector<uint8_t> dequantize_any_type(
    const void* data, const void* scales, const void* zero_points,  //
    QuantizationMethod method, bool is_asymm, size_t height, size_t width) {
    static_assert(is_floating_point<Output>);
    static_assert(is_integral<Input>);

    std::vector<uint8_t> dst;
    dst.resize(height * width * sizeof(Output));
    KAI_ASSUME(size_in_bits<Output> % 8 == 0);

    auto scale = read_array<Scale>(scales, 0);
    KAI_UNUSED(is_asymm);
    KAI_UNUSED(zero_points);
    auto zero_point = is_asymm ? read_array<ZeroPoint>(zero_points, 0) :  //
        -static_cast<ZeroPoint>(numeric_lowest<make_signed_t<Input>>);

    for (size_t y = 0; y < height; ++y) {
        if (method == QuantizationMethod::PER_ROW) {
            scale = read_array<Scale>(scales, y);

            if (is_asymm) {
                zero_point = read_array<ZeroPoint>(zero_points, y);
            }
        }

        for (size_t x = 0; x < width; ++x) {
            const ZeroPoint input = read_array<Input>(data, y * width + x);
            const Scale output = static_cast<Scale>(input - zero_point) * scale;
            write_array<Scale>(dst.data(), y * width + x, output);
        }
    }

    return dst;
}

}  // namespace

std::vector<uint8_t> dequantize(
    const void* data, const void* scales, const void* zero_points,  //
    DataType src_dt, DataType dst_dt, QuantizationMethod method,    //
    size_t height, size_t width) {
    switch (src_dt) {
        case DataType::QSU4:
            switch (dst_dt) {
                case DataType::FP32:
                    return dequantize_any_type<UInt4, float, int32_t, float>(
                        data, scales, zero_points, method, false, height, width);

                default:
                    KAI_ERROR("Unsupported destination data type!");
            }

        default:
            KAI_ERROR("Unsupported source data type!");
    }
}

}  // namespace kai::test
