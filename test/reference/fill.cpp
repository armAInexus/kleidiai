//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/fill.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/numeric_limits.hpp"
#include "test/common/round.hpp"
#include "test/common/type_traits.hpp"

namespace kai::test {

namespace {

template <typename T>
std::vector<uint8_t> fill_matrix_raw(size_t height, size_t width, std::function<T(size_t, size_t)> gen) {
    const auto size = height * width * size_in_bits<T> / 8;
    KAI_ASSUME(width * size_in_bits<T> % 8 == 0);

    std::vector<uint8_t> data;
    data.resize(size);
    auto ptr = reinterpret_cast<T*>(data.data());

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            write_array<T>(ptr, y * width + x, gen(y, x));
        }
    }

    return data;
}

template <typename T>
std::vector<uint8_t> fill_matrix_random_raw(size_t height, size_t width, uint64_t seed) {
    using TDist = std::conditional_t<
        std::is_floating_point_v<T>, std::uniform_real_distribution<float>, std::uniform_int_distribution<T>>;

    std::mt19937 rnd(seed);
    TDist dist;

    return fill_matrix_raw<T>(height, width, [&](size_t, size_t) { return dist(rnd); });
}

template <>
std::vector<uint8_t> fill_matrix_random_raw<Float16>(size_t height, size_t width, uint64_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist;

    return fill_matrix_raw<Float16>(height, width, [&](size_t, size_t) { return static_cast<Float16>(dist(rnd)); });
}

template <>
std::vector<uint8_t> fill_matrix_random_raw<BFloat16>(size_t height, size_t width, uint64_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist;

    return fill_matrix_raw<BFloat16>(height, width, [&](size_t, size_t) { return static_cast<BFloat16>(dist(rnd)); });
}

template <>
std::vector<uint8_t> fill_matrix_random_raw<Int4>(size_t height, size_t width, uint64_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_int_distribution<int8_t> dist(-8, 7);

    return fill_matrix_raw<Int4>(height, width, [&](size_t, size_t) { return Int4(dist(rnd)); });
}

template <>
std::vector<uint8_t> fill_matrix_random_raw<UInt4>(size_t height, size_t width, uint64_t seed) {
    std::mt19937 rnd(seed);
    std::uniform_int_distribution<int8_t> dist(0, 15);

    return fill_matrix_raw<UInt4>(height, width, [&](size_t, size_t) { return UInt4(dist(rnd)); });
}

}  // namespace

std::vector<uint8_t> fill_matrix_random(size_t height, size_t width, const DataFormat& format, uint64_t seed) {
    switch (format.pack_format()) {
        case DataFormat::PackFormat::NONE:
            switch (format.data_type()) {
                case DataType::FP32:
                    return fill_matrix_random_raw<float>(height, width, seed);

                case DataType::FP16:
                    return fill_matrix_random_raw<Float16>(height, width, seed);

                case DataType::BF16:
                    return fill_matrix_random_raw<BFloat16>(height, width, seed);

                case DataType::QSU4:
                    return fill_matrix_random_raw<UInt4>(height, width, seed);

                case DataType::QSI4:
                    return fill_matrix_random_raw<Int4>(height, width, seed);

                default:
                    KAI_ERROR("Unsupported data type!");
            }

            break;

        default:
            KAI_ERROR("Unsupported data format!");
    }
}

template <typename Value>
Value get_random(uint64_t seed, Value min_value, Value max_value) {
    static_assert(is_floating_point<Value> || is_integral<Value>);
    static_assert(size_in_bits<Value> <= 32);

    using Distribution = std::conditional_t<
        is_floating_point<Value>, std::uniform_real_distribution<float>,
        std::conditional_t<
            is_signed<Value>, std::uniform_int_distribution<int32_t>, std::uniform_int_distribution<uint32_t>>>;

    std::mt19937 rnd(seed);
    Distribution dist(min_value, max_value);

    return static_cast<Value>(dist(rnd));
}

template <typename Value>
Value get_random(uint64_t seed) {
    if constexpr (is_floating_point<Value>) {
        return get_random<Value>(seed, static_cast<Value>(0.0F), static_cast<Value>(1.0F));
    } else {
        return get_random<Value>(seed, numeric_lowest<Value>, numeric_highest<Value>);
    }
}

template float get_random(uint64_t seed);
template int32_t get_random(uint64_t seed);

template <typename Value>
std::vector<uint8_t> fill_random(size_t length, uint64_t seed, Value min_value, Value max_value) {
    static_assert(is_floating_point<Value> || is_integral<Value>);
    static_assert(size_in_bits<Value> <= 32);

    using Distribution = std::conditional_t<
        is_floating_point<Value>, std::uniform_real_distribution<float>,
        std::conditional_t<
            is_signed<Value>, std::uniform_int_distribution<int32_t>, std::uniform_int_distribution<uint32_t>>>;

    std::mt19937 rnd(seed);
    Distribution dist(min_value, max_value);

    std::vector<uint8_t> data(round_up_division(length * size_in_bits<Value>, 8));

    for (size_t i = 0; i < length; ++i) {
        write_array<Value>(data.data(), i, static_cast<Value>(dist(rnd)));
    }

    return data;
}

template <typename Value>
std::vector<uint8_t> fill_random(size_t length, uint64_t seed) {
    if constexpr (is_floating_point<Value>) {
        return fill_random<Value>(length, seed, static_cast<Value>(0.0F), static_cast<Value>(1.0F));
    } else {
        return fill_random<Value>(length, seed, numeric_lowest<Value>, numeric_highest<Value>);
    }
}

template std::vector<uint8_t> fill_random<float>(size_t length, uint64_t seed);
template std::vector<uint8_t> fill_random<int32_t>(size_t length, uint64_t seed);
template std::vector<uint8_t> fill_random<int8_t>(size_t length, uint64_t seed);

}  // namespace kai::test
