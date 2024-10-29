//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/transpose.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

std::vector<uint8_t> transpose(const void* data, DataType data_type, size_t height, size_t width) {
    KAI_ASSUME(data_type_size_in_bits(data_type) % 8 == 0);
    const auto element_size = data_type_size_in_bits(data_type) / 8;

    std::vector<uint8_t> output;
    output.resize(height * width * element_size);

    const auto* src_ptr = reinterpret_cast<const uint8_t*>(data);

    for (size_t y = 0; y < width; ++y) {
        for (size_t x = 0; x < height; ++x) {
            memcpy(
                output.data() + (y * height + x) * element_size, src_ptr + (x * width + y) * element_size,
                element_size);
        }
    }

    return output;
}

template <typename T>
std::vector<T> transpose(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size) {
    std::vector<T> output(dst_size, T(0));

    for (size_t y = 0; y < width; ++y) {
        for (size_t x = 0; x < height; ++x) {
            auto element = read_array<T>(data, (x * src_stride) + y);
            write_array<T>(reinterpret_cast<void*>(output.data()), (y * dst_stride) + x, element);
        }
    }

    return output;
}

template std::vector<Int4> transpose(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size);
}  // namespace kai::test
