//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

namespace {

template <typename From, typename To>
std::vector<uint8_t> cast_any_type(const void* src, size_t length) {
    std::vector<uint8_t> dst;
    dst.resize(length * size_in_bits<To> / 8);

    for (size_t i = 0; i < length; ++i) {
        write_array(dst.data(), i, static_cast<To>(read_array<From>(src, i)));
    }

    return dst;
}

}  // namespace

std::vector<uint8_t> cast(const void* src, kai::test::DataType src_dt, DataType dst_dt, size_t height, size_t width) {
    const auto length = height * width;

    if (src_dt == DataType::BF16 && dst_dt == DataType::FP32) {
        return cast_any_type<BFloat16, float>(src, length);
    }

    KAI_ERROR("Unsupported cast data type!");
}

}  // namespace kai::test
