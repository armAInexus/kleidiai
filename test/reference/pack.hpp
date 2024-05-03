//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kai::test {

class DataFormat;

/// Packs the matrix.
///
/// @param[in] dst_format Data format of the destination matrix.
/// @param[in] src Data buffer of the source matrix.
/// @param[in] src_format Data format of the source matrix.
/// @param[in] height Number of rows of the source matrix.
/// @param[in] width Number of columns of the source matrix.
std::vector<uint8_t> pack(
    const DataFormat& dst_format, const void* src, const void* scales, const void* zero_points,
    const DataFormat& src_format, size_t height, size_t width);

}  // namespace kai::test
