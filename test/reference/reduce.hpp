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

/// Reduction operator.
enum class ReductionOperator : uint32_t {
    ADD,  ///< Addition.
};

/// Reduces the matrix value using addition.
///
/// @param[in] src Input data.
/// @param[in] src_format Input data format.
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
/// @param[in] dst_foramt Output data format.
/// @param[in] dimension Reduction dimension.
///
/// @return The reduced matrix.
std::vector<uint8_t> reduce_add(
    const void* src, const DataFormat& src_format, size_t height, size_t width, const DataFormat& dst_format,
    size_t dimension);

}  // namespace kai::test
