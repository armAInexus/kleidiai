//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "test/common/data_type.hpp"

namespace kai::test {

/// Transposes the matrix.
///
/// @param[in] data Data buffer.
/// @param[in] data_type Element data type.
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
///
/// @return The transposed matrix.
std::vector<uint8_t> transpose(const void* data, DataType data_type, size_t height, size_t width);

}  // namespace kai::test
