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

/// Creates a new matrix filled with random data.
///
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
/// @param[in] format Data format.
/// @param[in] seed Random seed.
///
/// @return The data buffer for the matrix.
std::vector<uint8_t> fill_matrix_random(size_t height, size_t width, const DataFormat& format, uint64_t seed);

}  // namespace kai::test
