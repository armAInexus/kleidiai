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

template <typename T>
std::vector<uint8_t> clamp(const void* src, size_t len, T min_value, T max_value);

}  // namespace kai::test
