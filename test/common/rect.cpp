//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/rect.hpp"

#include <cstddef>

namespace kai::test {

Rect::Rect(size_t start_row, size_t start_col, size_t height, size_t width) :
    _start_row(start_row), _start_col(start_col), _height(height), _width(width) {
}

size_t Rect::start_row() const {
    return _start_row;
}

size_t Rect::start_col() const {
    return _start_col;
}

size_t Rect::height() const {
    return _height;
}

size_t Rect::width() const {
    return _width;
}

size_t Rect::end_row() const {
    return _start_row + _height;
}

size_t Rect::end_col() const {
    return _start_col + _width;
}

}  // namespace kai::test
