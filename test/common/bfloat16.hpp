//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iosfwd>
#include <type_traits>

#include "test/common/type_traits.hpp"

namespace kai::test {

/// Half-precision brain floating-point.
///
/// This class encapsulates `bfloat16_t` data type provided by `arm_bf16.h`.
class BFloat16 {
public:
    /// Constructor.
    BFloat16() = default;

    /// Destructor.
    ~BFloat16() = default;

    /// Copy constructor.
    BFloat16(const BFloat16&) = default;

    /// Copy assignment.
    BFloat16& operator=(const BFloat16&) = default;

    /// Move constructor.
    BFloat16(BFloat16&&) = default;

    /// Move assignment.
    BFloat16& operator=(BFloat16&&) = default;

    /// Creates a new object from the specified numeric value.
    template <typename T, std::enable_if_t<is_arithmetic<T>, bool> = true>
    explicit BFloat16(T value) : _data(0) {
        const auto value_f32 = static_cast<float>(value);
        asm("bfcvt %h[output], %s[input]" : [output] "=w"(_data) : [input] "w"(value_f32));
    }

    /// Assigns to the specified numeric value which will be converted to `bfloat16_t`.
    template <typename T, std::enable_if_t<is_arithmetic<T>, bool> = true>
    BFloat16& operator=(T value) {
        const auto value_f32 = static_cast<float>(value);
        asm("bfcvt %h[output], %s[input]" : [output] "=w"(_data) : [input] "w"(value_f32));
        return *this;
    }

    /// Converts to numeric type `T`.
    template <typename T, std::enable_if_t<is_arithmetic<T>, bool> = true>
    explicit operator T() const {
        union {
            float f32;
            uint32_t u32;
        } data;

        data.u32 = static_cast<uint32_t>(_data) << 16;

        return static_cast<T>(data.f32);
    }

    /// Equality operator.
    bool operator==(BFloat16 rhs) const {
        return _data == rhs._data;
    }

    /// Unequality operator.
    bool operator!=(BFloat16 rhs) const {
        return _data != rhs._data;
    }

    /// Writes the value to the output stream.
    ///
    /// @param[in] os Output stream to be written to.
    /// @param[in] value Value to be written.
    ///
    /// @return The output stream.
    friend std::ostream& operator<<(std::ostream& os, BFloat16 value);

private:
    uint16_t _data;
};

}  // namespace kai::test
