//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <tuple>

#include "test/common/MatMulMethod.hpp"
#include "test/common/matrix_portion.hpp"

namespace kai::test {
/// Matrix multiplication shape.
struct MatMulShape {
    size_t m;  ///< LHS height.
    size_t n;  ///< RHS width.
    size_t k;  ///< LHS width and RHS height.
};

/// Matrix multiplication test information.
using MatMulTestParams = std::tuple<MatMulMethod, MatMulShape, MatrixPortion>;

/// Prints the test information.
void PrintTo(const MatMulTestParams& param, std::ostream* os);
}  // namespace kai::test
