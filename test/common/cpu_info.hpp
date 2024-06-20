//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai::test {

/// Returns a value indicating whether the current CPU supports FEAT_SME.
bool cpu_has_sme();

/// Returns a value indicating whether the current CPU supports FEAT_SME2.
bool cpu_has_sme2();

}  // namespace kai::test
