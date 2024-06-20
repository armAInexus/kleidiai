//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/cpu_info.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "kai/kai_common.h"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif  // defined(__aarch64__) && defined(__linux__)

#if defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif  // defined(__aarch64__) && defined(__APPLE__)

namespace kai::test {

namespace {

#if defined(__aarch64__) && defined(__linux__)
constexpr uint64_t A64_HWCAP2_SME = 1UL << 23;
constexpr uint64_t A64_HWCAP2_SME2 = 1UL << 37;
#endif  // defined(__aarch64__) && defined(__linux__)

#if defined(__aarch64__) && defined(__APPLE__)
template <typename T>
T get_sysctl_by_name(std::string_view name) {
    T value{};
    size_t size = sizeof(T);

    KAI_ASSERT(sysctlbyname(name.data(), nullptr, &size, nullptr, 0) == 0);
    KAI_ASSERT(size == sizeof(T));

    [[maybe_unused]] int status = sysctlbyname(name.data(), &value, &size, nullptr, 0);
    KAI_ASSERT(status == 0);

    return value;
}
#endif  // defined(__aarch64__) && defined(__APPLE__)

/// Information about the CPU that is executing the program.
struct CpuInfo {
    CpuInfo() {
#if defined(__aarch64__) && defined(__linux__)
        const uint64_t hwcaps2 = getauxval(AT_HWCAP2);

        has_sme = (hwcaps2 & A64_HWCAP2_SME) != 0;
        has_sme2 = (hwcaps2 & A64_HWCAP2_SME2) != 0;
#endif  // defined(__aarch64__) && defined(__linux__)

#if defined(__aarch64__) && defined(__APPLE__)
        has_sme = get_sysctl_by_name<uint32_t>("hw.optional.arm.FEAT_SME") == 1;
        has_sme2 = get_sysctl_by_name<uint32_t>("hw.optional.arm.FEAT_SME2") == 1;
#endif  // defined(__aarch64__) && defined(__APPLE__)
    }

    /// Gets the singleton @ref CpuInfo object.
    static CpuInfo& current() {
        static CpuInfo cpu_info{};
        return cpu_info;
    }

    bool has_sme{};   ///< FEAT_SME is supported.
    bool has_sme2{};  ///< FEAT_SME2 is supported.
};

}  // namespace

bool cpu_has_sme() {
    return CpuInfo::current().has_sme;
}

bool cpu_has_sme2() {
    return CpuInfo::current().has_sme2;
}

}  // namespace kai::test
