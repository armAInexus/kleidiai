#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
include(FetchContent)

# Set timestamp of extracted contents to time of extraction.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

fetchcontent_declare(googlebench
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
    FIND_PACKAGE_ARGS NAMES benchmark
)

fetchcontent_makeavailable(googlebench)
