#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

workspace(name = "com_arm_kleidiai")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "08c0386f45821ce246bbbf77503c973246ed6ee5c3463e41efc197fa9bc3a7f4",
    strip_prefix = "bazel-skylib-288731ef9f7f688932bd50e704a91a45ec185f9b",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/288731ef9f7f688932bd50e704a91a45ec185f9b.zip"],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "1f357c27ca988c3f7c6b4bf68a9395005ac6761f034046e9dde0896e3aba00e4",
    strip_prefix = "googletest-1.14.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip"],
)

http_archive(
    name = "com_google_benchmark",
    sha256 = "84c49c4c07074f36fbf8b4f182ed7d75191a6fa72756ab4a17848455499f4286",
    strip_prefix = "benchmark-v1.8.4",
    urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.8.4.zip"],
)
