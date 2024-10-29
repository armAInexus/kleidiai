#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

FROM ubuntu:24.04

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        g++-aarch64-linux-gnu \
        cmake \
        git \
        ninja-build \
        python3 \
        python3-pip \
        wget \
        autoconf \
        automake \
        device-tree-compiler \
        flex \
        bison \
        bc \
        libssl-dev

RUN --mount=type=bind,source=build.sh,target=build.sh \
    ./build.sh
