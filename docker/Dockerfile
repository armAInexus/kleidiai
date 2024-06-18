#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

FROM ubuntu:24.04

RUN apt-get update  \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    build-essential \
    ca-certificates \
    clang \
    clang-tidy \
    llvm \
    libclang-rt-dev \
    cmake \
    git \
    ninja-build \
    python3 \
    reuse \
    pre-commit \
    gcovr
