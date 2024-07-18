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
    python3-pip \
    reuse \
    pre-commit \
    gcovr \
    e2tools \
    wget \
    autoconf \
    automake \
    device-tree-compiler \
    flex \
    bison \
    bc \
    libssl-dev \
    genext2fs \
    curl


RUN mkdir -p /opt/devtools
WORKDIR /opt/devtools

# =============================================================================
# Linux virtual machine on FVP.
# =============================================================================

# Downloads the latest Fixed Virtual Platform.
RUN mkdir FVP_Base_RevC-2xAEMvA_11.25_15_Linux64_armv8l && \
    wget -O- "https://developer.arm.com/-/media/Files/downloads/ecosystem-models/FM_11_25/FVP_Base_RevC-2xAEMvA_11.25_15_Linux64_armv8l.tgz" | tar xz -C FVP_Base_RevC-2xAEMvA_11.25_15_Linux64_armv8l
RUN ln -s /opt/devtools/FVP_Base_RevC-2xAEMvA_11.25_15_Linux64_armv8l/Base_RevC_AEMvA_pkg /opt/devtools/fvp_base_aemva

# Creates the root filesystem.
RUN wget "https://cdimage.ubuntu.com/ubuntu-base/releases/24.04/release/ubuntu-base-24.04-base-arm64.tar.gz" && \
    genext2fs -a ubuntu-base-24.04-base-arm64.tar.gz -B 4096 -b 51200 linux-rootfs.img && \
    tar cfJ linux-rootfs.img.xz linux-rootfs.img && \
    rm ubuntu-base-24.04-base-arm64.tar.gz linux-rootfs.img

# Compiles the latest Linux kernel and bootloader.
COPY build_linux_bootloader.sh .
RUN ./build_linux_bootloader.sh && \
    rm ./build_linux_bootloader.sh

WORKDIR /

ENV XDG_CACHE_HOME=/cache

RUN mkdir -p ${XDG_CACHE_HOME} && \
    wget "https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-$(dpkg --print-architecture)" -O /usr/bin/bazelisk && \
    chmod a+x /usr/bin/bazelisk && \
    bazelisk && \
    chmod -R a+rw ${XDG_CACHE_HOME}

ARG BAZEL_BUILD_TOOLS_VER=v7.1.2

RUN wget "https://github.com/bazelbuild/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VER}/buildifier-linux-$(dpkg --print-architecture)" -O /usr/bin/buildifier && \
    chmod a+x /usr/bin/buildifier

RUN wget "https://github.com/bazelbuild/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VER}/buildozer-linux-$(dpkg --print-architecture)" -O /usr/bin/buildozer && \
    chmod a+x /usr/bin/buildozer

RUN wget "https://github.com/bazelbuild/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VER}/unused_deps-linux-$(dpkg --print-architecture)" -O /usr/bin/unused_deps && \
    chmod a+x /usr/bin/unused_deps
