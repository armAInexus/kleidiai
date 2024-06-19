#!/bin/bash

#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

set -e

# This script is used by Dockerfile to create a Linux bootloader with the latest Linux kernel.

# Downloads tools and source code.
wget -O- "https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-aarch64-aarch64-none-elf.tar.xz" | tar xJ
wget -O- "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.9.4.tar.xz" | tar xJ
git clone --depth 1 "https://git.kernel.org/pub/scm/linux/kernel/git/mark/boot-wrapper-aarch64.git"
git clone --depth 1 --branch v6.9-dts "https://git.kernel.org/pub/scm/linux/kernel/git/devicetree/devicetree-rebasing.git"

# Builds the Linux kernel.
cd linux-6.9.4
make ARCH=arm64 defconfig
make ARCH=arm64 -j8 Image
cd ..

# Builds the device tree.
cd devicetree-rebasing
make CPP=/opt/devtools/arm-gnu-toolchain-13.2.Rel1-aarch64-aarch64-none-elf/bin/aarch64-none-elf-cpp src/arm64/arm/fvp-base-revc.dtb
cd ..

# Builds the bootloader.
cd boot-wrapper-aarch64
autoreconf -i
./configure \
    --enable-psci \
    --enable-gicv3 \
    --with-kernel-dir=../linux-6.9.4 \
    --with-dtb=../devicetree-rebasing/src/arm64/arm/fvp-base-revc.dtb \
    --with-cmdline="console=ttyAMA0 earlycon=pl011,0x1c090000 panic=1 root=/dev/vda rw init=/bin/bash -- /root/startup"
make -j8
cd ..

cp boot-wrapper-aarch64/linux-system.axf .

# Cleans up.
rm -rf \
    arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-elf \
    linux-6.9.4 \
    devicetree-rebasing \
    boot-wrapper-aarch64
