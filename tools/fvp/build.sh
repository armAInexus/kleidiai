#!/bin/bash

#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

set -e

GCC_VERSION=13.3.rel1

FVP_MAJOR=11
FVP_MINOR=26
FVP_PATCH=11

KERNEL_MAJOR=6
KERNEL_MINOR=10
KERNEL_PATCH=9

BOOTLOADER_VERSION=d62de19c866141cb450576040439de233438fb60

WORKDIR="/opt/devtools"

ARCH=$(uname -m)

mkdir -p "$WORKDIR"

pushd "$WORKDIR" > /dev/null

# ==================================================================================================
# Downloads the bare-metal toolchain.
# ==================================================================================================

mkdir toolchain-aarch64-none-elf
wget -O- "https://developer.arm.com/-/media/Files/downloads/gnu/${GCC_VERSION}/binrel/arm-gnu-toolchain-${GCC_VERSION}-${ARCH}-aarch64-none-elf.tar.xz" | tar xJ --strip-components=1 -C toolchain-aarch64-none-elf

export PATH="$PWD/toolchain-aarch64-none-elf/bin:$PATH"

# ==================================================================================================
# Downloads and builds the Linux kernel.
# ==================================================================================================

mkdir linux
wget -O- "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-${KERNEL_MAJOR}.${KERNEL_MINOR}.${KERNEL_PATCH}.tar.xz" | tar xJ --strip-components=1 -C linux

cd linux

make ARCH=arm64 CROSS_COMPILE=aarch64-none-elf- defconfig
make ARCH=arm64 CROSS_COMPILE=aarch64-none-elf- "-j$(nproc)" Image

cd ..

# ==================================================================================================
# Downloads and builds the device tree.
# ==================================================================================================

mkdir devicetree-rebasing
wget -O- "https://git.kernel.org/pub/scm/linux/kernel/git/devicetree/devicetree-rebasing.git/snapshot/devicetree-rebasing-${KERNEL_MAJOR}.${KERNEL_MINOR}-dts.tar.gz" | tar xz --strip-components=1 -C devicetree-rebasing

cd devicetree-rebasing

make CPP=aarch64-none-elf-cpp src/arm64/arm/fvp-base-revc.dtb

cd ..

# ==================================================================================================
# Downloads the builds the bootloader.
# ==================================================================================================

mkdir boot-wrapper-aarch64
wget -O- "https://git.kernel.org/pub/scm/linux/kernel/git/mark/boot-wrapper-aarch64.git/snapshot/boot-wrapper-aarch64-${BOOTLOADER_VERSION}.tar.gz" | tar xz --strip-components=1 -C boot-wrapper-aarch64

cd boot-wrapper-aarch64

autoreconf -i

./configure \
    --host=aarch64-linux-gnu \
    --enable-psci \
    --enable-gicv3 \
    --with-kernel-dir=../linux \
    --with-dtb=../devicetree-rebasing/src/arm64/arm/fvp-base-revc.dtb \
    --with-cmdline="console=ttyAMA0 earlycon=pl011,0x1c090000 root=/dev/vda2 rw"

make "-j$(nproc)"
mv linux-system.axf "$WORKDIR"

cd ..

# ==================================================================================================
# Downloads the lastest Fixed Virtual Platform.
# ==================================================================================================

case "$ARCH" in
    x86_64)
        FVP_PLATFORM=Linux64
        ;;

    arm64|aarch64)
        FVP_PLATFORM=Linux64_armv8l
        ;;

    *)
        echo "Unknown CPU architecture $ARCH!"
        exit 1
        ;;
esac

mkdir fvp
wget -O- "https://developer.arm.com/-/cdn-downloads/permalink/Fixed-Virtual-Platforms/FM-${FVP_MAJOR}.${FVP_MINOR}/FVP_Base_RevC-2xAEMvA_${FVP_MAJOR}.${FVP_MINOR}_${FVP_PATCH}_${FVP_PLATFORM}.tgz" | tar xz -C fvp

ln -s /opt/devtools/fvp/Base_RevC_AEMvA_pkg/models/${FVP_PLATFORM}_GCC-9.3 /opt/devtools/fvp/models
ln -s /opt/devtools/fvp/Base_RevC_AEMvA_pkg/plugins/${FVP_PLATFORM}_GCC-9.3 /opt/devtools/fvp/plugins

# ==================================================================================================
# Done.
# ==================================================================================================

popd > /dev/null
