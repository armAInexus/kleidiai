#!/bin/bash -eu

#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

TARGETARCH=${TARGETARCH:-amd64}

# This script is used by Dockerfile to create a Linux bootloader with the latest Linux kernel.
if [ "${TARGETARCH}" = "amd64" ] ; then
    HOST_ARCH=x86_64
elif [ "${TARGETARCH}" = "arm64" ] ; then
    HOST_ARCH=aarch64
else
    echo "Unknown $TARGETARCH" && exit 1
fi

TOOLCHAIN_VER=13.2.rel1
TOOLCHAIN_TYPE=aarch64-none-elf
TOOLCHAIN_DIR=$(pwd)/toolchain-${TOOLCHAIN_TYPE}/
CROSS_COMPILE=${TOOLCHAIN_DIR}/bin/${TOOLCHAIN_TYPE}-
KERNEL_VERSION=6.9.4

# Downloads Arm toolchain
mkdir -p ${TOOLCHAIN_DIR}
wget -O- "https://developer.arm.com/-/media/Files/downloads/gnu/${TOOLCHAIN_VER}/binrel/arm-gnu-toolchain-${TOOLCHAIN_VER}-${HOST_ARCH}-${TOOLCHAIN_TYPE}.tar.xz" | tar xJC ${TOOLCHAIN_DIR} --strip-components=1

# Download Linux Kernel
wget -O- "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-${KERNEL_VERSION}.tar.xz" | tar xJ

# Download booloader
# Revision 1fea854771f9aee405c4ae204c0e0f912318da6f supports bare metal gcc, otherwise hosted toolchain should be used
mkdir -p boot-wrapper-aarch64
wget -O- "https://git.kernel.org/pub/scm/linux/kernel/git/mark/boot-wrapper-aarch64.git/snapshot/boot-wrapper-aarch64-1fea854771f9aee405c4ae204c0e0f912318da6f.tar.gz" | tar xzC boot-wrapper-aarch64 --strip-components=1

# Download DTS tooling
mkdir -p devicetree-rebasing
wget -O- "https://git.kernel.org/pub/scm/linux/kernel/git/devicetree/devicetree-rebasing.git/snapshot/devicetree-rebasing-6.9-dts.tar.gz" | tar xzC devicetree-rebasing --strip-components=1


# Builds the Linux kernel.
cd linux-${KERNEL_VERSION}
make ARCH=arm64 CROSS_COMPILE=${CROSS_COMPILE} defconfig
make ARCH=arm64 CROSS_COMPILE=${CROSS_COMPILE} -j$(nproc) Image
cd ..

# Builds the device tree.
cd devicetree-rebasing
make CPP=${CROSS_COMPILE}cpp src/arm64/arm/fvp-base-revc.dtb
cd ..

# Builds the bootloader.
cd boot-wrapper-aarch64
export PATH=$(dirname ${CROSS_COMPILE}gcc):$PATH
autoreconf -i
./configure --host=${TOOLCHAIN_TYPE} \
    --enable-psci \
    --enable-gicv3 \
    --with-kernel-dir=../linux-${KERNEL_VERSION} \
    --with-dtb=../devicetree-rebasing/src/arm64/arm/fvp-base-revc.dtb \
    --with-cmdline="console=ttyAMA0 earlycon=pl011,0x1c090000 panic=1 root=/dev/vda rw init=/bin/bash -- /root/startup"
make -j$(nproc)
cd ..

mv boot-wrapper-aarch64/linux-system.axf .

# Cleans up.
rm -rf \
    ${TOOLCHAIN_DIR} \
    linux-${KERNEL_VERSION} \
    devicetree-rebasing \
    boot-wrapper-aarch64
