<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Create a Linux virtual machine on Fixed Virtual Platform

In order to develop and test KleidiAI microkernels without hardware supporting certain architectural features, we should use [Fixed Virtual Platform (FVP)](https://developer.arm.com/Tools%20and%20Software/Fixed%20Virtual%20Platforms). This guide shows how to create a minimum Linux virtual machine running on FVP with internet connection and the ability to access using SSH.

Note: In this guide, commands that run on the host machine starts with `host>` prompt and commands that run on the FVP starts with `fvp>` prompt.

## Prerequisites

The following tools are needed in the host system:

- `wget` to download the Linux image.
- `xz` to extract the Linux image.
- Container management tool such as Docker.
- Telnet client to connect to the FVP.
- SSH client to connect to the operating system running in the FVP.

## Prepare the FVP and the operating system

Build the Docker image:

```
host> docker build -t fvp .
```

Download a Linux image:

```
host> wget -O disk.img.xz "https://cdimage.ubuntu.com/releases/24.04.1/release/ubuntu-24.04.1-preinstalled-server-arm64+raspi.img.xz"
xz -d disk.img.xz
```

Run the FVP:

```
host> docker run \
  --rm -t -i \
  -p 5000:5000 \
  -p 5001:5001 \
  -p 5002:5002 \
  -p 5003:5003 \
  -p 8022:8022 \
  --mount=type=bind,source="$PWD/disk.img",target=/disk.img \
  fvp \
  /opt/devtools/fvp/models/FVP_Base_RevC-2xAEMvA \
    -C cache_state_modelled=0 \
    -C bp.refcounter.non_arch_start_at_default=1 \
    -C bp.secure_memory=0 \
    -C bp.pl011_uart0.out_file=output.txt \
    -C bp.pl011_uart0.shutdown_tag="System halted" \
    -C bp.terminal_0.mode=telnet \
    -C bp.terminal_0.start_telnet=0 \
    -C bp.terminal_1.mode=raw \
    -C bp.terminal_1.start_telnet=0 \
    -C bp.terminal_2.mode=raw \
    -C bp.terminal_2.start_telnet=0 \
    -C bp.terminal_3.mode=raw \
    -C bp.terminal_3.start_telnet=0 \
    -C bp.smsc_91c111.enabled=1 \
    -C bp.hostbridge.userNetPorts=8022=22 \
    -C bp.hostbridge.userNetworking=1 \
    -C cluster0.NUM_CORES=4 \
    -C cluster0.has_arm_v8-1=1 \
    -C cluster0.has_arm_v8-2=1 \
    -C cluster0.has_arm_v8-3=1 \
    -C cluster0.has_arm_v8-4=1 \
    -C cluster0.has_arm_v8-5=1 \
    -C cluster0.has_arm_v8-6=1 \
    -C cluster0.has_arm_v8-7=1 \
    -C cluster0.has_arm_v8-8=1 \
    -C cluster0.has_arm_v9-0=1 \
    -C cluster0.has_arm_v9-1=1 \
    -C cluster0.has_arm_v9-2=1 \
    -C cluster0.has_arm_v9-3=1 \
    -C cluster0.has_arm_v9-4=1 \
    -C cluster0.has_arm_v9-5=1 \
    -C cluster0.has_sve=1 \
    -C cluster0.sve.has_b16b16=1 \
    -C cluster0.sve.has_sve2=1 \
    -C cluster0.sve.has_sme=1 \
    -C cluster0.sve.has_sme2=1 \
    -C cluster0.sve.has_sme_f16f16=1 \
    -C cluster0.sve.has_sme_fa64=1 \
    -C cluster0.sve.has_sme_lutv2=1 \
    -C cluster0.sve.sme2_version=1 \
    -C cluster0.sve.veclen=2 \
    -C cluster0.sve.sme_veclens_implemented=4 \
    -C bp.virtio_rng.enabled=1 \
    -C bp.virtioblockdevice.image_path=/disk.img \
    -C bp.vis.disable_visualisation=1 \
    -a cluster*.cpu*=/opt/devtools/linux-system.axf
```

It should print out the port for each terminal. For example:

```
terminal_0: Listening for serial connection on port 5000
terminal_1: Listening for serial connection on port 5001
terminal_2: Listening for serial connection on port 5002
terminal_3: Listening for serial connection on port 5003
```

In this case, `terminal_0` is in port 500.
We need to connect to `terminal_0` using `telnet`:

```
host> telnet 127.0.0.1 5000
```

Login with the default username `ubuntu` and password `ubuntu`.
It will prompt to change the password after the first login.

Allow SSH login using password.

```
fvp> sudo rm /etc/ssh/sshd_config.d/50-cloud-init.conf
fvp> sudo systemctl restart ssh
```

## Setup SSH

Copy the public key into the machine:

```
host> ssh-copy-id -p 8022 ubuntu@127.0.0.1
```

Initiating a new SSH connection is very slow since the cryptography primitives used in the key exchange process is computationally heavy.
It's better if the long lived connection can be established and shared.

Putting the following config into `~/.ssh/config` file:

```
Host fvp
  User ubuntu
  HostName 127.0.0.1
  Port 8022
  ControlMaster auto
  ControlPath ~/.ssh/ssh_mux_%h_%p_%r
```

Start the first SSH connection:

```
host> ssh fvp
```

If this connection is kept alive, all subsequent SSH commands can be executed almost instantly.
