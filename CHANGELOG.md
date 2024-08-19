<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Changelog

KleidiAI follows the [Semantic Versioning](https://semver.org/) specification for releases.

## v0.2.0

- Micro-kernels to compute the matrix multiplication of dynamically quantized symmetric signed 8-bit integer with
  per-block quantization (QSI8D32) activations and quantized symmetric 4-bit signed integer with per-block quantization
  (QSI4C32) weights and the accumulation of the result into a single-precision (F32) output,
  optimized for Arm® Neon™ technology.
- Tensor packing micro-kernels to prepare the activations and weights for input to the above matrix multiplication
  micro-kernel.
- Unit test and example for integer micro-kernels.

## v0.1.0

The first release of KleidiAI includes:

- Micro-kernels to compute the matrix multiplication of:
  - Dynamically quantized 8-bit integer (QAI8DX) activations and quantized 4-bit integer (QSI4CX) weights and the
    accumulation of the result into a single-precision (F32) output, optimized for Arm® Neon™ technology.
  - Half precision floating-point (F16) activations and weights and the accumulation of the result into an F16 output,
    optimized for Neon technology.
  - F32 activations and weights and the accumulation of the result into an F32 output, optimized for SME2 technology.
- Tensor packing micro-kernels to prepare the activations and weights for input to the above matrix multiplication
  micro-kernels.
- Examples and documentation demonstrating the usage of the 4-bit integer and 16-bit floating point matrix
  multiplication micro-kernels.
- Testing suite.
- CMake and Bazel build system for micro kernels.
