<!--
    SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Changelog

KleidiAI follows the [Semantic Versioning](https://semver.org/) specification for releases.

## v0.4.0 -- Upcoming Release

- Add SME2 F32 GEMV micro-kernel.
- Micro-kernels to compute the matrix multiplication of dynamically quantized 8-bit integer (QAI8DX) LHS matrix, which typically holds the neural network activations, and quantized 4-bit integer (QSI4CX) RHS matrix, which typically holds the neural network weights, and the accumulation of the result into a single-precision (F32) output, optimized using the Arm® CPU feature FEAT_DotProd.

## v0.3.0

- Advanced SIMD FP32 GEMM micro-kernel.
- Micro-kernels to compute the matrix multiplication of dynamically quantized asymmetric signed 8-bit integer with per-row quantization (QAI8DX) LHS and quantized symmetric 4-bit signed integer with per-block quantization (QSI4C32) RHS. The destination matrix data type is single-precision floating-point (F32). The micro-kernels have been optimized using the Arm® CPU feature FEAT_I8MM for the matrix-by-matrix cases and the FEAT_DotProd for the vector-by-matrix cases.
- RHS matrix packing micro-kernels to pack the RHS matrix holding the QSI4C32 values.
- Unit test and example for integer micro-kernels.
- Extend support for signed 4-bit integer inputs in quantized symmetric 4-bit signed integer with per-channel quantization (QSI4CXP) RHS packing micro-kernel.
  - kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0 renamed to kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.
  - kai_rhs_pack_kxn_qsi4cxp_qsu4cxs1s0 renamed to kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.
- Remove FP16 GEMV micro-kernel optimized for Advanced SIMD.
  - Where a dedicated GEMV micro-kernel is not provided, it is recommended to use existing GEMM micro-kernels which have dedicated paths for M=1 (a "GEMV" operation).

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
