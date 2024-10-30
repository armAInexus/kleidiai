//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Example usage for matrix multiplication of two half-precision brain floating-point (BF16) matrices
// and the accumulation of the result into an FP16 destination matrix.
//
// The activations and the weights, stored in the LHS and RHS matrices respectively, are both non-transposed matrices.
// The matrix multiplication computation is performed using BF16 matrix multiply (BFMMLA)
// vector instructions present in the FEAT_BF16 Arm® architecture feature.
//
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_BF16.
#else
#include <arm_neon.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>

// Include micro-kernel variants
#include "kai/kai_common.h"
#include "kai_lhs_quant_pack_bf16p8x4_f16_neon.h"
#include "kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai_matmul_clamp_f16_bf16p_bf16p_interface.h"
#include "kai_rhs_quant_pack_kxn_bf16p12x4biasf16_f16_neon.h"

inline float bf16_to_float(const bfloat16_t* v) {
    const uint16_t uint_rep = *reinterpret_cast<const uint16_t*>(v);
    return kai_cast_f32_bf16(uint_rep);
}

namespace {
/// Micro-kernel interface
constexpr kai_matmul_clamp_f16_bf16p_bf16p_ukernel ukernel{
    kai_get_m_step_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_get_n_step_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_get_mr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_get_nr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_get_kr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_get_sr_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_get_lhs_packed_offset_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_get_rhs_packed_offset_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_get_dst_offset_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_get_dst_size_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    kai_run_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla};

/// @brief Truncate the 32-bit floating point number's least significant 16 mantissa bits
/// @param x floating-point number
/// @return truncated floating-point number
float truncate(float x) {
    uint32_t uval = (*reinterpret_cast<uint32_t*>(&x) & 0xffff0000);
    return *reinterpret_cast<float*>(&uval);
}

/// Reference implementation of matrix multiplication
template <typename Tin>
void run_matmul_ref(
    size_t m, size_t n, size_t k, const Tin* lhs, const Tin* rhs, const Tin* bias, float16_t* dst, float scalar_min,
    float scalar_max) {
    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            float acc = bias[col_idx];

            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                float lhs_val = truncate(static_cast<float>(lhs[row_idx * k + k_idx]));
                float rhs_val = truncate(static_cast<float>(rhs[col_idx + n * k_idx]));

                acc += lhs_val * rhs_val;
            }

            dst[row_idx * n + col_idx] = static_cast<float16_t>(std::clamp(acc, scalar_min, scalar_max));
        }
    }
}

/// Fills the matrix with incremental values
template <typename T>
void fill_matrix(size_t num_rows, size_t num_cols, T* dst, const float weight) {
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        dst[i] = static_cast<T>((i + 1) * weight);
    }
}

/// Print the matrix
template <typename T>
void print_matrix(size_t num_rows, size_t num_cols, const char* name, const T* src) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        for (size_t x = 0; x < num_cols; ++x) {
            std::cout << std::setprecision(2) << std::fixed << src[y * num_cols + x] << ", ";
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

void print_matrix(size_t num_rows, size_t num_cols, const char* name, const bfloat16_t* src) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        for (size_t x = 0; x < num_cols; ++x) {
            std::cout << std::setprecision(2) << std::fixed << bf16_to_float(&src[y * num_cols + x]) << ", ";
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

template <typename T>
void print_mixed_prec_matrix(
    size_t num_rows, size_t num_cols, const char* name, const uint8_t* src, int nr, int stride) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        const uint8_t* src_row = src + stride * y;
        for (size_t x = 0; x < num_cols; ++x) {
            if (x >= nr) {
                // print bfloat
                const bfloat16_t* src_elm =
                    reinterpret_cast<const bfloat16_t*>(src_row + nr * sizeof(T) + (x - nr) * sizeof(bfloat16_t));
                std::cout << std::setprecision(2) << std::fixed << bf16_to_float(src_elm) << ", ";
            } else {
                // print float
                const T* src_elm = reinterpret_cast<const T*>(src_row + x * sizeof(T));
                std::cout << std::setprecision(2) << std::fixed << *src_elm << ", ";
            }
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

template <typename T>
void print_bf_matrix(size_t num_rows, size_t num_cols, const char* name, const T* src) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        for (size_t x = 0; x < num_cols; ++x) {
            std::cout << std::setprecision(2) << std::fixed << truncate(static_cast<T>(src[y * num_cols + x])) << ", ";
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

/// Verify the micro-kernel output matches the reference implementation
bool is_output_correct(
    size_t num_rows, size_t num_cols, const float rel_tolerance, const float16_t* ref, const float16_t* act) {
    bool is_valid = true;

    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if (std::fabs(ref[i] - act[i]) / (act[i] + 1e-10) > rel_tolerance) {
            const size_t x = i % num_cols;
            const size_t y = i / num_cols;

            std::cout << std::setprecision(5) << std::fixed << "ERROR![" << y << "][" << x << "]: ref=" << ref[i]
                      << " vs. act=" << act[i] << "\n";

            is_valid = false;
        }
    }
    return is_valid;
}
}  // namespace

int main() {
    int ret = 0;

    // Parameters of the matrix multiplication. Change these values to see how the micro-kernels operate on different
    // sized matrices
    const size_t M = 23;  // Rows of LHS and DST matrices
    const size_t N = 37;  // Columns of RHS and DST matrices, and length of the Bias vector.
    const size_t K = 43;  // Columns of LHS, rows of RHS matrices

    const size_t lhs_size = M * K;
    const size_t rhs_size = N * K;
    const size_t bias_size = N;
    const size_t dst_size = M * N;

    // Allocate the memory
    float16_t* lhs = new float16_t[lhs_size];
    float16_t* rhs = new float16_t[rhs_size];
    float16_t* bias = new float16_t[bias_size];

    fill_matrix(M, K, lhs, 0.001);
    fill_matrix(K, N, rhs, 0.001);
    fill_matrix(1, N, bias, 0.01);

#ifdef KAI_DEBUG
    // std::cout << "Floats: " << std::endl;
    print_matrix(M, K, "lhs", lhs);
    print_matrix(K, N, "rhs", rhs);
    print_matrix(1, N, "bias", bias);

    // Print bf16 converted values
    print_bf_matrix(M, K, "lhs_bf", lhs);
    print_bf_matrix(K, N, "rhs_bf", rhs);
#endif  // KAI_DEBUG

    //----------- REFERENCE IMPLEMENTATION
    //------------------------------------
    //------------------------------------
    float16_t* dst_ref = new float16_t[dst_size];

    run_matmul_ref(
        M, N, K,           // Dimensions
        lhs,               // LHS buffer
        rhs,               // RHS buffer
        bias,              // Bias buffer
        dst_ref,           // DST
        -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
    );
    //----------- END REFERENCE IMPLEMENTATION
    //------------------------------------
    //------------------------------------

    //----------- MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------
    const size_t mr = ukernel.get_mr();
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();

    // In a single row, we pack nr bias values followed by K rows of nr RHS values
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf16_f16_neon(N, K);
    uint8_t* rhs_packed = new uint8_t[rhs_packed_size];

    const size_t lhs_stride = K * sizeof(float16_t);
    const size_t rhs_stride = N * sizeof(float16_t);
    const size_t dst_stride_row = N * sizeof(float16_t);
    const size_t dst_stride_col = sizeof(float16_t);

    const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f16_neon(M, K, mr, kr, sr);
    bfloat16_t* lhs_packed = new bfloat16_t[lhs_packed_size];

    // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
    kai_run_rhs_quant_pack_kxn_bf16p12x4biasf16_f16_neon(
        1, N, K, nr, kr, sr,  // Packing arguments
        rhs_stride,           // RHS stride
        rhs,                  // RHS
        bias,                 // Bias
        NULL,                 // Scale
        rhs_packed,           // RHS packed
        0, NULL);

    // The RHS and Bias buffers can be freed after packing, however we reuse them for the reference test below

#ifdef KAI_DEBUG
    const size_t rhs_packed_cols = nr + kai_roundup(K, kr) * nr;

    // Each col has nr float16s and then K*nr bfloats
    int rhs_packed_stride = nr * sizeof(float16_t) + kai_roundup(K, kr) * nr * sizeof(bfloat16_t);
    const size_t rhs_packed_rows = rhs_packed_size / rhs_packed_stride;

    print_mixed_prec_matrix<float16_t>(
        rhs_packed_rows, rhs_packed_cols, "rhs_packed", rhs_packed, nr, rhs_packed_stride);
#endif  // KAI_DEBUG

    float16_t* dst = new float16_t[dst_size];

    const auto timer_matmul_start = std::chrono::high_resolution_clock::now();

    kai_run_lhs_quant_pack_bf16p8x4_f16_neon(M, K, mr, kr, sr, 0 /* m_idx_start */, lhs, lhs_stride, lhs_packed);

    ukernel.run_matmul(
        M, N, K,           // Dimensions
        lhs_packed,        // LHS packed
        rhs_packed,        // RHS packed
        dst,               // DST
        dst_stride_row,    // DST stride (row)
        dst_stride_col,    // DST stride (col)
        -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
    );

    const auto timer_matmul_end = std::chrono::high_resolution_clock::now();
    const auto time_matmul =
        std::chrono::duration_cast<std::chrono::nanoseconds>(timer_matmul_end - timer_matmul_start);

#ifdef KAI_DEBUG
    int num_lhs_rows = (M + mr - 1) / mr;
    int num_lhs_cols = mr * kai_roundup(K, kr);

    print_matrix(num_lhs_rows, num_lhs_cols, "lhs_packed", lhs_packed);
    print_matrix(M, N, "dst", dst);
    print_matrix(M, N, "ref", dst_ref);
#endif  // KAI_DEBUG

    constexpr float rel_tolerance = 0.02;  // This value was chosen by experimentation
    const bool is_valid = is_output_correct(M, N, rel_tolerance, dst_ref, dst);

    std::cout << "TEST[matmul_clamp_f16_bf16p_bf16p]\n";
    std::cout << "- ukernel: matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla\n";
    if (is_valid) {
        std::cout << "- Status: PASSED\n";
        std::cout << "- Performance: " << time_matmul.count() << "ns\n";
    } else {
        std::cout << "- Status: FAILED\n";
        ret = 1;
    }

    //----------- END MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------

    delete[] lhs;
    delete[] rhs;
    delete[] bias;
    delete[] lhs_packed;
    delete[] rhs_packed;
    delete[] dst;
    delete[] dst_ref;

    return ret;
}

#endif  // Architectural features check.
