//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Example usage for matrix multiplication of two single precision floating-point (FP32) matrices and the accumulation
// of the result into an FP32 destination matrix.
//
// The activations and the weights, stored in the LHS and RHS matrices respectively, are both non-transposed matrices.
// The matrix multiplication computation is performed using vector instructions present in the FEAT_BF16 Arm®
// architecture feature.
//
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_BF16.
#else
#include <arm_neon.h>

#include <algorithm>
#include <bitset>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

// Include micro-kernel variants
#include "kai/kai_common.h"
#include "kai_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla.h"
#include "kai_rhs_pack_kxn_f32p4x24biasf32_f32_bf16_neon.h"
#include "matmul_clamp_f32_f32_bf16p_interface.h"

#define FLOAT16_MIN (FLT_MIN)
#define FLOAT16_MAX (FLT_MAX)

/** Convert bfloat16 to float
 *
 * @param[in] v Bfloat16 value to convert to float
 *
 * @return Converted value
 */
inline float bf16_to_float(const bfloat16_t* v) {
    const uint32_t lv = ((*reinterpret_cast<const uint16_t*>(v)) << 16);
    float fp;
    memcpy(&fp, &lv, sizeof(lv));
    return fp;
}

inline float bf16_to_float(uint16_t v) {
    const uint32_t lv = (v << 16);
    float fp;
    memcpy(&fp, &lv, sizeof(lv));
    return fp;
}

namespace {
/// Micro-kernel interface

// a64
constexpr kai_matmul_clamp_f32_f32_bf16p_ukernel ukernel{
    kai_get_m_step_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_get_n_step_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_get_mr_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_get_nr_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_get_kr_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_get_sr_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_get_lhs_offset_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_get_rhs_packed_offset_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_get_dst_offset_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_get_dst_size_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla,
    kai_run_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla};

float truncate(float x) {
    uint32_t uval = (*reinterpret_cast<uint32_t*>(&x) & 0xffff0000);
    return *reinterpret_cast<float*>(&uval);
}

/// Reference implementation of matrix multiplication
void run_matmul_ref(
    size_t m, size_t n, size_t k, const float* lhs, const float* rhs, const float* bias, float* dst, float scalar_min,
    float scalar_max) {
    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            float acc = bias[col_idx];

            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                float lhs_val = truncate(lhs[row_idx * k + k_idx]);
                float rhs_val = truncate(rhs[col_idx + n * k_idx]);

                acc += lhs_val * rhs_val;
            }
            acc = std::max(acc, scalar_min);
            acc = std::min(acc, scalar_max);

            dst[row_idx * n + col_idx] = acc;
        }
    }
}

/// Fills the matrix with incremental values
void fill_matrix(size_t num_rows, size_t num_cols, float* dst, const float weight) {
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        dst[i] = float((i + 1) * weight);
    }
}

void fill_identity(size_t num_rows, size_t num_cols, float* dst, const float weight) {
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        int col = i % num_cols;
        int row = i / num_cols;

        dst[i] = (col == row ? 1.f : 0.f);
    }
}

/// Print the matrix
void print_matrix(size_t num_rows, size_t num_cols, const char* name, const float* src) {
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

void print_mixed_prec_matrix(
    size_t num_rows, size_t num_cols, const char* name, const bfloat16_t* src, int nr, int stride) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        uint8_t* src_row = ((uint8_t*)src) + stride * y;
        for (size_t x = 0; x < num_cols; ++x) {
            if (x >= nr) {
                // print bfloat
                bfloat16_t* src_elm =
                    reinterpret_cast<bfloat16_t*>(src_row + nr * sizeof(float) + (x - nr) * sizeof(bfloat16_t));
                std::cout << std::setprecision(2) << std::fixed << bf16_to_float(src_elm) << ", ";
            } else {
                // print float
                float* src_elm = reinterpret_cast<float*>(src_row + x * sizeof(float));
                std::cout << std::setprecision(2) << std::fixed << *src_elm << ", ";
            }
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

void print_bf_matrix(size_t num_rows, size_t num_cols, const char* name, const float* src) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        for (size_t x = 0; x < num_cols; ++x) {
            std::cout << std::setprecision(2) << std::fixed << truncate(src[y * num_cols + x]) << ", ";
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

// void print_matrix(size_t num_rows, size_t num_cols, const char* name, const bfloat16_t* src) {
//     std::cout << name << " = [\n";
//     for (size_t y = 0; y < num_rows; ++y) {
//         std::cout << "  [";
//         for (size_t x = 0; x < num_cols; ++x) {
//             std::cout << std::setprecision(2) << std::fixed << src[y * num_cols + x] << ", ";
//         }
//         std::cout << ("],\n");
//     }
//     std::cout << ("]\n\n");
// }

/// Verify the micro-kernel output matches the reference implementation
bool is_output_correct(
    size_t num_rows, size_t num_cols, const float rel_tolerance, const float* ref, const float* act) {
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
    // Parameters of the matrix multiplication. Change these values to see how the micro-kernels operate on different
    // sized matrices
    const size_t M = 4;   // Rows of LHS and DST matrices
    const size_t N = 10;  // Columns of RHS and DST matrices, and length of the Bias vector.
    const size_t K = 5;   // Columns of LHS, rows of RHS matrices

    const size_t lhs_size = M * K;
    const size_t rhs_size = N * K;
    const size_t bias_size = N;
    const size_t dst_size = M * N;

    // Allocate the memory
    float* lhs = new float[lhs_size];
    float* rhs = new float[rhs_size];
    float* bias = new float[bias_size];

    fill_matrix(M, K, lhs, 0.1);
    fill_matrix(K, N, rhs, 0.1);
    fill_matrix(1, N, bias, 1);

#ifdef KAI_DEBUG
    print_matrix(M, K, "lhs", lhs);
    print_matrix(K, N, "rhs", rhs);
    print_matrix(1, N, "bias", bias);

    // Print bf16 converted values
    print_bf_matrix(M, K, "lhs", lhs);
    print_bf_matrix(K, N, "rhs", rhs);
#endif  // KAI_DEBUG

    //----------- REFERENCE IMPLEMENTATION
    //------------------------------------
    //------------------------------------
    float* dst_ref = new float[dst_size];

    run_matmul_ref(
        M, N, K,                  // Dimensions
        lhs,                      // LHS buffer
        rhs,                      // RHS buffer
        bias,                     // Bias buffer
        dst_ref,                  // DST
        FLOAT16_MIN, FLOAT16_MAX  // Min and max for the clamp operation
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
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p4x24biasf32_f32_bf16_neon(N, K);
    const size_t rhs_packed_cols = nr + kai_roundup(K, kr) * nr;

    int rhs_packed_stride = nr * sizeof(float) + kai_roundup(K, kr) * nr * sizeof(bfloat16_t);

    // Each col has nr floats and then K*nr bfloats
    bfloat16_t* rhs_packed = new bfloat16_t[rhs_packed_size];

    const size_t lhs_stride = K * sizeof(float);
    const size_t rhs_stride = N * sizeof(float);
    const size_t dst_stride_row = N * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
    kai_run_rhs_pack_kxn_f32p4x24biasf32_f32_bf16_neon(
        1, N, K, nr, kr, sr,  // Packing arguments
        rhs_stride,           // RHS stride
        rhs,                  // RHS
        bias,                 // Bias
        NULL,                 // Scale
        rhs_packed,           // RHS packed
        0, NULL);

    // The RHS and Bias buffers can be freed after packing, however we reuse them for the reference test below
#ifdef KAI_DEBUG
    const size_t rhs_packed_rows = rhs_packed_size / rhs_packed_stride;
    print_mixed_prec_matrix(rhs_packed_rows, rhs_packed_cols, "rhs_packed", rhs_packed, nr, rhs_packed_stride);
#endif  // KAI_DEBUG

    float* dst = new float[dst_size];

    const auto timer_matmul_start = std::chrono::high_resolution_clock::now();

    ukernel.run_matmul(
        M, N, K,          // Dimensions
        lhs,              // LHS packed
        lhs_stride,       // Lhs stride
        rhs_packed,       // RHS packed
        dst,              // DST
        dst_stride_row,   // DST stride (row)
        dst_stride_col,   // DST stride (col)
        FLT_MIN, FLT_MAX  // Min and max for the clamp operation
    );

    const auto timer_matmul_end = std::chrono::high_resolution_clock::now();
    const auto time_matmul =
        std::chrono::duration_cast<std::chrono::nanoseconds>(timer_matmul_end - timer_matmul_start);

#ifdef KAI_DEBUG
    print_matrix(M, N, "dst", dst);
    print_matrix(M, N, "ref", dst_ref);
#endif  // KAI_DEBUG

    const bool is_valid = is_output_correct(M, N, 0.02 /* rel tol */, dst_ref, dst);

    std::cout << "TEST[matmul_clamp_f32_f32_bf16p]\n";
    std::cout << "- ukernel: kai_matmul_clamp_f32_f32_bf16p4x1biasf32_4x24x4_neon_mmla\n";
    if (is_valid) {
        std::cout << "- Status: PASSED\n";
        std::cout << "- Performance: " << time_matmul.count() << "ns\n";
    } else {
        std::cout << "- Status: FAILED\n";
        return 1;
    }

    //----------- END MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------

    delete[] lhs;
    delete[] rhs;
    delete[] bias;
    delete[] rhs_packed;
    delete[] dst;
    delete[] dst_ref;

    return 0;
}

#endif  // Architectural features check.
