/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include "blas3/Tensile/gemm.hpp"

template <typename T>
void testing_gemm_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_fn = arg.api == FORTRAN ? rocblas_gemm<T, true> : rocblas_gemm<T, false>;

        const rocblas_int M = 100;
        const rocblas_int N = 101;
        const rocblas_int K = 102;

        const rocblas_int lda = 103;
        const rocblas_int ldb = 103;
        const rocblas_int ldc = 103;

        device_vector<T> alpha_d(1), beta_d(1), one_d(1), zero_d(1);

        const T alpha_h(1), beta_h(2), one_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* one   = &one_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        rocblas_int A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
        rocblas_int A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
        rocblas_int B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
        rocblas_int B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // Allocate device memory
        device_matrix<T> dA(A_row, A_col, lda);
        device_matrix<T> dB(B_row, B_col, ldb);
        device_matrix<T> dC(M, N, ldc);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());

        // clang-format off

// check for invalid enum
EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, (rocblas_operation) rocblas_side_both, transB, M, N, K, nullptr,
nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc), rocblas_status_invalid_value);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, (rocblas_operation) rocblas_side_both, M, N, K, nullptr,
nullptr, lda, nullptr, ldb, beta, nullptr, ldc), rocblas_status_invalid_value);

// check for invalid size
EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, -1, N, K, nullptr,
nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, M, -1, K, nullptr,
nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, M, N, -1, nullptr,
nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc), rocblas_status_invalid_size);

// check for invalid leading dimension
EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, rocblas_operation_none, rocblas_operation_none, M, N, K, nullptr,
nullptr, M-1, nullptr, ldb, nullptr, nullptr, ldc), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, rocblas_operation_none, rocblas_operation_none, M, N, K, nullptr,
nullptr, lda, nullptr, K-1, nullptr, nullptr, ldc), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, nullptr,
nullptr, K-1, nullptr, ldb, nullptr, nullptr, ldc), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, nullptr,
nullptr, lda, nullptr, N-1, nullptr, nullptr, ldc), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, M, N, K, nullptr,
nullptr, lda, nullptr, ldb, nullptr, nullptr, M-1), rocblas_status_invalid_size);

// check that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( nullptr, transA, transB, M, N, K, alpha,
dA, lda, dB, ldb, beta, dC, ldc), rocblas_status_invalid_handle);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, M, N, K, nullptr,
dA, lda, dB, ldb, beta, dC, ldc), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, M, N, K, alpha,
nullptr, lda, dB, ldb, beta, dC, ldc), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, M, N, K, alpha,
dA, lda, nullptr, ldb, beta, dC, ldc), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, M, N, K, alpha,
dA, lda, dB, ldb, nullptr, dC, ldc), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, M, N, K, alpha,
dA, lda, dB, ldb, beta, nullptr, ldc), rocblas_status_invalid_pointer);

// If M==0, then all pointers can be nullptr without issue.
EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle, transA, transB, 0, N, K, nullptr,
nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc), rocblas_status_success);

// If N==0, then all pointers can be nullptr without issue.
EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle, transA, transB, M, 0, K, nullptr,
nullptr, lda, nullptr, ldb, nullptr, nullptr, ldc), rocblas_status_success);

// If alpha==0 && beta==1, then A, B and C can be nullptr without issue.
EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle, transA, transB, M, N, K, zero,
nullptr, lda, nullptr, ldb, one, nullptr, ldc), rocblas_status_success);

// the following tests still output to C

// If K==0, then alpha, A and B can be nullptr without issue.
EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle, transA, transB, M, N, 0, nullptr,
nullptr, lda, nullptr, ldb, beta, dC, ldc), rocblas_status_success);

// If alpha==0, then A and B can be nullptr without issue.
EXPECT_ROCBLAS_STATUS( rocblas_gemm_fn( handle, transA, transB, M, N, K, zero,
nullptr, lda, nullptr, ldb, beta, dC, ldc), rocblas_status_success);

        // clang-format on
    }
}

template <typename T>
void testing_gemm(const Arguments& arg)
{
    auto rocblas_gemm_fn = arg.api == FORTRAN ? rocblas_gemm<T, true> : rocblas_gemm<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used      = 0.0;
    double               rocblas_error = 0.0;
    bool                 HMM           = arg.HMM;
    rocblas_local_handle handle{arg};

    rocblas_math_mode math_mode = rocblas_math_mode(arg.math_mode);
    CHECK_ROCBLAS_ERROR(rocblas_set_math_mode(handle, math_mode));
    CHECK_ROCBLAS_ERROR(rocblas_get_math_mode(handle, &math_mode));

    rocblas_int A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
    rocblas_int A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
    rocblas_int B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

    // check here to prevent undefined memory allocation error
    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M;
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle,
                                              transA,
                                              transB,
                                              M,
                                              N,
                                              K,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              nullptr,
                                              ldc),
                              rocblas_status_invalid_size);

        return;
    }

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        device_vector<T> dA(1);
        device_vector<T> dB(1);
        device_vector<T> dC(1);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));
        return;
    }
#endif

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(A_row, A_col, lda);
    host_matrix<T> hB(B_row, B_col, ldb);
    host_matrix<T> hC_1(M, N, ldc);

    // Allocate device memory
    device_matrix<T> dA(A_row, A_col, lda, HMM);
    device_matrix<T> dB(B_row, B_col, ldb, HMM);
    device_matrix<T> dC(M, N, ldc, HMM);
    device_vector<T> d_alpha(1, 1, HMM);
    device_vector<T> d_beta(1, 1, HMM);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC_1, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_1));

    if(arg.unit_check || arg.norm_check)
    {
        host_matrix<T> hC_gold(M, N, ldc);
        hC_gold = hC_1;

        // ROCBLAS rocblas_pointer_mode_host
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            handle.pre_test(arg);
            if(arg.api != INTERNAL)
            {
                CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
                    handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));
            }
            else
            {
                // using arg.stride_x,y,d for offset testing
                rocblas_stride           offsetA = arg.stride_x;
                rocblas_stride           offsetB = arg.stride_y;
                rocblas_stride           offsetC = arg.stride_d;
                constexpr rocblas_stride strideA = 0, strideB = 0, strideC = 0;
                constexpr rocblas_int    batch_count = 1;

                CHECK_ROCBLAS_ERROR(rocblas_internal_gemm_template<T>(handle,
                                                                      transA,
                                                                      transB,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      &h_alpha,
                                                                      (const T*)dA + offsetA,
                                                                      -offsetA,
                                                                      lda,
                                                                      strideA,
                                                                      (const T*)dB + offsetB,
                                                                      -offsetB,
                                                                      ldb,
                                                                      strideB,
                                                                      &h_beta,
                                                                      (T*)dC + offsetC,
                                                                      -offsetC,
                                                                      ldc,
                                                                      strideC,
                                                                      batch_count));
            }
            handle.post_test(arg);
            CHECK_HIP_ERROR(hC_1.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            // gold has copy of hC1
            CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

            // ROCBLAS rocblas_pointer_mode_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

            CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
                handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
        }

        // For the xf32 xdl math op, cast type of A/B from float to xfloat32 .
        if(std::is_same<T, float>{} && math_mode == rocblas_xf32_xdl_math_op)
        {
            type_to_xdl_math_op_type<rocblas_xfloat32, float>(hA.data(), hA.size());
            type_to_xdl_math_op_type<rocblas_xfloat32, float>(hB.data(), hB.size());
        }

        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        ref_gemm<T>(transA, transB, M, N, K, h_alpha, hA, lda, hB, ldb, h_beta, (T*)hC_gold, ldc);

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        //releasing already used host memory
        hA = host_matrix<T>();
        hB = host_matrix<T>();

        // check host error and norm
        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<T,
                                  rocblas_half> && (rocblas_handle(handle)->getArchMajor() == 11))
                {
                    const double tol = K * sum_error_tolerance_for_gfx11<T, T, T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
                }
                else if(std::is_same_v<T, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
                }
            }

            if(arg.norm_check)
            {
                auto err1 = std::abs(norm_check_general<T>('F', M, N, ldc, (T*)hC_gold, (T*)hC_1));
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
            }
        }

        if(arg.pointer_mode_device)
        {
            // fetch device mode GPU results
            CHECK_HIP_ERROR(hC_1.transfer_from(dC));

            if(arg.unit_check)
            {
                if(std::is_same_v<T,
                                  rocblas_half> && (rocblas_handle(handle)->getArchMajor() == 11))
                {
                    const double tol = K * sum_error_tolerance_for_gfx11<T, T, T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
                }
                else if(std::is_same_v<T, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
                }
            }

            if(arg.norm_check)
            {
                auto err1 = std::abs(norm_check_general<T>('F', M, N, ldc, (T*)hC_gold, (T*)hC_1));
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_fn(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_K, e_alpha, e_lda, e_beta, e_ldb, e_ldc>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gemm_gflop_count<T>(M, N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error);
    }
}
