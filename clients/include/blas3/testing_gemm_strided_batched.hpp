/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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

/* ============================================================================================ */
template <typename T>
void testing_gemm_strided_batched_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_strided_batched_fn = arg.api == FORTRAN
                                                   ? rocblas_gemm_strided_batched<T, true>
                                                   : rocblas_gemm_strided_batched<T, false>;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const rocblas_int M = 100;
        const rocblas_int N = 101;
        const rocblas_int K = 102;

        const rocblas_int lda = 103;
        const rocblas_int ldb = 103;
        const rocblas_int ldc = 103;

        const rocblas_int batch_count = 1;

        device_vector<T> alpha_d(1), beta_d(1), one_d(1), zero_d(1);

        const T alpha_h(1), beta_h(2), one_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* one   = &one_h;
        const T* zero  = &zero_h;

        rocblas_int A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
        rocblas_int A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
        rocblas_int B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
        rocblas_int B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

        const rocblas_int stride_a = lda * A_col;
        const rocblas_int stride_b = ldb * B_col;
        const rocblas_int stride_c = ldc * N;

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

        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
        int32_t           solution_index = 0;
        rocblas_int       flags          = 0;

        const size_t safe_size = stride_c;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // Allocate device memory
        device_strided_batch_matrix<T> dA(A_row, A_col, lda, stride_a, batch_count);
        device_strided_batch_matrix<T> dB(B_row, B_col, ldb, stride_b, batch_count);
        device_strided_batch_matrix<T> dC(M, N, ldc, stride_c, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());

        // clang-format off

// check for valid enum
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, (rocblas_operation) rocblas_side_both, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_value);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, (rocblas_operation) rocblas_side_both, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_value);

// check for invalid size
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, -1, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, -1, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, -1, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, -1), rocblas_status_invalid_size);

// check for invalid leading dimension
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, alpha,
dA, M-1, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, alpha,
dA, lda, stride_a, dB, K-1, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, alpha,
dA, K-1, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, alpha,
dA, lda, stride_a, dB, N-1, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, M-1, stride_c, batch_count), rocblas_status_invalid_size);

// check that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(nullptr, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_handle);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, nullptr,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, alpha,
nullptr, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, nullptr, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, nullptr, dC, ldc, stride_c, batch_count), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, nullptr, ldc, stride_c, batch_count), rocblas_status_invalid_pointer);

// If batch_count==0, then all pointers can be nullptr without issue
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, nullptr,
nullptr, lda, stride_a, nullptr, ldb, stride_b, nullptr, nullptr, ldc, stride_c, 0), rocblas_status_success);

// If M==0, then all pointers can be nullptr without issue
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, 0, N, K, nullptr,
nullptr, lda, stride_a, nullptr, ldb, stride_b, nullptr, nullptr, ldc, stride_c, batch_count), rocblas_status_success);

// If N==0, then all pointers can be nullptr without issue
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, 0, K, nullptr,
nullptr, lda, stride_a, nullptr, ldb, stride_b, nullptr, nullptr, ldc, stride_c, batch_count), rocblas_status_success);

// If alpha==0 and beta==1, then A, B and C can be nullptr without issue
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, zero,
nullptr, lda, stride_a, nullptr, ldb, stride_b, one, nullptr, ldc, stride_c, batch_count), rocblas_status_success);

// the following tests still output to C

// If K==0, then alpha, A and B can both be nullptr without issue.
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, 0, nullptr,
nullptr, lda, stride_a, nullptr, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_success);

// If alpha==0, then A and B can both be nullptr without issue.
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle, transA, transB, M, N, K, zero,
nullptr, lda, stride_a, nullptr, ldb, stride_b, beta, dC, ldc, stride_c, batch_count), rocblas_status_success);

        // clang-format on
    }
}

template <typename T>
void testing_gemm_strided_batched(const Arguments& arg)
{
    auto rocblas_gemm_strided_batched_fn = arg.api == FORTRAN
                                               ? rocblas_gemm_strided_batched<T, true>
                                               : rocblas_gemm_strided_batched<T, false>;

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    rocblas_int stride_a    = arg.stride_a;
    rocblas_int stride_b    = arg.stride_b;
    rocblas_int stride_c    = arg.stride_c;
    rocblas_int batch_count = arg.batch_count;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_local_handle handle{arg};

    rocblas_int A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
    rocblas_int A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
    rocblas_int B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

    // check here to prevent undefined memory allocation error
    // Note: K==0 is not an early exit, since C must still be multiplied by beta
    bool invalid_size
        = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              K,
                                                              nullptr,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              ldb,
                                                              stride_b,
                                                              nullptr,
                                                              nullptr,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;

    double rocblas_error = 0.0;

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        device_vector<T> dA(1);
        device_vector<T> dB(1);
        device_vector<T> dC(1);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_fn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            &h_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            &h_beta,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));
        return;
    }
#endif

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(A_row, A_col, lda, stride_a, batch_count);
    host_strided_batch_matrix<T> hB(B_row, B_col, ldb, stride_b, batch_count);
    host_strided_batch_matrix<T> hC_1(M, N, ldc, stride_c, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(A_row, A_col, lda, stride_a, batch_count);
    device_strided_batch_matrix<T> dB(B_row, B_col, ldb, stride_b, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_c, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);

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

    if(arg.unit_check || arg.norm_check)
    {
        host_strided_batch_matrix<T> hC_2(M, N, ldc, stride_c, batch_count);
        host_strided_batch_matrix<T> hC_gold(M, N, ldc, stride_c, batch_count);
        CHECK_HIP_ERROR(hC_2.memcheck());
        CHECK_HIP_ERROR(hC_gold.memcheck());

        // copy data from CPU to device
        hC_2.copy_from(hC_1);
        hC_gold.copy_from(hC_1);

        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dC.transfer_from(hC_1));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_fn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            &h_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            &h_beta,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));
        handle.post_test(arg);

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dC.transfer_from(hC_2));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_fn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            d_beta,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < batch_count; b++)
        {
            cblas_gemm<T>(
                transA, transB, M, N, K, h_alpha, hA[b], lda, hB[b], ldb, h_beta, hC_gold[b], ldc);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // fetch GPU
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        if(arg.unit_check)
        {
            if(std::is_same_v<T, rocblas_half> && (rocblas_handle(handle)->getArchMajor() == 11))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<T, T, T>;
                near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_1, batch_count, tol);
                near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_2, batch_count, tol);
            }
            else if(std::is_same_v<T, rocblas_half> && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<T>;
                near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_1, batch_count, tol);
                near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_2, batch_count, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
                unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_2, batch_count);
            }
        }

        if(arg.norm_check)
        {
            double error_hst_ptr = std::abs(
                norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC_1, batch_count));
            double error_dev_ptr = std::abs(
                norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC_2, batch_count));
            rocblas_error = error_hst_ptr > error_dev_ptr ? error_hst_ptr : error_dev_ptr;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_fn(handle,
                                                                transA,
                                                                transB,
                                                                M,
                                                                N,
                                                                K,
                                                                &h_alpha,
                                                                dA,
                                                                lda,
                                                                stride_a,
                                                                dB,
                                                                ldb,
                                                                stride_b,
                                                                &h_beta,
                                                                dC,
                                                                ldc,
                                                                stride_c,
                                                                batch_count));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_strided_batched_fn(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            &h_alpha,
                                            dA,
                                            lda,
                                            stride_a,
                                            dB,
                                            ldb,
                                            stride_b,
                                            &h_beta,
                                            dC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_beta,
                      e_ldb,
                      e_stride_b,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gemm_gflop_count<T>(M, N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error);
    }
}
