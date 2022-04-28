/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "bytes.hpp"
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

template <typename T>
void testing_herk_bad_arg(const Arguments& arg)
{
    auto rocblas_herk_fn
        = arg.fortran ? rocblas_herk<T, real_t<T>, true> : rocblas_herk<T, real_t<T>, false>;

    rocblas_local_handle    handle{arg};
    const rocblas_fill      uplo   = rocblas_fill_upper;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_int       N      = 100;
    const rocblas_int       K      = 100;
    const rocblas_int       lda    = 100;
    const rocblas_int       ldc    = 100;
    using U                        = real_t<T>;
    const U alpha                  = 1.0;
    const U beta                   = 1.0;

    size_t cols = (transA == rocblas_operation_none ? std::max(K, 1) : N);
    size_t rows = (transA != rocblas_operation_none ? std::max(K, 1) : N);

    // Allocate device memory
    device_matrix<T> dA(rows, cols, lda);
    device_matrix<T> dC(N, N, ldc);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_herk_fn)(nullptr, uplo, transA, N, K, &alpha, dA, lda, &beta, dC, ldc),
        rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_herk_fn)(handle, rocblas_fill_full, transA, N, K, &alpha, dA, lda, &beta, dC, ldc),
        rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS((rocblas_herk_fn)(handle,
                                            uplo,
                                            rocblas_operation_transpose,
                                            N,
                                            K,
                                            &alpha,
                                            dA,
                                            lda,
                                            &beta,
                                            dC,
                                            ldc),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_herk_fn)(handle, uplo, transA, N, K, nullptr, dA, lda, &beta, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_herk_fn)(handle, uplo, transA, N, K, &alpha, nullptr, lda, &beta, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_herk_fn)(handle, uplo, transA, N, K, &alpha, dA, lda, nullptr, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_herk_fn)(handle, uplo, transA, N, K, &alpha, dA, lda, &beta, nullptr, ldc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        (rocblas_herk_fn)(handle, uplo, transA, 0, K, nullptr, nullptr, lda, nullptr, nullptr, ldc),
        rocblas_status_success);
}

template <typename T>
void testing_herk(const Arguments& arg)
{
    auto rocblas_herk_fn
        = arg.fortran ? rocblas_herk<T, real_t<T>, true> : rocblas_herk<T, real_t<T>, false>;

    rocblas_local_handle handle{arg};
    rocblas_fill         uplo   = char2rocblas_fill(arg.uplo);
    rocblas_operation    transA = char2rocblas_operation(arg.transA);
    rocblas_int          N      = arg.N;
    rocblas_int          K      = arg.K;
    rocblas_int          lda    = arg.lda;
    rocblas_int          ldc    = arg.ldc;
    using U                     = real_t<T>;
    U alpha                     = arg.get_alpha<U>();
    U beta                      = arg.get_beta<U>();

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = N < 0 || K < 0 || ldc < N || (transA == rocblas_operation_none && lda < N)
                        || (transA != rocblas_operation_none && lda < K);
    if(N == 0 || invalid_size)
    {
        // ensure invalid sizes checked before pointer check
        EXPECT_ROCBLAS_STATUS(
            (rocblas_herk<
                T>)(handle, uplo, transA, N, K, nullptr, nullptr, lda, nullptr, nullptr, ldc),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    size_t cols = (transA == rocblas_operation_none ? K : N);
    size_t rows = (transA == rocblas_operation_none ? N : K);

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(rows, cols, lda);
    host_matrix<T> hC_1(N, N, ldc);
    host_matrix<T> hC_2(N, N, ldc);
    host_matrix<T> hC_gold(N, N, ldc);
    host_vector<U> h_alpha(1);
    host_vector<U> h_beta(1);

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Allocate device memory
    device_matrix<T> dA(rows, cols, lda);
    device_matrix<T> dC(N, N, ldc);
    device_vector<U> d_alpha(1);
    device_vector<U> d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_triangular_matrix, true, true);
    rocblas_init_matrix(
        hC_1, arg, rocblas_client_beta_sets_nan, rocblas_client_hermitian_matrix, false, true);

    hC_2    = hC_1;
    hC_gold = hC_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dC.transfer_from(hC_1));

    if(arg.unit_check || arg.norm_check)
    {
        // host alpha/beta
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(
            (rocblas_herk<
                T>)(handle, uplo, transA, N, K, &h_alpha[0], dA, lda, &h_beta[0], dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // device alpha/beta
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));

        CHECK_ROCBLAS_ERROR(
            (rocblas_herk_fn)(handle, uplo, transA, N, K, d_alpha, dA, lda, d_beta, dC, ldc));

        // CPU BLAS
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        cblas_herk<T>(uplo, transA, N, K, h_alpha[0], hA, lda, h_beta[0], hC_gold, ldc);

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        if(arg.unit_check)
        {
            const double tol = K * sum_error_tolerance<T>;
            near_check_general<T>(N, N, ldc, hC_gold, hC_1, tol);
            near_check_general<T>(N, N, ldc, hC_gold, hC_2, tol);
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC_1));
            auto err2     = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC_2));
            rocblas_error = err1 > err2 ? err1 : err2;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_herk_fn(handle, uplo, transA, N, K, h_alpha, dA, lda, h_beta, dC, ldc);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_herk_fn(handle, uplo, transA, N, K, h_alpha, dA, lda, h_beta, dC, ldc);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_transA, e_N, e_K, e_alpha, e_lda, e_beta, e_ldc>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            herk_gflop_count<T>(N, K),
            ArgumentLogging::NA_value,
            cpu_time_used,
            rocblas_error);
    }
}
