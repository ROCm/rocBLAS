/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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
void testing_herk_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_herk_batched_fn = arg.api == FORTRAN ? rocblas_herk_batched<T, real_t<T>, true>
                                                      : rocblas_herk_batched<T, real_t<T>, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_fill      uplo        = rocblas_fill_upper;
        const rocblas_operation transA      = rocblas_operation_none;
        const rocblas_int       N           = 100;
        const rocblas_int       K           = 99;
        const rocblas_int       lda         = 100;
        const rocblas_int       ldc         = 100;
        rocblas_int             batch_count = 2;

        using U = real_t<T>;

        device_vector<U> alpha_d(1), zero_d(1);
        device_vector<U> beta_d(1), one_d(1);

        const U alpha_h(1), zero_h(0);
        const U beta_h(2), one_h(1);

        const U* alpha = &alpha_h;
        const U* zero  = &zero_h;
        const U* beta  = &beta_h;
        const U* one   = &one_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
        }

        size_t cols = (transA == rocblas_operation_none ? std::max(K, 1) : N);
        size_t rows = (transA != rocblas_operation_none ? std::max(K, 1) : N);

        // Allocate device memory
        device_batch_matrix<T> dA(rows, cols, lda, batch_count);
        device_batch_matrix<T> dC(N, N, ldc, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());

        EXPECT_ROCBLAS_STATUS(
            (rocblas_herk_batched<
                T>)(nullptr, uplo, transA, N, K, alpha, dA, lda, beta, dC, ldc, batch_count),
            rocblas_status_invalid_handle);

        // values
        EXPECT_ROCBLAS_STATUS((rocblas_herk_batched_fn)(handle,
                                                        rocblas_fill_full,
                                                        transA,
                                                        N,
                                                        K,
                                                        alpha,
                                                        dA,
                                                        lda,
                                                        beta,
                                                        dC,
                                                        ldc,
                                                        batch_count),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS((rocblas_herk_batched_fn)(handle,
                                                        uplo,
                                                        (rocblas_operation)rocblas_fill_full,
                                                        N,
                                                        K,
                                                        alpha,
                                                        dA,
                                                        lda,
                                                        beta,
                                                        dC,
                                                        ldc,
                                                        batch_count),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS((rocblas_herk_batched_fn)(handle,
                                                        uplo,
                                                        rocblas_operation_transpose,
                                                        N,
                                                        K,
                                                        alpha,
                                                        dA,
                                                        lda,
                                                        beta,
                                                        dC,
                                                        ldc,
                                                        batch_count),
                              rocblas_status_invalid_value);

        // size
        EXPECT_ROCBLAS_STATUS(
            (rocblas_herk_batched<
                T>)(handle, uplo, transA, N, K, alpha, dA, lda - 1, beta, dC, ldc, batch_count),
            rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(
            (rocblas_herk_batched<
                T>)(handle, uplo, transA, N, K, alpha, dA, lda, beta, dC, ldc - 1, batch_count),
            rocblas_status_invalid_size);

        // alpha/beta
        EXPECT_ROCBLAS_STATUS(
            (rocblas_herk_batched<
                T>)(handle, uplo, transA, N, K, nullptr, dA, lda, beta, dC, ldc, batch_count),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            (rocblas_herk_batched<
                T>)(handle, uplo, transA, N, K, alpha, dA, lda, nullptr, dC, ldc, batch_count),
            rocblas_status_invalid_pointer);

        // invalid pointers
        if(pointer_mode == rocblas_pointer_mode_host)
        {
            EXPECT_ROCBLAS_STATUS((rocblas_herk_batched<T>)(handle,
                                                            uplo,
                                                            transA,
                                                            N,
                                                            K,
                                                            alpha,
                                                            nullptr,
                                                            lda,
                                                            beta,
                                                            dC,
                                                            ldc,
                                                            batch_count),
                                  rocblas_status_invalid_pointer);

            EXPECT_ROCBLAS_STATUS((rocblas_herk_batched<T>)(handle,
                                                            uplo,
                                                            transA,
                                                            N,
                                                            K,
                                                            alpha,
                                                            dA,
                                                            lda,
                                                            beta,
                                                            nullptr,
                                                            ldc,
                                                            batch_count),
                                  rocblas_status_invalid_pointer);
        }

        // batch_count==0 quick return for no ops with null pointers
        EXPECT_ROCBLAS_STATUS((rocblas_herk_batched_fn)(handle,
                                                        uplo,
                                                        transA,
                                                        N,
                                                        K,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        nullptr,
                                                        nullptr,
                                                        ldc,
                                                        0),
                              rocblas_status_success);

        // N==0 quick return for no ops with null pointers
        EXPECT_ROCBLAS_STATUS((rocblas_herk_batched_fn)(handle,
                                                        uplo,
                                                        transA,
                                                        0,
                                                        K,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        nullptr,
                                                        nullptr,
                                                        ldc,
                                                        batch_count),
                              rocblas_status_success);

        // k==0 and beta==1 quick return with null pointers
        EXPECT_ROCBLAS_STATUS((rocblas_herk_batched_fn)(handle,
                                                        uplo,
                                                        transA,
                                                        N,
                                                        0,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        one,
                                                        nullptr,
                                                        ldc,
                                                        batch_count),
                              rocblas_status_success);

        // alpha==0 and beta==1 quick return with null pointers
        EXPECT_ROCBLAS_STATUS((rocblas_herk_batched_fn)(handle,
                                                        uplo,
                                                        transA,
                                                        N,
                                                        K,
                                                        zero,
                                                        nullptr,
                                                        lda,
                                                        one,
                                                        nullptr,
                                                        ldc,
                                                        batch_count),
                              rocblas_status_success);
    }
}

template <typename T>
void testing_herk_batched(const Arguments& arg)
{
    auto rocblas_herk_batched_fn = arg.api == FORTRAN ? rocblas_herk_batched<T, real_t<T>, true>
                                                      : rocblas_herk_batched<T, real_t<T>, false>;

    rocblas_local_handle handle{arg};
    rocblas_fill         uplo   = char2rocblas_fill(arg.uplo);
    rocblas_operation    transA = char2rocblas_operation(arg.transA);
    rocblas_int          N      = arg.N;
    rocblas_int          K      = arg.K;
    rocblas_int          lda    = arg.lda;
    rocblas_int          ldc    = arg.ldc;
    using U                     = real_t<T>;
    U           alpha           = arg.get_alpha<U>();
    U           beta            = arg.get_beta<U>();
    rocblas_int batch_count     = arg.batch_count;

    double gpu_time_used, cpu_time_used;
    double error_host   = 0.0;
    double error_device = 0.0;

    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = N < 0 || K < 0 || ldc < N || (transA == rocblas_operation_none && lda < N)
                        || (transA != rocblas_operation_none && lda < K) || batch_count < 0;
    if(N == 0 || batch_count == 0 || invalid_size)
    {
        // ensure invalid sizes checked before pointer check
        EXPECT_ROCBLAS_STATUS((rocblas_herk_batched_fn)(handle,
                                                        uplo,
                                                        transA,
                                                        N,
                                                        K,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        nullptr,
                                                        nullptr,
                                                        ldc,
                                                        batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    size_t cols = (transA == rocblas_operation_none ? std::max(K, 1) : N);
    size_t rows = (transA != rocblas_operation_none ? std::max(K, 1) : N);

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(rows, cols, lda, batch_count);
    host_batch_matrix<T> hC(N, N, ldc, batch_count);
    host_batch_matrix<T> hC_gold(N, N, ldc, batch_count);
    host_vector<U>       h_alpha(1);
    host_vector<U>       h_beta(1);

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(rows, cols, lda, batch_count);
    device_batch_matrix<T> dC(N, N, ldc, batch_count);
    device_vector<U>       d_alpha(1);
    device_vector<U>       d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true, true);
    rocblas_init_matrix(
        hC, arg, rocblas_client_beta_sets_nan, rocblas_client_hermitian_matrix, false, true);

    hC_gold.copy_from(hC);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // host alpha/beta
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(dC.transfer_from(hC));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR((rocblas_herk_batched_fn)(handle,
                                                          uplo,
                                                          transA,
                                                          N,
                                                          K,
                                                          &h_alpha[0],
                                                          dA.ptr_on_device(),
                                                          lda,
                                                          &h_beta[0],
                                                          dC.ptr_on_device(),
                                                          ldc,
                                                          batch_count));
            handle.post_test(arg);
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            // device alpha/beta
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
            CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));

            CHECK_ROCBLAS_ERROR((rocblas_herk_batched_fn)(handle,
                                                          uplo,
                                                          transA,
                                                          N,
                                                          K,
                                                          d_alpha,
                                                          dA.ptr_on_device(),
                                                          lda,
                                                          d_beta,
                                                          dC.ptr_on_device(),
                                                          ldc,
                                                          batch_count));
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        // cpu reference
        for(int b = 0; b < batch_count; b++)
        {
            ref_herk<T>(uplo, transA, N, K, h_alpha[0], hA[b], lda, h_beta[0], hC_gold[b], ldc);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<
                       T,
                       rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, ldc, hC_gold, hC, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(N, N, ldc, hC_gold, hC, batch_count);
                }
            }

            if(arg.norm_check)
            {
                error_host
                    = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC, batch_count));
            }
        }

        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));

            if(arg.unit_check)
            {
                if(std::is_same_v<
                       T,
                       rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, ldc, hC_gold, hC, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(N, N, ldc, hC_gold, hC, batch_count);
                }
            }

            if(arg.norm_check)
            {
                error_device
                    = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC, batch_count));
            }
        }
    }
    else
    {
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_herk_batched_fn(handle,
                                    uplo,
                                    transA,
                                    N,
                                    K,
                                    h_alpha,
                                    dA.ptr_on_device(),
                                    lda,
                                    h_beta,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_herk_batched_fn(handle,
                                    uplo,
                                    transA,
                                    N,
                                    K,
                                    h_alpha,
                                    dA.ptr_on_device(),
                                    lda,
                                    h_beta,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_transA, e_N, e_K, e_alpha, e_lda, e_beta, e_ldc, e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         herk_gflop_count<T>(N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         error_host,
                         error_device);
    }
}
