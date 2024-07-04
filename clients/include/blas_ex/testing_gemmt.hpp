/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_common.hpp"

template <typename T>
void testing_gemmt_bad_arg(const Arguments& arg)
{
    auto rocblas_gemmt_fn
        = arg.api & c_API_FORTRAN ? rocblas_gemmt<T, true> : rocblas_gemmt<T, false>;
    auto rocblas_gemmt_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_gemmt_64<T, true> : rocblas_gemmt_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_fill      uplo   = rocblas_fill_upper;
        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;
        const int64_t           N      = 100;
        const int64_t           K      = 100;
        const int64_t           lda    = 100;
        const int64_t           ldb    = 100;
        const int64_t           ldc    = 100;

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

        size_t A_row = (transA == rocblas_operation_none ? N : std::max(K, int64_t(1)));
        size_t A_col = (transA == rocblas_operation_none ? std::max(K, int64_t(1)) : N);
        size_t B_row = (transB == rocblas_operation_none ? std::max(K, int64_t(1)) : N);
        size_t B_col = (transB == rocblas_operation_none ? N : std::max(K, int64_t(1)));

        // Allocate device memory
        device_matrix<T> dA(A_row, A_col, lda);
        device_matrix<T> dB(B_row, B_col, ldb);
        device_matrix<T> dC(N, N, ldc);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());

        // invalid values
        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_gemmt_fn,
                    (nullptr, uplo, transA, transB, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_gemmt_fn,
                    (handle,
                     rocblas_fill_full,
                     transA,
                     transB,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_gemmt_fn,
                    (handle,
                     uplo,
                     (rocblas_operation)rocblas_fill_full,
                     transB,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_gemmt_fn,
                    (handle,
                     uplo,
                     transA,
                     (rocblas_operation)rocblas_fill_full,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc));

        // alpha/beta pointer checks
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_gemmt_fn,
                    (handle, uplo, transA, transB, N, K, nullptr, dA, lda, dB, ldb, beta, dC, ldc));

        DAPI_EXPECT(
            rocblas_status_invalid_pointer,
            rocblas_gemmt_fn,
            (handle, uplo, transA, transB, N, K, alpha, dA, lda, dB, ldb, nullptr, dC, ldc));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            // alpha and beta can only be inspected in host_mode so A and B validated
            DAPI_EXPECT(
                rocblas_status_invalid_pointer,
                rocblas_gemmt_fn,
                (handle, uplo, transA, transB, N, K, alpha, nullptr, lda, dB, ldb, beta, dC, ldc));

            DAPI_EXPECT(
                rocblas_status_invalid_pointer,
                rocblas_gemmt_fn,
                (handle, uplo, transA, transB, N, K, alpha, dA, lda, nullptr, ldb, beta, dC, ldc));

            DAPI_EXPECT(
                rocblas_status_invalid_pointer,
                rocblas_gemmt_fn,
                (handle, uplo, transA, transB, N, K, alpha, dA, lda, dB, ldb, beta, nullptr, ldc));
        }

        // size
        DAPI_EXPECT(
            rocblas_status_invalid_size,
            rocblas_gemmt_fn,
            (handle, uplo, transA, transB, N, K, alpha, dA, lda - 1, dB, ldb, beta, dC, ldc));

        DAPI_EXPECT(
            rocblas_status_invalid_size,
            rocblas_gemmt_fn,
            (handle, uplo, transA, transB, N, K, alpha, dA, lda, dB, ldb - 1, beta, dC, ldc));

        DAPI_EXPECT(
            rocblas_status_invalid_size,
            rocblas_gemmt_fn,
            (handle, uplo, transA, transB, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc - 1));

        // N==0 quick return for no ops with null pointers
        DAPI_CHECK(rocblas_gemmt_fn,
                   (handle,
                    uplo,
                    transA,
                    transB,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    dB,
                    ldb,
                    nullptr,
                    nullptr,
                    ldc));

        // k==0 and beta==1 all A, B, C pointers may be null
        DAPI_CHECK(rocblas_gemmt_fn,
                   (handle,
                    uplo,
                    transA,
                    transB,
                    N,
                    0,
                    alpha,
                    nullptr,
                    lda,
                    nullptr,
                    ldb,
                    one,
                    nullptr,
                    ldc));

        // alpha==0 and beta==1 all pointers may be null
        DAPI_CHECK(rocblas_gemmt_fn,
                   (handle,
                    uplo,
                    transA,
                    transB,
                    N,
                    K,
                    zero,
                    nullptr,
                    lda,
                    nullptr,
                    ldb,
                    one,
                    nullptr,
                    ldc));
    }
}

template <typename T>
void testing_gemmt(const Arguments& arg)
{
    auto rocblas_gemmt_fn
        = arg.api & c_API_FORTRAN ? rocblas_gemmt<T, true> : rocblas_gemmt<T, false>;
    auto rocblas_gemmt_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_gemmt_64<T, true> : rocblas_gemmt_64<T, false>;

    rocblas_local_handle handle{arg};
    rocblas_fill         uplo   = char2rocblas_fill(arg.uplo);
    rocblas_operation    transA = char2rocblas_operation(arg.transA);
    rocblas_operation    transB = char2rocblas_operation(arg.transB);
    int64_t              N      = arg.N;
    int64_t              K      = arg.K;
    int64_t              lda    = arg.lda;
    int64_t              ldb    = arg.ldb;
    int64_t              ldc    = arg.ldc;
    T                    alpha  = arg.get_alpha<T>();
    T                    beta   = arg.get_beta<T>();

    double cpu_time_used;
    double error_host   = 0.0;
    double error_device = 0.0;

    size_t A_row = (transA == rocblas_operation_none ? N : std::max(K, int64_t(1)));
    size_t A_col = (transA == rocblas_operation_none ? std::max(K, int64_t(1)) : N);
    size_t B_row = (transB == rocblas_operation_none ? std::max(K, int64_t(1)) : N);
    size_t B_col = (transB == rocblas_operation_none ? N : std::max(K, int64_t(1)));

    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = N < 0 || K < 0 || ldc < N || lda < A_row || ldb < B_row || ldc < N;
    if(N == 0 || invalid_size)
    {
        // ensure invalid sizes checked before pointer check
        DAPI_EXPECT((invalid_size ? rocblas_status_invalid_size : rocblas_status_success),
                    rocblas_gemmt_fn,
                    (handle,
                     uplo,
                     transA,
                     transB,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     nullptr,
                     ldc));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(A_row, A_col, lda);
    host_matrix<T> hB(B_row, B_col, ldb);
    host_matrix<T> hC(N, N, ldc);
    host_matrix<T> hC_gold(N, N, ldc);
    host_vector<T> h_alpha(1);
    host_vector<T> h_beta(1);

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Allocate device memory
    device_matrix<T> dA(A_row, A_col, lda);
    device_matrix<T> dB(B_row, B_col, ldb);
    device_matrix<T> dC(N, N, ldc);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

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
    rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_triangular_matrix);

    hC_gold = hC;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // host alpha/beta
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(dC.transfer_from(hC));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_gemmt_fn,
                       (handle,
                        uplo,
                        transA,
                        transB,
                        N,
                        K,
                        &h_alpha[0],
                        dA,
                        lda,
                        dB,
                        ldb,
                        &h_beta[0],
                        dC,
                        ldc));
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

            DAPI_CHECK(
                rocblas_gemmt_fn,
                (handle, uplo, transA, transB, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

            if(arg.repeatability_check)
            {
                host_matrix<T> hC_copy(N, N, ldc);
                CHECK_HIP_ERROR(hC_copy.memcheck());

                CHECK_HIP_ERROR(hC.transfer_from(dC));
                for(int i = 0; i < arg.iters; i++)
                {
                    CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
                    DAPI_CHECK(rocblas_gemmt_fn,
                               (handle,
                                uplo,
                                transA,
                                transB,
                                N,
                                K,
                                d_alpha,
                                dA,
                                lda,
                                dB,
                                ldb,
                                d_beta,
                                dC,
                                ldc));
                    CHECK_HIP_ERROR(hC_copy.transfer_from(dC));
                    unit_check_general<T>(N, N, ldc, hC, hC_copy);
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        ref_gemmt<T>(
            uplo, transA, transB, N, K, h_alpha[0], hA, lda, hB, ldb, h_beta[0], hC_gold, ldc);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        auto compare_hC_to_gold = [&] {
            if(arg.unit_check)
            {
                unit_check_general<T>(N, N, ldc, hC_gold, hC);
            }

            double error = 0;
            if(arg.norm_check)
            {
                error = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC));
            }
            return error;
        };

        if(arg.pointer_mode_host)
        {
            error_host = compare_hC_to_gold();
        }

        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));

            error_device = compare_hC_to_gold();
        }
    }
    else
    {
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream); // in microseconds

            DAPI_DISPATCH(
                rocblas_gemmt_fn,
                (handle, uplo, transA, transB, N, K, h_alpha, dA, lda, dB, ldb, h_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        //Using the same flops as syrkx as both these functions as same operations
        double gflops = syrkx_gflop_count<T>(N, K);

        ArgumentModel<e_uplo, e_transA, e_transB, e_N, e_K, e_alpha, e_lda, e_ldb, e_beta, e_ldc>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gflops,
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         error_host,
                         error_device);
    }
}
