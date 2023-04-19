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
void testing_geam_bad_arg(const Arguments& arg)
{
    auto rocblas_geam_fn = arg.api == FORTRAN ? rocblas_geam<T, true> : rocblas_geam<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_int M = 100;
        const rocblas_int N = 99;

        const rocblas_int lda = 100;
        const rocblas_int ldb = 100;
        const rocblas_int ldc = 100;

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

        rocblas_int A_row = transA == rocblas_operation_none ? M : N;
        rocblas_int A_col = transA == rocblas_operation_none ? N : M;
        rocblas_int B_row = transB == rocblas_operation_none ? M : N;
        rocblas_int B_col = transB == rocblas_operation_none ? N : M;

        // Allocate device memory
        device_matrix<T> dA(A_row, A_col, lda);
        device_matrix<T> dB(B_row, B_col, ldb);
        device_matrix<T> dC(M, N, ldc);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());

        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_fn(nullptr, transA, transB, M, N, alpha, dA, lda, beta, dB, ldb, dC, ldc),
            rocblas_status_invalid_handle);

        // invalid values
        EXPECT_ROCBLAS_STATUS(rocblas_geam_fn(handle,
                                              (rocblas_operation)rocblas_fill_full,
                                              transB,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              beta,
                                              dB,
                                              ldb,
                                              dC,
                                              ldc),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_geam_fn(handle,
                                              transA,
                                              (rocblas_operation)rocblas_fill_full,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              beta,
                                              dB,
                                              ldb,
                                              dC,
                                              ldc),
                              rocblas_status_invalid_value);

        // invalid sizes not done in yaml test

        // A == C leading dims must match
        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_fn(
                handle, transA, transB, M, N, alpha, dC, ldc + 1, beta, dB, ldb, dC, ldc),
            rocblas_status_invalid_size);

        // B == C leading dims must match
        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_fn(
                handle, transA, transB, M, N, alpha, dA, lda, beta, dC, ldc + 1, dC, ldc),
            rocblas_status_invalid_size);

        // alpha/beta
        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_fn(handle, transA, transB, M, N, nullptr, dA, lda, beta, dB, ldb, dC, ldc),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_fn(
                handle, transA, transB, M, N, alpha, dA, lda, nullptr, dB, ldb, dC, ldc),
            rocblas_status_invalid_pointer);

        // invalid pointers
        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_fn(
                handle, transA, transB, M, N, alpha, dA, lda, beta, dB, ldb, nullptr, ldc),
            rocblas_status_invalid_pointer);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            EXPECT_ROCBLAS_STATUS(
                rocblas_geam_fn(
                    handle, transA, transB, M, N, alpha, nullptr, lda, beta, dB, ldb, dC, ldc),
                rocblas_status_invalid_pointer);

            EXPECT_ROCBLAS_STATUS(
                rocblas_geam_fn(
                    handle, transA, transB, M, N, alpha, dA, lda, beta, nullptr, ldb, dC, ldc),
                rocblas_status_invalid_pointer);
        }

        // M==0 then all may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_fn(handle,
                                              transA,
                                              transB,
                                              0,
                                              N,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              ldc),
                              rocblas_status_success);

        // N==0 then all may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_fn(handle,
                                              transA,
                                              transB,
                                              M,
                                              0,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              ldc),
                              rocblas_status_success);

        // alpha==0 then A may be nullptr
        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_fn(
                handle, transA, transB, M, N, zero, nullptr, lda, beta, dB, ldb, dC, ldc),
            rocblas_status_success);

        // beta==0 then B may be nullptr
        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_fn(
                handle, transA, transB, M, N, alpha, dA, lda, zero, nullptr, ldb, dC, ldc),
            rocblas_status_success);
    }
}

template <typename T>
void testing_geam(const Arguments& arg)
{
    auto rocblas_geam_fn = arg.api == FORTRAN ? rocblas_geam<T, true> : rocblas_geam<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T alpha = arg.get_alpha<T>();
    T beta  = arg.get_beta<T>();

    T* dC_in_place;

    rocblas_int A_row = transA == rocblas_operation_none ? M : N;
    rocblas_int A_col = transA == rocblas_operation_none ? N : M;
    rocblas_int B_row = transB == rocblas_operation_none ? M : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : M;

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;

    double rocblas_error_1 = std::numeric_limits<double>::max();
    double rocblas_error_2 = std::numeric_limits<double>::max();
    double rocblas_error   = std::numeric_limits<double>::max();

    rocblas_local_handle handle{arg};

    size_t size_C = size_t(ldc) * N;

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < A_row || ldb < B_row || ldc < M;
    if(invalid_size || !M || !N)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_fn(
                handle,
                transA,
                transB,
                M,
                N,
                nullptr,
                nullptr,
                lda,
                nullptr,
                nullptr,
                ldb,
                (T*)0x1, // defeat C==A or B leading dim invalid checks, C nullptr in bad_arg
                ldc),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(A_row, A_col, lda), hA_copy(A_row, A_col, lda);
    host_matrix<T> hB(B_row, B_col, ldb), hB_copy(B_row, B_col, ldb);
    host_matrix<T> hC_1(M, N, ldc);
    host_matrix<T> hC_2(M, N, ldc);
    host_matrix<T> hC_gold(M, N, ldc);
    host_vector<T> h_alpha(1);
    host_vector<T> h_beta(1);

    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hA_copy.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hB_copy.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_2.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Allocate device memory
    device_matrix<T> dA(A_row, A_col, lda);
    device_matrix<T> dB(B_row, B_col, ldb);
    device_matrix<T> dC(M, N, ldc);
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
    rocblas_init_matrix(hB, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    hA_copy = hA;
    hB_copy = hB;

    // copy data from CPU to device
    CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
    CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_geam_fn(
            handle, transA, transB, M, N, &alpha, dA, lda, &beta, dB, ldb, dC, ldc));
        handle.post_test(arg);
        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_geam_fn(
            handle, transA, transB, M, N, d_alpha, dA, lda, d_beta, dB, ldb, dC, ldc));

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        cblas_geam(transA,
                   transB,
                   M,
                   N,
                   (T*)h_alpha,
                   (T*)hA,
                   lda,
                   (T*)h_beta,
                   (T*)hB,
                   ldb,
                   (T*)hC_gold,
                   ldc);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // fetch GPU
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
            unit_check_general<T>(M, N, ldc, hC_gold, hC_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1);
            rocblas_error_2 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2);
        }

        // inplace check for dC == dA
        {
            dC_in_place = dA;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam_fn(
                handle, transA, transB, M, N, &alpha, dA, lda, &beta, dB, ldb, dC_in_place, ldc);

            if(lda != ldc || transA != rocblas_operation_none)
            {
                EXPECT_ROCBLAS_STATUS(status_h, rocblas_status_invalid_size);
            }
            else
            {
                CHECK_HIP_ERROR(
                    hipMemcpy(hC_1, dC_in_place, sizeof(T) * size_C, hipMemcpyDeviceToHost));
                // dA was clobbered by dC_in_place, so copy hA back to dA
                CHECK_HIP_ERROR(dA.transfer_from(hA));

                // reference calculation
                cblas_geam(transA,
                           transB,
                           M,
                           N,
                           (T*)h_alpha,
                           (T*)hA_copy,
                           lda,
                           (T*)h_beta,
                           (T*)hB,
                           ldb,
                           (T*)hC_gold,
                           ldc);

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
                }

                if(arg.norm_check)
                {
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1);
                }
            }
        }

        // inplace check for dC == dB
        {
            dC_in_place = dB;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam_fn(
                handle, transA, transB, M, N, &alpha, dA, lda, &beta, dB, ldb, dC_in_place, ldc);

            if(ldb != ldc || transB != rocblas_operation_none)
            {
                EXPECT_ROCBLAS_STATUS(status_h, rocblas_status_invalid_size);
            }
            else
            {
                CHECK_ROCBLAS_ERROR(status_h);

                CHECK_HIP_ERROR(
                    hipMemcpy(hC_1, dC_in_place, sizeof(T) * size_C, hipMemcpyDeviceToHost));

                // reference calculation
                cblas_geam(transA,
                           transB,
                           M,
                           N,
                           (T*)h_alpha,
                           (T*)hA_copy,
                           lda,
                           (T*)h_beta,
                           (T*)hB_copy,
                           ldb,
                           (T*)hC_gold,
                           ldc);

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
                }

                if(arg.norm_check)
                {
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1);
                }
            }
        }
    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_geam_fn(handle, transA, transB, M, N, &alpha, dA, lda, &beta, dB, ldb, dC, ldc);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_geam_fn(handle, transA, transB, M, N, &alpha, dA, lda, &beta, dB, ldb, dC, ldc);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_alpha, e_lda, e_beta, e_ldb, e_ldc>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         geam_gflop_count<T>(M, N),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
