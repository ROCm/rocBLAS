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

template <typename T>
void testing_trmm_bad_arg(const Arguments& arg)
{
    auto rocblas_trmm_fn = arg.api == FORTRAN ? rocblas_trmm<T, true> : rocblas_trmm<T, false>;
    // trmm has both inplace and outofplace versions.
    // inplace == true for inplace, inplace == false for outofplace
    bool inplace = !arg.outofplace;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_int M     = 100;
        const rocblas_int N     = 100;
        const rocblas_int lda   = 100;
        const rocblas_int ldb   = 100;
        const rocblas_int ldc   = 100;
        const rocblas_int ldOut = inplace ? ldb : ldc;

        device_vector<T> alpha_d(1), zero_d(1);

        const T alpha_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        const rocblas_side      side   = rocblas_side_left;
        const rocblas_fill      uplo   = rocblas_fill_upper;
        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

        rocblas_int K = side == rocblas_side_left ? M : N;

        // Allocate device memory
        device_matrix<T> dA(K, K, lda);
        device_matrix<T> dB(M, N, ldb);

        rocblas_int dC_M   = inplace ? 1 : M;
        rocblas_int dC_N   = inplace ? 1 : N;
        rocblas_int dC_ldc = inplace ? 1 : ldc;

        device_matrix<T> dC(dC_M, dC_N, dC_ldc);

        device_matrix<T>* dOut = inplace ? &dB : &dC;

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());

        // check for invalid enum
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              rocblas_side_both,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              *dOut,
                                              ldOut),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              side,
                                              (rocblas_fill)rocblas_side_both,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              *dOut,
                                              ldOut),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              side,
                                              uplo,
                                              (rocblas_operation)rocblas_side_both,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              *dOut,
                                              ldOut),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              side,
                                              uplo,
                                              transA,
                                              (rocblas_diagonal)rocblas_side_both,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              *dOut,
                                              ldOut),
                              rocblas_status_invalid_value);

        // check for invalid size
        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_fn(
                handle, side, uplo, transA, diag, -1, N, alpha, dA, lda, dB, ldb, *dOut, ldOut),
            rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_fn(
                handle, side, uplo, transA, diag, M, -1, alpha, dA, lda, dB, ldb, *dOut, ldOut),
            rocblas_status_invalid_size);

        // check for invalid leading dimension
        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_fn(
                handle, side, uplo, transA, diag, M, N, alpha, dA, lda, dB, M - 1, *dOut, ldOut),
            rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_fn(
                handle, side, uplo, transA, diag, M, N, alpha, dA, lda, dB, ldb, *dOut, M - 1),
            rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              rocblas_side_left,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              M - 1,
                                              dB,
                                              ldb,
                                              *dOut,
                                              ldOut),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              rocblas_side_right,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              N - 1,
                                              dB,
                                              ldb,
                                              *dOut,
                                              ldOut),
                              rocblas_status_invalid_size);

        // check that nullpointer gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_fn(
                nullptr, side, uplo, transA, diag, M, N, alpha, dA, lda, dB, ldb, *dOut, ldOut),
            rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_fn(
                handle, side, uplo, transA, diag, M, N, nullptr, dA, lda, dB, ldb, *dOut, ldOut),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_fn(
                handle, side, uplo, transA, diag, M, N, alpha, nullptr, lda, dB, ldb, *dOut, ldOut),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_fn(
                handle, side, uplo, transA, diag, M, N, alpha, dA, lda, nullptr, ldb, *dOut, ldOut),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_fn(
                handle, side, uplo, transA, diag, M, N, alpha, dA, lda, dB, ldb, nullptr, ldOut),
            rocblas_status_invalid_pointer);

        // quick return: If alpha==0, then A and B can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              zero,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              *dOut,
                                              ldOut),
                              rocblas_status_success);

        // quick return: If M==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              0,
                                              N,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              ldOut),
                              rocblas_status_success);

        // quick return: If N==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              0,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              ldOut),
                              rocblas_status_success);
        // in-place only checks
        if(inplace)
        {
            // if inplace, must have ldb == ldc
            EXPECT_ROCBLAS_STATUS(
                rocblas_trmm_fn(
                    handle, side, uplo, transA, diag, M, N, alpha, dA, lda, dB, ldb, dB, ldb + 1),
                rocblas_status_invalid_value);
        }
    }
}

template <typename T>
void testing_trmm(const Arguments& arg)
{
    auto rocblas_trmm_fn = arg.api == FORTRAN ? rocblas_trmm<T, true> : rocblas_trmm<T, false>;

    // trmm has both inplace and outofplace versions.
    // inplace == true for inplace, inplace == false for outofplace
    bool inplace = !arg.outofplace;

    rocblas_int M     = arg.M;
    rocblas_int N     = arg.N;
    rocblas_int lda   = arg.lda;
    rocblas_int ldb   = arg.ldb;
    rocblas_int ldc   = arg.ldc;
    rocblas_int ldOut = inplace ? ldb : ldc;

    char char_side   = arg.side;
    char char_uplo   = arg.uplo;
    char char_transA = arg.transA;
    char char_diag   = arg.diag;
    T    h_alpha_T   = arg.get_alpha<T>();

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_int K = side == rocblas_side_left ? M : N;

    rocblas_local_handle handle{arg};

    // ensure invalid sizes and quick return checked before pointer check
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M || ldc < M;
    if(M == 0 || N == 0 || invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              ldc),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(K, K, lda);
    host_matrix<T> hB(M, N, ldb);
    host_matrix<T> hC(M, N, ldc);
    host_matrix<T> hC_gold(M, N, ldc);

    // Allocate device memory
    device_matrix<T> dA(K, K, lda);
    device_matrix<T> dB(M, N, ldb);

    device_vector<T> alpha_d(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(alpha_d.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    // inplace    trmm is given by B <- alpha * op(A) * B so  matrix C is not used
    // outofplace trmm is given by C <- alpha * op(A) * B and matrix C is used
    rocblas_int dC_M   = inplace ? 1 : M;
    rocblas_int dC_N   = inplace ? 1 : N;
    rocblas_int dC_ldc = inplace ? 1 : ldc;

    device_matrix<T> dC(dC_M, dC_N, dC_ldc);
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    device_matrix<T>* dOut = inplace ? &dB : &dC;

    if(inplace)
    {
        if(ldb != ldc)
        {
            EXPECT_ROCBLAS_STATUS(rocblas_trmm_fn(handle,
                                                  side,
                                                  uplo,
                                                  transA,
                                                  diag,
                                                  M,
                                                  N,
                                                  &h_alpha_T,
                                                  dA,
                                                  lda,
                                                  dB,
                                                  ldb,
                                                  *dOut,
                                                  ldc),
                                  rocblas_status_invalid_value);
            return;
        }

        hC_gold = hB;
    }
    else
    {
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        hC_gold = hC;
    }

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double err_host = 0.0, err_device = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR(rocblas_trmm_fn(handle,
                                                side,
                                                uplo,
                                                transA,
                                                diag,
                                                M,
                                                N,
                                                &h_alpha_T,
                                                dA,
                                                lda,
                                                dB,
                                                ldb,
                                                *dOut,
                                                ldOut));
            handle.post_test(arg);
            CHECK_HIP_ERROR(hC.transfer_from(*dOut));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR((*dOut).transfer_from(hC_gold));
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, &h_alpha_T, sizeof(T), hipMemcpyHostToDevice));

            CHECK_ROCBLAS_ERROR(rocblas_trmm_fn(
                handle, side, uplo, transA, diag, M, N, alpha_d, dA, lda, dB, ldb, *dOut, ldOut));
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        ref_trmm<T>(side, uplo, transA, diag, M, N, h_alpha_T, hA, lda, hB, ldb);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy B matrix into C matrix
        copy_matrix_with_different_leading_dimensions(hB, hC_gold);

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<T, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC);
                }
            }
            if(arg.norm_check)
            {
                err_host = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC));
            }
        }
        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hC.transfer_from(*dOut));

            if(arg.unit_check)
            {
                if(std::is_same_v<T, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC);
                }
            }
            if(arg.norm_check)
            {
                err_device = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC));
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
            CHECK_ROCBLAS_ERROR(rocblas_trmm_fn(handle,
                                                side,
                                                uplo,
                                                transA,
                                                diag,
                                                M,
                                                N,
                                                &h_alpha_T,
                                                dA,
                                                lda,
                                                dB,
                                                ldb,
                                                *dOut,
                                                ldOut));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_trmm_fn(handle,
                                                side,
                                                uplo,
                                                transA,
                                                diag,
                                                M,
                                                N,
                                                &h_alpha_T,
                                                dA,
                                                lda,
                                                dB,
                                                ldb,
                                                *dOut,
                                                ldOut));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side,
                      e_uplo,
                      e_transA,
                      e_diag,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_ldb,
                      e_ldc,
                      e_outofplace>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         trmm_gflop_count<T>(M, N, side),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         err_host,
                         err_device);
    }
}
