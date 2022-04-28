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

template <typename T, bool HERM>
void testing_symm_hemm_bad_arg(const Arguments& arg)
{
    // clang-format off
    auto rocblas_fn = HERM ? (arg.fortran ? rocblas_hemm<T, true> : rocblas_hemm<T, false>)
                           : (arg.fortran ? rocblas_symm<T, true> : rocblas_symm<T, false>);
    // clang-format on

    rocblas_local_handle handle{arg};
    const rocblas_side   side  = rocblas_side_left;
    const rocblas_fill   uplo  = rocblas_fill_upper;
    const rocblas_int    M     = 100;
    const rocblas_int    N     = 100;
    const rocblas_int    lda   = 100;
    const rocblas_int    ldb   = 100;
    const rocblas_int    ldc   = 100;
    const T              alpha = 1.0;
    const T              beta  = 1.0;

    size_t rows = (side == rocblas_side_left ? N : M);
    size_t cols = (side == rocblas_side_left ? M : N);

    // Allocate device memory
    device_matrix<T> dA(rows, cols, lda);
    device_matrix<T> dB(M, N, ldb);
    device_matrix<T> dC(M, N, ldc);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(nullptr, side, uplo, M, N, &alpha, dA, lda, dB, ldb, &beta, dC, ldc),
        rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(handle, rocblas_side_both, uplo, M, N, &alpha, dA, lda, dB, ldb, &beta, dC, ldc),
        rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(handle, side, rocblas_fill_full, M, N, &alpha, dA, lda, dB, ldb, &beta, dC, ldc),
        rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(handle, side, uplo, M, N, nullptr, dA, lda, dB, ldb, &beta, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(handle, side, uplo, M, N, &alpha, nullptr, lda, dB, ldb, &beta, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(handle, side, uplo, M, N, &alpha, dA, lda, nullptr, ldb, &beta, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(handle, side, uplo, M, N, &alpha, dA, lda, dB, ldb, nullptr, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(handle, side, uplo, M, N, &alpha, dA, lda, dB, ldb, &beta, nullptr, ldc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(handle, side, uplo, 0, N, nullptr, nullptr, lda, dB, ldb, nullptr, nullptr, ldc),
        rocblas_status_success);
}

template <typename T, bool HERM>
void testing_symm_hemm(const Arguments& arg)
{
    // clang-format off
    auto rocblas_fn = HERM ? (arg.fortran ? rocblas_hemm<T, true> : rocblas_hemm<T, false>)
                           : (arg.fortran ? rocblas_symm<T, true> : rocblas_symm<T, false>);
    auto gflop_count_fn = HERM ? hemm_gflop_count<T> : symm_gflop_count<T>;
    // clang-format on

    rocblas_local_handle handle{arg};
    rocblas_side         side  = char2rocblas_side(arg.side);
    rocblas_fill         uplo  = char2rocblas_fill(arg.uplo);
    rocblas_int          M     = arg.M;
    rocblas_int          N     = arg.N;
    rocblas_int          lda   = arg.lda;
    rocblas_int          ldb   = arg.ldb;
    rocblas_int          ldc   = arg.ldc;
    T                    alpha = arg.get_alpha<T>();
    T                    beta  = arg.get_beta<T>();

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    // Note: N==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = M < 0 || N < 0 || ldc < M || ldb < M
                        || (side == rocblas_side_left && (lda < M))
                        || (side != rocblas_side_left && (lda < N));
    if(M == 0 || N == 0 || invalid_size)
    {
        // ensure invalid sizes checked before pointer check
        EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                         side,
                                         uplo,
                                         M,
                                         N,
                                         nullptr,
                                         nullptr,
                                         lda,
                                         nullptr,
                                         ldb,
                                         nullptr,
                                         nullptr,
                                         ldc),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }
    size_t rows = (side == rocblas_side_left ? N : M);
    size_t cols = (side == rocblas_side_left ? M : N);

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(rows, cols, lda);
    host_matrix<T> hB(M, N, ldb);
    host_matrix<T> hC_1(M, N, ldc);
    host_matrix<T> hC_2(M, N, ldc);
    host_matrix<T> hC_gold(M, N, ldc);
    host_vector<T> h_alpha(1);
    host_vector<T> h_beta(1);

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Allocate device memory
    device_matrix<T> dA(rows, cols, lda);
    device_matrix<T> dB(M, N, ldb);
    device_matrix<T> dC(M, N, ldc);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    if(HERM)
    {
        rocblas_init_matrix<T>(
            hA, arg, rocblas_client_never_set_nan, rocblas_client_hermitian_matrix, true);
    }
    else
    {
        rocblas_init_matrix<T>(
            hA, arg, rocblas_client_never_set_nan, rocblas_client_symmetric_matrix, true);
    }
    rocblas_init_matrix<T>(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix<T>(hC_1, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    hC_2    = hC_1;
    hC_gold = hC_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_1));

    if(arg.unit_check || arg.norm_check)
    {
        // host alpha/beta
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_fn(
            handle, side, uplo, M, N, &h_alpha[0], dA, lda, dB, ldb, &h_beta[0], dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // device alpha/beta
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));

        CHECK_ROCBLAS_ERROR(
            rocblas_fn(handle, side, uplo, M, N, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        // CPU BLAS
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        if(HERM)
        {
            cblas_hemm<T>(side, uplo, M, N, h_alpha, hA, lda, hB, ldb, h_beta, hC_gold, ldc);
        }
        else
        {
            cblas_symm<T>(side, uplo, M, N, h_alpha[0], hA, lda, hB, ldb, h_beta[0], hC_gold, ldc);
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_float_complex>{}
               || std::is_same<T, rocblas_double_complex>{})
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
                near_check_general<T>(M, N, ldc, hC_gold, hC_2, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
                unit_check_general<T>(M, N, ldc, hC_gold, hC_2);
            }
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1));
            auto err2     = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2));
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
            rocblas_fn(handle, side, uplo, M, N, h_alpha, dA, lda, dB, ldb, h_beta, dC, ldc);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_fn(handle, side, uplo, M, N, h_alpha, dA, lda, dB, ldb, h_beta, dC, ldc);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side, e_uplo, e_M, e_N, e_alpha, e_lda, e_ldb, e_beta, e_ldc>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            gflop_count_fn(side, M, N),
            ArgumentLogging::NA_value,
            cpu_time_used,
            rocblas_error);
    }
}
