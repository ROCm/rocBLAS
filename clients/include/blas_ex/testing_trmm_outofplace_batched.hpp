/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
void testing_trmm_outofplace_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_trmm_outofplace_batched_fn = arg.fortran
                                                  ? rocblas_trmm_outofplace_batched<T, true>
                                                  : rocblas_trmm_outofplace_batched<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_int M           = 100;
        const rocblas_int N           = 101;
        const rocblas_int lda         = 101;
        const rocblas_int ldb         = 101;
        const rocblas_int ldc         = 101;
        const rocblas_int batch_count = 2;

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
        device_batch_matrix<T> dA(K, K, lda, batch_count);
        device_batch_matrix<T> dB(M, N, ldb, batch_count);
        device_batch_matrix<T> dC(M, N, ldc, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());

        // check for invalid enum
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
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
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
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
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_outofplace_batched_fn(handle,
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
                                               dC,
                                               ldc,
                                               batch_count),
            rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm_outofplace_batched_fn(handle,
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
                                               dC,
                                               ldc,
                                               batch_count),
            rocblas_status_invalid_value);

        // check for invalid size
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 -1,
                                                                 N,
                                                                 alpha,
                                                                 dA,
                                                                 lda,
                                                                 dB,
                                                                 ldb,
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 -1,
                                                                 alpha,
                                                                 dA,
                                                                 lda,
                                                                 dB,
                                                                 ldb,
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_size);

        // check for invalid leading dimension
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 N,
                                                                 alpha,
                                                                 dA,
                                                                 lda,
                                                                 dB,
                                                                 M - 1,
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
                                                                 side,
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
                                                                 dC,
                                                                 M - 1,
                                                                 batch_count),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
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
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
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
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_size);

        // check that nullpointer gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(nullptr,
                                                                 side,
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
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 N,
                                                                 nullptr,
                                                                 dA,
                                                                 lda,
                                                                 dB,
                                                                 ldb,
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 N,
                                                                 alpha,
                                                                 nullptr,
                                                                 lda,
                                                                 dB,
                                                                 ldb,
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 N,
                                                                 alpha,
                                                                 dA,
                                                                 lda,
                                                                 nullptr,
                                                                 ldb,
                                                                 dC,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
                                                                 side,
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
                                                                 nullptr,
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_invalid_pointer);

        // quick return: If alpha==0, then A and B can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
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
                                                                 dC.ptr_on_device(),
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_success);

        // quick return: If M==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
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
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_success);

        // quick return: If N==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
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
                                                                 ldc,
                                                                 batch_count),
                              rocblas_status_success);

        // quick return: If batch_count==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
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
                                                                 ldc,
                                                                 0),
                              rocblas_status_success);
    }
}

template <typename T>
void testing_trmm_outofplace_batched(const Arguments& arg)
{
    auto rocblas_trmm_outofplace_batched_fn = arg.fortran
                                                  ? rocblas_trmm_outofplace_batched<T, true>
                                                  : rocblas_trmm_outofplace_batched<T, false>;

    bool nantest = rocblas_isnan(arg.alpha) || rocblas_isnan(arg.alphai);
    if(!std::is_same<T, float>{} && !std::is_same<T, double>{} && !std::is_same<T, rocblas_half>{}
       && !rocblas_is_complex<T> && nantest)
        return; // Exclude integers or other types which don't support NaN

    rocblas_local_handle handle{arg};
    rocblas_int          M           = arg.M;
    rocblas_int          N           = arg.N;
    rocblas_int          lda         = arg.lda;
    rocblas_int          ldb         = arg.ldb;
    rocblas_int          ldc         = arg.ldc;
    rocblas_int          batch_count = arg.batch_count;

    char char_side   = arg.side;
    char char_uplo   = arg.uplo;
    char char_transA = arg.transA;
    char char_diag   = arg.diag;
    T    alpha       = arg.get_alpha<T>();

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_int K = side == rocblas_side_left ? M : N;

    // ensure invalid sizes and quick return checked before pointer check
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M || ldc < M || batch_count < 0;
    if(M == 0 || N == 0 || batch_count == 0 || invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_batched_fn(handle,
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
                                                                 ldc,
                                                                 batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double rocblas_error          = 0.0;

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(K, K, lda, batch_count);
    host_batch_matrix<T> hB(M, N, ldb, batch_count);
    host_batch_matrix<T> hB_gold(M, N, ldb, batch_count);
    host_batch_matrix<T> hC(M, N, ldc, batch_count);
    host_batch_matrix<T> hC_1(M, N, ldc, batch_count);
    host_batch_matrix<T> hC_2(M, N, ldc, batch_count);
    host_batch_matrix<T> hC_gold(M, N, ldc, batch_count);
    host_vector<T>       h_alpha(1);

    //  Initialize data on CPU
    h_alpha[0] = alpha;

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hB_gold.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_2.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(K, K, lda, batch_count);
    device_batch_matrix<T> dB(M, N, ldb, batch_count);
    device_batch_matrix<T> dC(M, N, ldc, batch_count);
    device_vector<T>       d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix);

    hB_gold.copy_from(hB);
    hC_1.copy_from(hC);
    hC_2.copy_from(hC);
    hC_gold.copy_from(hC);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dC.transfer_from(hC_1));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_trmm_outofplace_batched_fn(handle,
                                                               side,
                                                               uplo,
                                                               transA,
                                                               diag,
                                                               M,
                                                               N,
                                                               &h_alpha[0],
                                                               dA.ptr_on_device(),
                                                               lda,
                                                               dB.ptr_on_device(),
                                                               ldb,
                                                               dC.ptr_on_device(),
                                                               ldc,
                                                               batch_count));
        handle.post_test(arg);
        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // calculate dB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));

        CHECK_ROCBLAS_ERROR(rocblas_trmm_outofplace_batched_fn(handle,
                                                               side,
                                                               uplo,
                                                               transA,
                                                               diag,
                                                               M,
                                                               N,
                                                               d_alpha,
                                                               dA.ptr_on_device(),
                                                               lda,
                                                               dB.ptr_on_device(),
                                                               ldb,
                                                               dC.ptr_on_device(),
                                                               ldc,
                                                               batch_count));

        // fetch GPU
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        // CPU BLAS
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        for(rocblas_int i = 0; i < batch_count; i++)
        {
            cblas_trmm<T>(side, uplo, transA, diag, M, N, alpha, hA[i], lda, hB_gold[i], ldb);
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // copy B matrix into C matrix
        copy_matrix_with_different_leading_dimensions(hB_gold, hC_gold);

        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<T>;
                near_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count, tol);
                near_check_general<T>(M, N, ldc, hC_gold, hC_2, batch_count, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
                unit_check_general<T>(M, N, ldc, hC_gold, hC_2, batch_count);
            }
        }

        if(arg.norm_check)
        {
            auto err1 = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count));
            auto err2 = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2, batch_count));
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
            CHECK_ROCBLAS_ERROR(rocblas_trmm_outofplace_batched_fn(handle,
                                                                   side,
                                                                   uplo,
                                                                   transA,
                                                                   diag,
                                                                   M,
                                                                   N,
                                                                   &h_alpha[0],
                                                                   dA.ptr_on_device(),
                                                                   lda,
                                                                   dB.ptr_on_device(),
                                                                   ldb,
                                                                   dC.ptr_on_device(),
                                                                   ldc,
                                                                   batch_count));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_trmm_outofplace_batched_fn(handle,
                                               side,
                                               uplo,
                                               transA,
                                               diag,
                                               M,
                                               N,
                                               &h_alpha[0],
                                               dA.ptr_on_device(),
                                               lda,
                                               dB.ptr_on_device(),
                                               ldb,
                                               dC.ptr_on_device(),
                                               ldc,
                                               batch_count);
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
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         trmm_gflop_count<T>(M, N, side),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error);
    }
}
