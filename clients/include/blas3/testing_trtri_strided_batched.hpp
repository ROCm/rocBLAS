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
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_trtri_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_trtri_strided_batched_fn = arg.api == FORTRAN
                                                ? rocblas_trtri_strided_batched<T, true>
                                                : rocblas_trtri_strided_batched<T, false>;

    rocblas_local_handle handle{arg};

    const rocblas_int    N           = 100;
    const rocblas_int    lda         = 100;
    const rocblas_stride stride_A    = lda * N;
    const rocblas_int    batch_count = 2;

    const rocblas_fill     uplo = rocblas_fill_upper;
    const rocblas_diagonal diag = rocblas_diagonal_non_unit;

    // Allocate device memory
    device_strided_batch_matrix<T> dA(N, N, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dinvA(N, N, lda, stride_A, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dinvA.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_trtri_strided_batched_fn(handle,
                                                           rocblas_fill_full,
                                                           diag,
                                                           N,
                                                           dA,
                                                           lda,
                                                           stride_A,
                                                           dinvA,
                                                           lda,
                                                           stride_A,
                                                           batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_trtri_strided_batched_fn(handle,
                                                           uplo,
                                                           (rocblas_diagonal)rocblas_side_both,
                                                           N,
                                                           dA,
                                                           lda,
                                                           stride_A,
                                                           dinvA,
                                                           lda,
                                                           stride_A,
                                                           batch_count),
                          rocblas_status_invalid_value);

    // check for invalid sizes
    EXPECT_ROCBLAS_STATUS(
        rocblas_trtri_strided_batched_fn(
            handle, uplo, diag, -1, dA, lda, stride_A, dinvA, lda, stride_A, batch_count),
        rocblas_status_invalid_size);

    EXPECT_ROCBLAS_STATUS(rocblas_trtri_strided_batched_fn(
                              handle, uplo, diag, N, dA, lda, stride_A, dinvA, lda, stride_A, -1),
                          rocblas_status_invalid_size);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trtri_strided_batched_fn(
            handle, uplo, diag, N, dA, lda - 1, stride_A, dinvA, lda, stride_A, batch_count),
        rocblas_status_invalid_size);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trtri_strided_batched_fn(
            handle, uplo, diag, N, dA, lda, stride_A, dinvA, lda - 1, stride_A, batch_count),
        rocblas_status_invalid_size);

    // nullptr tests
    EXPECT_ROCBLAS_STATUS(
        rocblas_trtri_strided_batched_fn(
            nullptr, uplo, diag, N, dA, lda, stride_A, dinvA, lda, stride_A, batch_count),
        rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trtri_strided_batched_fn(
            handle, uplo, diag, N, nullptr, lda, stride_A, dinvA, lda, stride_A, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trtri_strided_batched_fn(
            handle, uplo, diag, N, dA, lda, stride_A, nullptr, lda, stride_A, batch_count),
        rocblas_status_invalid_pointer);

    // quick return: If N==0, then all pointers can be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_trtri_strided_batched_fn(
            handle, uplo, diag, 0, nullptr, lda, stride_A, nullptr, lda, stride_A, batch_count),
        rocblas_status_success);

    // quick return: If batch_count==0, then all pointers can be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_trtri_strided_batched_fn(
            handle, uplo, diag, N, nullptr, lda, stride_A, nullptr, lda, stride_A, 0),
        rocblas_status_success);
}

template <typename T>
void testing_trtri_strided_batched(const Arguments& arg)
{
    auto rocblas_trtri_strided_batched_fn = arg.api == FORTRAN
                                                ? rocblas_trtri_strided_batched<T, true>
                                                : rocblas_trtri_strided_batched<T, false>;

    rocblas_int N           = arg.N;
    rocblas_int lda         = arg.lda;
    rocblas_int batch_count = arg.batch_count;

    rocblas_stride stride_A = lda * N;

    char char_uplo = arg.uplo;
    char char_diag = arg.diag;

    // char_uplo = 'U';
    rocblas_fill     uplo = char2rocblas_fill(char_uplo);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle{arg};

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || lda < N || batch_count < 0;
    if(invalid_size || N == 0 || batch_count == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_trtri_strided_batched_fn(
                handle, uplo, diag, N, nullptr, lda, stride_A, nullptr, lda, stride_A, batch_count),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_strided_batch_matrix<T> hA(N, N, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hB(N, N, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hA_2(N, N, lda, stride_A, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hA_2.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(N, N, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dinvA(N, N, lda, stride_A, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dinvA.memcheck());

    // Initial Data on CPU
    // Explicitly set the unused side of matrix `hA` to 0 when using it for temp storage.
    // Used rocblas_client_triangular_matrix type initialization, which will ensure the unused side is set to 0 or could be done manually
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true, true);

    for(size_t b = 0; b < batch_count; b++)
    {
        for(size_t i = 0; i < N; i++)
        {
            for(size_t j = 0; j < N; j++)
            {
                T* A = hA[b];
                A[i + j * lda] *= 0.01;

                if(j % 2)
                    A[i + j * lda] *= -1;

                if(i == j)
                {
                    if(diag == rocblas_diagonal_unit)
                        A[i + j * lda] = 1.0; // need to preprocess matrix for clbas_trtri
                    else
                        A[i + j * lda] *= 100.0;
                }
            }
        }
    }
    hB.copy_from(hA);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dinvA.transfer_from(hA));

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double rocblas_error          = 0.0;

    if(!ROCBLAS_REALLOC_ON_DEMAND)
    {
        // Compute size
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));

        CHECK_ALLOC_QUERY(rocblas_trtri_strided_batched_fn(
            handle, uplo, diag, N, dA, lda, stride_A, dinvA, lda, stride_A, batch_count));

        // Test in place
        CHECK_ALLOC_QUERY(rocblas_trtri_strided_batched_fn(
            handle, uplo, diag, N, dA, lda, stride_A, dA, lda, stride_A, batch_count));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        // Allocate memory
        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    hipStream_t stream;
    if(arg.timing)
    {
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
    }

    handle.pre_test(arg);

    CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched_fn(
        handle, uplo, diag, N, dA, lda, stride_A, dinvA, lda, stride_A, batch_count));

    // Test in place
    CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched_fn(
        handle, uplo, diag, N, dA, lda, stride_A, dA, lda, stride_A, batch_count));

    handle.post_test(arg);

    if(arg.timing)
    {
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hA.transfer_from(dinvA));
    CHECK_HIP_ERROR(hA_2.transfer_from(dA));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        for(size_t b = 0; b < batch_count; b++)
        {
            // CBLAS doesn't have trtri implementation so using the LAPACK trtri
            lapack_xtrtri<T>(char_uplo, char_diag, N, hB[b], lda);
        }
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

#if 0
        rocblas_print_matrix(hB, hA, N, N, lda, 1);
#endif

        if(arg.unit_check)
        {
            const double rel_error = trtri_tolerance<T>(N);
            near_check_general<T>(N, N, lda, stride_A, hB, hA, batch_count, rel_error);
            near_check_general<T>(N, N, lda, stride_A, hB, hA_2, batch_count, rel_error);
        }

        if(arg.norm_check)
        {
            for(size_t b = 0; b < batch_count; b++)
            {
                rocblas_error = fmax(rocblas_error,
                                     norm_check_symmetric<T>('F', char_uplo, N, lda, hB[b], hA[b]));
            }
            rocblas_error = 0.0;
            for(size_t b = 0; b < batch_count; b++)
            {
                rocblas_error = fmax(
                    rocblas_error, norm_check_symmetric<T>('F', char_uplo, N, lda, hB[b], hA_2[b]));
            }
        }
    } // end of norm_check

    if(arg.timing)
    {
        ArgumentModel<e_uplo, e_diag, e_N, e_lda, e_stride_a, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            trtri_gflop_count<T>(N),
            ArgumentLogging::NA_value,
            cpu_time_used,
            rocblas_error);
    }
}
