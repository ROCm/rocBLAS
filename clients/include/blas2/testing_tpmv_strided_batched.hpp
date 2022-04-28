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
void testing_tpmv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_tpmv_strided_batched_fn = arg.fortran ? rocblas_tpmv_strided_batched<T, true>
                                                       : rocblas_tpmv_strided_batched<T, false>;

    const rocblas_int       M           = 100;
    const rocblas_int       incx        = 1;
    const rocblas_int       batch_count = 1;
    const rocblas_stride    stride_a    = (M * (M + 1)) / 2;
    const rocblas_stride    stride_x    = M;
    const rocblas_operation transA      = rocblas_operation_none;
    const rocblas_fill      uplo        = rocblas_fill_lower;
    const rocblas_diagonal  diag        = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_strided_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(M), 1, stride_a, batch_count);
    device_strided_batch_vector<T> dx(M, incx, stride_x, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    // Checks.
    EXPECT_ROCBLAS_STATUS(rocblas_tpmv_strided_batched_fn(handle,
                                                          rocblas_fill_full,
                                                          transA,
                                                          diag,
                                                          M,
                                                          dAp,
                                                          stride_a,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          batch_count),
                          rocblas_status_invalid_value);
    // arg_checks code shared so transA, diag tested only in non-batched

    EXPECT_ROCBLAS_STATUS(
        rocblas_tpmv_strided_batched_fn(
            handle, uplo, transA, diag, M, nullptr, stride_a, dx, incx, stride_x, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_tpmv_strided_batched_fn(
            handle, uplo, transA, diag, M, dAp, stride_a, nullptr, incx, stride_x, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_tpmv_strided_batched_fn(
            nullptr, uplo, transA, diag, M, dAp, stride_a, dx, incx, stride_x, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_tpmv_strided_batched(const Arguments& arg)
{
    auto rocblas_tpmv_strided_batched_fn = arg.fortran ? rocblas_tpmv_strided_batched<T, true>
                                                       : rocblas_tpmv_strided_batched<T, false>;

    rocblas_int    M = arg.M, incx = arg.incx, batch_count = arg.batch_count;
    rocblas_stride stride_a = arg.stride_a, stride_x = arg.stride_x;

    char char_uplo = arg.uplo, char_transA = arg.transA, char_diag = arg.diag;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || !incx || batch_count < 0;
    if(invalid_size || !M || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_tpmv_strided_batched_fn(handle,
                                                              uplo,
                                                              transA,
                                                              diag,
                                                              M,
                                                              nullptr,
                                                              stride_a,
                                                              nullptr,
                                                              incx,
                                                              stride_x,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(M, M, M, stride_a, batch_count);
    host_strided_batch_matrix<T> hAp(1, rocblas_packed_matrix_size(M), 1, stride_a, batch_count);
    host_strided_batch_vector<T> hx(M, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hres(M, incx, stride_x, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAp.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hres.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(M), 1, stride_a, batch_count);
    device_strided_batch_vector<T> dx(M, incx, stride_x, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    // Helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, M);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used, rocblas_error;

    /* =====================================================================
     ROCBLAS
     =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_tpmv_strided_batched_fn(
            handle, uplo, transA, diag, M, dAp, stride_a, dx, incx, stride_x, batch_count));

        // CPU BLAS
        {
            cpu_time_used = get_time_us_no_sync();
            for(rocblas_int b = 0; b < batch_count; ++b)
            {
                cblas_tpmv<T>(uplo, transA, diag, M, hAp[b], hx[b], incx);
            }
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // fetch GPU
        CHECK_HIP_ERROR(hres.transfer_from(dx));

        // Unit check.
        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, abs_incx, stride_x, hx, hres, batch_count);
        }

        // Norm check.
        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', 1, M, abs_incx, stride_x, hx, hres, batch_count);
        }
    }

    if(arg.timing)
    {

        // Warmup
        {
            int number_cold_calls = arg.cold_iters;
            for(int iter = 0; iter < number_cold_calls; iter++)
            {
                rocblas_tpmv_strided_batched_fn(
                    handle, uplo, transA, diag, M, dAp, stride_a, dx, incx, stride_x, batch_count);
            }
        }

        // Go !
        {
            hipStream_t stream;
            CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
            gpu_time_used        = get_time_us_sync(stream); // in microseconds
            int number_hot_calls = arg.iters;
            for(int iter = 0; iter < number_hot_calls; iter++)
            {
                rocblas_tpmv_strided_batched_fn(
                    handle, uplo, transA, diag, M, dAp, stride_a, dx, incx, stride_x, batch_count);
            }
            gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        }

        // Log performance.
        ArgumentModel<e_uplo,
                      e_transA,
                      e_diag,
                      e_M,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         tpmv_gflop_count<T>(M),
                         tpmv_gbyte_count<T>(M),
                         cpu_time_used,
                         rocblas_error);
    }
}
