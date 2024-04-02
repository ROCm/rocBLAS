/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
void testing_tpmv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_tpmv_batched_fn
        = arg.api == FORTRAN ? rocblas_tpmv_batched<T, true> : rocblas_tpmv_batched<T, false>;

    auto rocblas_tpmv_batched_fn_64 = arg.api == FORTRAN_64 ? rocblas_tpmv_batched_64<T, true>
                                                            : rocblas_tpmv_batched_64<T, false>;

    const int64_t           N           = 100;
    const int64_t           incx        = 1;
    const int64_t           batch_count = 1;
    const rocblas_operation transA      = rocblas_operation_none;
    const rocblas_fill      uplo        = rocblas_fill_lower;
    const rocblas_diagonal  diag        = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    // Checks.
    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_tpmv_batched_fn,
                (nullptr,
                 uplo,
                 transA,
                 diag,
                 N,
                 dAp.ptr_on_device(),
                 dx.ptr_on_device(),
                 incx,
                 batch_count));
    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_tpmv_batched_fn,
                (handle,
                 rocblas_fill_full,
                 transA,
                 diag,
                 N,
                 dAp.ptr_on_device(),
                 dx.ptr_on_device(),
                 incx,
                 batch_count));

    // arg_checks code shared so transA, diag tested only in non-batched
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_tpmv_batched_fn,
                (handle, uplo, transA, diag, N, nullptr, dx.ptr_on_device(), incx, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_tpmv_batched_fn,
                (handle, uplo, transA, diag, N, dAp.ptr_on_device(), nullptr, incx, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_tpmv_batched_fn,
                (handle, uplo, transA, diag, N, dAp.ptr_on_device(), nullptr, incx, batch_count));
}

template <typename T>
void testing_tpmv_batched(const Arguments& arg)
{
    auto rocblas_tpmv_batched_fn
        = arg.api == FORTRAN ? rocblas_tpmv_batched<T, true> : rocblas_tpmv_batched<T, false>;

    auto rocblas_tpmv_batched_fn_64 = arg.api == FORTRAN_64 ? rocblas_tpmv_batched_64<T, true>
                                                            : rocblas_tpmv_batched_64<T, false>;

    int64_t N = arg.N, incx = arg.incx, batch_count = arg.batch_count;

    char char_uplo = arg.uplo, char_transA = arg.transA, char_diag = arg.diag;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle{arg};

    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_tpmv_batched_fn,
                    (handle, uplo, transA, diag, N, nullptr, nullptr, incx, batch_count));

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_batch_matrix<T> hA(N, N, N, batch_count);
    host_batch_matrix<T> hAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hres(N, incx, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAp.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hres.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    auto dAp_on_device = dAp.ptr_on_device();
    auto dx_on_device  = dx.ptr_on_device();

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, N);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double cpu_time_used, rocblas_error;

    /* =====================================================================
     ROCBLAS
     =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        handle.pre_test(arg);
        // GPU BLAS
        DAPI_CHECK(rocblas_tpmv_batched_fn,
                   (handle, uplo, transA, diag, N, dAp_on_device, dx_on_device, incx, batch_count));
        handle.post_test(arg);

        // fetch GPU
        CHECK_HIP_ERROR(hres.transfer_from(dx));

        if(arg.repeatability_check)
        {
            host_batch_vector<T> hres_copy(N, incx, batch_count);
            CHECK_HIP_ERROR(hres_copy.memcheck());
            for(int i = 0; i < arg.iters; i++)
            {
                CHECK_HIP_ERROR(dAp.transfer_from(hAp));
                CHECK_HIP_ERROR(dx.transfer_from(hx));
                DAPI_CHECK(rocblas_tpmv_batched_fn,
                           (handle,
                            uplo,
                            transA,
                            diag,
                            N,
                            dAp_on_device,
                            dx_on_device,
                            incx,
                            batch_count));
                CHECK_HIP_ERROR(hres_copy.transfer_from(dx));
                unit_check_general<T>(1, N, incx, hres, hres_copy, batch_count);
            }
            return;
        }

        // CPU BLAS
        {
            cpu_time_used = get_time_us_no_sync();
            for(size_t b = 0; b < batch_count; ++b)
            {
                ref_tpmv<T>(uplo, transA, diag, N, hAp[b], hx[b], incx);
            }
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // Unit check.
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incx, hx, hres, batch_count);
        }

        // Norm check.
        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, incx, hx, hres, batch_count);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(
                rocblas_tpmv_batched_fn,
                (handle, uplo, transA, diag, N, dAp_on_device, dx_on_device, incx, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // Log performance.
        ArgumentModel<e_uplo, e_transA, e_diag, e_N, e_incx, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            tpmv_gflop_count<T>(N),
            tpmv_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error);
    }
}
