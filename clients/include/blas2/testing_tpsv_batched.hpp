/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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
void testing_tpsv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_tpsv_batched_fn
        = arg.api == FORTRAN ? rocblas_tpsv_batched<T, true> : rocblas_tpsv_batched<T, false>;

    const rocblas_int       N           = 100;
    const rocblas_int       incx        = 1;
    const rocblas_int       batch_count = 3;
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
    EXPECT_ROCBLAS_STATUS(
        rocblas_tpsv_batched_fn(
            handle, rocblas_fill_full, transA, diag, N, dAp, dx, incx, batch_count),
        rocblas_status_invalid_value);
    // arg_checks code shared so transA, diag tested only in non-batched

    EXPECT_ROCBLAS_STATUS(
        rocblas_tpsv_batched_fn(handle, uplo, transA, diag, N, nullptr, dx, incx, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_tpsv_batched_fn(handle, uplo, transA, diag, N, dAp, nullptr, incx, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_tpsv_batched_fn(nullptr, uplo, transA, diag, N, dAp, dx, incx, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_tpsv_batched(const Arguments& arg)
{
    auto rocblas_tpsv_batched_fn
        = arg.api == FORTRAN ? rocblas_tpsv_batched<T, true> : rocblas_tpsv_batched<T, false>;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    char        char_uplo   = arg.uplo;
    char        char_transA = arg.transA;
    char        char_diag   = arg.diag;
    rocblas_int batch_count = arg.batch_count;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_tpsv_batched_fn(
                handle, uplo, transA, diag, N, nullptr, nullptr, incx, batch_count),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier    = 40.0;
    double residual_eps_multiplier = 20.0;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_batch_matrix<T> hA(N, N, N, batch_count);
    host_batch_matrix<T> hAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
    host_batch_vector<T> hb(N, incx, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hx_or_b(N, incx, batch_count);
    host_batch_vector<T> cpu_x_or_b(N, incx, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAp.memcheck());
    CHECK_HIP_ERROR(hb.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hx_or_b.memcheck());
    CHECK_HIP_ERROR(cpu_x_or_b.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
    device_batch_vector<T> dx_or_b(N, incx, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_or_b.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(hA,
                        arg,
                        rocblas_client_never_set_nan,
                        rocblas_client_diagonally_dominant_triangular_matrix,
                        true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(diag == rocblas_diagonal_unit)
    {
        make_unit_diagonal(uplo, hA);
    }

    hb.copy_from(hx);
    for(int b = 0; b < batch_count; b++)
    {
        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, N, hA[b], N, hb[b], incx);
    }

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, N);

    cpu_x_or_b.copy_from(hb);
    hx_or_b.copy_from(hb);

    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b));

    double max_err = 0.0;
    double max_res = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_tpsv_batched_fn(handle,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    N,
                                                    dAp.ptr_on_device(),
                                                    dx_or_b.ptr_on_device(),
                                                    incx,
                                                    batch_count));
        handle.post_test(arg);
        CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));

        max_err = vector_norm_1(N, incx, hx, hx_or_b);

        if(arg.unit_check)
            trsm_err_res_check<T>(max_err, N, error_eps_multiplier, eps);

        // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmv<T>(uplo, transA, diag, N, hA[b], N, hx_or_b[b], incx);
        }

        max_res = vector_norm_1(N, incx, hx_or_b, hb);

        if(arg.unit_check)
            trsm_err_res_check<T>(max_res, N, residual_eps_multiplier, eps);
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(dx_or_b.transfer_from(cpu_x_or_b));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_tpsv_batched_fn(handle,
                                    uplo,
                                    transA,
                                    diag,
                                    N,
                                    dAp.ptr_on_device(),
                                    dx_or_b.ptr_on_device(),
                                    incx,
                                    batch_count);

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_tpsv_batched_fn(handle,
                                    uplo,
                                    transA,
                                    diag,
                                    N,
                                    dAp.ptr_on_device(),
                                    dx_or_b.ptr_on_device(),
                                    incx,
                                    batch_count);

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // CPU cblas
        cpu_time_used = get_time_us_no_sync();

        if(arg.norm_check)
            for(int b = 0; b < batch_count; b++)
                cblas_tpsv<T>(uplo, transA, diag, N, hAp[b], cpu_x_or_b[b], incx);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        ArgumentModel<e_uplo, e_transA, e_diag, e_N, e_incx, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            tpsv_gflop_count<T>(N),
            ArgumentLogging::NA_value,
            cpu_time_used,
            max_err,
            max_res);
    }
}
