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
void testing_trsv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_trsv_strided_batched_fn = arg.fortran ? rocblas_trsv_strided_batched<T, true>
                                                       : rocblas_trsv_strided_batched<T, false>;

    const rocblas_int       M           = 100;
    const rocblas_int       lda         = 100;
    const rocblas_int       incx        = 1;
    const rocblas_int       batch_count = 1;
    const rocblas_stride    stride_a    = M * lda;
    const rocblas_stride    stride_x    = M;
    const rocblas_operation transA      = rocblas_operation_none;
    const rocblas_fill      uplo        = rocblas_fill_lower;
    const rocblas_diagonal  diag        = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(M, M, lda, stride_a, batch_count);
    host_strided_batch_vector<T> hx(M, incx, stride_x, batch_count);

    // Check host memory allocation
    CHECK_DEVICE_ALLOCATION(hA.memcheck());
    CHECK_DEVICE_ALLOCATION(hx.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(M, M, lda, stride_a, batch_count);
    device_strided_batch_vector<T> dx(M, incx, stride_x, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    // Checks.
    EXPECT_ROCBLAS_STATUS(rocblas_trsv_strided_batched_fn(handle,
                                                          rocblas_fill_full,
                                                          transA,
                                                          diag,
                                                          M,
                                                          dA,
                                                          lda,
                                                          stride_a,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          batch_count),
                          rocblas_status_invalid_value);
    // arg_checks code shared so transA, diag tested only in non-batched

    EXPECT_ROCBLAS_STATUS(
        rocblas_trsv_strided_batched_fn(
            handle, uplo, transA, diag, M, nullptr, lda, stride_a, dx, incx, stride_x, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trsv_strided_batched_fn(
            handle, uplo, transA, diag, M, dA, lda, stride_a, nullptr, incx, stride_x, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trsv_strided_batched_fn(
            nullptr, uplo, transA, diag, M, dA, lda, stride_a, dx, incx, stride_x, batch_count),
        rocblas_status_invalid_handle);
}

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 40

template <typename T>
void testing_trsv_strided_batched(const Arguments& arg)
{
    auto rocblas_trsv_strided_batched_fn = arg.fortran ? rocblas_trsv_strided_batched<T, true>
                                                       : rocblas_trsv_strided_batched<T, false>;

    rocblas_int M           = arg.M;
    rocblas_int lda         = arg.lda;
    rocblas_int incx        = arg.incx;
    char        char_uplo   = arg.uplo;
    char        char_transA = arg.transA;
    char        char_diag   = arg.diag;
    rocblas_int stride_a    = arg.stride_a;
    rocblas_int stride_x    = arg.stride_x;
    rocblas_int batch_count = arg.batch_count;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || lda < M || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !M || !batch_count)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(rocblas_trsv_strided_batched_fn(handle,
                                                              uplo,
                                                              transA,
                                                              diag,
                                                              M,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              incx,
                                                              stride_x,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(M, M, lda, stride_a, batch_count);
    host_strided_batch_matrix<T> hAAT(M, M, lda, stride_a, batch_count);
    host_strided_batch_vector<T> hb(M, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx(M, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx_or_b_1(M, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx_or_b_2(M, incx, stride_x, batch_count);
    host_strided_batch_vector<T> cpu_x_or_b(M, incx, stride_x, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAAT.memcheck());
    CHECK_HIP_ERROR(hb.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hx_or_b_1.memcheck());
    CHECK_HIP_ERROR(hx_or_b_2.memcheck());
    CHECK_HIP_ERROR(cpu_x_or_b.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(M, M, lda, stride_a, batch_count);
    device_strided_batch_vector<T> dx_or_b(M, incx, stride_x, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
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

    // Calculate hb = hA*hx;
    for(int b = 0; b < batch_count; b++)
    {
        cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hb + stride_x * b, incx);
    }

    cpu_x_or_b.copy_from(hb);
    hx_or_b_1.copy_from(hb);
    hx_or_b_2.copy_from(hb);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

    double error_host       = 0.0;
    double error_device     = 0.0;
    double max_error_host   = 0.0;
    double max_error_device = 0.0;
    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    if(!ROCBLAS_REALLOC_ON_DEMAND)
    {
        // Compute size
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));

        CHECK_ALLOC_QUERY(rocblas_trsv_strided_batched_fn(handle,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          dA,
                                                          lda,
                                                          stride_a,
                                                          dx_or_b,
                                                          incx,
                                                          stride_x,
                                                          batch_count));
        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        // Allocate memory
        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_trsv_strided_batched_fn(handle,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dx_or_b,
                                                            incx,
                                                            stride_x,
                                                            batch_count));
        handle.post_test(arg);
        CHECK_HIP_ERROR(hx_or_b_1.transfer_from(dx_or_b));

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_2));
        CHECK_ROCBLAS_ERROR(rocblas_trsv_strided_batched_fn(handle,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dx_or_b,
                                                            incx,
                                                            stride_x,
                                                            batch_count));
        CHECK_HIP_ERROR(hx_or_b_2.transfer_from(dx_or_b));

        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate norm 1 of vector E
        for(int b = 0; b < batch_count; b++)
        {
            error_host       = rocblas_abs(vector_norm_1<T>(M, incx, hx[b], hx_or_b_1[b]));
            error_device     = rocblas_abs(vector_norm_1<T>(M, incx, hx[b], hx_or_b_2[b]));
            max_error_host   = std::max(max_error_host, error_host);
            max_error_device = std::max(max_error_device, error_device);

            //unit test
            trsm_err_res_check<T>(error_host, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(error_device, M, error_eps_multiplier, eps);
        }

        // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hx_or_b_1[b], incx);
            cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hx_or_b_2[b], incx);
        }

        //calculate norm 1 of res
        for(int b = 0; b < batch_count; b++)
        {
            error_host   = rocblas_abs(vector_norm_1<T>(M, incx, hx_or_b_1[b], hb[b]));
            error_device = rocblas_abs(vector_norm_1<T>(M, incx, hx_or_b_1[b], hb[b]));

            //unit test
            trsm_err_res_check<T>(error_host, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(error_device, M, error_eps_multiplier, eps);
        }
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_trsv_strided_batched_fn(handle,
                                            uplo,
                                            transA,
                                            diag,
                                            M,
                                            dA,
                                            lda,
                                            stride_a,
                                            dx_or_b,
                                            incx,
                                            stride_x,
                                            batch_count);

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_trsv_strided_batched_fn(handle,
                                            uplo,
                                            transA,
                                            diag,
                                            M,
                                            dA,
                                            lda,
                                            stride_a,
                                            dx_or_b,
                                            incx,
                                            stride_x,
                                            batch_count);

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // CPU cblas
        cpu_time_used = get_time_us_no_sync();

        if(arg.norm_check)
            for(int b = 0; b < batch_count; b++)
                cblas_trsv<T>(uplo, transA, diag, M, hA[b], lda, cpu_x_or_b[b], incx);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        ArgumentModel<e_uplo,
                      e_transA,
                      e_diag,
                      e_M,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         trsv_gflop_count<T>(M),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         max_error_host,
                         max_error_device);
    }
}
