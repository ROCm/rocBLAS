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
void testing_hemv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_hemv_batched_fn
        = arg.fortran ? rocblas_hemv_batched<T, true> : rocblas_hemv_batched<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_fill uplo        = rocblas_fill_upper;
        const rocblas_int  N           = 100;
        const rocblas_int  lda         = 100;
        const rocblas_int  incx        = 1;
        const rocblas_int  incy        = 1;
        const rocblas_int  batch_count = 2;

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

        // Allocate device memory
        device_batch_matrix<T> dA(N, N, lda, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);
        device_batch_vector<T> dy(N, incy, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dx.memcheck());
        CHECK_DEVICE_ALLOCATION(dy.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(nullptr,
                                                      uplo,
                                                      N,
                                                      alpha,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dx.ptr_on_device(),
                                                      incx,
                                                      beta,
                                                      dy.ptr_on_device(),
                                                      incy,
                                                      batch_count),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                      rocblas_fill_full,
                                                      N,
                                                      alpha,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dx.ptr_on_device(),
                                                      incx,
                                                      beta,
                                                      dy.ptr_on_device(),
                                                      incy,
                                                      batch_count),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                      uplo,
                                                      N,
                                                      nullptr,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dx.ptr_on_device(),
                                                      incx,
                                                      beta,
                                                      dy.ptr_on_device(),
                                                      incy,
                                                      batch_count),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                      uplo,
                                                      N,
                                                      alpha,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dx.ptr_on_device(),
                                                      incx,
                                                      nullptr,
                                                      dy.ptr_on_device(),
                                                      incy,
                                                      batch_count),
                              rocblas_status_invalid_pointer);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          alpha,
                                                          nullptr,
                                                          lda,
                                                          dx.ptr_on_device(),
                                                          incx,
                                                          beta,
                                                          dy.ptr_on_device(),
                                                          incy,
                                                          batch_count),
                                  rocblas_status_invalid_pointer);

            EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          alpha,
                                                          dA.ptr_on_device(),
                                                          lda,
                                                          nullptr,
                                                          incx,
                                                          beta,
                                                          dy.ptr_on_device(),
                                                          incy,
                                                          batch_count),
                                  rocblas_status_invalid_pointer);

            EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          alpha,
                                                          dA.ptr_on_device(),
                                                          lda,
                                                          dx.ptr_on_device(),
                                                          incx,
                                                          beta,
                                                          nullptr,
                                                          incy,
                                                          batch_count),
                                  rocblas_status_invalid_pointer);
        }

        // When N==0, all pointers may be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                      uplo,
                                                      0,
                                                      nullptr,
                                                      nullptr,
                                                      lda,
                                                      nullptr,
                                                      incx,
                                                      nullptr,
                                                      nullptr,
                                                      incy,
                                                      batch_count),
                              rocblas_status_success);

        // When alpha==0, A and x can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                      uplo,
                                                      N,
                                                      zero,
                                                      nullptr,
                                                      lda,
                                                      nullptr,
                                                      incx,
                                                      beta,
                                                      dy.ptr_on_device(),
                                                      incy,
                                                      batch_count),
                              rocblas_status_success);

        // When alpha==0 && beta==1, A, x and y can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                      uplo,
                                                      N,
                                                      zero,
                                                      nullptr,
                                                      lda,
                                                      nullptr,
                                                      incx,
                                                      one,
                                                      nullptr,
                                                      incy,
                                                      batch_count),
                              rocblas_status_success);

        // When batch_count==0, all pointers may be nullptr without error
        EXPECT_ROCBLAS_STATUS(
            rocblas_hemv_batched_fn(
                handle, uplo, N, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy, 0),
            rocblas_status_success);
    }
}

template <typename T>
void testing_hemv_batched(const Arguments& arg)
{
    auto rocblas_hemv_batched_fn
        = arg.fortran ? rocblas_hemv_batched<T, true> : rocblas_hemv_batched<T, false>;

    rocblas_int  N           = arg.N;
    rocblas_int  lda         = arg.lda;
    rocblas_int  incx        = arg.incx;
    rocblas_int  incy        = arg.incy;
    rocblas_int  batch_count = arg.batch_count;
    T            h_alpha     = arg.get_alpha<T>();
    T            h_beta      = arg.get_beta<T>();
    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_hemv_batched_fn(handle,
                                                      uplo,
                                                      N,
                                                      nullptr,
                                                      nullptr,
                                                      lda,
                                                      nullptr,
                                                      incx,
                                                      nullptr,
                                                      nullptr,
                                                      incy,
                                                      batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(N, N, lda, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy_1(N, incy, batch_count);
    host_batch_vector<T> hy_2(N, incy, batch_count);
    host_batch_vector<T> hy_gold(N, incy, batch_count);
    host_vector<T>       halpha(1);
    host_vector<T>       hbeta(1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy_1.memcheck());
    CHECK_HIP_ERROR(hy_2.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(N, N, lda, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy_1(N, incy, batch_count);
    device_batch_vector<T> dy_2(N, incy, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_hermitian_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy_1, arg, rocblas_client_beta_sets_nan);
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    hy_gold.copy_from(hy_1);
    hy_2.copy_from(hy_1);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy_1.transfer_from(hy_1));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(dy_1.transfer_from(hy_1));
        CHECK_HIP_ERROR(dy_2.transfer_from(hy_2));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_hemv_batched_fn(handle,
                                                    uplo,
                                                    N,
                                                    &h_alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    &h_beta,
                                                    dy_1.ptr_on_device(),
                                                    incy,
                                                    batch_count));
        handle.post_test(arg);

        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_hemv_batched_fn(handle,
                                                    uplo,
                                                    N,
                                                    d_alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    d_beta,
                                                    dy_2.ptr_on_device(),
                                                    incy,
                                                    batch_count));
        handle.post_test(arg);

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        for(int b = 0; b < batch_count; b++)
            cblas_hemv<T>(uplo, N, h_alpha, hA[b], lda, hx[b], incx, h_beta, hy_gold[b], incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy_1));
        CHECK_HIP_ERROR(hy_2.transfer_from(dy_2));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incy, hy_gold, hy_1, batch_count);
            unit_check_general<T>(1, N, incy, hy_gold, hy_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, N, incy, hy_gold, hy_1, batch_count);
            rocblas_error_2 = norm_check_general<T>('F', 1, N, incy, hy_gold, hy_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_hemv_batched_fn(handle,
                                    uplo,
                                    N,
                                    &h_alpha,
                                    dA.ptr_on_device(),
                                    lda,
                                    dx.ptr_on_device(),
                                    incx,
                                    &h_beta,
                                    dy_1.ptr_on_device(),
                                    incy,
                                    batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_hemv_batched_fn(handle,
                                    uplo,
                                    N,
                                    &h_alpha,
                                    dA.ptr_on_device(),
                                    lda,
                                    dx.ptr_on_device(),
                                    incx,
                                    &h_beta,
                                    dy_1.ptr_on_device(),
                                    incy,
                                    batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_N, e_alpha, e_lda, e_incx, e_beta, e_incy, e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         hemv_gflop_count<T>(N),
                         hemv_gbyte_count<T>(N),
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
