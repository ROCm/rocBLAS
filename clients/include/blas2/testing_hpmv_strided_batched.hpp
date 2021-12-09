/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
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
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_hpmv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_hpmv_strided_batched_fn = arg.fortran ? rocblas_hpmv_strided_batched<T, true>
                                                       : rocblas_hpmv_strided_batched<T, false>;

    const rocblas_int    N           = 100;
    const rocblas_int    incx        = 1;
    const rocblas_int    incy        = 1;
    const rocblas_int    batch_count = 5;
    const rocblas_stride stride_A    = 10000;
    const rocblas_stride stride_x    = 100;
    const rocblas_stride stride_y    = 100;
    const T              alpha       = 0.5;
    const T              beta        = 2.0;
    const T              zero        = 0.0;
    const T              one         = 1.0;

    const rocblas_fill   uplo = rocblas_fill_upper;
    rocblas_local_handle handle{arg};

    size_t size_A = size_t(N) * N;

    // allocate memory on device
    device_strided_batch_vector<T> dA(size_A, 1, stride_A, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          nullptr,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dA,
                                                          stride_A,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dA,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          nullptr,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          nullptr,
                                                          dA,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dA,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          nullptr,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(nullptr,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dA,
                                                          stride_A,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_invalid_handle);

    // If batch_count==0, then all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          nullptr,
                                                          nullptr,
                                                          stride_A,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          nullptr,
                                                          nullptr,
                                                          incy,
                                                          stride_y,
                                                          0),
                          rocblas_status_success);

    // If N==0, then all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                          uplo,
                                                          0,
                                                          nullptr,
                                                          nullptr,
                                                          stride_A,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          nullptr,
                                                          nullptr,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_success);

    // If alpha==0, then A and x may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          &zero,
                                                          nullptr,
                                                          stride_A,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          &beta,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_success);

    // If alpha==0 && beta==1, then A, x and y may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          &zero,
                                                          nullptr,
                                                          stride_A,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          &one,
                                                          nullptr,
                                                          incy,
                                                          stride_y,
                                                          batch_count),
                          rocblas_status_success);
}

template <typename T>
void testing_hpmv_strided_batched(const Arguments& arg)
{
    auto rocblas_hpmv_strided_batched_fn = arg.fortran ? rocblas_hpmv_strided_batched<T, true>
                                                       : rocblas_hpmv_strided_batched<T, false>;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_int    incy        = arg.incy;
    T              h_alpha     = arg.get_alpha<T>();
    T              h_beta      = arg.get_beta<T>();
    rocblas_fill   uplo        = char2rocblas_fill(arg.uplo);
    rocblas_stride stride_A    = arg.stride_a;
    rocblas_stride stride_x    = arg.stride_x;
    rocblas_stride stride_y    = arg.stride_y;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_hpmv_strided_batched_fn(handle,
                                                              uplo,
                                                              N,
                                                              nullptr,
                                                              nullptr,
                                                              stride_A,
                                                              nullptr,
                                                              incx,
                                                              stride_x,
                                                              nullptr,
                                                              nullptr,
                                                              incy,
                                                              stride_y,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = (size_t(N * (N + 1)) / 2);
    size_t size_x   = N;
    size_t size_y   = N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_strided_batch_vector<T> hA(size_A, 1, stride_A, batch_count);
    host_strided_batch_vector<T> hx(size_x, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy_1(size_y, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_2(size_y, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_gold(size_y, incy, stride_y, batch_count);
    host_vector<T>               halpha(1);
    host_vector<T>               hbeta(1);
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy_1.memcheck());
    CHECK_HIP_ERROR(hy_2.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());
    CHECK_HIP_ERROR(halpha.memcheck());
    CHECK_HIP_ERROR(hbeta.memcheck());

    device_strided_batch_vector<T> dA(size_A, 1, stride_A, batch_count);
    device_strided_batch_vector<T> dx(size_x, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy_1(size_y, incy, stride_y, batch_count);
    device_strided_batch_vector<T> dy_2(size_y, incy, stride_y, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hA, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy_1, arg, rocblas_client_beta_sets_nan);

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
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_hpmv_strided_batched_fn(handle,
                                                            uplo,
                                                            N,
                                                            &h_alpha,
                                                            dA,
                                                            stride_A,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            &h_beta,
                                                            dy_1,
                                                            incy,
                                                            stride_y,
                                                            batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_hpmv_strided_batched_fn(handle,
                                                            uplo,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            stride_A,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            d_beta,
                                                            dy_2,
                                                            incy,
                                                            stride_y,
                                                            batch_count));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        for(int b = 0; b < batch_count; b++)
            cblas_hpmv<T>(uplo,
                          N,
                          h_alpha,
                          hA + b * stride_A,
                          hx + b * stride_x,
                          incx,
                          h_beta,
                          hy_gold + b * stride_y,
                          incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy_1));
        CHECK_HIP_ERROR(hy_2.transfer_from(dy_2));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, stride_y, hy_gold, hy_1, batch_count);
            unit_check_general<T>(1, N, abs_incy, stride_y, hy_gold, hy_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', 1, N, abs_incy, stride_y, hy_gold, hy_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('F', 1, N, abs_incy, stride_y, hy_gold, hy_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_hpmv_strided_batched_fn(handle,
                                            uplo,
                                            N,
                                            &h_alpha,
                                            dA,
                                            stride_A,
                                            dx,
                                            incx,
                                            stride_x,
                                            &h_beta,
                                            dy_1,
                                            incy,
                                            stride_y,
                                            batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_hpmv_strided_batched_fn(handle,
                                            uplo,
                                            N,
                                            &h_alpha,
                                            dA,
                                            stride_A,
                                            dx,
                                            incx,
                                            stride_x,
                                            &h_beta,
                                            dy_1,
                                            incy,
                                            stride_y,
                                            batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_beta,
                      e_incy,
                      e_stride_y,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         hpmv_gflop_count<T>(N),
                         hpmv_gbyte_count<T>(N),
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
