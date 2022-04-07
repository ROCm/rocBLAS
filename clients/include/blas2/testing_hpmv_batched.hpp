/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
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
void testing_hpmv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_hpmv_batched_fn
        = arg.fortran ? rocblas_hpmv_batched<T, true> : rocblas_hpmv_batched<T, false>;

    const rocblas_int N           = 100;
    const rocblas_int incx        = 1;
    const rocblas_int incy        = 1;
    const rocblas_int batch_count = 5;
    const T           alpha       = 0.5;
    const T           beta        = 2.0;
    const T           zero        = 0.0;
    const T           one         = 1.0;

    const rocblas_fill   uplo = rocblas_fill_upper;
    rocblas_local_handle handle{arg};

    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    // Allocate device memory
    device_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_batched_fn(handle,
                                                  rocblas_fill_full,
                                                  N,
                                                  &alpha,
                                                  dAp.ptr_on_device(),
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_batched_fn(handle,
                                                  uplo,
                                                  N,
                                                  &alpha,
                                                  nullptr,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_batched_fn(handle,
                                                  uplo,
                                                  N,
                                                  &alpha,
                                                  dAp.ptr_on_device(),
                                                  nullptr,
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_batched_fn(handle,
                                                  uplo,
                                                  N,
                                                  &alpha,
                                                  dAp.ptr_on_device(),
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  nullptr,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_batched_fn(handle,
                                                  uplo,
                                                  N,
                                                  nullptr,
                                                  dAp.ptr_on_device(),
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_batched_fn(handle,
                                                  uplo,
                                                  N,
                                                  &alpha,
                                                  dAp.ptr_on_device(),
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  nullptr,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_batched_fn(nullptr,
                                                  uplo,
                                                  N,
                                                  &alpha,
                                                  dAp.ptr_on_device(),
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_handle);

    // If batch_count==0, then all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_hpmv_batched_fn(
            handle, uplo, N, nullptr, nullptr, nullptr, incx, nullptr, nullptr, incy, 0),
        rocblas_status_success);

    // If N==0, then all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_hpmv_batched_fn(
            handle, uplo, 0, nullptr, nullptr, nullptr, incx, nullptr, nullptr, incy, batch_count),
        rocblas_status_success);

    // If alpha==0, then A and x may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_hpmv_batched_fn(handle,
                                                  uplo,
                                                  N,
                                                  &zero,
                                                  nullptr,
                                                  nullptr,
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_success);

    // If alpha==0 && beta==1, then A, x and y may be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_hpmv_batched_fn(
            handle, uplo, N, &zero, nullptr, nullptr, incx, &one, nullptr, incy, batch_count),
        rocblas_status_success);
}

template <typename T>
void testing_hpmv_batched(const Arguments& arg)
{
    auto rocblas_hpmv_batched_fn
        = arg.fortran ? rocblas_hpmv_batched<T, true> : rocblas_hpmv_batched<T, false>;

    rocblas_int  N           = arg.N;
    rocblas_int  incx        = arg.incx;
    rocblas_int  incy        = arg.incy;
    rocblas_int  batch_count = arg.batch_count;
    T            h_alpha     = arg.get_alpha<T>();
    T            h_beta      = arg.get_beta<T>();
    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_hpmv_batched_fn(handle,
                                                      uplo,
                                                      N,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      incx,
                                                      nullptr,
                                                      nullptr,
                                                      incy,
                                                      batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = N * abs_incx;
    size_t size_y   = N * abs_incy;

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_batch_matrix<T> hA(N, N, N, batch_count);
    host_batch_matrix<T> hAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
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

    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Allocate device memory
    device_batch_matrix<T> dA(N, N, N, batch_count);
    device_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
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

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, N);

    hy_gold.copy_from(hy_1);
    hy_2.copy_from(hy_1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
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
        CHECK_HIP_ERROR(dy_2.transfer_from(hy_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_hpmv_batched_fn(handle,
                                                    uplo,
                                                    N,
                                                    &h_alpha,
                                                    dAp.ptr_on_device(),
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    &h_beta,
                                                    dy_1.ptr_on_device(),
                                                    incy,
                                                    batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_hpmv_batched_fn(handle,
                                                    uplo,
                                                    N,
                                                    d_alpha,
                                                    dAp.ptr_on_device(),
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    d_beta,
                                                    dy_2.ptr_on_device(),
                                                    incy,
                                                    batch_count));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        for(int b = 0; b < batch_count; b++)
            cblas_hpmv<T>(uplo, N, h_alpha, hAp[b], hx[b], incx, h_beta, hy_gold[b], incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy_1));
        CHECK_HIP_ERROR(hy_2.transfer_from(dy_2));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_1, batch_count);
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_hpmv_batched_fn(handle,
                                    uplo,
                                    N,
                                    &h_alpha,
                                    dAp.ptr_on_device(),
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
            rocblas_hpmv_batched_fn(handle,
                                    uplo,
                                    N,
                                    &h_alpha,
                                    dAp.ptr_on_device(),
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
                         hpmv_gflop_count<T>(N),
                         hpmv_gbyte_count<T>(N),
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
