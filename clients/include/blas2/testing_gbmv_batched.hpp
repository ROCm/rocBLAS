/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
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
void testing_gbmv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_gbmv_batched_fn
        = arg.fortran ? rocblas_gbmv_batched<T, true> : rocblas_gbmv_batched<T, false>;

    const rocblas_int M                 = 100;
    const rocblas_int N                 = 100;
    const rocblas_int KL                = 5;
    const rocblas_int KU                = 5;
    const rocblas_int lda               = 100;
    const rocblas_int incx              = 1;
    const rocblas_int incy              = 1;
    const T           alpha             = 1.0;
    const T           beta              = 1.0;
    const rocblas_int batch_count       = 5;
    const rocblas_int safe_size         = 100;
    const rocblas_int banded_matrix_row = KL + KU + 1;

    const rocblas_operation transA = rocblas_operation_none;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_batch_matrix<T> dAb(banded_matrix_row, N, lda, batch_count);
    device_batch_vector<T> dx(safe_size, 1, batch_count);
    device_batch_vector<T> dy(safe_size, 1, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAb.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    auto dA_dev = dAb.ptr_on_device();
    auto dx_dev = dx.ptr_on_device();
    auto dy_dev = dy.ptr_on_device();

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_batched_fn(handle,
                                                  (rocblas_operation)rocblas_fill_full,
                                                  M,
                                                  N,
                                                  KL,
                                                  KU,
                                                  &alpha,
                                                  dA_dev,
                                                  lda,
                                                  dx_dev,
                                                  incx,
                                                  &beta,
                                                  dy_dev,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_batched_fn(handle,
                                                  transA,
                                                  M,
                                                  N,
                                                  KL,
                                                  KU,
                                                  &alpha,
                                                  nullptr,
                                                  lda,
                                                  dx_dev,
                                                  incx,
                                                  &beta,
                                                  dy_dev,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_batched_fn(handle,
                                                  transA,
                                                  M,
                                                  N,
                                                  KL,
                                                  KU,
                                                  &alpha,
                                                  dA_dev,
                                                  lda,
                                                  nullptr,
                                                  incx,
                                                  &beta,
                                                  dy_dev,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_batched_fn(handle,
                                                  transA,
                                                  M,
                                                  N,
                                                  KL,
                                                  KU,
                                                  &alpha,
                                                  dA_dev,
                                                  lda,
                                                  dx_dev,
                                                  incx,
                                                  &beta,
                                                  nullptr,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_batched_fn(handle,
                                                  transA,
                                                  M,
                                                  N,
                                                  KL,
                                                  KU,
                                                  nullptr,
                                                  dA_dev,
                                                  lda,
                                                  dx_dev,
                                                  incx,
                                                  &beta,
                                                  dy_dev,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_batched_fn(handle,
                                                  transA,
                                                  M,
                                                  N,
                                                  KL,
                                                  KU,
                                                  &alpha,
                                                  dA_dev,
                                                  lda,
                                                  dx_dev,
                                                  incx,
                                                  nullptr,
                                                  dy_dev,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_batched_fn(nullptr,
                                                  transA,
                                                  M,
                                                  N,
                                                  KL,
                                                  KU,
                                                  &alpha,
                                                  dA_dev,
                                                  lda,
                                                  dx_dev,
                                                  incx,
                                                  &beta,
                                                  dy_dev,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_gbmv_batched(const Arguments& arg)
{
    auto rocblas_gbmv_batched_fn
        = arg.fortran ? rocblas_gbmv_batched<T, true> : rocblas_gbmv_batched<T, false>;

    rocblas_int       M                 = arg.M;
    rocblas_int       N                 = arg.N;
    rocblas_int       KL                = arg.KL;
    rocblas_int       KU                = arg.KU;
    rocblas_int       lda               = arg.lda;
    rocblas_int       incx              = arg.incx;
    rocblas_int       incy              = arg.incy;
    T                 h_alpha           = arg.get_alpha<T>();
    T                 h_beta            = arg.get_beta<T>();
    rocblas_operation transA            = char2rocblas_operation(arg.transA);
    rocblas_int       batch_count       = arg.batch_count;
    rocblas_int       banded_matrix_row = KL + KU + 1;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < banded_matrix_row || !incx || !incy
                        || batch_count < 0 || KL < 0 || KU < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gbmv_batched_fn(handle,
                                                      transA,
                                                      M,
                                                      N,
                                                      KL,
                                                      KU,
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

    size_t dim_x, abs_incx;
    size_t dim_y, abs_incy;

    if(transA == rocblas_operation_none)
    {
        dim_x = N;
        dim_y = M;
    }
    else
    {
        dim_x = M;
        dim_y = N;
    }

    abs_incx = incx >= 0 ? incx : -incx;
    abs_incy = incy >= 0 ? incy : -incy;

    // Naming: `h` is in CPU (host) memory(eg hAb), `d` is in GPU (device) memory (eg dAb).
    // Allocate host memory
    host_batch_matrix<T> hAb(banded_matrix_row, N, lda, batch_count);
    host_batch_vector<T> hx(dim_x, incx, batch_count);
    host_batch_vector<T> hy_1(dim_y, incy, batch_count);
    host_batch_vector<T> hy_2(dim_y, incy, batch_count);
    host_batch_vector<T> hy_gold(dim_y, incy, batch_count);
    host_vector<T>       halpha(1);
    host_vector<T>       hbeta(1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hAb.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy_1.memcheck());
    CHECK_HIP_ERROR(hy_2.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());

    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Allocate device memory
    device_batch_matrix<T> dAb(banded_matrix_row, N, lda, batch_count);
    device_batch_vector<T> dx(dim_x, incx, batch_count);
    device_batch_vector<T> dy_1(dim_y, incy, batch_count);
    device_batch_vector<T> dy_2(dim_y, incy, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAb.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hAb, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy_1, arg, rocblas_client_beta_sets_nan);

    hy_2.copy_from(hy_1);
    hy_gold.copy_from(hy_1);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAb.transfer_from(hAb));
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
        CHECK_ROCBLAS_ERROR(rocblas_gbmv_batched_fn(handle,
                                                    transA,
                                                    M,
                                                    N,
                                                    KL,
                                                    KU,
                                                    &h_alpha,
                                                    dAb.ptr_on_device(),
                                                    lda,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    &h_beta,
                                                    dy_1.ptr_on_device(),
                                                    incy,
                                                    batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_gbmv_batched_fn(handle,
                                                    transA,
                                                    M,
                                                    N,
                                                    KL,
                                                    KU,
                                                    d_alpha,
                                                    dAb.ptr_on_device(),
                                                    lda,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    d_beta,
                                                    dy_2.ptr_on_device(),
                                                    incy,
                                                    batch_count));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_gbmv<T>(
                transA, M, N, KL, KU, h_alpha, hAb[b], lda, hx[b], incx, h_beta, hy_gold[b], incy);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy_1));
        CHECK_HIP_ERROR(hy_2.transfer_from(dy_2));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold, hy_1, batch_count);
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold, hy_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold, hy_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold, hy_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_gbmv_batched_fn(handle,
                                    transA,
                                    M,
                                    N,
                                    KL,
                                    KU,
                                    &h_alpha,
                                    dAb.ptr_on_device(),
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
            rocblas_gbmv_batched_fn(handle,
                                    transA,
                                    M,
                                    N,
                                    KL,
                                    KU,
                                    &h_alpha,
                                    dAb.ptr_on_device(),
                                    lda,
                                    dx.ptr_on_device(),
                                    incx,
                                    &h_beta,
                                    dy_1.ptr_on_device(),
                                    incy,
                                    batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_M,
                      e_N,
                      e_KL,
                      e_KU,
                      e_alpha,
                      e_lda,
                      e_incx,
                      e_beta,
                      e_incy,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gbmv_gflop_count<T>(transA, M, N, KL, KU),
                         gbmv_gbyte_count<T>(transA, M, N, KL, KU),
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
