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
void testing_gbmv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_gbmv_strided_batched_fn = arg.fortran ? rocblas_gbmv_strided_batched<T, true>
                                                       : rocblas_gbmv_strided_batched<T, false>;

    const rocblas_int       M                 = 100;
    const rocblas_int       N                 = 100;
    const rocblas_int       KL                = 5;
    const rocblas_int       KU                = 5;
    const rocblas_int       lda               = 100;
    const rocblas_int       incx              = 1;
    const rocblas_int       incy              = 1;
    const T                 alpha             = 0.5;
    const T                 beta              = 1.5;
    const T                 zero              = 0.0;
    const T                 one               = 1.0;
    const rocblas_int       stride_A          = 10000;
    const rocblas_int       stride_x          = 100;
    const rocblas_int       stride_y          = 100;
    const rocblas_int       batch_count       = 5;
    const rocblas_int       banded_matrix_row = KL + KU + 1;
    const rocblas_operation transA            = rocblas_operation_none;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_strided_batch_matrix<T> dAb(banded_matrix_row, N, lda, stride_A, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count),
        dy(M, incy, stride_y, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAb.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          nullptr,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          dAb,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          dAb,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          nullptr,
                                                          dAb,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          dAb,
                                                          lda,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(nullptr,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &alpha,
                                                          dAb,
                                                          lda,
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

    // If batch_count == 0, then all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          nullptr,
                                                          nullptr,
                                                          lda,
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

    // If M==0, then all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          0,
                                                          N,
                                                          KL,
                                                          KU,
                                                          nullptr,
                                                          nullptr,
                                                          lda,
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

    // If N==0, then all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          M,
                                                          0,
                                                          KL,
                                                          KU,
                                                          nullptr,
                                                          nullptr,
                                                          lda,
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

    // If alpha==0, then A and X may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &zero,
                                                          nullptr,
                                                          lda,
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

    // If alpha==0 && beta==1, then A, X and Y may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          KL,
                                                          KU,
                                                          &zero,
                                                          nullptr,
                                                          lda,
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
void testing_gbmv_strided_batched(const Arguments& arg)
{
    auto rocblas_gbmv_strided_batched_fn = arg.fortran ? rocblas_gbmv_strided_batched<T, true>
                                                       : rocblas_gbmv_strided_batched<T, false>;

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
    rocblas_int       stride_A          = arg.stride_a;
    rocblas_int       stride_x          = arg.stride_x;
    rocblas_int       stride_y          = arg.stride_y;
    rocblas_int       batch_count       = arg.batch_count;
    rocblas_int       banded_matrix_row = KL + KU + 1;

    rocblas_local_handle handle{arg};
    size_t               dim_x;
    size_t               dim_y, abs_incy;

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

    abs_incy = incy >= 0 ? incy : -incy;

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < banded_matrix_row || !incx || !incy || KL < 0
                        || KU < 0 || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gbmv_strided_batched_fn(handle,
                                                              transA,
                                                              M,
                                                              N,
                                                              KL,
                                                              KU,
                                                              nullptr,
                                                              nullptr,
                                                              lda,
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

    // Naming: `h` is in CPU (host) memory(eg hAb), `d` is in GPU (device) memory (eg dAb).
    // Allocate host memory
    host_strided_batch_matrix<T> hAb(banded_matrix_row, N, lda, stride_A, batch_count);
    host_strided_batch_vector<T> hx(dim_x, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy_1(dim_y, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_2(dim_y, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_gold(dim_y, incy, stride_y, batch_count);
    host_vector<T>               halpha(1);
    host_vector<T>               hbeta(1);
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Check host memory allocation
    CHECK_HIP_ERROR(hAb.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy_1.memcheck());
    CHECK_HIP_ERROR(hy_2.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dAb(banded_matrix_row, N, lda, stride_A, batch_count);
    device_strided_batch_vector<T> dx(dim_x, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy_1(dim_y, incy, stride_y, batch_count);
    device_strided_batch_vector<T> dy_2(dim_y, incy, stride_y, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);

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

    // copy vector is easy in STL; hy_gold = hy_1: save a copy in hy_gold which will be output of
    // CPU BLAS
    hy_gold.copy_from(hy_1);
    hy_2.copy_from(hy_1);

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
        CHECK_ROCBLAS_ERROR(rocblas_gbmv_strided_batched_fn(handle,
                                                            transA,
                                                            M,
                                                            N,
                                                            KL,
                                                            KU,
                                                            &h_alpha,
                                                            dAb,
                                                            lda,
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
        CHECK_ROCBLAS_ERROR(rocblas_gbmv_strided_batched_fn(handle,
                                                            transA,
                                                            M,
                                                            N,
                                                            KL,
                                                            KU,
                                                            d_alpha,
                                                            dAb,
                                                            lda,
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
            unit_check_general<T>(1, dim_y, abs_incy, stride_y, hy_gold, hy_1, batch_count);
            unit_check_general<T>(1, dim_y, abs_incy, stride_y, hy_gold, hy_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>(
                'F', 1, dim_y, abs_incy, stride_y, hy_gold, hy_1, batch_count);
            rocblas_error_2 = norm_check_general<T>(
                'F', 1, dim_y, abs_incy, stride_y, hy_gold, hy_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_gbmv_strided_batched_fn(handle,
                                            transA,
                                            M,
                                            N,
                                            KL,
                                            KU,
                                            &h_alpha,
                                            dAb,
                                            lda,
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
            rocblas_gbmv_strided_batched_fn(handle,
                                            transA,
                                            M,
                                            N,
                                            KL,
                                            KU,
                                            &h_alpha,
                                            dAb,
                                            lda,
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

        ArgumentModel<e_transA,
                      e_M,
                      e_N,
                      e_KL,
                      e_KU,
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
                         gbmv_gflop_count<T>(transA, M, N, KL, KU),
                         gbmv_gbyte_count<T>(transA, M, N, KL, KU),
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
