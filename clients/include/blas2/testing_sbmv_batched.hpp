/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_sbmv_batched_bad_arg(const Arguments& arg)
{
    rocblas_fill uplo        = rocblas_fill_upper;
    rocblas_int  N           = 100;
    rocblas_int  K           = 2;
    rocblas_int  incx        = 1;
    rocblas_int  incy        = 1;
    rocblas_int  lda         = 100;
    T            alpha       = 0.6;
    T            beta        = 0.6;
    rocblas_int  batch_count = 2;

    rocblas_local_handle handle{arg};

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = lda * N;
    size_t size_x   = N * abs_incx * batch_count;
    size_t size_y   = N * abs_incy * batch_count;

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_batched<T>(nullptr,
                                                  uplo,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_batched<T>(handle,
                                                  rocblas_fill_full,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_batched<T>(handle,
                                                  uplo,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_batched<T>(handle,
                                                  uplo,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  nullptr,
                                                  lda,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_batched<T>(handle,
                                                  uplo,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  nullptr,
                                                  incx,
                                                  &beta,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_batched<T>(handle,
                                                  uplo,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  nullptr,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_sbmv_batched<T>(handle,
                                                  uplo,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  &beta,
                                                  nullptr,
                                                  incy,
                                                  batch_count),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_sbmv_batched(const Arguments& arg)
{
    rocblas_int N    = arg.N;
    rocblas_int lda  = arg.lda;
    rocblas_int K    = arg.K;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;

    host_vector<T> alpha(1);
    host_vector<T> beta(1);
    alpha[0] = arg.alpha;
    beta[0]  = arg.beta;

    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);
    rocblas_int  batch_count = arg.batch_count;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    size_t size_A = size_t(lda) * N;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || lda < K + 1 || K < 0 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_sbmv_batched<T>(handle,
                                                      uplo,
                                                      N,
                                                      K,
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

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hy2(N, incy, batch_count);
    host_batch_vector<T> hg(N, incy, batch_count);

    double gpu_time_used, cpu_time_used;
    double h_error, d_error;

    char char_fill = arg.uplo;

    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hA, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

    // save a copy in hg which will later get output of CPU BLAS
    hg.copy_from(hy);
    hy2.copy_from(hy);

    if(arg.unit_check || arg.norm_check)
    {
        cpu_time_used = get_time_us_no_sync();

        // cpu reference
        for(int i = 0; i < batch_count; i++)
        {
            cblas_sbmv<T>(uplo, N, K, alpha[0], hA[i], lda, hx[i], incx, beta[0], hg[i], incy);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;
    }

    // copy data from CPU to device
    dx.transfer_from(hx);
    dy.transfer_from(hy);
    dA.transfer_from(hA);

    if(arg.unit_check || arg.norm_check)
    {

        //
        // rocblas_pointer_mode_host test
        //
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_sbmv_batched<T>(handle,
                                                    uplo,
                                                    N,
                                                    K,
                                                    alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    beta,
                                                    dy.ptr_on_device(),
                                                    incy,
                                                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        //
        // rocblas_pointer_mode_device test
        //
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(d_alpha.transfer_from(alpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(beta));

        dy.transfer_from(hy2);

        CHECK_ROCBLAS_ERROR(rocblas_sbmv_batched<T>(handle,
                                                    uplo,
                                                    N,
                                                    K,
                                                    d_alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    d_beta,
                                                    dy.ptr_on_device(),
                                                    incy,
                                                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy2.transfer_from(dy));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hg, hy, batch_count);
            unit_check_general<T>(1, N, abs_incy, hg, hy2, batch_count);
        }

        if(arg.norm_check)
        {
            h_error = norm_check_general<T>('F', 1, N, abs_incy, hg, hy, batch_count);
            d_error = norm_check_general<T>('F', 1, N, abs_incy, hg, hy2, batch_count);
        }
    }

    if(arg.timing)
    {

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_sbmv_batched<T>(handle,
                                                        uplo,
                                                        N,
                                                        K,
                                                        alpha,
                                                        dA.ptr_on_device(),
                                                        lda,
                                                        dx.ptr_on_device(),
                                                        incx,
                                                        beta,
                                                        dy.ptr_on_device(),
                                                        incy,
                                                        batch_count));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_sbmv_batched<T>(handle,
                                                        uplo,
                                                        N,
                                                        K,
                                                        alpha,
                                                        dA.ptr_on_device(),
                                                        lda,
                                                        dx.ptr_on_device(),
                                                        incx,
                                                        beta,
                                                        dy.ptr_on_device(),
                                                        incy,
                                                        batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_N, e_K, e_alpha, e_lda, e_incx, e_beta, e_incy, e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         sbmv_gflop_count<T>(N, K),
                         sbmv_gbyte_count<T>(N, K),
                         cpu_time_used,
                         h_error,
                         d_error);
    }
}
