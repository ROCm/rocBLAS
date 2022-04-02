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
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_her2_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_her2_batched_fn
        = arg.fortran ? rocblas_her2_batched<T, true> : rocblas_her2_batched<T, false>;

    rocblas_fill         uplo        = rocblas_fill_upper;
    rocblas_int          N           = 100;
    rocblas_int          lda         = 100;
    rocblas_int          incx        = 1;
    rocblas_int          incy        = 1;
    T                    alpha       = 0.6;
    rocblas_int          batch_count = 2;
    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_batch_matrix<T> dA_1(N, N, lda, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_her2_batched<
            T>)(handle, rocblas_fill_full, N, &alpha, dx, incx, dy, incy, dA_1, lda, batch_count),
        rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_her2_batched<
            T>)(handle, uplo, N, &alpha, nullptr, incx, dy, incy, dA_1, lda, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_her2_batched<
            T>)(handle, uplo, N, &alpha, dx, incx, nullptr, incy, dA_1, lda, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_her2_batched<
            T>)(handle, uplo, N, &alpha, dx, incx, dy, incy, nullptr, lda, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_her2_batched<
            T>)(nullptr, uplo, N, &alpha, dx, incx, dy, incy, dA_1, lda, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_her2_batched(const Arguments& arg)
{
    auto rocblas_her2_batched_fn
        = arg.fortran ? rocblas_her2_batched<T, true> : rocblas_her2_batched<T, false>;

    rocblas_int  N           = arg.N;
    rocblas_int  lda         = arg.lda;
    rocblas_int  incx        = arg.incx;
    rocblas_int  incy        = arg.incy;
    T            h_alpha     = arg.get_alpha<T>();
    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);
    rocblas_int  batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument check before allocating invalid memory
    bool invalid_size = N < 0 || lda < 1 || lda < N || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS((rocblas_her2_batched<T>)(handle,
                                                        uplo,
                                                        N,
                                                        nullptr,
                                                        nullptr,
                                                        incx,
                                                        nullptr,
                                                        incy,
                                                        nullptr,
                                                        lda,
                                                        batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA_1), `d` is in GPU (device) memory (eg dA_1).
    // Allocate host memory
    host_batch_matrix<T> hA_1(N, N, lda, batch_count);
    host_batch_matrix<T> hA_2(N, N, lda, batch_count);
    host_batch_matrix<T> hA_gold(N, N, lda, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_vector<T>       halpha(1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA_1.memcheck());
    CHECK_HIP_ERROR(hA_2.memcheck());
    CHECK_HIP_ERROR(hA_gold.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());

    halpha[0] = h_alpha;

    // Allocate device memory
    device_batch_matrix<T> dA_1(N, N, lda, batch_count);
    device_batch_matrix<T> dA_2(N, N, lda, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<T>       d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA_1, arg, rocblas_client_never_set_nan, rocblas_client_hermitian_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan);

    hA_2.copy_from(hA_1);
    hA_gold.copy_from(hA_1);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA_1.transfer_from(hA_1));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1;
    double rocblas_error_2;

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
        CHECK_HIP_ERROR(dA_2.transfer_from(hA_1));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_her2_batched<T>)(handle,
                                                      uplo,
                                                      N,
                                                      &h_alpha,
                                                      dx.ptr_on_device(),
                                                      incx,
                                                      dy.ptr_on_device(),
                                                      incy,
                                                      dA_1.ptr_on_device(),
                                                      lda,
                                                      batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_her2_batched<T>)(handle,
                                                      uplo,
                                                      N,
                                                      d_alpha,
                                                      dx.ptr_on_device(),
                                                      incx,
                                                      dy.ptr_on_device(),
                                                      incy,
                                                      dA_2.ptr_on_device(),
                                                      lda,
                                                      batch_count));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_her2<T>(uplo, N, h_alpha, hx[b], incx, hy[b], incy, hA_gold[b], lda);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_1));
        CHECK_HIP_ERROR(hA_2.transfer_from(dA_2));

        if(arg.unit_check)
        {
            const double tol = N * sum_error_tolerance<T>;
            near_check_general<T>(N, N, lda, hA_gold, hA_1, batch_count, tol);
            near_check_general<T>(N, N, lda, hA_gold, hA_2, batch_count, tol);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', N, N, lda, hA_gold, hA_1, batch_count);
            rocblas_error_2 = norm_check_general<T>('F', N, N, lda, hA_gold, hA_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_her2_batched<T>(handle,
                                    uplo,
                                    N,
                                    &h_alpha,
                                    dx.ptr_on_device(),
                                    incx,
                                    dy.ptr_on_device(),
                                    incy,
                                    dA_1.ptr_on_device(),
                                    lda,
                                    batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_her2_batched<T>(handle,
                                    uplo,
                                    N,
                                    &h_alpha,
                                    dx.ptr_on_device(),
                                    incx,
                                    dy.ptr_on_device(),
                                    incy,
                                    dA_1.ptr_on_device(),
                                    lda,
                                    batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_N, e_alpha, e_lda, e_incx, e_incy, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            her2_gflop_count<T>(N),
            her2_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}
