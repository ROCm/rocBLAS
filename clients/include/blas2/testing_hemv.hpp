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
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_hemv_bad_arg(const Arguments& arg)
{
    auto rocblas_hemv_fn = arg.fortran ? rocblas_hemv<T, true> : rocblas_hemv<T, false>;

    const rocblas_int N     = 100;
    const rocblas_int lda   = 100;
    const rocblas_int incx  = 1;
    const rocblas_int incy  = 1;
    const T           alpha = 0.5;
    const T           beta  = 1.5;
    const T           zero  = 0.0;
    const T           one   = 1.0;

    const rocblas_fill   uplo = rocblas_fill_upper;
    rocblas_local_handle handle{arg};

    size_t size_A = lda * size_t(N);
    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_hemv_fn(handle, uplo, N, &alpha, nullptr, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hemv_fn(handle, uplo, N, &alpha, dA, lda, nullptr, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hemv_fn(handle, uplo, N, &alpha, dA, lda, dx, incx, &beta, nullptr, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hemv_fn(handle, uplo, N, nullptr, dA, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hemv_fn(handle, uplo, N, &alpha, dA, lda, dx, incx, nullptr, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_hemv_fn(nullptr, uplo, N, &alpha, dA, lda, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_handle);

    // When N==0, all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_hemv_fn(
            handle, uplo, 0, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy),
        rocblas_status_success);

    // When alpha==0, A and x may be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_hemv_fn(handle, uplo, N, &zero, nullptr, lda, nullptr, incx, &beta, dy, incy),
        rocblas_status_success);

    // When alpha==0 && beta==1, A, x and y may be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_hemv_fn(handle, uplo, N, &zero, nullptr, lda, nullptr, incx, &one, nullptr, incy),
        rocblas_status_success);
}

template <typename T>
void testing_hemv(const Arguments& arg)
{
    auto rocblas_hemv_fn = arg.fortran ? rocblas_hemv<T, true> : rocblas_hemv<T, false>;

    rocblas_int  N       = arg.N;
    rocblas_int  lda     = arg.lda;
    rocblas_int  incx    = arg.incx;
    rocblas_int  incy    = arg.incy;
    T            h_alpha = arg.get_alpha<T>();
    T            h_beta  = arg.get_beta<T>();
    rocblas_fill uplo    = char2rocblas_fill(arg.uplo);

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N < 0 || lda < N || lda < 1 || !incx || !incy)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_hemv_fn(
                handle, uplo, N, &h_alpha, nullptr, lda, nullptr, incx, &h_beta, nullptr, incy),
            rocblas_status_invalid_size);

        return;
    }

    size_t size_A   = lda * size_t(N);
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = N * abs_incx;
    size_t size_y   = N * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hx(size_x);
    host_vector<T> hy_1(size_y);
    host_vector<T> hy_2(size_y);
    host_vector<T> hy_gold(size_y);
    host_vector<T> halpha(1);
    host_vector<T> hbeta(1);
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy_1(size_y);
    device_vector<T> dy_2(size_y);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(hA, arg, N, N, lda, 0, 1, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hx, arg, N, abs_incx, 0, 1, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy_1, arg, N, abs_incy, 0, 1, rocblas_client_beta_sets_nan);

    // copy vector is easy in STL; hy_gold = hy_1: save a copy in hy_gold which will be output of
    // CPU BLAS
    hy_gold = hy_1;
    hy_2    = hy_1;

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
        CHECK_ROCBLAS_ERROR(
            rocblas_hemv_fn(handle, uplo, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            rocblas_hemv_fn(handle, uplo, N, d_alpha, dA, lda, dx, incx, d_beta, dy_2, incy));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        cblas_hemv<T>(uplo, N, h_alpha, hA, lda, hx, incx, h_beta, hy_gold, incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy_1));
        CHECK_HIP_ERROR(hy_2.transfer_from(dy_2));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_1);
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_1);
            rocblas_error_2 = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_hemv_fn(handle, uplo, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_hemv_fn(handle, uplo, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_N, e_alpha, e_lda, e_incx, e_beta, e_incy>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            hemv_gflop_count<T>(N),
            hemv_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}
