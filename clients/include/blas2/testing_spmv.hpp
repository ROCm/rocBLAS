/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
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
void testing_spmv_bad_arg(const Arguments& arg)
{
    auto rocblas_spmv_fn = arg.fortran ? rocblas_spmv<T, true> : rocblas_spmv<T, false>;

    rocblas_fill         uplo  = rocblas_fill_upper;
    rocblas_int          N     = 100;
    rocblas_int          incx  = 1;
    rocblas_int          incy  = 1;
    T                    alpha = 0.6;
    T                    beta  = 0.6;
    rocblas_local_handle handle{arg};

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = N * abs_incx;
    size_t size_y   = N * abs_incy;

    // Allocate device memory
    device_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_spmv_fn(nullptr, uplo, N, &alpha, dAp, dx, incx, &beta, dy, incy),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(
        rocblas_spmv_fn(handle, rocblas_fill_full, N, &alpha, dAp, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_spmv_fn(handle, uplo, N, nullptr, dAp, dx, incx, &beta, dy, incy),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_spmv_fn(handle, uplo, N, &alpha, nullptr, dx, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_spmv_fn(handle, uplo, N, &alpha, dAp, nullptr, incx, &beta, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_spmv_fn(handle, uplo, N, &alpha, dAp, dx, incx, nullptr, dy, incy),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_spmv_fn(handle, uplo, N, &alpha, dAp, dx, incx, &beta, nullptr, incy),
        rocblas_status_invalid_pointer);
}

template <typename T>
void testing_spmv(const Arguments& arg)
{
    auto rocblas_spmv_fn = arg.fortran ? rocblas_spmv<T, true> : rocblas_spmv<T, false>;

    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;

    host_vector<T> alpha(1);
    host_vector<T> beta(1);
    alpha[0] = arg.get_alpha<T>();
    beta[0]  = arg.get_beta<T>();

    rocblas_fill uplo = char2rocblas_fill(arg.uplo);

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    size_t size_X = size_t(N) * abs_incx;
    size_t size_Y = size_t(N) * abs_incy;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || !incx || !incy;
    if(invalid_size || !N)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_spmv_fn(
                handle, uplo, N, nullptr, nullptr, nullptr, incx, nullptr, nullptr, incy),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_matrix<T> hA(N, N, N);
    host_matrix<T> hAp(1, rocblas_packed_matrix_size(N), 1);
    host_vector<T> hx(size_X);
    host_vector<T> hy_1(size_Y);
    host_vector<T> hy_2(size_Y);
    host_vector<T> hy_gold(size_Y); // gold standard

    // Allocate device memory
    device_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1);
    device_vector<T> dx(size_X);
    device_vector<T> dy(size_Y);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_symmetric_matrix, true);
    rocblas_init_vector(hx, arg, N, abs_incx, 0, 1, rocblas_client_alpha_sets_nan, false, false);
    rocblas_init_vector(hy_1, arg, N, abs_incy, 0, 1, rocblas_client_beta_sets_nan);

    // helper function to convert regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, N);

    // make copy in hy_gold which will later be used with CPU BLAS
    hy_gold = hy_1;
    hy_2    = hy_1; // device memory re-test

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy_1));
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));

    double gpu_time_used, cpu_time_used;
    double h_error, d_error;

    if(arg.unit_check || arg.norm_check)
    {
        //
        // rocblas_pointer_mode_host test
        //
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_spmv_fn(handle, uplo, N, alpha, dAp, dx, incx, beta, dy, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy));

        //
        // rocblas_pointer_mode_device test
        //
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(d_alpha.transfer_from(alpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(beta));

        dy.transfer_from(hy_2);

        CHECK_ROCBLAS_ERROR(
            rocblas_spmv_fn(handle, uplo, N, d_alpha, dAp, dx, incx, d_beta, dy, incy));

        // cpu ref
        cpu_time_used = get_time_us_no_sync();

        // cpu reference
        cblas_spmv<T>(uplo, N, alpha[0], hAp, hx, incx, beta[0], hy_gold, incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_2.transfer_from(dy));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_1);
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_2);
        }

        if(arg.norm_check)
        {
            h_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_1);
            d_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(
                rocblas_spmv_fn(handle, uplo, N, alpha, dAp, dx, incx, beta, dy, incy));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            CHECK_ROCBLAS_ERROR(
                rocblas_spmv_fn(handle, uplo, N, alpha, dAp, dx, incx, beta, dy, incy));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_N, e_alpha, e_lda, e_incx, e_beta, e_incy>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            spmv_gflop_count<T>(N),
            spmv_gbyte_count<T>(N),
            cpu_time_used,
            h_error,
            d_error);
    }
}
