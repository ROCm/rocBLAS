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
void testing_syr_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_syr_strided_batched_fn = arg.fortran ? rocblas_syr_strided_batched<T, true>
                                                      : rocblas_syr_strided_batched<T, false>;

    rocblas_fill   uplo        = rocblas_fill_upper;
    rocblas_int    N           = 100;
    rocblas_int    incx        = 1;
    rocblas_int    lda         = 100;
    T              alpha       = 0.6;
    rocblas_int    batch_count = 5;
    rocblas_stride stridex     = 1;
    rocblas_stride strideA     = 1;

    rocblas_local_handle handle{arg};

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t size_A   = size_t(lda) * N;
    size_t size_x   = N * abs_incx;

    // allocate memory on device
    device_vector<T> dA_1(size_A);
    device_vector<T> dx(size_x);
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_syr_strided_batched_fn(
            handle, uplo, N, &alpha, nullptr, incx, stridex, dA_1, lda, strideA, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_syr_strided_batched_fn(
            handle, uplo, N, &alpha, dx, incx, stridex, nullptr, lda, strideA, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_syr_strided_batched_fn(
            nullptr, uplo, N, &alpha, dx, incx, stridex, dA_1, lda, strideA, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_syr_strided_batched(const Arguments& arg)
{
    auto rocblas_syr_strided_batched_fn = arg.fortran ? rocblas_syr_strided_batched<T, true>
                                                      : rocblas_syr_strided_batched<T, false>;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_int    lda         = arg.lda;
    T              h_alpha     = arg.get_alpha<T>();
    rocblas_fill   uplo        = char2rocblas_fill(arg.uplo);
    rocblas_stride stridex     = arg.stride_x;
    rocblas_stride strideA     = arg.stride_a;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument check before allocating invalid memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_syr_strided_batched_fn(handle,
                                                             uplo,
                                                             N,
                                                             nullptr,
                                                             nullptr,
                                                             incx,
                                                             stridex,
                                                             nullptr,
                                                             lda,
                                                             strideA,
                                                             batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t size_A   = size_t(lda) * N * batch_count;
    size_t size_x   = size_t(N) * abs_incx * batch_count;

    strideA = std::max(strideA, rocblas_stride(size_t(lda) * N));
    stridex = std::max(stridex, rocblas_stride(size_t(N) * abs_incx));

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA_1(size_A);
    host_vector<T> hA_2(size_A);
    host_vector<T> hA_gold(size_A);
    host_vector<T> hx(size_x);

    // allocate memory on device
    device_vector<T> dA_1(size_A);
    device_vector<T> dA_2(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> d_alpha(1);
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    // Initialize data on host memory
    rocblas_init_matrix(hA_1,
                        arg,
                        N,
                        N,
                        lda,
                        1,
                        batch_count,
                        rocblas_client_never_set_nan,
                        rocblas_client_symmetric_matrix,
                        true);
    rocblas_init_vector(
        hx, arg, N, abs_incx, 1, batch_count, rocblas_client_alpha_sets_nan, false, true);

    // copy matrix is easy in STL; hA_gold = hA_1: save a copy in hA_gold which will be output of
    // CPU BLAS
    hA_gold = hA_1;
    hA_2    = hA_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA_1, hA_1, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dA_2, hA_2, sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_syr_strided_batched_fn(
            handle, uplo, N, &h_alpha, dx, incx, stridex, dA_1, lda, strideA, batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_syr_strided_batched_fn(
            handle, uplo, N, d_alpha, dx, incx, stridex, dA_2, lda, strideA, batch_count));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_syr<T>(uplo, N, h_alpha, hx + i * stridex, incx, hA_gold + i * strideA, lda);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        hipMemcpy(hA_1, dA_1, sizeof(T) * size_A, hipMemcpyDeviceToHost);
        hipMemcpy(hA_2, dA_2, sizeof(T) * size_A, hipMemcpyDeviceToHost);

        if(arg.unit_check)
        {
            if(std::is_same<T, float>{} || std::is_same<T, double>{})
            {
                unit_check_general<T>(N, N, lda, strideA, hA_gold, hA_1, batch_count);
                unit_check_general<T>(N, N, lda, strideA, hA_gold, hA_2, batch_count);
            }
            else
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(N, N, lda, strideA, hA_gold, hA_1, batch_count, tol);
                near_check_general<T>(N, N, lda, strideA, hA_gold, hA_2, batch_count, tol);
            }
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', N, N, lda, strideA, hA_gold, hA_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('F', N, N, lda, strideA, hA_gold, hA_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_syr_strided_batched_fn(
                handle, uplo, N, &h_alpha, dx, incx, stridex, dA_1, lda, strideA, batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_syr_strided_batched_fn(
                handle, uplo, N, &h_alpha, dx, incx, stridex, dA_1, lda, strideA, batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        Arguments targ(arg);
        targ.stride_a = strideA;
        targ.stride_x = stridex;
        ArgumentModel<e_uplo, e_N, e_alpha, e_lda, e_stride_a, e_incx, e_stride_x, e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         targ,
                         gpu_time_used,
                         syr_gflop_count<T>(N),
                         syr_gbyte_count<T>(N),
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
