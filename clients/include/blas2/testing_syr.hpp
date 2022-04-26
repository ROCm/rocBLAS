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
void testing_syr_bad_arg(const Arguments& arg)
{
    auto rocblas_syr_fn = arg.fortran ? rocblas_syr<T, true> : rocblas_syr<T, false>;

    rocblas_fill         uplo  = rocblas_fill_upper;
    rocblas_int          N     = 100;
    rocblas_int          incx  = 1;
    rocblas_int          lda   = 100;
    T                    alpha = 0.6;
    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_matrix<T> dA_1(N, N, lda);
    device_vector<T> dx(N, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(handle, rocblas_fill_full, N, &alpha, dx, incx, dA_1, lda),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(handle, uplo, N, &alpha, nullptr, incx, dA_1, lda),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(handle, uplo, N, &alpha, dx, incx, nullptr, lda),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(nullptr, uplo, N, &alpha, dx, incx, dA_1, lda),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_syr(const Arguments& arg)
{
    auto rocblas_syr_fn = arg.fortran ? rocblas_syr<T, true> : rocblas_syr<T, false>;

    rocblas_int          N       = arg.N;
    rocblas_int          incx    = arg.incx;
    rocblas_int          lda     = arg.lda;
    T                    h_alpha = arg.get_alpha<T>();
    rocblas_fill         uplo    = char2rocblas_fill(arg.uplo);
    rocblas_local_handle handle{arg};

    // argument check before allocating invalid memory
    if(N < 0 || lda < N || lda < 1 || !incx)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(handle, uplo, N, nullptr, nullptr, incx, nullptr, lda),
                              rocblas_status_invalid_size);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA_1), `d` is in GPU (device) memory (eg dA_1).
    // Allocate host memory
    host_matrix<T> hA_1(N, N, lda);
    host_matrix<T> hA_2(N, N, lda);
    host_matrix<T> hA_gold(N, N, lda);
    host_vector<T> hx(N, incx);

    // Allocate device memory
    device_matrix<T> dA_1(N, N, lda);
    device_matrix<T> dA_2(N, N, lda);
    device_vector<T> dx(N, incx);
    device_vector<T> d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA_1, arg, rocblas_client_never_set_nan, rocblas_client_symmetric_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);

    // copy matrix is easy in STL; hA_gold = hA_1: save a copy in hA_gold which will be output of
    // CPU BLAS
    hA_gold = hA_1;
    hA_2    = hA_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA_1.transfer_from(hA_1));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1;
    double rocblas_error_2;

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA_2.transfer_from(hA_2));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_syr_fn(handle, uplo, N, &h_alpha, dx, incx, dA_1, lda));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_syr_fn(handle, uplo, N, d_alpha, dx, incx, dA_2, lda));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        cblas_syr<T>(uplo, N, h_alpha, hx, incx, hA_gold, lda);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        hipMemcpy(hA_1, dA_1, sizeof(T) * N * lda, hipMemcpyDeviceToHost);
        hipMemcpy(hA_2, dA_2, sizeof(T) * N * lda, hipMemcpyDeviceToHost);

        if(arg.unit_check)
        {
            if(std::is_same<T, float>{} || std::is_same<T, double>{})
            {
                unit_check_general<T>(N, N, lda, hA_gold, hA_1);
                unit_check_general<T>(N, N, lda, hA_gold, hA_2);
            }
            else
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(N, N, lda, hA_gold, hA_1, tol);
                near_check_general<T>(N, N, lda, hA_gold, hA_2, tol);
            }
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', N, N, lda, hA_gold, hA_1);
            rocblas_error_2 = norm_check_general<T>('F', N, N, lda, hA_gold, hA_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_syr_fn(handle, uplo, N, &h_alpha, dx, incx, dA_1, lda);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_syr_fn(handle, uplo, N, &h_alpha, dx, incx, dA_1, lda);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_N, e_alpha, e_lda, e_incx>{}.log_args<T>(rocblas_cout,
                                                                         arg,
                                                                         gpu_time_used,
                                                                         syr_gflop_count<T>(N),
                                                                         syr_gbyte_count<T>(N),
                                                                         cpu_time_used,
                                                                         rocblas_error_1,
                                                                         rocblas_error_2);
    }
}
