/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
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
    auto rocblas_syr_fn = arg.api == FORTRAN ? rocblas_syr<T, true> : rocblas_syr<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_fill uplo = rocblas_fill_upper;
        rocblas_int  N    = 100;
        rocblas_int  incx = 1;
        rocblas_int  lda  = 100;

        device_vector<T> alpha_d(1), zero_d(1);

        const T alpha_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        // Allocate device memory
        device_matrix<T> dA(N, N, lda);
        device_vector<T> dx(N, incx);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dx.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(nullptr, uplo, N, alpha, dx, incx, dA, lda),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(
            rocblas_syr_fn(handle, rocblas_fill_full, N, alpha, dx, incx, dA, lda),
            rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(handle, uplo, N, nullptr, dx, incx, dA, lda),
                              rocblas_status_invalid_pointer);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(handle, uplo, N, alpha, nullptr, incx, dA, lda),
                                  rocblas_status_invalid_pointer);

            EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(handle, uplo, N, alpha, dx, incx, nullptr, lda),
                                  rocblas_status_invalid_pointer);
        }

        // N==0 all pointers may be null
        EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(handle, uplo, 0, nullptr, nullptr, incx, nullptr, lda),
                              rocblas_status_success);

        // alpha==0 all pointers may be null
        EXPECT_ROCBLAS_STATUS(rocblas_syr_fn(handle, uplo, N, zero, nullptr, incx, nullptr, lda),
                              rocblas_status_success);
    }
}

template <typename T>
void testing_syr(const Arguments& arg)
{
    auto rocblas_syr_fn = arg.api == FORTRAN ? rocblas_syr<T, true> : rocblas_syr<T, false>;

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

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(N, N, lda);
    host_matrix<T> hA_gold(N, N, lda);
    host_vector<T> hx(N, incx);

    // Allocate device memory
    device_matrix<T> dA(N, N, lda);
    device_vector<T> dx(N, incx);
    device_vector<T> d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_symmetric_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);

    // copy matrix is easy in STL; hA_gold = hA: save a copy in hA_gold which will be output of
    // CPU BLAS
    hA_gold = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1;
    double rocblas_error_2;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR(rocblas_syr_fn(handle, uplo, N, &h_alpha, dx, incx, dA, lda));
            handle.post_test(arg);

            // copy output from device to CPU
            CHECK_HIP_ERROR(hA.transfer_from(dA));
        }

        if(arg.pointer_mode_device)
        {
            // copy data from CPU to device
            CHECK_HIP_ERROR(dA.transfer_from(hA_gold));
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR(rocblas_syr_fn(handle, uplo, N, d_alpha, dx, incx, dA, lda));
            handle.post_test(arg);
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        cblas_syr<T>(uplo, N, h_alpha, hx, incx, hA_gold, lda);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(std::is_same<T, float>{} || std::is_same<T, double>{})
                {
                    unit_check_general<T>(N, N, lda, hA_gold, hA);
                }
                else
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, lda, hA_gold, hA, tol);
                }
            }

            if(arg.norm_check)
            {
                rocblas_error_1 = norm_check_general<T>('F', N, N, lda, hA_gold, hA);
            }
        }

        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hA.transfer_from(dA));

            if(arg.unit_check)
            {
                if(std::is_same<T, float>{} || std::is_same<T, double>{})
                {
                    unit_check_general<T>(N, N, lda, hA_gold, hA);
                }
                else
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, lda, hA_gold, hA, tol);
                }
            }

            if(arg.norm_check)
            {
                rocblas_error_2 = norm_check_general<T>('F', N, N, lda, hA_gold, hA);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_syr_fn(handle, uplo, N, &h_alpha, dx, incx, dA, lda);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_syr_fn(handle, uplo, N, &h_alpha, dx, incx, dA, lda);
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
