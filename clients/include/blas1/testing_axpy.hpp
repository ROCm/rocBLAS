/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

/* ============================================================================================ */
template <typename T>
void testing_axpy_bad_arg(const Arguments& arg)
{
    auto rocblas_axpy_fn = arg.fortran ? rocblas_axpy<T, true> : rocblas_axpy<T, false>;

    rocblas_int N         = 100;
    rocblas_int incx      = 1;
    rocblas_int incy      = 1;
    size_t      safe_size = 100;
    T           alpha     = 0.6;
    T           zero      = 0.0;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_axpy_fn(handle, N, &alpha, nullptr, incx, dy, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_axpy_fn(handle, N, &alpha, dx, incx, nullptr, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_axpy_fn(handle, N, nullptr, dx, incx, dy, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_axpy_fn(nullptr, N, &alpha, dx, incx, dy, incy),
                          rocblas_status_invalid_handle);
    // If N == 0, then alpha, X and Y can be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_axpy_fn(handle, 0, nullptr, nullptr, incx, nullptr, incy),
                          rocblas_status_success);
    // If alpha == 0, then X and Y can be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_axpy_fn(handle, N, &zero, nullptr, incx, nullptr, incy),
                          rocblas_status_success);
}

template <typename T>
void testing_axpy(const Arguments& arg)
{
    auto rocblas_axpy_fn = arg.fortran ? rocblas_axpy<T, true> : rocblas_axpy<T, false>;

    rocblas_int          N       = arg.N;
    rocblas_int          incx    = arg.incx;
    rocblas_int          incy    = arg.incy;
    T                    h_alpha = arg.get_alpha<T>();
    bool                 HMM     = arg.HMM;
    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_axpy_fn(handle, N, nullptr, nullptr, incx, nullptr, incy));
        return;
    }

    rocblas_int abs_incy = incy > 0 ? incy : -incy;

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_vector<T> hx(N, incx ? incx : 1);
    host_vector<T> hy_1(N, incy ? incy : 1);
    host_vector<T> hy_2(N, incy ? incy : 1);
    host_vector<T> hy_gold(N, incy ? incy : 1);

    // Allocate device memory
    device_vector<T> dx(N, incx ? incx : 1, HMM);
    device_vector<T> dy_1(N, incy ? incy : 1, HMM);
    device_vector<T> dy_2(N, incy ? incy : 1, HMM);
    device_vector<T> d_alpha(1, 1, HMM);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy_1, arg, rocblas_client_alpha_sets_nan, false, true);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    // BLAS
    hy_2    = hy_1;
    hy_gold = hy_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy_1.transfer_from(hy_1));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(dy_2.transfer_from(hy_2));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        // ROCBLAS pointer mode host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_axpy_fn(handle, N, &h_alpha, dx, incx, dy_1, incy));

        // ROCBLAS pointer mode device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_axpy_fn(handle, N, d_alpha, dx, incx, dy_2, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy_1));
        CHECK_HIP_ERROR(hy_2.transfer_from(dy_2));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        cblas_axpy<T>(N, h_alpha, hx, incx, hy_gold, incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

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
            rocblas_axpy_fn(handle, N, &h_alpha, dx, incx, dy_1, incy);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_axpy_fn(handle, N, &h_alpha, dx, incx, dy_1, incy);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_incy>{}.log_args<T>(rocblas_cout,
                                                                  arg,
                                                                  gpu_time_used,
                                                                  axpy_gflop_count<T>(N),
                                                                  axpy_gbyte_count<T>(N),
                                                                  cpu_time_used,
                                                                  rocblas_error_1,
                                                                  rocblas_error_2);
    }
}
