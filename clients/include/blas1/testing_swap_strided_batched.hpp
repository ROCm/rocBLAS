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

#include "cblas_interface.hpp"
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
void testing_swap_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_swap_strided_batched_fn = arg.fortran ? rocblas_swap_strided_batched<T, true>
                                                       : rocblas_swap_strided_batched<T, false>;

    rocblas_int    N           = 100;
    rocblas_int    incx        = 1;
    rocblas_int    incy        = 1;
    rocblas_stride stride_x    = 1;
    rocblas_stride stride_y    = 1;
    rocblas_int    batch_count = 2;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched_fn(
                              nullptr, N, dx, incx, stride_x, dy, incy, stride_y, batch_count),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched_fn(
                              handle, N, nullptr, incx, stride_x, dy, incy, stride_y, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched_fn(
                              handle, N, dx, incx, stride_x, nullptr, incy, stride_y, batch_count),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_swap_strided_batched(const Arguments& arg)
{
    auto rocblas_swap_strided_batched_fn = arg.fortran ? rocblas_swap_strided_batched<T, true>
                                                       : rocblas_swap_strided_batched<T, false>;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_int    incy        = arg.incy;
    rocblas_stride stride_x    = arg.stride_x;
    rocblas_stride stride_y    = arg.stride_y;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_swap_strided_batched_fn(
                handle, N, nullptr, incx, stride_x, nullptr, incy, stride_y, batch_count),
            rocblas_status_success);
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_strided_batch_vector<T> hx(N, incx ? incx : 1, stride_x, batch_count);
    host_strided_batch_vector<T> hy(N, incy ? incy : 1, stride_y, batch_count);
    host_strided_batch_vector<T> hx_gold(N, incx ? incx : 1, stride_x, batch_count);
    host_strided_batch_vector<T> hy_gold(N, incy ? incy : 1, stride_y, batch_count);

    // Allocate device memory
    device_strided_batch_vector<T> dx(N, incx ? incx : 1, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy ? incy : 1, stride_y, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // Initialize the host vector.
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);

    hx_gold.copy_from(hx);
    hy_gold.copy_from(hy);

    // Transfer data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_swap_strided_batched_fn(
            handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));

        // Transfer data from device to CPU
        CHECK_HIP_ERROR(hx.transfer_from(dx));
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_swap<T>(N, hx_gold[b], incx, hy_gold[b], incy);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incx, stride_x, hx_gold, hx, batch_count);
            unit_check_general<T>(1, N, abs_incy, stride_y, hy_gold, hy, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', 1, N, abs_incx, stride_x, hx_gold, hx, batch_count);
            rocblas_error
                = norm_check_general<T>('F', 1, N, abs_incy, stride_y, hy_gold, hy, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_swap_strided_batched_fn(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap_strided_batched_fn(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            ArgumentLogging::NA_value,
            swap_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error);
    }
}
