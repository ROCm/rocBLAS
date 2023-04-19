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
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_iamax_iamin_ref.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_iamax_iamin_bad_arg(const Arguments& arg, rocblas_iamax_iamin_t<T> func)
{
    rocblas_int N    = 100;
    rocblas_int incx = 1;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_vector<T> dx(N, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    rocblas_int h_rocblas_result;

    EXPECT_ROCBLAS_STATUS(func(nullptr, N, dx, incx, &h_rocblas_result),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(func(handle, N, nullptr, incx, &h_rocblas_result),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(func(handle, N, dx, incx, nullptr), rocblas_status_invalid_pointer);
}

template <typename T>
void testing_iamax_bad_arg(const Arguments& arg)
{
    auto rocblas_iamax_fn = arg.api == FORTRAN ? rocblas_iamax<T, true> : rocblas_iamax<T, false>;
    testing_iamax_iamin_bad_arg<T>(arg, rocblas_iamax_fn);
}

template <typename T>
void testing_iamin_bad_arg(const Arguments& arg)
{
    auto rocblas_iamin_fn = arg.api == FORTRAN ? rocblas_iamin<T, true> : rocblas_iamin<T, false>;
    testing_iamax_iamin_bad_arg<T>(arg, rocblas_iamin_fn);
}

template <typename T, void REFBLAS_FUNC(int64_t, const T*, int64_t, int64_t*)>
void testing_iamax_iamin(const Arguments& arg, rocblas_iamax_iamin_t<T> func)
{
    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;

    rocblas_int h_rocblas_result_1;
    rocblas_int h_rocblas_result_2;

    rocblas_int rocblas_error_1;
    rocblas_int rocblas_error_2;

    rocblas_local_handle handle{arg};

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        using R = rocblas_int;
        device_vector<R> d_rocblas_result(1);
        CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

        host_vector<R> h_rocblas_result(1);
        CHECK_HIP_ERROR(h_rocblas_result.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(func(handle, N, nullptr, incx, d_rocblas_result));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(func(handle, N, nullptr, incx, h_rocblas_result));

        R cpu_0 = R(0);
        R gpu_0, gpu_1;
        CHECK_HIP_ERROR(hipMemcpy(&gpu_0, d_rocblas_result, sizeof(T), hipMemcpyDeviceToHost));
        gpu_1 = h_rocblas_result[0];
        unit_check_general<R>(1, 1, 1, &cpu_0, &gpu_0);
        unit_check_general<R>(1, 1, 1, &cpu_0, &gpu_1);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_vector<T> hx(N, incx);

    // Allocate device memory
    device_vector<T>           dx(N, incx);
    device_vector<rocblas_int> d_rocblas_result(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

    // Initial Data on CPU
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(func(handle, N, dx, incx, &h_rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(func(handle, N, dx, incx, d_rocblas_result));
        handle.post_test(arg);
        CHECK_HIP_ERROR(hipMemcpy(
            &h_rocblas_result_2, d_rocblas_result, sizeof(rocblas_int), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        int64_t result_i64;
        REFBLAS_FUNC(N, hx, incx, &result_i64);
        rocblas_int cpu_result = rocblas_int(result_i64);
        cpu_time_used          = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &h_rocblas_result_1);
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &h_rocblas_result_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = h_rocblas_result_1 - cpu_result;
            rocblas_error_2 = h_rocblas_result_2 - cpu_result;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            func(handle, N, dx, incx, d_rocblas_result);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            func(handle, N, dx, incx, d_rocblas_result);
        }

        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used);

        ArgumentModel<e_N, e_incx>{}.log_args<T>(rocblas_cout,
                                                 arg,
                                                 gpu_time_used,
                                                 ArgumentLogging::NA_value,
                                                 iamax_iamin_gbyte_count<T>(N),
                                                 cpu_time_used,
                                                 rocblas_error_1,
                                                 rocblas_error_2);
    }
}

template <typename T>
void testing_iamax(const Arguments& arg)
{
    auto rocblas_iamax_fn = arg.api == FORTRAN ? rocblas_iamax<T, true> : rocblas_iamax<T, false>;
    testing_iamax_iamin<T, rocblas_iamax_iamin_ref::iamax<T>>(arg, rocblas_iamax_fn);
}

template <typename T>
void testing_iamin(const Arguments& arg)
{
    auto rocblas_iamin_fn = arg.api == FORTRAN ? rocblas_iamin<T, true> : rocblas_iamin<T, false>;
    testing_iamax_iamin<T, rocblas_iamax_iamin_ref::iamin<T>>(arg, rocblas_iamin_fn);
}
