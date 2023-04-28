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
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T, typename R>
using rocblas_reduction_batched_t = rocblas_status (*)(rocblas_handle  handle,
                                                       rocblas_int     n,
                                                       const T* const* x,
                                                       rocblas_int     incx,
                                                       rocblas_int     batch_count,
                                                       R*              result);

template <typename T, typename R>
void template_testing_reduction_batched_bad_arg(const Arguments&                  arg,
                                                rocblas_reduction_batched_t<T, R> func)
{

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_int N = 100, incx = 1, batch_count = 2;

        // Allocate device memory
        device_batch_vector<T> dx(N, incx, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dx.memcheck());

        R h_rocblas_result; // host address as all early returns

        EXPECT_ROCBLAS_STATUS(func(nullptr, N, dx, incx, batch_count, &h_rocblas_result),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(func(handle, N, nullptr, incx, batch_count, &h_rocblas_result),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(func(handle, N, dx, incx, batch_count, nullptr),
                              rocblas_status_invalid_pointer);
    }
}

template <typename T, typename R>
void template_testing_reduction_batched(const Arguments&                  arg,
                                        rocblas_reduction_batched_t<T, R> func,
                                        void (*REFBLAS_FUNC)(int64_t, const T*, int64_t, int64_t*))
{
    rocblas_int          N = arg.N, incx = arg.incx, batch_count = arg.batch_count;
    rocblas_stride       stride_x = arg.stride_x;
    double               rocblas_error_1, rocblas_error_2;
    rocblas_local_handle handle{arg};

    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        device_vector<R> d_rocblas_result(std::max(batch_count, 1));
        CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

        host_vector<R> h_rocblas_result(std::max(batch_count, 1));
        CHECK_HIP_ERROR(h_rocblas_result.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(func(handle, N, nullptr, incx, batch_count, d_rocblas_result));
        handle.post_test(arg);

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(func(handle, N, nullptr, incx, batch_count, h_rocblas_result));

        if(batch_count > 0)
        {
            host_vector<R> cpu_0(batch_count);
            host_vector<R> gpu_0(batch_count);
            CHECK_HIP_ERROR(gpu_0.transfer_from(d_rocblas_result));
            unit_check_general<R>(1, 1, 1, 1, cpu_0, gpu_0, batch_count);
            unit_check_general<R>(1, 1, 1, 1, cpu_0, h_rocblas_result, batch_count);
        }

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_batch_vector<T> hx(N, incx, batch_count);
    host_vector<R>       hr1(batch_count);
    host_vector<R>       hr2(batch_count);
    host_vector<R>       cpu_result(batch_count);
    host_vector<int64_t> i64_result(batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hx.memcheck());

    // Allocate device memory
    device_batch_vector<T> dx(N, incx, batch_count);
    device_vector<R>       dr(batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dr.memcheck());

    // Initialize memory on host.
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // Copy data from host to device.
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR(func(handle, N, dx.ptr_on_device(), incx, batch_count, hr1));
        }

        // GPU BLAS, rocblas_pointer_mode_device
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR(func(handle, N, dx.ptr_on_device(), incx, batch_count, dr));
            CHECK_HIP_ERROR(hr2.transfer_from(dr));
        }

        // CPU BLAS
        {
            cpu_time_used = get_time_us_no_sync();

            for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                REFBLAS_FUNC(N, hx[batch_index], incx, i64_result + batch_index);
                cpu_result[batch_index] = rocblas_int(i64_result[batch_index]);
            }

            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        if(arg.unit_check)
        {
            unit_check_general<R>(batch_count, 1, 1, cpu_result, hr1);
            unit_check_general<R>(batch_count, 1, 1, cpu_result, hr2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = 0.0;
            rocblas_error_2 = 0.0;
            for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                double a1       = double(hr1[batch_index]);
                double a2       = double(hr2[batch_index]);
                double c        = double(cpu_result[batch_index]);
                rocblas_error_1 = std::max(rocblas_error_1, std::abs((c - a1) / c));
                rocblas_error_2 = std::max(rocblas_error_2, std::abs((c - a2) / c));
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
            func(handle, N, dx.ptr_on_device(), incx, batch_count, hr2);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            func(handle, N, dx.ptr_on_device(), incx, batch_count, hr2);
        }

        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used);

        ArgumentModel<e_N, e_incx, e_batch_count>{}.log_args<T>(rocblas_cout,
                                                                arg,
                                                                gpu_time_used,
                                                                ArgumentLogging::NA_value,
                                                                iamax_iamin_gbyte_count<T>(N),
                                                                cpu_time_used,
                                                                rocblas_error_1,
                                                                rocblas_error_2);
    }
}
