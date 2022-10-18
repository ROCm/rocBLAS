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
#include "near.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_asum_bad_arg(const Arguments& arg)
{
    auto rocblas_asum_fn = arg.fortran ? rocblas_asum<T, true> : rocblas_asum<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_int N                = 100;
        rocblas_int incx             = 1;
        real_t<T>   rocblas_result   = 10;
        real_t<T>*  h_rocblas_result = &rocblas_result;

        // Allocate device memory
        device_vector<T> dx(N, incx);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dx.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_asum_fn(nullptr, N, dx, incx, h_rocblas_result),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_asum_fn(handle, N, nullptr, incx, h_rocblas_result),
                              rocblas_status_invalid_pointer);
        EXPECT_ROCBLAS_STATUS(rocblas_asum_fn(handle, N, dx, incx, nullptr),
                              rocblas_status_invalid_pointer);
    }
}

template <typename T>
void testing_asum(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_asum_fn = FORTRAN ? rocblas_asum<T, true> : rocblas_asum<T, false>;

    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;

    real_t<T>            rocblas_result_1;
    real_t<T>            rocblas_result_2;
    real_t<T>            cpu_result;
    double               rocblas_error_1;
    double               rocblas_error_2;
    rocblas_local_handle handle{arg};

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dx(safe_size);
        CHECK_DEVICE_ALLOCATION(dx.memcheck());

        device_vector<real_t<T>> dr(1);
        CHECK_DEVICE_ALLOCATION(dr.memcheck());

        host_vector<real_t<T>> hr_1(1);
        host_vector<real_t<T>> hr_2(1);
        host_vector<real_t<T>> result_0(1);
        CHECK_HIP_ERROR(hr_1.memcheck());
        CHECK_HIP_ERROR(hr_2.memcheck());
        CHECK_HIP_ERROR(result_0.memcheck());
        result_0[0] = real_t<T>(0);

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_asum_fn(handle, N, dx, incx, dr));
        CHECK_HIP_ERROR(hr_1.transfer_from(dr));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_asum_fn(handle, N, dx, incx, hr_2));

        // check that result is set to 0
        unit_check_general<real_t<T>, real_t<T>>(1, 1, 1, result_0, hr_1);
        unit_check_general<real_t<T>, real_t<T>>(1, 1, 1, result_0, hr_2);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_vector<T> hx(N, incx);

    // Allocate device memory
    device_vector<T>         dx(N, incx);
    device_vector<real_t<T>> dr(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dr.memcheck());

    // Initial Data on CPU
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_asum_fn(handle, N, dx, incx, &rocblas_result_1));

        // GPU BLAS rocblas_pointer_mode_device
        CHECK_HIP_ERROR(dx.transfer_from(hx));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_asum_fn(handle, N, dx, incx, dr));
        handle.post_test(arg);
        CHECK_HIP_ERROR(hipMemcpy(&rocblas_result_2, dr, sizeof(real_t<T>), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        cblas_asum<T>(N, hx, incx, &cpu_result);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<real_t<T>, real_t<T>>(1, 1, 1, &cpu_result, &rocblas_result_1);
            unit_check_general<real_t<T>, real_t<T>>(1, 1, 1, &cpu_result, &rocblas_result_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = std::abs((cpu_result - rocblas_result_1) / cpu_result);
            rocblas_error_2 = std::abs((cpu_result - rocblas_result_2) / cpu_result);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_asum_fn(handle, N, dx, incx, dr);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_asum_fn(handle, N, dx, incx, dr);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx>{}.log_args<T>(rocblas_cout,
                                                 arg,
                                                 gpu_time_used,
                                                 asum_gflop_count<T>(N),
                                                 asum_gflop_count<T>(N),
                                                 cpu_time_used,
                                                 rocblas_error_1,
                                                 rocblas_error_2);
    }
}
