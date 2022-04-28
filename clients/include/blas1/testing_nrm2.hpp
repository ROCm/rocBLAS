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
void testing_nrm2_bad_arg(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_nrm2_fn = FORTRAN ? rocblas_nrm2<T, true> : rocblas_nrm2<T, false>;

    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    static const size_t safe_size = 100;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_vector<T>         dx(N, incx);
    device_vector<real_t<T>> d_rocblas_result(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    EXPECT_ROCBLAS_STATUS(rocblas_nrm2_fn(handle, N, nullptr, incx, d_rocblas_result),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_nrm2_fn(handle, N, dx, incx, nullptr),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_nrm2_fn(nullptr, N, dx, incx, d_rocblas_result),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_nrm2(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_nrm2_fn = FORTRAN ? rocblas_nrm2<T, true> : rocblas_nrm2<T, false>;

    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_local_handle handle{arg};

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        device_vector<real_t<T>> d_rocblas_result_0(1);
        host_vector<real_t<T>>   h_rocblas_result_0(1);
        CHECK_HIP_ERROR(d_rocblas_result_0.memcheck());
        CHECK_HIP_ERROR(h_rocblas_result_0.memcheck());

        rocblas_init_nan(h_rocblas_result_0, 1, 1, 1);
        CHECK_HIP_ERROR(hipMemcpy(
            d_rocblas_result_0, h_rocblas_result_0, sizeof(real_t<T>), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_fn(handle, N, nullptr, incx, d_rocblas_result_0));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_fn(handle, N, nullptr, incx, h_rocblas_result_0));

        host_vector<real_t<T>> cpu_0(1);
        host_vector<real_t<T>> gpu_0(1);
        CHECK_HIP_ERROR(cpu_0.memcheck());
        CHECK_HIP_ERROR(gpu_0.memcheck());

        CHECK_HIP_ERROR(
            hipMemcpy(gpu_0, d_rocblas_result_0, sizeof(real_t<T>), hipMemcpyDeviceToHost));
        unit_check_general<real_t<T>>(1, 1, 1, cpu_0, gpu_0);
        unit_check_general<real_t<T>>(1, 1, 1, cpu_0, h_rocblas_result_0);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_vector<T>         hx(N, incx);
    host_vector<real_t<T>> rocblas_result_1(1, 1);
    host_vector<real_t<T>> rocblas_result_2(1, 1);
    host_vector<real_t<T>> cpu_result(1, 1);

    // Allocate device memory
    device_vector<T>         dx(N, incx);
    device_vector<real_t<T>> d_rocblas_result_2(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result_2.memcheck());

    // Initial Data on CPU
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_fn(handle, N, dx, incx, rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_nrm2_fn(handle, N, dx, incx, d_rocblas_result_2));

        CHECK_HIP_ERROR(rocblas_result_2.transfer_from(d_rocblas_result_2));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        cblas_nrm2<T>(N, hx, incx, cpu_result);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        real_t<T> abs_result = cpu_result[0] > 0 ? cpu_result[0] : -cpu_result[0];
        real_t<T> abs_error;
        if(abs_result > 0)
        {
            abs_error = std::numeric_limits<real_t<T>>::epsilon() * N * abs_result;
        }
        else
        {
            abs_error = std::numeric_limits<real_t<T>>::epsilon() * N;
        }
        real_t<T> tolerance = 2.0; //  accounts for rounding in reduction sum. depends on n.
            //  If test fails, try decreasing n or increasing tolerance.
        abs_error *= tolerance;

        if(!rocblas_isnan(arg.alpha))
        {
            if(arg.unit_check)
            {
                near_check_general<real_t<T>, real_t<T>>(
                    1, 1, 1, cpu_result, rocblas_result_1, abs_error);
                near_check_general<real_t<T>, real_t<T>>(
                    1, 1, 1, cpu_result, rocblas_result_2, abs_error);
            }
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = rocblas_abs((cpu_result[0] - rocblas_result_1[0]) / cpu_result[0]);
            rocblas_error_2 = rocblas_abs((cpu_result[0] - rocblas_result_2[0]) / cpu_result[0]);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_nrm2_fn(handle, N, dx, incx, d_rocblas_result_2);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_nrm2_fn(handle, N, dx, incx, d_rocblas_result_2);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx>{}.log_args<T>(rocblas_cout,
                                                 arg,
                                                 gpu_time_used,
                                                 nrm2_gflop_count<T>(N),
                                                 nrm2_gbyte_count<T>(N),
                                                 cpu_time_used,
                                                 rocblas_error_1,
                                                 rocblas_error_2);
    }
}
