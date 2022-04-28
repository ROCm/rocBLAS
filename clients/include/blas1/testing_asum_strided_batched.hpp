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
void testing_asum_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_asum_strided_batched_fn
        = FORTRAN ? rocblas_asum_strided_batched<T, true> : rocblas_asum_strided_batched<T, false>;

    rocblas_int    N           = 100;
    rocblas_int    incx        = 1;
    rocblas_stride stridex     = N;
    rocblas_int    batch_count = 5;
    real_t<T>      h_rocblas_result[1];

    // Allocate device memory
    device_strided_batch_vector<T> dx(N, incx, stridex, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    rocblas_local_handle handle{arg};
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    EXPECT_ROCBLAS_STATUS(rocblas_asum_strided_batched_fn(
                              handle, N, nullptr, incx, stridex, batch_count, h_rocblas_result),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_asum_strided_batched_fn(handle, N, dx, incx, stridex, batch_count, nullptr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_asum_strided_batched_fn(
                              nullptr, N, dx, incx, stridex, batch_count, h_rocblas_result),
                          rocblas_status_invalid_handle);
};

template <typename T>
void testing_asum_strided_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_asum_strided_batched_fn
        = FORTRAN ? rocblas_asum_strided_batched<T, true> : rocblas_asum_strided_batched<T, false>;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_stride stridex     = arg.stride_x;
    rocblas_int    batch_count = arg.batch_count;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_local_handle handle{arg};

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        host_vector<real_t<T>> hr_1(std::max(1, std::abs(batch_count)));
        host_vector<real_t<T>> hr_2(std::max(1, std::abs(batch_count)));
        host_vector<real_t<T>> result_0(std::max(1, std::abs(batch_count)));
        CHECK_HIP_ERROR(hr_1.memcheck());
        CHECK_HIP_ERROR(hr_2.memcheck());
        CHECK_HIP_ERROR(result_0.memcheck());

        device_vector<real_t<T>> dr(std::max(1, std::abs(batch_count)));
        CHECK_DEVICE_ALLOCATION(dr.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        EXPECT_ROCBLAS_STATUS(
            rocblas_asum_strided_batched_fn(handle, N, nullptr, incx, stridex, batch_count, dr),
            rocblas_status_success);

        CHECK_HIP_ERROR(hr_1.transfer_from(dr));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_asum_strided_batched_fn(handle, N, nullptr, incx, stridex, batch_count, hr_2),
            rocblas_status_success);

        if(batch_count > 0)
        {
            unit_check_general<real_t<T>, real_t<T>>(1, batch_count, 1, result_0, hr_1);
            unit_check_general<real_t<T>, real_t<T>>(1, batch_count, 1, result_0, hr_2);
        }

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_strided_batch_vector<T> hx(N, incx, stridex, batch_count);
    host_vector<real_t<T>>       hr_1(batch_count);
    host_vector<real_t<T>>       hr_2(batch_count);
    host_vector<real_t<T>>       hr_gold(batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hx.memcheck());

    // Allocate device memory
    device_strided_batch_vector<T> dx(N, incx, stridex, batch_count);
    device_vector<real_t<T>>       dr(batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dr.memcheck());

    // Initialize memory on host.
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(
            rocblas_asum_strided_batched_fn(handle, N, dx, incx, stridex, batch_count, hr_2));

        // GPU BgdLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            rocblas_asum_strided_batched_fn(handle, N, dx, incx, stridex, batch_count, dr));

        CHECK_HIP_ERROR(hr_1.transfer_from(dr));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_asum<T>(N, hx[b], incx, hr_gold + b);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<real_t<T>, real_t<T>>(batch_count, 1, 1, hr_gold, hr_2);
            unit_check_general<real_t<T>, real_t<T>>(batch_count, 1, 1, hr_gold, hr_1);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = std::abs((hr_gold[0] - hr_2[0]) / hr_gold[0]);
            rocblas_error_2 = std::abs((hr_gold[0] - hr_1[0]) / hr_gold[0]);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_asum_strided_batched_fn(handle, N, dx, incx, stridex, batch_count, dr);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_asum_strided_batched_fn(handle, N, dx, incx, stridex, batch_count, dr);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_stride_x, e_batch_count>{}.log_args<T>(rocblas_cout,
                                                                            arg,
                                                                            gpu_time_used,
                                                                            asum_gflop_count<T>(N),
                                                                            asum_gbyte_count<T>(N),
                                                                            cpu_time_used,
                                                                            rocblas_error_1,
                                                                            rocblas_error_2);
    }
}
