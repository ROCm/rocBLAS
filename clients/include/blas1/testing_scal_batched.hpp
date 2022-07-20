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

template <typename T, typename U = T>
void testing_scal_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_scal_batched_fn
        = arg.fortran ? rocblas_scal_batched<T, U, true> : rocblas_scal_batched<T, U, false>;

    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    U           h_alpha     = U(1.0);
    rocblas_int batch_count = 2;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_batch_vector<T> dx(N, incx, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_batched_fn)(nullptr, N, &h_alpha, dx.ptr_on_device(), incx, batch_count),
        rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_batched_fn)(handle, N, nullptr, dx.ptr_on_device(), incx, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_batched_fn)(handle, N, &h_alpha, nullptr, incx, batch_count),
        rocblas_status_invalid_pointer);
}

template <typename T, typename U = T>
void testing_scal_batched(const Arguments& arg)
{
    auto rocblas_scal_batched_fn
        = arg.fortran ? rocblas_scal_batched<T, U, true> : rocblas_scal_batched<T, U, false>;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    U           h_alpha     = arg.get_alpha<U>();
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N < 0 || incx <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            (rocblas_scal_batched_fn)(handle, N, nullptr, nullptr, incx, batch_count),
            rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx_1), `d` is in GPU (device) memory (eg dx_1).
    // Allocate host memory
    host_batch_vector<T> hx_1(N, incx, batch_count);
    host_batch_vector<T> hx_2(N, incx, batch_count);
    host_batch_vector<T> hx_gold(N, incx, batch_count);
    host_vector<U>       halpha(1);
    halpha[0] = h_alpha;

    // Allocate device memory
    device_batch_vector<T> dx_1(N, incx, batch_count);
    device_batch_vector<T> dx_2(N, incx, batch_count);
    device_vector<U>       d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initialize memory on host.
    rocblas_init_vector(hx_1, arg, rocblas_client_alpha_sets_nan, true);

    hx_2.copy_from(hx_1);
    hx_gold.copy_from(hx_1);

    // copy data from CPU to device
    // 1. User intermediate arrays to access device memory from host
    CHECK_HIP_ERROR(dx_1.transfer_from(hx_1));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(dx_2.transfer_from(hx_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((
            rocblas_scal_batched_fn(handle, N, &h_alpha, dx_1.ptr_on_device(), incx, batch_count)));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (rocblas_scal_batched_fn(handle, N, d_alpha, dx_2.ptr_on_device(), incx, batch_count)));

        // Transfer output from device to CPU
        CHECK_HIP_ERROR(hx_1.transfer_from(dx_1));
        CHECK_HIP_ERROR(hx_2.transfer_from(dx_2));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal(N, h_alpha, (T*)hx_gold[b], incx);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incx, hx_gold, hx_1, batch_count);
            unit_check_general<T>(1, N, incx, hx_gold, hx_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, N, incx, hx_gold, hx_1, batch_count);
            rocblas_error_2 = norm_check_general<T>('F', 1, N, incx, hx_gold, hx_2, batch_count);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_scal_batched_fn(handle, N, &h_alpha, dx_1.ptr_on_device(), incx, batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_scal_batched_fn(handle, N, &h_alpha, dx_1.ptr_on_device(), incx, batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_batch_count>{}.log_args<T>(rocblas_cout,
                                                                         arg,
                                                                         gpu_time_used,
                                                                         scal_gflop_count<T, U>(N),
                                                                         scal_gbyte_count<T>(N),
                                                                         cpu_time_used,
                                                                         rocblas_error_1,
                                                                         rocblas_error_2);
    }
}
