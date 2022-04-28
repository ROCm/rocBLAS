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
void testing_axpy_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_axpy_strided_batched_fn = arg.fortran ? rocblas_axpy_strided_batched<T, true>
                                                       : rocblas_axpy_strided_batched<T, false>;

    rocblas_local_handle handle{arg};
    rocblas_int          N = 100, incx = 1, incy = 1, batch_count = 2;

    rocblas_stride stridex = N * incx, stridey = N * incy;

    T alpha = 0.6, zero = 0.0;

    // Allocate device memory
    device_strided_batch_vector<T> dx(N, incx, stridex, batch_count),
        dy(N, incy, stridey, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_strided_batched_fn(
            handle, N, &alpha, nullptr, incx, stridex, dy, incy, stridey, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_strided_batched_fn(
            handle, N, &alpha, dx, incx, stridex, nullptr, incy, stridey, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_strided_batched_fn(
            handle, N, nullptr, dx, incx, stridex, dy, incy, stridey, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_strided_batched_fn(
            nullptr, N, &alpha, dx, incx, stridex, dy, incy, stridey, batch_count),
        rocblas_status_invalid_handle);

    // When batch_count==0, all pointers can be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_strided_batched_fn(
            handle, N, nullptr, nullptr, incx, stridex, nullptr, incy, stridey, 0),
        rocblas_status_success);

    // When N==0, all pointers can be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_strided_batched_fn(
            handle, 0, nullptr, nullptr, incx, stridex, nullptr, incy, stridey, batch_count),
        rocblas_status_success);

    // When alpha==0, X and Y can be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_axpy_strided_batched_fn(
            handle, N, &zero, nullptr, incx, stridex, nullptr, incy, stridey, batch_count),
        rocblas_status_success);
}

template <typename T>
void testing_axpy_strided_batched(const Arguments& arg)
{
    auto rocblas_axpy_strided_batched_fn = arg.fortran ? rocblas_axpy_strided_batched<T, true>
                                                       : rocblas_axpy_strided_batched<T, false>;

    rocblas_int N = arg.N, incx = arg.incx, incy = arg.incy, batch_count = arg.batch_count;

    rocblas_stride stridex = arg.stride_x, stridey = arg.stride_y;
    if(!stridex)
        stridex = N;
    if(!stridey)
        stridey = N;

    T                    h_alpha = arg.get_alpha<T>();
    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_axpy_strided_batched_fn(
                handle, N, nullptr, nullptr, incx, stridex, nullptr, incy, stridey, batch_count),
            rocblas_status_success);
        return;
    }

    rocblas_int abs_incy = incy > 0 ? incy : -incy;

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_strided_batch_vector<T> hx(N, incx ? incx : 1, stridex, batch_count);
    host_strided_batch_vector<T> hy_1(N, incy ? incy : 1, stridey, batch_count);
    host_strided_batch_vector<T> hy_2(N, incy ? incy : 1, stridey, batch_count);
    host_strided_batch_vector<T> hy_gold(N, incy ? incy : 1, stridey, batch_count);
    host_vector<T>               halpha(1);

    halpha[0] = h_alpha;

    // Check host memory allocation
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy_1.memcheck());
    CHECK_HIP_ERROR(hy_2.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());

    // Allocate device memory
    device_strided_batch_vector<T> dx(N, incx ? incx : 1, stridex, batch_count);
    device_strided_batch_vector<T> dy(N, incy ? incy : 1, stridey, batch_count);
    device_vector<T>               dalpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dalpha.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy_1, arg, rocblas_client_alpha_sets_nan, false, true);

    hy_gold.copy_from(hy_1);
    hy_2.copy_from(hy_1);

    // Device memory.
    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {

        // Transfer host to device
        CHECK_HIP_ERROR(dx.transfer_from(hx));

        // Call routine with pointer mode on host.
        {

            // Pointer mode host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            // Transfer host to device
            CHECK_HIP_ERROR(dy.transfer_from(hy_1));

            // Call routine.
            CHECK_ROCBLAS_ERROR(rocblas_axpy_strided_batched_fn(
                handle, N, halpha, dx, incx, stridex, dy, incy, stridey, batch_count));

            CHECK_HIP_ERROR(hy_1.transfer_from(dy));

            // Pointer mode device.
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

            // Transfer host to device
            CHECK_HIP_ERROR(dy.transfer_from(hy_2));
            CHECK_HIP_ERROR(dalpha.transfer_from(halpha));

            // Call routine.
            CHECK_ROCBLAS_ERROR(rocblas_axpy_strided_batched_fn(
                handle, N, dalpha, dx, incx, stridex, dy, incy, stridey, batch_count));

            // Transfer from device to host.
            CHECK_HIP_ERROR(hy_2.transfer_from(dy));

            // CPU BLAS
            {
                cpu_time_used = get_time_us_no_sync();

                // Compute the host solution.
                for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
                {
                    cblas_axpy<T>(N, h_alpha, hx[batch_index], incx, hy_gold[batch_index], incy);
                }
                cpu_time_used = get_time_us_no_sync() - cpu_time_used;
            }

            // Compare with with the solution.
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, abs_incy, stridey, hy_gold, hy_1, batch_count);
                unit_check_general<T>(1, N, abs_incy, stridey, hy_gold, hy_2, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_1 = norm_check_general<T>(
                    'I', 1, N, abs_incy, stridey, hy_gold, hy_1, batch_count);
                rocblas_error_2 = norm_check_general<T>(
                    'I', 1, N, abs_incy, stridey, hy_gold, hy_2, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        // Transfer from host to device.
        CHECK_HIP_ERROR(dy.transfer_from(hy_gold));

        // Cold.
        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_axpy_strided_batched_fn(
                handle, N, &h_alpha, dx, incx, stridex, dy, incy, stridey, batch_count);
        }

        // Transfer from host to device.
        CHECK_HIP_ERROR(dy.transfer_from(hy_gold));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_axpy_strided_batched_fn(
                handle, N, &h_alpha, dx, incx, stridex, dy, incy, stridey, batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         axpy_gflop_count<T>(N),
                         axpy_gbyte_count<T>(N),
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
