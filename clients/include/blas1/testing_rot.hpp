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
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T, typename U = T, typename V = T>
void testing_rot_bad_arg(const Arguments& arg)
{
    auto rocblas_rot_fn
        = arg.api == FORTRAN ? rocblas_rot<T, U, V, true> : rocblas_rot<T, U, V, false>;

    rocblas_int          N    = 100;
    rocblas_int          incx = 1;
    rocblas_int          incy = 1;
    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);
    device_vector<U> dc(1, 1);
    device_vector<V> ds(1, 1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_rot_fn(nullptr, N, dx, incx, dy, incy, dc, ds)),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_fn(handle, N, nullptr, incx, dy, incy, dc, ds)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_fn(handle, N, dx, incx, nullptr, incy, dc, ds)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_fn(handle, N, dx, incx, dy, incy, nullptr, ds)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_fn(handle, N, dx, incx, dy, incy, dc, nullptr)),
                          rocblas_status_invalid_pointer);
}

template <typename T, typename U = T, typename V = T>
void testing_rot(const Arguments& arg)
{

    auto rocblas_rot_fn
        = arg.api == FORTRAN ? rocblas_rot<T, U, V, true> : rocblas_rot<T, U, V, false>;

    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;

    rocblas_local_handle handle{arg};
    double               gpu_time_used, cpu_time_used;
    double norm_error_host_x = 0.0, norm_error_host_y = 0.0, norm_error_device_x = 0.0,
           norm_error_device_y = 0.0;

    // check to prevent undefined memory allocation error
    if(N <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            (rocblas_rot_fn(handle, N, nullptr, incx, nullptr, incy, nullptr, nullptr)));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_vector<T> hx(N, incx);
    host_vector<T> hy(N, incy);
    host_vector<U> hc(1);
    host_vector<V> hs(1);

    // Allocate device memory
    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);
    device_vector<U> dc(1, 1);
    device_vector<V> ds(1, 1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);
    rocblas_init_vector(hc, arg, rocblas_client_alpha_sets_nan, false);
    rocblas_init_vector(hs, arg, rocblas_client_alpha_sets_nan, false);

    // CPU BLAS reference data
    host_vector<T> cx = hx;
    host_vector<T> cy = hy;
    // cblas_rotg<T, U>(cx, cy, hc, hs);
    // cx[0] = hx[0];
    // cy[0] = hy[0];
    cpu_time_used = get_time_us_no_sync();
    cblas_rot<T, T, U, V>(N, cx, incx, cy, incy, hc, hs);
    cpu_time_used = get_time_us_no_sync() - cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // Test rocblas_pointer_mode_host
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR((rocblas_rot_fn(handle, N, dx, incx, dy, incy, hc, hs)));
            handle.post_test(arg);
            // Allocate host memory
            host_vector<T> rx(N, incx);
            host_vector<T> ry(N, incy);

            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incx, cx, rx);
                unit_check_general<T>(1, N, incy, cy, ry);
            }
            if(arg.norm_check)
            {
                norm_error_host_x = norm_check_general<T>('F', 1, N, incx, cx, rx);
                norm_error_host_y = norm_check_general<T>('F', 1, N, incy, cy, ry);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            CHECK_HIP_ERROR(dc.transfer_from(hc));
            CHECK_HIP_ERROR(ds.transfer_from(hs));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR((rocblas_rot_fn(handle, N, dx, incx, dy, incy, dc, ds)));
            handle.post_test(arg);

            // Allocate host memory
            host_vector<T> rx(N, incx);
            host_vector<T> ry(N, incy);

            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incx, cx, rx);
                unit_check_general<T>(1, N, incy, cy, ry);
            }
            if(arg.norm_check)
            {
                norm_error_device_x = norm_check_general<T>('F', 1, N, incx, cx, rx);
                norm_error_device_y = norm_check_general<T>('F', 1, N, incy, cy, ry);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_HIP_ERROR(dc.transfer_from(hc));
        CHECK_HIP_ERROR(ds.transfer_from(hs));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rot_fn(handle, N, dx, incx, dy, incy, dc, ds);
        }
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rot_fn(handle, N, dx, incx, dy, incy, dc, ds);
        }
        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used);

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<T>(rocblas_cout,
                                                         arg,
                                                         gpu_time_used,
                                                         rot_gflop_count<T, T, U, V>(N),
                                                         rot_gbyte_count<T>(N),
                                                         cpu_time_used,
                                                         norm_error_host_x,
                                                         norm_error_device_x,
                                                         norm_error_host_y,
                                                         norm_error_device_y);
    }
}
