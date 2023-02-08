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

template <typename Tx, typename Ty, typename Tcs, typename Tex>
void testing_rot_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_rot_ex_fn = arg.fortran ? rocblas_rot_ex_fortran : rocblas_rot_ex;

    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype y_type         = rocblas_datatype_f32_r;
    rocblas_datatype cs_type        = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_int          N    = 100;
    rocblas_int          incx = 1;
    rocblas_int          incy = 1;
    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_vector<Tx>  dx(N, incx);
    device_vector<Ty>  dy(N, incy);
    device_vector<Tcs> dc(1, 1);
    device_vector<Tcs> ds(1, 1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_ex_fn(
            nullptr, N, dx, x_type, incx, dy, y_type, incy, dc, ds, cs_type, execution_type)),
        rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_ex_fn(
            handle, N, nullptr, x_type, incx, dy, y_type, incy, dc, ds, cs_type, execution_type)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_ex_fn(
            handle, N, dx, x_type, incx, nullptr, y_type, incy, dc, ds, cs_type, execution_type)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_ex_fn(
            handle, N, dx, x_type, incx, dy, y_type, incy, nullptr, ds, cs_type, execution_type)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rot_ex_fn(
            handle, N, dx, x_type, incx, dy, y_type, incy, dc, nullptr, cs_type, execution_type)),
        rocblas_status_invalid_pointer);
}

template <typename Tx, typename Ty, typename Tcs, typename Tex>
void testing_rot_ex(const Arguments& arg)
{
    auto rocblas_rot_ex_fn = arg.fortran ? rocblas_rot_ex_fortran : rocblas_rot_ex;

    rocblas_datatype x_type         = arg.a_type;
    rocblas_datatype y_type         = arg.b_type;
    rocblas_datatype cs_type        = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

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
        CHECK_ROCBLAS_ERROR((rocblas_rot_ex_fn(handle,
                                               N,
                                               nullptr,
                                               x_type,
                                               incx,
                                               nullptr,
                                               y_type,
                                               incy,
                                               nullptr,
                                               nullptr,
                                               cs_type,
                                               execution_type)));
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_vector<Tx>  hx(N, incx ? incx : 1);
    host_vector<Ty>  hy(N, incy ? incy : 1);
    host_vector<Tcs> hc(1, 1);
    host_vector<Tcs> hs(1, 1);

    // Allocate device memory
    device_vector<Tx>  dx(N, incx ? incx : 1);
    device_vector<Ty>  dy(N, incy ? incy : 1);
    device_vector<Tcs> dc(1, 1);
    device_vector<Tcs> ds(1, 1);

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
    host_vector<Tx> hx_gold = hx;
    host_vector<Ty> hy_gold = hy;

    cpu_time_used = get_time_us_no_sync();
    cblas_rot<Tx, Ty, Tcs, Tcs>(N, hx_gold, incx, hy_gold, incy, hc, hs);
    cpu_time_used = get_time_us_no_sync() - cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // Test rocblas_pointer_mode_host
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR((rocblas_rot_ex_fn(
                handle, N, dx, x_type, incx, dy, y_type, incy, hc, hs, cs_type, execution_type)));
            handle.post_test(arg);
            host_vector<Tx> rx(N, incx ? incx : 1);
            host_vector<Ty> ry(N, incy ? incy : 1);

            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, abs_incx, hx_gold, rx);
                unit_check_general<Ty>(1, N, abs_incy, hy_gold, ry);
            }
            if(arg.norm_check)
            {
                norm_error_host_x = norm_check_general<Tx>('F', 1, N, abs_incx, hx_gold, rx);
                norm_error_host_y = norm_check_general<Ty>('F', 1, N, abs_incy, hy_gold, ry);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            CHECK_HIP_ERROR(dc.transfer_from(hc));
            CHECK_HIP_ERROR(ds.transfer_from(hs));

            CHECK_ROCBLAS_ERROR((rocblas_rot_ex_fn(
                handle, N, dx, x_type, incx, dy, y_type, incy, dc, ds, cs_type, execution_type)));

            host_vector<Tx> rx(N, incx ? incx : 1);
            host_vector<Ty> ry(N, incy ? incy : 1);

            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));
            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, abs_incx, hx_gold, rx);
                unit_check_general<Ty>(1, N, abs_incy, hy_gold, ry);
            }
            if(arg.norm_check)
            {
                norm_error_device_x = norm_check_general<Tx>('F', 1, N, abs_incx, hx_gold, rx);
                norm_error_device_y = norm_check_general<Ty>('F', 1, N, abs_incy, hy_gold, ry);
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
            rocblas_rot_ex_fn(
                handle, N, dx, x_type, incx, dy, y_type, incy, dc, ds, cs_type, execution_type);
        }
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rot_ex_fn(
                handle, N, dx, x_type, incx, dy, y_type, incy, dc, ds, cs_type, execution_type);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<Tx>(rocblas_cout,
                                                          arg,
                                                          gpu_time_used,
                                                          rot_gflop_count<Tx, Ty, Tcs, Tcs>(N),
                                                          rot_gbyte_count<Tx>(N),
                                                          cpu_time_used,
                                                          norm_error_host_x,
                                                          norm_error_device_x,
                                                          norm_error_host_y,
                                                          norm_error_device_y);
    }
}
