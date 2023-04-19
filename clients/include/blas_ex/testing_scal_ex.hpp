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
#include "type_dispatch.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_scal_ex_fn = arg.api == FORTRAN ? rocblas_scal_ex_fortran : rocblas_scal_ex;

    rocblas_datatype alpha_type     = rocblas_type2datatype<Ta>();
    rocblas_datatype x_type         = rocblas_type2datatype<Tx>();
    rocblas_datatype execution_type = rocblas_type2datatype<Tex>();

    rocblas_int N    = 100;
    rocblas_int incx = 1;
    Ta          alpha(0.6);

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_vector<Tx> dx(N, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_ex_fn(nullptr, N, &alpha, alpha_type, dx, x_type, incx, execution_type)),
        rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_ex_fn(handle, N, nullptr, alpha_type, dx, x_type, incx, execution_type)),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_ex_fn(handle, N, &alpha, alpha_type, nullptr, x_type, incx, execution_type)),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS((rocblas_scal_ex_fn(handle,
                                              N,
                                              nullptr,
                                              rocblas_datatype_f32_c,
                                              nullptr,
                                              rocblas_datatype_f64_r,
                                              incx,
                                              rocblas_datatype_f32_c)),
                          rocblas_status_not_implemented);
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_ex(const Arguments& arg)
{
    auto rocblas_scal_ex_fn = arg.api == FORTRAN ? rocblas_scal_ex_fortran : rocblas_scal_ex;

    rocblas_datatype alpha_type     = arg.a_type;
    rocblas_datatype x_type         = arg.b_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_int N       = arg.N;
    rocblas_int incx    = arg.incx;
    Ta          h_alpha = arg.get_alpha<Ta>();

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N <= 0 || incx <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_scal_ex_fn(
            handle, N, nullptr, alpha_type, nullptr, x_type, incx, execution_type)));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_vector<Tx> hx(N, incx);
    host_vector<Tx> hx_gold(N, incx);
    host_vector<Ta> halpha(1);
    halpha[0] = h_alpha;

    // Initial Data on CPU
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // allocate memory on device
    device_vector<Tx> dx(N, incx);
    device_vector<Ta> d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // copy vector is easy in STL; hx_gold = hx: save a copy in hx_gold which will be output of CPU
    // BLAS
    hx_gold = hx;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // GPU BLAS, rocblas_pointer_mode_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR((rocblas_scal_ex_fn(
                handle, N, &h_alpha, alpha_type, dx, x_type, incx, execution_type)));
            handle.post_test(arg);

            // Transfer output from device to CPU
            CHECK_HIP_ERROR(hx.transfer_from(dx));
        }

        if(arg.pointer_mode_device)
        {
            // copy data from CPU to device for rocblas_pointer_mode_device test
            CHECK_HIP_ERROR(dx.transfer_from(hx_gold));
            CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

            // GPU BLAS, rocblas_pointer_mode_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR((rocblas_scal_ex_fn(
                handle, N, d_alpha, alpha_type, dx, x_type, incx, execution_type)));
            handle.post_test(arg);
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        cblas_scal(N, h_alpha, (Tx*)hx_gold, incx);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, incx, hx_gold, hx);
            }
            if(arg.norm_check)
            {
                rocblas_error_1 = norm_check_general<Tx>('F', 1, N, incx, hx_gold, hx);
            }
        }

        if(arg.pointer_mode_device)
        {
            // Transfer output from device to CPU
            CHECK_HIP_ERROR(hx.transfer_from(dx));

            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, incx, hx_gold, hx);
            }
            if(arg.norm_check)
            {
                rocblas_error_2 = norm_check_general<Tx>('F', 1, N, incx, hx_gold, hx);
            }
        }
    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_scal_ex_fn(handle, N, &h_alpha, alpha_type, dx, x_type, incx, execution_type);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_scal_ex_fn(handle, N, &h_alpha, alpha_type, dx, x_type, incx, execution_type);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx>{}.log_args<Tx>(rocblas_cout,
                                                           arg,
                                                           gpu_time_used,
                                                           scal_gflop_count<Tx, Ta>(N),
                                                           scal_gbyte_count<Tx>(N),
                                                           cpu_time_used,
                                                           rocblas_error_1,
                                                           rocblas_error_2);
    }
}
