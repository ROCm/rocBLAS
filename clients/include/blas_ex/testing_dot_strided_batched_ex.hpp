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
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_strided_batched_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_dot_strided_batched_ex_fn
        = arg.fortran ? (CONJ ? rocblas_dotc_strided_batched_ex_fortran
                              : rocblas_dot_strided_batched_ex_fortran)
                      : (CONJ ? rocblas_dotc_strided_batched_ex : rocblas_dot_strided_batched_ex);

    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype y_type         = rocblas_datatype_f32_r;
    rocblas_datatype result_type    = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    rocblas_int incy        = 1;
    rocblas_int stride_x    = incx * N;
    rocblas_int stride_y    = incy * N;
    rocblas_int batch_count = 2;
    size_t      size_x      = stride_x * batch_count;
    size_t      size_y      = stride_y * batch_count;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_strided_batch_vector<Tx> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<Ty> dy(N, incy, stride_y, batch_count);
    device_vector<Tr>               d_rocblas_result(batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    EXPECT_ROCBLAS_STATUS((rocblas_dot_strided_batched_ex_fn)(nullptr,
                                                              N,
                                                              dx,
                                                              x_type,
                                                              incx,
                                                              stride_x,
                                                              dy,
                                                              y_type,
                                                              incy,
                                                              stride_y,
                                                              batch_count,
                                                              d_rocblas_result,
                                                              result_type,
                                                              execution_type),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS((rocblas_dot_strided_batched_ex_fn)(handle,
                                                              N,
                                                              nullptr,
                                                              x_type,
                                                              incx,
                                                              stride_x,
                                                              dy,
                                                              y_type,
                                                              incy,
                                                              stride_y,
                                                              batch_count,
                                                              d_rocblas_result,
                                                              result_type,
                                                              execution_type),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_dot_strided_batched_ex_fn)(handle,
                                                              N,
                                                              dx,
                                                              x_type,
                                                              incx,
                                                              stride_x,
                                                              nullptr,
                                                              y_type,
                                                              incy,
                                                              stride_y,
                                                              batch_count,
                                                              d_rocblas_result,
                                                              result_type,
                                                              execution_type),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_dot_strided_batched_ex_fn)(handle,
                                                              N,
                                                              dx,
                                                              x_type,
                                                              incx,
                                                              stride_x,
                                                              dy,
                                                              y_type,
                                                              incy,
                                                              stride_y,
                                                              batch_count,
                                                              nullptr,
                                                              result_type,
                                                              execution_type),
                          rocblas_status_invalid_pointer);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr>
void testing_dotc_strided_batched_ex_bad_arg(const Arguments& arg)
{
    testing_dot_strided_batched_ex_bad_arg<Tx, Ty, Tr, Tex, true>(arg);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_strided_batched_ex(const Arguments& arg)
{
    auto rocblas_dot_strided_batched_ex_fn
        = arg.fortran ? (CONJ ? rocblas_dotc_strided_batched_ex_fortran
                              : rocblas_dot_strided_batched_ex_fortran)
                      : (CONJ ? rocblas_dotc_strided_batched_ex : rocblas_dot_strided_batched_ex);

    rocblas_datatype x_type         = arg.a_type;
    rocblas_datatype y_type         = arg.b_type;
    rocblas_datatype result_type    = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_int    incy        = arg.incy;
    rocblas_int    batch_count = arg.batch_count;
    rocblas_int    abs_incx    = incx >= 0 ? incx : -incx;
    rocblas_int    abs_incy    = incy >= 0 ? incy : -incy;
    rocblas_stride stride_x    = arg.stride_x;
    rocblas_stride stride_y    = arg.stride_y;

    double               rocblas_error_1 = 0;
    double               rocblas_error_2 = 0;
    rocblas_local_handle handle{arg};

    // check to prevent undefined memmory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        device_vector<Tr> d_rocblas_result(std::max(batch_count, 1));
        CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

        host_vector<Tr> h_rocblas_result(std::max(batch_count, 1));
        CHECK_HIP_ERROR(h_rocblas_result.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        EXPECT_ROCBLAS_STATUS((rocblas_dot_strided_batched_ex_fn)(handle,
                                                                  N,
                                                                  nullptr,
                                                                  x_type,
                                                                  incx,
                                                                  stride_x,
                                                                  nullptr,
                                                                  y_type,
                                                                  incy,
                                                                  stride_y,
                                                                  batch_count,
                                                                  d_rocblas_result,
                                                                  result_type,
                                                                  execution_type),
                              rocblas_status_success);

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS((rocblas_dot_strided_batched_ex_fn)(handle,
                                                                  N,
                                                                  nullptr,
                                                                  x_type,
                                                                  incx,
                                                                  stride_x,
                                                                  nullptr,
                                                                  y_type,
                                                                  incy,
                                                                  stride_y,
                                                                  batch_count,
                                                                  h_rocblas_result,
                                                                  result_type,
                                                                  execution_type),
                              rocblas_status_success);

        if(batch_count > 0)
        {
            host_vector<Tr> cpu_0(batch_count);
            host_vector<Tr> gpu_0(batch_count);
            CHECK_HIP_ERROR(gpu_0.transfer_from(d_rocblas_result));
            unit_check_general<Tr>(1, 1, 1, 1, cpu_0, gpu_0, batch_count);
            unit_check_general<Tr>(1, 1, 1, 1, cpu_0, h_rocblas_result, batch_count);
        }

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_strided_batch_vector<Tx> hx(N, incx ? incx : 1, stride_x, batch_count);
    host_strided_batch_vector<Ty> hy(N, incy ? incy : 1, stride_y, batch_count);
    host_vector<Tr>               cpu_result(batch_count);
    host_vector<Tr>               rocblas_result_1(batch_count);
    host_vector<Tr>               rocblas_result_2(batch_count);

    // Allocate device memory
    device_strided_batch_vector<Tx> dx(N, incx ? incx : 1, stride_x, batch_count);
    device_strided_batch_vector<Ty> dy(N, incy ? incy : 1, stride_y, batch_count);
    device_vector<Tr>               d_rocblas_result_2(batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result_2.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false, true);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;

    // arg.algo indicates to force optimized x dot x kernel algorithm with equal inc
    auto dy_ptr = (arg.algo) ? (Tx*)(dx) : (Ty*)(dy);
    auto hy_ptr = (arg.algo) ? (Tx*)(hx) : (Ty*)(hy);

    if(arg.algo)
    {
        incy     = incx;
        stride_y = stride_x;
    }

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR((rocblas_dot_strided_batched_ex_fn)(handle,
                                                                N,
                                                                dx,
                                                                x_type,
                                                                incx,
                                                                stride_x,
                                                                dy_ptr,
                                                                y_type,
                                                                incy,
                                                                stride_y,
                                                                batch_count,
                                                                rocblas_result_1,
                                                                result_type,
                                                                execution_type));
        handle.post_test(arg);
        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR((rocblas_dot_strided_batched_ex_fn)(handle,
                                                                N,
                                                                dx,
                                                                x_type,
                                                                incx,
                                                                stride_x,
                                                                dy_ptr,
                                                                y_type,
                                                                incy,
                                                                stride_y,
                                                                batch_count,
                                                                d_rocblas_result_2,
                                                                result_type,
                                                                execution_type));
        handle.post_test(arg);
        CHECK_HIP_ERROR(rocblas_result_2.transfer_from(d_rocblas_result_2));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; ++b)
        {
            (CONJ ? cblas_dotc<Tx>
                  : cblas_dot<Tx>)(N, hx[b], incx, hy_ptr + b * stride_y, incy, &cpu_result[b]);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            if(std::is_same<Tex, rocblas_half>{} && N > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = N * sum_error_tolerance<Tex>;

                near_check_general<Tr>(1, 1, 1, 1, cpu_result, rocblas_result_1, batch_count, tol);
                near_check_general<Tr>(1, 1, 1, 1, cpu_result, rocblas_result_2, batch_count, tol);
            }
            else
            {
                unit_check_general<Tr>(1, 1, 1, 1, cpu_result, rocblas_result_1, batch_count);
                unit_check_general<Tr>(1, 1, 1, 1, cpu_result, rocblas_result_2, batch_count);
            }
        }

        if(arg.norm_check)
        {
            rocblas_cout << "cpu=" << cpu_result << ", gpu_host_ptr=" << rocblas_result_1
                         << ", gpu_device_ptr=" << rocblas_result_2 << std::endl;
            for(int b = 0; b < batch_count; ++b)
            {
                rocblas_error_1
                    += rocblas_abs((cpu_result[b] - rocblas_result_1[b]) / cpu_result[b]);
                rocblas_error_2
                    += rocblas_abs((cpu_result[b] - rocblas_result_2[b]) / cpu_result[b]);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            (rocblas_dot_strided_batched_ex_fn)(handle,
                                                N,
                                                dx,
                                                x_type,
                                                incx,
                                                stride_x,
                                                dy_ptr,
                                                y_type,
                                                incy,
                                                stride_y,
                                                batch_count,
                                                d_rocblas_result_2,
                                                result_type,
                                                execution_type);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            (rocblas_dot_strided_batched_ex_fn)(handle,
                                                N,
                                                dx,
                                                x_type,
                                                incx,
                                                stride_x,
                                                dy_ptr,
                                                y_type,
                                                incy,
                                                stride_y,
                                                batch_count,
                                                d_rocblas_result_2,
                                                result_type,
                                                execution_type);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count, e_algo>{}
            .log_args<Tx>(rocblas_cout,
                          arg,
                          gpu_time_used,
                          dot_gflop_count<CONJ, Tx>(N),
                          dot_gbyte_count<Tx>(N),
                          cpu_time_used,
                          rocblas_error_1,
                          rocblas_error_2);
    }
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr>
void testing_dotc_strided_batched_ex(const Arguments& arg)
{
    testing_dot_strided_batched_ex<Tx, Ty, Tr, Tex, true>(arg);
}
