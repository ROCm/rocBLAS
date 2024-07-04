/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_common.hpp"

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_strided_batched_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_dot_strided_batched_ex_fn
        = arg.api & c_API_FORTRAN
              ? (CONJ ? rocblas_dotc_strided_batched_ex_fortran
                      : rocblas_dot_strided_batched_ex_fortran)
              : (CONJ ? rocblas_dotc_strided_batched_ex : rocblas_dot_strided_batched_ex);
    auto rocblas_dot_strided_batched_ex_fn_64
        = arg.api & c_API_FORTRAN
              ? (CONJ ? rocblas_dotc_strided_batched_ex_64_fortran
                      : rocblas_dot_strided_batched_ex_64_fortran)
              : (CONJ ? rocblas_dotc_strided_batched_ex_64 : rocblas_dot_strided_batched_ex_64);

    rocblas_datatype x_type         = rocblas_type2datatype<Tx>();
    rocblas_datatype y_type         = rocblas_type2datatype<Ty>();
    rocblas_datatype result_type    = rocblas_type2datatype<Tr>();
    rocblas_datatype execution_type = rocblas_type2datatype<Tex>();

    int64_t        N           = 100;
    int64_t        incx        = 1;
    int64_t        incy        = 1;
    rocblas_stride stride_x    = incx * N;
    rocblas_stride stride_y    = incy * N;
    int64_t        batch_count = 2;

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

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_dot_strided_batched_ex_fn,
                (nullptr,
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
                 execution_type));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dot_strided_batched_ex_fn,
                (handle,
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
                 execution_type));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dot_strided_batched_ex_fn,
                (handle,
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
                 execution_type));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dot_strided_batched_ex_fn,
                (handle,
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
                 execution_type));
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_strided_batched_ex(const Arguments& arg)
{
    auto rocblas_dot_strided_batched_ex_fn
        = arg.api & c_API_FORTRAN
              ? (CONJ ? rocblas_dotc_strided_batched_ex_fortran
                      : rocblas_dot_strided_batched_ex_fortran)
              : (CONJ ? rocblas_dotc_strided_batched_ex : rocblas_dot_strided_batched_ex);
    auto rocblas_dot_strided_batched_ex_fn_64
        = arg.api & c_API_FORTRAN
              ? (CONJ ? rocblas_dotc_strided_batched_ex_64_fortran
                      : rocblas_dot_strided_batched_ex_64_fortran)
              : (CONJ ? rocblas_dotc_strided_batched_ex_64 : rocblas_dot_strided_batched_ex_64);

    rocblas_datatype x_type         = arg.a_type;
    rocblas_datatype y_type         = arg.b_type;
    rocblas_datatype result_type    = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    int64_t        N           = arg.N;
    int64_t        incx        = arg.incx;
    int64_t        incy        = arg.incy;
    int64_t        batch_count = arg.batch_count;
    rocblas_stride stride_x    = arg.stride_x;
    rocblas_stride stride_y    = arg.stride_y;

    double               rocblas_error_host   = 0;
    double               rocblas_error_device = 0;
    rocblas_local_handle handle{arg};

    // check to prevent undefined memmory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        device_vector<Tr> d_rocblas_result(std::max(batch_count, int64_t(1)));
        CHECK_DEVICE_ALLOCATION(d_rocblas_result.memcheck());

        host_vector<Tr> h_rocblas_result(std::max(batch_count, int64_t(1)));
        CHECK_HIP_ERROR(h_rocblas_result.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_dot_strided_batched_ex_fn,
                   (handle,
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
                    execution_type));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(rocblas_dot_strided_batched_ex_fn,
                   (handle,
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
                    execution_type));

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
    host_strided_batch_vector<Tx> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<Ty> hy(N, incy, stride_y, batch_count);
    host_vector<Tr>               cpu_result(batch_count);
    host_vector<Tr>               rocblas_result_host(batch_count);
    host_vector<Tr>               rocblas_result_device(batch_count);

    // Allocate device memory
    device_strided_batch_vector<Tx> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<Ty> dy(N, incy, stride_y, batch_count);
    device_vector<Tr>               d_rocblas_result_device(batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_rocblas_result_device.memcheck());

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
        if(arg.pointer_mode_host)
        {
            // GPU BLAS, rocblas_pointer_mode_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            DAPI_CHECK(rocblas_dot_strided_batched_ex_fn,
                       (handle,
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
                        rocblas_result_host,
                        result_type,
                        execution_type));
        }

        if(arg.pointer_mode_device)
        {
            // GPU BLAS, rocblas_pointer_mode_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_dot_strided_batched_ex_fn,
                       (handle,
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
                        d_rocblas_result_device,
                        result_type,
                        execution_type));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                host_vector<Tr> rocblas_result_device_copy(batch_count);
                CHECK_HIP_ERROR(rocblas_result_device_copy.memcheck());

                CHECK_HIP_ERROR(rocblas_result_device.transfer_from(d_rocblas_result_device));
                for(int i = 0; i < arg.iters; i++)
                {
                    DAPI_CHECK(rocblas_dot_strided_batched_ex_fn,
                               (handle,
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
                                d_rocblas_result_device,
                                result_type,
                                execution_type));
                    CHECK_HIP_ERROR(
                        rocblas_result_device_copy.transfer_from(d_rocblas_result_device));
                    unit_check_general<Tr>(
                        1, 1, 1, 1, rocblas_result_device, rocblas_result_device_copy, batch_count);
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(size_t b = 0; b < batch_count; ++b)
        {
            (CONJ ? ref_dotc<Tx, Tr>
                  : ref_dot<Tx, Tr>)(N, hx[b], incx, hy_ptr + b * stride_y, incy, &cpu_result[b]);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // For large N, rocblas_half tends to diverge proportional to N
        // Tolerance is slightly greater than 1 / 1024.0
        bool near_check = arg.initialization == rocblas_initialization::hpl
                          || (std::is_same_v<Tex, rocblas_half> && N > 10000);

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(near_check)
                {
                    const double tol = N * sum_error_tolerance<Tex>;
                    near_check_general<Tr>(
                        1, 1, 1, 1, cpu_result, rocblas_result_host, batch_count, tol);
                }
                else
                {
                    unit_check_general<Tr>(
                        1, 1, 1, 1, cpu_result, rocblas_result_host, batch_count);
                }
            }

            if(arg.norm_check)
            {
                for(size_t b = 0; b < batch_count; ++b)
                {
                    rocblas_error_host
                        += rocblas_abs((cpu_result[b] - rocblas_result_host[b]) / cpu_result[b]);
                }
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(rocblas_result_device.transfer_from(d_rocblas_result_device));

            if(arg.unit_check)
            {
                if(near_check)
                {
                    const double tol = N * sum_error_tolerance<Tex>;
                    near_check_general<Tr>(
                        1, 1, 1, 1, cpu_result, rocblas_result_device, batch_count, tol);
                }
                else
                {
                    unit_check_general<Tr>(
                        1, 1, 1, 1, cpu_result, rocblas_result_device, batch_count);
                }
            }

            if(arg.norm_check)
            {
                for(size_t b = 0; b < batch_count; ++b)
                {
                    rocblas_error_device
                        += rocblas_abs((cpu_result[b] - rocblas_result_device[b]) / cpu_result[b]);
                }
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_dot_strided_batched_ex_fn,
                          (handle,
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
                           d_rocblas_result_device,
                           result_type,
                           execution_type));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count, e_algo>{}
            .log_args<Tx>(rocblas_cout,
                          arg,
                          gpu_time_used,
                          dot_gflop_count<CONJ, Tx>(N),
                          dot_gbyte_count<Tx>(N),
                          cpu_time_used,
                          rocblas_error_host,
                          rocblas_error_device);
    }
}
