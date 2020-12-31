/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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
void testing_rot_strided_batched_ex_bad_arg(const Arguments& arg)
{
    // clang-format off
    const bool FORTRAN                        = arg.fortran;
    auto       rocblas_rot_strided_batched_ex_fn = FORTRAN ? rocblas_rot_strided_batched_ex_fortran
                                                           : rocblas_rot_strided_batched_ex;
    // clang-format on

    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype y_type         = rocblas_datatype_f32_r;
    rocblas_datatype cs_type        = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_int         N           = 100;
    rocblas_int         incx        = 1;
    rocblas_stride      stride_x    = 1;
    rocblas_int         incy        = 1;
    rocblas_stride      stride_y    = 1;
    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 100;

    rocblas_local_handle handle{arg};
    device_vector<Tx>    dx(safe_size);
    device_vector<Ty>    dy(safe_size);
    device_vector<Tcs>   dc(1);
    device_vector<Tcs>   ds(1);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_rot_strided_batched_ex_fn(nullptr,
                                                             N,
                                                             dx,
                                                             x_type,
                                                             incx,
                                                             stride_x,
                                                             dy,
                                                             y_type,
                                                             incy,
                                                             stride_y,
                                                             dc,
                                                             ds,
                                                             cs_type,
                                                             batch_count,
                                                             execution_type)),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_strided_batched_ex_fn(handle,
                                                             N,
                                                             nullptr,
                                                             x_type,
                                                             incx,
                                                             stride_x,
                                                             dy,
                                                             y_type,
                                                             incy,
                                                             stride_y,
                                                             dc,
                                                             ds,
                                                             cs_type,
                                                             batch_count,
                                                             execution_type)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_strided_batched_ex_fn(handle,
                                                             N,
                                                             dx,
                                                             x_type,
                                                             incx,
                                                             stride_x,
                                                             nullptr,
                                                             y_type,
                                                             incy,
                                                             stride_y,
                                                             dc,
                                                             ds,
                                                             cs_type,
                                                             batch_count,
                                                             execution_type)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_strided_batched_ex_fn(handle,
                                                             N,
                                                             dx,
                                                             x_type,
                                                             incx,
                                                             stride_x,
                                                             dy,
                                                             y_type,
                                                             incy,
                                                             stride_y,
                                                             nullptr,
                                                             ds,
                                                             cs_type,
                                                             batch_count,
                                                             execution_type)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_strided_batched_ex_fn(handle,
                                                             N,
                                                             dx,
                                                             x_type,
                                                             incx,
                                                             stride_x,
                                                             dy,
                                                             y_type,
                                                             incy,
                                                             stride_y,
                                                             dc,
                                                             nullptr,
                                                             cs_type,
                                                             batch_count,
                                                             execution_type)),
                          rocblas_status_invalid_pointer);
}

template <typename Tx, typename Ty, typename Tcs, typename Tex>
void testing_rot_strided_batched_ex(const Arguments& arg)
{
    // clang-format off
    const bool FORTRAN                        = arg.fortran;
    auto       rocblas_rot_strided_batched_ex_fn = FORTRAN ? rocblas_rot_strided_batched_ex_fortran
                                                           : rocblas_rot_strided_batched_ex;
    // clang-format on

    rocblas_datatype x_type         = arg.a_type;
    rocblas_datatype y_type         = arg.b_type;
    rocblas_datatype cs_type        = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int stride_x    = arg.stride_x;
    rocblas_int stride_y    = arg.stride_y;
    rocblas_int incy        = arg.incy;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};
    double               gpu_time_used, cpu_time_used;
    double norm_error_host_x = 0.0, norm_error_host_y = 0.0, norm_error_device_x = 0.0,
           norm_error_device_y = 0.0;

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        EXPECT_ROCBLAS_STATUS((rocblas_rot_strided_batched_ex_fn)(handle,
                                                                  N,
                                                                  nullptr,
                                                                  x_type,
                                                                  incx,
                                                                  stride_x,
                                                                  nullptr,
                                                                  y_type,
                                                                  incy,
                                                                  stride_y,
                                                                  nullptr,
                                                                  nullptr,
                                                                  cs_type,
                                                                  batch_count,
                                                                  execution_type),
                              rocblas_status_success);
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_x   = N * size_t(abs_incx) + size_t(stride_x) * size_t(batch_count - 1);
    size_t      size_y   = N * size_t(abs_incy) + size_t(stride_y) * size_t(batch_count - 1);

    device_vector<Tx>  dx(size_x);
    device_vector<Ty>  dy(size_y);
    device_vector<Tcs> dc(1);
    device_vector<Tcs> ds(1);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    // Initial Data on CPU
    host_vector<Tx>  hx(size_x);
    host_vector<Ty>  hy(size_y);
    host_vector<Tcs> hc(1);
    host_vector<Tcs> hs(1);
    rocblas_seedrand();
    rocblas_init<Tx>(hx, 1, N, abs_incx, stride_x, batch_count);
    rocblas_init<Ty>(hy, 1, N, abs_incy, stride_y, batch_count);

    rocblas_init<Tcs>(hc, 1, 1, 1);
    rocblas_init<Tcs>(hs, 1, 1, 1);

    // CPU BLAS reference data
    host_vector<Tx> cx = hx;
    host_vector<Ty> cy = hy;
    // cblas_rotg<T, U>(cx, cy, hc, hs);
    // cx[0] = hx[0];
    // cy[0] = hy[0];
    cpu_time_used = get_time_us_no_sync();
    for(int b = 0; b < batch_count; b++)
    {
        cblas_rot<Tx, Ty, Tcs, Tcs>(N, cx + b * stride_x, incx, cy + b * stride_y, incy, hc, hs);
    }
    cpu_time_used = get_time_us_no_sync() - cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // Test rocblas_pointer_mode_host
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(Ty) * size_y, hipMemcpyHostToDevice));
            CHECK_ROCBLAS_ERROR((rocblas_rot_strided_batched_ex_fn(handle,
                                                                   N,
                                                                   dx,
                                                                   x_type,
                                                                   incx,
                                                                   stride_x,
                                                                   dy,
                                                                   y_type,
                                                                   incy,
                                                                   stride_y,
                                                                   hc,
                                                                   hs,
                                                                   cs_type,
                                                                   batch_count,
                                                                   execution_type)));
            host_vector<Tx> rx(size_x);
            host_vector<Ty> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(Ty) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, abs_incx, stride_x, cx, rx, batch_count);
                unit_check_general<Ty>(1, N, abs_incy, stride_y, cy, ry, batch_count);
            }
            if(arg.norm_check)
            {
                norm_error_host_x
                    = norm_check_general<Tx>('F', 1, N, abs_incx, stride_x, cx, rx, batch_count);
                norm_error_host_y
                    = norm_check_general<Ty>('F', 1, N, abs_incy, stride_x, cy, ry, batch_count);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(Ty) * size_y, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(Tcs), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(Tcs), hipMemcpyHostToDevice));
            CHECK_ROCBLAS_ERROR((rocblas_rot_strided_batched_ex_fn(handle,
                                                                   N,
                                                                   dx,
                                                                   x_type,
                                                                   incx,
                                                                   stride_x,
                                                                   dy,
                                                                   y_type,
                                                                   incy,
                                                                   stride_y,
                                                                   dc,
                                                                   ds,
                                                                   cs_type,
                                                                   batch_count,
                                                                   execution_type)));
            host_vector<Tx> rx(size_x);
            host_vector<Ty> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(Ty) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, abs_incx, stride_x, cx, rx, batch_count);
                unit_check_general<Ty>(1, N, abs_incy, stride_y, cy, ry, batch_count);
            }
            if(arg.norm_check)
            {
                norm_error_device_x
                    = norm_check_general<Tx>('F', 1, N, abs_incx, stride_x, cx, rx, batch_count);
                norm_error_device_y
                    = norm_check_general<Ty>('F', 1, N, abs_incy, stride_y, cy, ry, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(Ty) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(Tcs), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(Tcs), hipMemcpyHostToDevice));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_rot_strided_batched_ex_fn(handle,
                                              N,
                                              dx,
                                              x_type,
                                              incx,
                                              stride_x,
                                              dy,
                                              y_type,
                                              incy,
                                              stride_y,
                                              dc,
                                              ds,
                                              cs_type,
                                              batch_count,
                                              execution_type);
        }
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_rot_strided_batched_ex_fn(handle,
                                              N,
                                              dx,
                                              x_type,
                                              incx,
                                              stride_x,
                                              dy,
                                              y_type,
                                              incy,
                                              stride_y,
                                              dc,
                                              ds,
                                              cs_type,
                                              batch_count,
                                              execution_type);
        }
        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used) / number_hot_calls;

        ArgumentModel<e_N, e_incx, e_stride_x, e_incy, e_stride_y, e_batch_count>{}.log_args<Tx>(
            rocblas_cout,
            arg,
            gpu_time_used,
            ArgumentLogging::NA_value,
            ArgumentLogging::NA_value,
            cpu_time_used,
            norm_error_host_x,
            norm_error_device_x,
            norm_error_host_y,
            norm_error_device_y);
    }
}
