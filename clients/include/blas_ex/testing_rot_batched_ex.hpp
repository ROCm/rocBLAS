/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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
void testing_rot_batched_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_rot_batched_ex_fn
        = arg.fortran ? rocblas_rot_batched_ex_fortran : rocblas_rot_batched_ex;

    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype y_type         = rocblas_datatype_f32_r;
    rocblas_datatype cs_type        = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    rocblas_int incy        = 1;
    rocblas_int batch_count = 5;

    rocblas_local_handle    handle{arg};
    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    device_vector<Tcs>      dc(1);
    device_vector<Tcs>      ds(1);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_rot_batched_ex_fn(nullptr,
                                                     N,
                                                     dx.ptr_on_device(),
                                                     x_type,
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     y_type,
                                                     incy,
                                                     dc,
                                                     ds,
                                                     cs_type,
                                                     batch_count,
                                                     execution_type)),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_batched_ex_fn(handle,
                                                     N,
                                                     nullptr,
                                                     x_type,
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     y_type,
                                                     incy,
                                                     dc,
                                                     ds,
                                                     cs_type,
                                                     batch_count,
                                                     execution_type)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_batched_ex_fn(handle,
                                                     N,
                                                     dx.ptr_on_device(),
                                                     x_type,
                                                     incx,
                                                     nullptr,
                                                     y_type,
                                                     incy,
                                                     dc,
                                                     ds,
                                                     cs_type,
                                                     batch_count,
                                                     execution_type)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_batched_ex_fn(handle,
                                                     N,
                                                     dx.ptr_on_device(),
                                                     x_type,
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     y_type,
                                                     incy,
                                                     nullptr,
                                                     ds,
                                                     cs_type,
                                                     batch_count,
                                                     execution_type)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rot_batched_ex_fn(handle,
                                                     N,
                                                     dx.ptr_on_device(),
                                                     x_type,
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     y_type,
                                                     incy,
                                                     dc,
                                                     nullptr,
                                                     cs_type,
                                                     batch_count,
                                                     execution_type)),
                          rocblas_status_invalid_pointer);
}

template <typename Tx, typename Ty, typename Tcs, typename Tex>
void testing_rot_batched_ex(const Arguments& arg)
{
    auto rocblas_rot_batched_ex_fn
        = arg.fortran ? rocblas_rot_batched_ex_fortran : rocblas_rot_batched_ex;

    rocblas_datatype x_type         = arg.a_type;
    rocblas_datatype y_type         = arg.b_type;
    rocblas_datatype cs_type        = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
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
        CHECK_ROCBLAS_ERROR((rocblas_rot_batched_ex_fn(handle,
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
                                                       batch_count,
                                                       execution_type)));
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_x   = N * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    device_vector<Tcs>      dc(1);
    device_vector<Tcs>      ds(1);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    // Initial Data on CPU
    host_batch_vector<Tx> hx(N, incx, batch_count);
    host_batch_vector<Ty> hy(N, incy, batch_count);
    host_vector<Tcs>      hc(1);
    host_vector<Tcs>      hs(1);

    rocblas_init(hx, true);
    rocblas_init(hy, false);

    rocblas_init<Tcs>(hc, 1, 1, 1);
    rocblas_init<Tcs>(hs, 1, 1, 1);

    // CPU BLAS reference data
    host_batch_vector<Tx> cx(N, incx, batch_count);
    host_batch_vector<Ty> cy(N, incy, batch_count);
    cx.copy_from(hx);
    cy.copy_from(hy);

    // cblas_rotg<T, U>(cx, cy, hc, hs);
    // cx[0] = hx[0];
    // cy[0] = hy[0];
    cpu_time_used = get_time_us_no_sync();
    for(int b = 0; b < batch_count; b++)
    {
        cblas_rot<Tx, Ty, Tcs, Tcs>(N, cx[b], incx, cy[b], incy, hc, hs);
    }
    cpu_time_used = get_time_us_no_sync() - cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // Test rocblas_pointer_mode_host
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));

            CHECK_ROCBLAS_ERROR((rocblas_rot_batched_ex_fn(handle,
                                                           N,
                                                           dx.ptr_on_device(),
                                                           x_type,
                                                           incx,
                                                           dy.ptr_on_device(),
                                                           y_type,
                                                           incy,
                                                           hc,
                                                           hs,
                                                           cs_type,
                                                           batch_count,
                                                           execution_type)));

            host_batch_vector<Tx> rx(N, incx, batch_count);
            host_batch_vector<Ty> ry(N, incy, batch_count);

            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, abs_incx, cx, rx, batch_count);
                unit_check_general<Ty>(1, N, abs_incy, cy, ry, batch_count);
            }
            if(arg.norm_check)
            {
                norm_error_host_x
                    = norm_check_general<Tx>('F', 1, N, abs_incx, cx, rx, batch_count);
                norm_error_host_y
                    = norm_check_general<Ty>('F', 1, N, abs_incy, cy, ry, batch_count);
            }
        }

        // Test rocblas_pointer_mode_device
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));

            CHECK_HIP_ERROR(dc.transfer_from(hc));
            CHECK_HIP_ERROR(ds.transfer_from(hs));

            CHECK_ROCBLAS_ERROR((rocblas_rot_batched_ex_fn(handle,
                                                           N,
                                                           dx.ptr_on_device(),
                                                           x_type,
                                                           incx,
                                                           dy.ptr_on_device(),
                                                           y_type,
                                                           incy,
                                                           dc,
                                                           ds,
                                                           cs_type,
                                                           batch_count,
                                                           execution_type)));

            host_batch_vector<Tx> rx(N, incx, batch_count);
            host_batch_vector<Ty> ry(N, incy, batch_count);
            CHECK_HIP_ERROR(rx.transfer_from(dx));
            CHECK_HIP_ERROR(ry.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, abs_incx, cx, rx, batch_count);
                unit_check_general<Ty>(1, N, abs_incy, cy, ry, batch_count);
            }
            if(arg.norm_check)
            {
                norm_error_device_x
                    = norm_check_general<Tx>('F', 1, N, abs_incx, cx, rx, batch_count);
                norm_error_device_y
                    = norm_check_general<Ty>('F', 1, N, abs_incy, cy, ry, batch_count);
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
            rocblas_rot_batched_ex_fn(handle,
                                      N,
                                      dx.ptr_on_device(),
                                      x_type,
                                      incx,
                                      dy.ptr_on_device(),
                                      y_type,
                                      incy,
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
            rocblas_rot_batched_ex_fn(handle,
                                      N,
                                      dx.ptr_on_device(),
                                      x_type,
                                      incx,
                                      dy.ptr_on_device(),
                                      y_type,
                                      incy,
                                      dc,
                                      ds,
                                      cs_type,
                                      batch_count,
                                      execution_type);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_batch_count>{}.log_args<Tx>(rocblas_cout,
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
