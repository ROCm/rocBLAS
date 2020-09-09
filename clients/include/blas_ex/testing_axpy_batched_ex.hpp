/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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
template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_batched_ex_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_axpy_batched_ex_fn
        = FORTRAN ? rocblas_axpy_batched_ex_fortran : rocblas_axpy_batched_ex;

    rocblas_datatype alpha_type     = rocblas_datatype_f32_r;
    rocblas_datatype x_type         = rocblas_datatype_f32_r;
    rocblas_datatype y_type         = rocblas_datatype_f32_r;
    rocblas_datatype execution_type = rocblas_datatype_f32_r;

    rocblas_local_handle handle(arg.atomics_mode);

    rocblas_int N = 100, incx = 1, incy = 1, batch_count = 2;

    Ta                      alpha = 0.6;
    device_batch_vector<Tx> dx(10, 1, 2);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    device_batch_vector<Ty> dy(10, 1, 2);
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched_ex_fn(handle,
                                                     N,
                                                     &alpha,
                                                     alpha_type,
                                                     nullptr,
                                                     x_type,
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     y_type,
                                                     incy,
                                                     batch_count,
                                                     execution_type),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched_ex_fn(handle,
                                                     N,
                                                     &alpha,
                                                     alpha_type,
                                                     dx.ptr_on_device(),
                                                     x_type,
                                                     incx,
                                                     nullptr,
                                                     y_type,
                                                     incy,
                                                     batch_count,
                                                     execution_type),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched_ex_fn(handle,
                                                     N,
                                                     nullptr,
                                                     alpha_type,
                                                     dx.ptr_on_device(),
                                                     x_type,
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     y_type,
                                                     incy,
                                                     batch_count,
                                                     execution_type),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched_ex_fn(nullptr,
                                                     N,
                                                     &alpha,
                                                     alpha_type,
                                                     dx.ptr_on_device(),
                                                     x_type,
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     y_type,
                                                     incy,
                                                     batch_count,
                                                     execution_type),
                          rocblas_status_invalid_handle);
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_batched_ex(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_axpy_batched_ex_fn
        = FORTRAN ? rocblas_axpy_batched_ex_fortran : rocblas_axpy_batched_ex;

    rocblas_datatype alpha_type     = arg.a_type;
    rocblas_datatype x_type         = arg.b_type;
    rocblas_datatype y_type         = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_local_handle handle(arg.atomics_mode);
    rocblas_int          N = arg.N, incx = arg.incx, incy = arg.incy, batch_count = arg.batch_count;

    Ta  h_alpha    = arg.get_alpha<Ta>();
    Tex h_alpha_ex = (Tex)h_alpha;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched_ex_fn(handle,
                                                         N,
                                                         nullptr,
                                                         alpha_type,
                                                         nullptr,
                                                         x_type,
                                                         incx,
                                                         nullptr,
                                                         y_type,
                                                         incy,
                                                         batch_count,
                                                         execution_type),
                              rocblas_status_success);
        return;
    }

    rocblas_int abs_incx = std::abs(incx);
    rocblas_int abs_incy = std::abs(incy);
    size_t      size_x   = N * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;

    //
    // Host memory.
    //
    host_batch_vector<Tx> hx(N, incx ? incx : 1, batch_count);
    host_batch_vector<Ty> hy(N, incy ? incy : 1, batch_count), hy1(N, incy ? incy : 1, batch_count),
        hy2(N, incy ? incy : 1, batch_count);
    host_batch_vector<Tex> hy_ex(N, incy ? incy : 1, batch_count);
    host_batch_vector<Tex> hx_ex(N, incx ? incx : 1, batch_count);
    host_vector<Ta>        halpha(1);

    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());
    CHECK_HIP_ERROR(hy1.memcheck());
    CHECK_HIP_ERROR(hy2.memcheck());
    CHECK_HIP_ERROR(halpha.memcheck());

    //
    // Device memory.
    //
    device_batch_vector<Tx> dx(N, incx ? incx : 1, batch_count);
    device_batch_vector<Ty> dy(N, incy ? incy : 1, batch_count);
    device_vector<Ta>       dalpha(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dalpha.memcheck());

    //
    // Assign host alpha.
    //
    halpha[0] = h_alpha;

    //
    // Initialize host memory.
    // TODO: add NaN testing when roblas_isnan(arg.alpha) returns true.
    //
    rocblas_init(hx, true);
    rocblas_init(hy, false);

    for(rocblas_int b = 0; b < batch_count; b++)
    {
        for(size_t i = 0; i < size_y; i++)
            hy_ex[b][i] = (Tex)hy[b][i];
        for(size_t i = 0; i < size_x; i++)
            hx_ex[b][i] = (Tex)hx[b][i];
    }

    //
    // Device memory.
    //

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        //
        // Transfer host to device
        //
        CHECK_HIP_ERROR(dx.transfer_from(hx));

        //
        // Call routine with pointer mode on host.
        //

        //
        // Pointer mode.
        //
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        //
        // Call routine.
        //
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_ROCBLAS_ERROR(rocblas_axpy_batched_ex_fn(handle,
                                                       N,
                                                       halpha,
                                                       alpha_type,
                                                       dx.ptr_on_device(),
                                                       x_type,
                                                       incx,
                                                       dy.ptr_on_device(),
                                                       y_type,
                                                       incy,
                                                       batch_count,
                                                       execution_type));

        //
        // Transfer from device to host.
        //
        CHECK_HIP_ERROR(hy1.transfer_from(dy));

        //
        // Pointer mode.
        //
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        //
        // Call routine.
        //
        CHECK_HIP_ERROR(dalpha.transfer_from(halpha));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_ROCBLAS_ERROR(rocblas_axpy_batched_ex_fn(handle,
                                                       N,
                                                       dalpha,
                                                       alpha_type,
                                                       dx.ptr_on_device(),
                                                       x_type,
                                                       incx,
                                                       dy.ptr_on_device(),
                                                       y_type,
                                                       incy,
                                                       batch_count,
                                                       execution_type));
        //
        // Transfer from device to host.
        //
        CHECK_HIP_ERROR(hy2.transfer_from(dy));

        //
        // CPU BLAS
        //
        {
            cpu_time_used = get_time_us_no_sync();

            //
            // Compute the host solution.
            //
            for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                cblas_axpy<Tex>(N, h_alpha_ex, hx_ex[batch_index], incx, hy_ex[batch_index], incy);
            }
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;

            for(rocblas_int b = 0; b < batch_count; b++)
            {
                for(size_t i = 0; i < size_y; i++)
                    hy[b][i] = hy_ex[b][i];
            }
        }

        //
        // Compare with with hsolution.
        //
        if(arg.unit_check)
        {
            unit_check_general<Ty>(1, N, abs_incy, hy, hy1, batch_count);

            unit_check_general<Ty>(1, N, abs_incy, hy, hy2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<Ty>('I', 1, N, abs_incy, hy, hy1, batch_count);
            rocblas_error_2 = norm_check_general<Ty>('I', 1, N, abs_incy, hy, hy2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        //
        // Transfer from host to device.
        //
        CHECK_HIP_ERROR(dy.transfer_from(hy));

        //
        // Cold.
        //
        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_axpy_batched_ex_fn(handle,
                                       N,
                                       &h_alpha,
                                       alpha_type,
                                       dx.ptr_on_device(),
                                       x_type,
                                       incx,
                                       dy.ptr_on_device(),
                                       y_type,
                                       incy,
                                       batch_count,
                                       execution_type);
        }

        //
        // Transfer from host to device.
        //
        CHECK_HIP_ERROR(dy.transfer_from(hy));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_axpy_batched_ex_fn(handle,
                                       N,
                                       &h_alpha,
                                       alpha_type,
                                       dx.ptr_on_device(),
                                       x_type,
                                       incx,
                                       dy.ptr_on_device(),
                                       y_type,
                                       incy,
                                       batch_count,
                                       execution_type);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_incy, e_batch_count>{}.log_args<Ta>(
            rocblas_cout,
            arg,
            gpu_time_used,
            axpy_gflop_count<Ta>(N),
            axpy_gbyte_count<Ta>(N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}
