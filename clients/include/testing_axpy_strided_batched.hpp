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
template <typename T>
void testing_axpy_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_axpy_strided_batched_fn
        = FORTRAN ? rocblas_axpy_strided_batched<T, true> : rocblas_axpy_strided_batched<T, false>;

    rocblas_local_handle handle;
    rocblas_int          N = 100, incx = 1, incy = 1, batch_count = arg.batch_count;

    rocblas_stride stridex = arg.stride_x, stridey = arg.stride_y;

    T alpha = 0.6;

    device_strided_batch_vector<T> dx(10, 1, 10, 2), dy(10, 1, 10, 2);

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
}

template <typename T>
void testing_axpy_strided_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_axpy_strided_batched_fn
        = FORTRAN ? rocblas_axpy_strided_batched<T, true> : rocblas_axpy_strided_batched<T, false>;

    rocblas_int N = arg.N, incx = arg.incx, incy = arg.incy, batch_count = arg.batch_count;

    rocblas_stride stridex = arg.stride_x, stridey = arg.stride_y;
    if(!stridex)
        stridex = N;
    if(!stridey)
        stridey = N;

    T                    h_alpha = arg.get_alpha<T>();
    rocblas_local_handle handle;

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

    rocblas_int abs_incy = std::abs(incy);

    //
    // Host memory.
    //
    host_strided_batch_vector<T> hx(N, incx ? incx : 1, stridex, batch_count),
        hy(N, incy ? incy : 1, stridey, batch_count), hy1(N, incy ? incy : 1, stridey, batch_count),
        hy2(N, incy ? incy : 1, stridey, batch_count);
    host_vector<T> halpha(1);

    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());
    CHECK_HIP_ERROR(hy1.memcheck());
    CHECK_HIP_ERROR(hy2.memcheck());
    CHECK_HIP_ERROR(halpha.memcheck());

    device_strided_batch_vector<T> dx(N, incx ? incx : 1, stridex, batch_count),
        dy(N, incy ? incy : 1, stridey, batch_count);
    device_vector<T> dalpha(1);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dalpha.memcheck());

    halpha[0] = h_alpha;

    //
    // Initialize host memory.
    // TODO: add NaN testing when roblas_isnan(arg.alpha) returns true.
    //

    rocblas_init(hx, true);
    rocblas_init(hy, false);

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
        {

            //
            // Pointer mode.
            //
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            //
            // Transfer host to device
            //
            CHECK_HIP_ERROR(dy.transfer_from(hy));

            //
            // Call routine.
            //
            CHECK_ROCBLAS_ERROR(rocblas_axpy_strided_batched_fn(
                handle, N, halpha, dx, incx, stridex, dy, incy, stridey, batch_count));

            CHECK_HIP_ERROR(hy1.transfer_from(dy));

            //
            // Pointer mode.
            //
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            //
            // Transfer host to device
            //
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            CHECK_HIP_ERROR(dalpha.transfer_from(halpha));
            //
            // Call routine.
            //
            CHECK_ROCBLAS_ERROR(rocblas_axpy_strided_batched_fn(
                handle, N, dalpha, dx, incx, stridex, dy, incy, stridey, batch_count));

            //
            // Transfer from device to host.
            //
            CHECK_HIP_ERROR(hy2.transfer_from(dy));

            //
            // CPU BLAS
            //
            {
                cpu_time_used = get_time_us();

                //
                // Compute the host solution.
                //
                for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
                {
                    cblas_axpy<T>(N, h_alpha, hx[batch_index], incx, hy[batch_index], incy);
                }
                cpu_time_used = get_time_us() - cpu_time_used;
            }

            //
            // Compare with with the solution.
            //
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, abs_incy, stridey, hy, hy1, batch_count);
                unit_check_general<T>(1, N, abs_incy, stridey, hy, hy2, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_1
                    = norm_check_general<T>('I', 1, N, abs_incy, stridey, hy, hy1, batch_count);
                rocblas_error_2
                    = norm_check_general<T>('I', 1, N, abs_incy, stridey, hy, hy2, batch_count);
            }
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
            rocblas_axpy_strided_batched_fn(
                handle, N, &h_alpha, dx, incx, stridex, dy, incy, stridey, batch_count);
        }

        //
        // Transfer from host to device.
        //
        CHECK_HIP_ERROR(dy.transfer_from(hy));

        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_axpy_strided_batched_fn(
                handle, N, &h_alpha, dx, incx, stridex, dy, incy, stridey, batch_count);
        }
        gpu_time_used = get_time_us() - gpu_time_used;

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
