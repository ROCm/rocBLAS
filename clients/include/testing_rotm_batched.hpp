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

template <typename T>
void testing_rotm_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_rotm_batched_fn
        = FORTRAN ? rocblas_rotm_batched<T, true> : rocblas_rotm_batched<T, false>;

    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    rocblas_int incy        = 1;
    rocblas_int batch_count = 5;

    rocblas_local_handle   handle;
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_batch_vector<T> dparam(1, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparam.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
    EXPECT_ROCBLAS_STATUS((rocblas_rotm_batched_fn(nullptr,
                                                   N,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   dy.ptr_on_device(),
                                                   incy,
                                                   dparam.ptr_on_device(),
                                                   batch_count)),
                          rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS((rocblas_rotm_batched_fn(handle,
                                                   N,
                                                   nullptr,
                                                   incx,
                                                   dy.ptr_on_device(),
                                                   incy,
                                                   dparam.ptr_on_device(),
                                                   batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_rotm_batched_fn(handle,
                                                   N,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   nullptr,
                                                   incy,
                                                   dparam.ptr_on_device(),
                                                   batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotm_batched_fn(
            handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, nullptr, batch_count)),
        rocblas_status_invalid_pointer);
}

template <typename T>
void testing_rotm_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_rotm_batched_fn
        = FORTRAN ? rocblas_rotm_batched<T, true> : rocblas_rotm_batched<T, false>;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;
    double               gpu_time_used, cpu_time_used;
    double norm_error_host_x = 0.0, norm_error_host_y = 0.0, norm_error_device_x = 0.0,
           norm_error_device_y = 0.0;
    const T rel_error          = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_rotm_batched_fn(
            handle, N, nullptr, incx, nullptr, incy, nullptr, batch_count)));
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_x   = N * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_batch_vector<T> dparam(5, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dparam.memcheck());

    // Initial Data on CPU
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hdata(4, 1, batch_count);
    host_batch_vector<T> hparam(5, 1, batch_count);

    rocblas_init(hx, true);
    rocblas_init(hy, false);
    rocblas_init(hdata, false);
    for(int b = 0; b < batch_count; b++)
    {
        // CPU BLAS reference data
        cblas_rotmg<T>(&hdata[b][0], &hdata[b][1], &hdata[b][2], &hdata[b][3], hparam[b]);
    }

    constexpr int FLAG_COUNT        = 4;
    const T       FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};

    for(int i = 0; i < FLAG_COUNT; i++)
    {
        for(int b = 0; b < batch_count; b++)
            hparam[b][0] = FLAGS[i];

        host_batch_vector<T> cx(N, incx, batch_count);
        host_batch_vector<T> cy(N, incy, batch_count);
        cx.copy_from(hx);
        cy.copy_from(hy);
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotm<T>(N, cx[b], incx, cy[b], incy, hparam[b]);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check || arg.norm_check)
        {
            // Test rocblas_pointer_mode_host
            // TODO: THIS IS NO LONGER SUPPORTED
            // {
            //     CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            //     for(int b = 0; b < batch_count; b++)
            //     {
            //         CHECK_HIP_ERROR(
            //             hipMemcpy(bx[b], hx[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
            //         CHECK_HIP_ERROR(
            //             hipMemcpy(by[b], hy[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
            //     }
            //     CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            //     CHECK_HIP_ERROR(hipMemcpy(dy, by, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

            //     CHECK_ROCBLAS_ERROR(
            //         (rocblas_rotm_batched_fn(handle, N, dx, incx, dy, incy, hparam, batch_count)));

            //     host_vector<T> rx[batch_count];
            //     host_vector<T> ry[batch_count];
            //     for(int b = 0; b < batch_count; b++)
            //     {
            //         rx[b] = host_vector<T>(size_x);
            //         ry[b] = host_vector<T>(size_y);
            //         CHECK_HIP_ERROR(
            //             hipMemcpy(rx[b], bx[b], sizeof(T) * size_x, hipMemcpyDeviceToHost));
            //         CHECK_HIP_ERROR(
            //             hipMemcpy(ry[b], by[b], sizeof(T) * size_y, hipMemcpyDeviceToHost));
            //     }

            //     if(arg.unit_check)
            //     {
            //         T rel_error = std::numeric_limits<T>::epsilon() * 1000;
            //         near_check_general<T,T>(1, N, batch_count, incx, cx, rx, rel_error);
            //         near_check_general<T,T>(1, N, batch_count, incy, cy, ry, rel_error);
            //     }
            //     if(arg.norm_check)
            //     {
            //         norm_error_host_x = norm_check_general<T>('F', 1, N, batch_count, incx, cx, rx);
            //         norm_error_host_y = norm_check_general<T>('F', 1, N, batch_count, incy, cy, ry);
            //     }
            // }

            // Test rocblas_pointer_mode_device
            {
                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
                CHECK_HIP_ERROR(dx.transfer_from(hx));
                CHECK_HIP_ERROR(dy.transfer_from(hy));
                CHECK_HIP_ERROR(dparam.transfer_from(hparam));

                CHECK_ROCBLAS_ERROR((rocblas_rotm_batched_fn(handle,
                                                             N,
                                                             dx.ptr_on_device(),
                                                             incx,
                                                             dy.ptr_on_device(),
                                                             incy,
                                                             dparam.ptr_on_device(),
                                                             batch_count)));

                host_batch_vector<T> rx(N, incx, batch_count);
                host_batch_vector<T> ry(N, incy, batch_count);
                CHECK_HIP_ERROR(rx.transfer_from(dx));
                CHECK_HIP_ERROR(ry.transfer_from(dy));

                if(arg.unit_check)
                {
                    near_check_general<T>(1, N, abs_incx, cx, rx, batch_count, rel_error);
                    near_check_general<T>(1, N, abs_incy, cy, ry, batch_count, rel_error);
                }
                if(arg.norm_check)
                {
                    norm_error_device_x
                        = norm_check_general<T>('F', 1, N, abs_incx, cx, rx, batch_count);
                    norm_error_device_y
                        = norm_check_general<T>('F', 1, N, abs_incy, cy, ry, batch_count);
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
            CHECK_HIP_ERROR(dparam.transfer_from(hparam));

            for(int iter = 0; iter < number_cold_calls; iter++)
            {
                rocblas_rotm_batched_fn(handle,
                                        N,
                                        dx.ptr_on_device(),
                                        incx,
                                        dy.ptr_on_device(),
                                        incy,
                                        dparam.ptr_on_device(),
                                        batch_count);
            }
            gpu_time_used = get_time_us(); // in microseconds
            for(int iter = 0; iter < number_hot_calls; iter++)
            {
                rocblas_rotm_batched_fn(handle,
                                        N,
                                        dx.ptr_on_device(),
                                        incx,
                                        dy.ptr_on_device(),
                                        incy,
                                        dparam.ptr_on_device(),
                                        batch_count);
            }
            gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

            rocblas_cout << "N,incx,incy,rocblas(us),cpu(us)";
            if(arg.norm_check)
                rocblas_cout
                    << ",norm_error_host_x,norm_error_host_y,norm_error_device_x,norm_error_"
                       "device_y";
            rocblas_cout << std::endl;
            rocblas_cout << N << "," << incx << "," << incy << "," << gpu_time_used << ","
                         << cpu_time_used;
            if(arg.norm_check)
                rocblas_cout << ',' << norm_error_host_x << ',' << norm_error_host_y << ","
                             << norm_error_device_x << "," << norm_error_device_y;
            rocblas_cout << std::endl;
        }
    }
}
