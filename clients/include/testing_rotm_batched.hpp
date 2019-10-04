/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
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
    rocblas_int         N           = 100;
    rocblas_int         incx        = 1;
    rocblas_int         incy        = 1;
    rocblas_int         batch_count = 5;
    static const size_t safe_size   = 100;

    rocblas_local_handle    handle;
    device_vector<T*, 0, T> dx(safe_size);
    device_vector<T*, 0, T> dy(safe_size);
    device_vector<T*, 0, T> dparam(safe_size);
    if(!dx || !dy || !dparam)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotm_batched<T>(nullptr, N, dx, incx, dy, incy, dparam, batch_count)),
        rocblas_status_invalid_handle);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotm_batched<T>(handle, N, nullptr, incx, dy, incy, dparam, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotm_batched<T>(handle, N, dx, incx, nullptr, incy, dparam, batch_count)),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        (rocblas_rotm_batched<T>(handle, N, dx, incx, dy, incy, nullptr, batch_count)),
        rocblas_status_invalid_pointer);
}

template <typename T>
void testing_rotm_batched(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;
    double               gpu_time_used, cpu_time_used;
    double norm_error_host_x = 0.0, norm_error_host_y = 0.0, norm_error_device_x = 0.0,
           norm_error_device_y = 0.0;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0 || batch_count <= 0)
    {
        static const size_t     safe_size = 100; // arbitrarily set to 100
        device_vector<T*, 0, T> dx(safe_size);
        device_vector<T*, 0, T> dy(safe_size);
        device_vector<T*, 0, T> dparam(safe_size);
        if(!dx || !dy || !dparam)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        if(batch_count < 0)
            EXPECT_ROCBLAS_STATUS(
                (rocblas_rotm_batched<T>(handle, N, dx, incx, dy, incy, dparam, batch_count)),
                rocblas_status_invalid_size);
        else
            CHECK_ROCBLAS_ERROR(
                (rocblas_rotm_batched<T>(handle, N, dx, incx, dy, incy, dparam, batch_count)));
        return;
    }

    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dy(batch_count);
    device_vector<T*, 0, T> dparam(batch_count);

    if(!dx || !dy || !dparam)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    host_vector<T> hx[batch_count];
    host_vector<T> hy[batch_count];
    host_vector<T> hdata[batch_count]; //(4);
    host_vector<T> hparam[batch_count]; //(5);

    device_batch_vector<T> bx(batch_count, size_x);
    device_batch_vector<T> by(batch_count, size_y);
    device_batch_vector<T> bdata(batch_count, 4);
    device_batch_vector<T> bparam(batch_count, 5);

    for(int b = 0; b < batch_count; b++)
    {
        hx[b]     = host_vector<T>(size_x);
        hy[b]     = host_vector<T>(size_y);
        hdata[b]  = host_vector<T>(4);
        hparam[b] = host_vector<T>(5);
    }

    int last = batch_count - 1;
    if((!bx[last] && size_x) || (!by[last] && size_y) || !bdata[last] || !bparam[last])
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocblas_seedrand();
    for(int b = 0; b < batch_count; b++)
    {
        rocblas_init<T>(hx[b], 1, N, incx);
        rocblas_init<T>(hy[b], 1, N, incy);
        rocblas_init<T>(hdata[b], 1, 4, 1);

        // CPU BLAS reference data
        cblas_rotmg<T>(&hdata[b][0], &hdata[b][1], &hdata[b][2], &hdata[b][3], hparam[b]);
    }

    constexpr int FLAG_COUNT        = 4;
    const T       FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};

    for(int i = 0; i < FLAG_COUNT; i++)
    {
        for(int b = 0; b < batch_count; b++)
            hparam[b][0] = FLAGS[i];

        host_vector<T> cx[batch_count];
        host_vector<T> cy[batch_count];
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; b++)
        {
            cx[b] = hx[b];
            cy[b] = hy[b];

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
            //         (rocblas_rotm_batched<T>(handle, N, dx, incx, dy, incy, hparam, batch_count)));

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
            //         near_check_general<T>(1, N, batch_count, incx, cx, rx, rel_error);
            //         near_check_general<T>(1, N, batch_count, incy, cy, ry, rel_error);
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
                for(int b = 0; b < batch_count; b++)
                {
                    CHECK_HIP_ERROR(
                        hipMemcpy(bx[b], hx[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(
                        hipMemcpy(by[b], hy[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(
                        hipMemcpy(bparam[b], hparam[b], sizeof(T) * 5, hipMemcpyHostToDevice));
                }
                CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dy, by, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(
                    hipMemcpy(dparam, bparam, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

                CHECK_ROCBLAS_ERROR(
                    (rocblas_rotm_batched<T>(handle, N, dx, incx, dy, incy, dparam, batch_count)));

                host_vector<T> rx[batch_count];
                host_vector<T> ry[batch_count];
                for(int b = 0; b < batch_count; b++)
                {
                    rx[b] = host_vector<T>(size_x);
                    ry[b] = host_vector<T>(size_y);
                    CHECK_HIP_ERROR(
                        hipMemcpy(rx[b], bx[b], sizeof(T) * size_x, hipMemcpyDeviceToHost));
                    CHECK_HIP_ERROR(
                        hipMemcpy(ry[b], by[b], sizeof(T) * size_y, hipMemcpyDeviceToHost));
                }

                if(arg.unit_check)
                {
                    T rel_error = std::numeric_limits<T>::epsilon() * 1000;
                    near_check_general<T>(1, N, batch_count, incx, cx, rx, rel_error);
                    near_check_general<T>(1, N, batch_count, incy, cy, ry, rel_error);
                }
                if(arg.norm_check)
                {
                    norm_error_device_x
                        = norm_check_general<T>('F', 1, N, batch_count, incx, cx, rx);
                    norm_error_device_y
                        = norm_check_general<T>('F', 1, N, batch_count, incy, cy, ry);
                }
            }
        }

        if(arg.timing)
        {
            int number_cold_calls = 2;
            int number_hot_calls  = 100;
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            for(int b = 0; b < batch_count; b++)
            {
                CHECK_HIP_ERROR(hipMemcpy(bx[b], hx[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(by[b], hy[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(
                    hipMemcpy(bparam[b], hparam[b], sizeof(T) * 5, hipMemcpyHostToDevice));
            }
            CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, by, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(
                hipMemcpy(dparam, bparam, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

            for(int iter = 0; iter < number_cold_calls; iter++)
            {
                rocblas_rotm_batched<T>(handle, N, dx, incx, dy, incy, dparam, batch_count);
            }
            gpu_time_used = get_time_us(); // in microseconds
            for(int iter = 0; iter < number_hot_calls; iter++)
            {
                rocblas_rotm_batched<T>(handle, N, dx, incx, dy, incy, dparam, batch_count);
            }
            gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

            std::cout << "N,incx,incy,rocblas(us),cpu(us)";
            if(arg.norm_check)
                std::cout << ",norm_error_host_x,norm_error_host_y,norm_error_device_x,norm_error_"
                             "device_y";
            std::cout << std::endl;
            std::cout << N << "," << incx << "," << incy << "," << gpu_time_used << ","
                      << cpu_time_used;
            if(arg.norm_check)
                std::cout << ',' << norm_error_host_x << ',' << norm_error_host_y << ","
                          << norm_error_device_x << "," << norm_error_device_y;
            std::cout << std::endl;
        }
    }
}
