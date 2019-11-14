/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
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

template <typename T>
void testing_swap_batched_bad_arg(const Arguments& arg)
{
    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    rocblas_int incy        = 1;
    rocblas_int batch_count = 1;

    rocblas_local_handle handle;

    device_vector<T*, 0, T> dxt(1);
    device_vector<T*, 0, T> dyt(1);
    if(!dxt || !dyt)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(handle, N, nullptr, incx, dyt, incy, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(handle, N, dxt, incx, nullptr, incy, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(nullptr, N, dxt, incx, dyt, incy, batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_swap_batched(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        static const size_t     safe_size = 100; //  arbitrarily set to 100
        device_vector<T*, 0, T> dxt(safe_size);
        device_vector<T*, 0, T> dyt(safe_size);
        if(!dxt || !dyt)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(rocblas_swap_batched<T>(handle, N, dxt, incx, dyt, incy, batch_count),
                              N > 0 && batch_count < 0 ? rocblas_status_invalid_size
                                                       : rocblas_status_success);
        return;
    }

    ssize_t abs_incx = incx >= 0 ? incx : -incx;
    ssize_t abs_incy = incy >= 0 ? incy : -incy;

    size_t size_x = N * abs_incx;
    size_t size_y = N * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx[batch_count];
    host_vector<T> hy[batch_count];
    host_vector<T> hx_gold[batch_count];
    host_vector<T> hy_gold[batch_count];

    for(int i = 0; i < batch_count; i++)
    {
        hx[i]      = host_vector<T>(size_x);
        hy[i]      = host_vector<T>(size_y);
        hx_gold[i] = host_vector<T>(size_x);
        hy_gold[i] = host_vector<T>(size_y);
    }

    // Initial Data on CPU
    rocblas_seedrand();
    for(int i = 0; i < batch_count; i++)
    {
        rocblas_init<T>(hx[i], 1, N, abs_incx);
        // make hy different to hx
        for(size_t j = 0; j < N; j++)
        {
            hy[i][j * abs_incy] = hx[i][j * abs_incx] + 1.0;
        }
        hx_gold[i] = hx[i]; // swapped later by cblas_swap
        hy_gold[i] = hy[i];
    }

    device_batch_vector<T> dxvec(batch_count, size_x);
    device_batch_vector<T> dyvec(batch_count, size_y);

    // copy data from host to device
    for(int i = 0; i < batch_count; i++)
    {
        CHECK_HIP_ERROR(hipMemcpy(dxvec[i], hx[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dyvec[i], hy[i], sizeof(T) * size_y, hipMemcpyHostToDevice));
    }

    // vector pointers on gpu
    device_vector<T*, 0, T> dx_pvec(batch_count);
    device_vector<T*, 0, T> dy_pvec(batch_count);
    if(!dx_pvec || !dy_pvec)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy gpu vector pointers from host to device pointer array
    CHECK_HIP_ERROR(hipMemcpy(dx_pvec, dxvec, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_pvec, dyvec, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(
            rocblas_swap_batched<T>(handle, N, dx_pvec, incx, dy_pvec, incy, batch_count));

        // copy data from device to CPU
        for(int i = 0; i < batch_count; i++)
        {
            CHECK_HIP_ERROR(hipMemcpy(hx[i], dxvec[i], sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hy[i], dyvec[i], sizeof(T) * size_y, hipMemcpyDeviceToHost));
        }

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_swap<T>(N, hx_gold[i], incx, hy_gold[i], incy);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            for(int i = 0; i < batch_count; i++)
            {
                unit_check_general<T>(1, N, abs_incx, hx_gold[i], hx[i]);
                unit_check_general<T>(1, N, abs_incy, hy_gold[i], hy[i]);
            }
        }

        if(arg.norm_check)
        {
            for(int i = 0; i < batch_count; i++)
            {
                rocblas_error = norm_check_general<T>('F', 1, N, abs_incx, hx_gold[i], hx[i]);
                rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold[i], hy[i]);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_swap_batched<T>(handle, N, dx_pvec, incx, dy_pvec, incy, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap_batched<T>(handle, N, dx_pvec, incx, dy_pvec, incy, batch_count);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,incy,batch_count,rocblas-us" << std::endl;
        std::cout << N << "," << incx << "," << incy << "," << batch_count << "," << gpu_time_used
                  << std::endl;
    }
}
