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

template <typename T>
void testing_swap_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_swap_batched_fn
        = FORTRAN ? rocblas_swap_batched<T, true> : rocblas_swap_batched<T, false>;

    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    rocblas_int incy        = 1;
    rocblas_int batch_count = 1;

    rocblas_local_handle handle;

    device_batch_vector<T> dxt(N, incx, batch_count);
    device_batch_vector<T> dyt(N, incy, batch_count);
    CHECK_DEVICE_ALLOCATION(dxt.memcheck());
    CHECK_DEVICE_ALLOCATION(dyt.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_swap_batched_fn(handle, N, nullptr, incx, dyt.ptr_on_device(), incy, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_swap_batched_fn(handle, N, dxt.ptr_on_device(), incx, nullptr, incy, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_swap_batched_fn(
            nullptr, N, dxt.ptr_on_device(), incx, dyt.ptr_on_device(), incy, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_swap_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_swap_batched_fn
        = FORTRAN ? rocblas_swap_batched<T, true> : rocblas_swap_batched<T, false>;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_swap_batched_fn(handle, N, nullptr, incx, nullptr, incy, batch_count),
            rocblas_status_success);
        return;
    }

    ssize_t abs_incx = incx >= 0 ? incx : -incx;
    ssize_t abs_incy = incy >= 0 ? incy : -incy;

    size_t size_x = N * abs_incx;
    size_t size_y = N * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hx_gold(N, incx, batch_count);
    host_batch_vector<T> hy_gold(N, incy, batch_count);

    // Initial Data on CPU
    rocblas_init(hx, true);
    for(int i = 0; i < batch_count; i++)
    {
        // make hy different to hx
        for(size_t j = 0; j < N; j++)
        {
            hy[i][j * abs_incy] = hx[i][j * abs_incx] + 1.0;
        }
    }

    hx_gold.copy_from(hx); // swapped later by cblas_swap
    hy_gold.copy_from(hy);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_swap_batched_fn(
            handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count));

        // copy data from device to CPU
        CHECK_HIP_ERROR(hx.transfer_from(dx));
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_swap<T>(N, hx_gold[i], incx, hy_gold[i], incy);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incx, hx_gold, hx, batch_count);
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incx, hx_gold, hx, batch_count);
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_swap_batched_fn(
                handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap_batched_fn(
                handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocblas_cout << "N,incx,incy,batch_count,rocblas-us" << std::endl;
        rocblas_cout << N << "," << incx << "," << incy << "," << batch_count << ","
                     << gpu_time_used << std::endl;
    }
}
