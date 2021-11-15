/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
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
void testing_swap_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_swap_strided_batched_fn
        = FORTRAN ? rocblas_swap_strided_batched<T, true> : rocblas_swap_strided_batched<T, false>;

    rocblas_int    N           = 100;
    rocblas_int    incx        = 1;
    rocblas_int    incy        = 1;
    rocblas_stride stride_x    = 1;
    rocblas_stride stride_y    = 1;
    rocblas_int    batch_count = 5;

    static const size_t safe_size = 100; //  arbitrarily set to 100

    rocblas_local_handle handle{arg};

    // allocate memory on device
    device_vector<T> dx(safe_size);
    device_vector<T> dy(safe_size);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched_fn(
                              handle, N, nullptr, incx, stride_x, dy, incy, stride_y, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched_fn(
                              handle, N, dx, incx, stride_x, nullptr, incy, stride_y, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched_fn(
                              nullptr, N, dx, incx, stride_x, dy, incy, stride_y, batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_swap_strided_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_swap_strided_batched_fn
        = FORTRAN ? rocblas_swap_strided_batched<T, true> : rocblas_swap_strided_batched<T, false>;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_int    incy        = arg.incy;
    rocblas_stride stride_x    = arg.stride_x;
    rocblas_stride stride_y    = arg.stride_y;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_swap_strided_batched_fn(
                handle, N, nullptr, incx, stride_x, nullptr, incy, stride_y, batch_count),
            rocblas_status_success);
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    size_t size_x = (size_t)(stride_x >= 0 ? stride_x : -stride_x);
    size_t size_y = (size_t)(stride_y >= 0 ? stride_y : -stride_y);
    // not testing non-standard strides
    size_x = std::max(size_x, N * abs_incx);
    size_y = std::max(size_y, N * abs_incy);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x * batch_count);
    host_vector<T> hy(size_y * batch_count);
    host_vector<T> hx_gold(size_x * batch_count);
    host_vector<T> hy_gold(size_y * batch_count);

    // Initialize the host vector.
    rocblas_init_vector(hx, arg, N, abs_incx, size_x, batch_count, true);
    rocblas_init_vector(hy, arg, N, abs_incy, size_y, batch_count, false);

    hx_gold = hx;
    hy_gold = hy;
    // using cpu BLAS to compute swap gold later on

    // allocate memory on device
    device_vector<T> dx(size_x * batch_count);
    device_vector<T> dy(size_y * batch_count);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    size_t dataSizeX = sizeof(T) * size_x * batch_count;
    size_t dataSizeY = sizeof(T) * size_y * batch_count;

    // copy vector data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, dataSizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, dataSizeY, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_swap_strided_batched_fn(
            handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(hx, dx, dataSizeX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy, dy, dataSizeY, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_swap<T>(N, hx_gold + i * stride_x, incx, hy_gold + i * stride_y, incy);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incx, stride_x, hx_gold, hx, batch_count);
            unit_check_general<T>(1, N, abs_incy, stride_y, hy_gold, hy, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', 1, N, abs_incx, stride_x, hx_gold, hx, batch_count);
            rocblas_error
                = norm_check_general<T>('F', 1, N, abs_incy, stride_y, hy_gold, hy, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_swap_strided_batched_fn(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap_strided_batched_fn(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            ArgumentLogging::NA_value,
            swap_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error);
    }
}
