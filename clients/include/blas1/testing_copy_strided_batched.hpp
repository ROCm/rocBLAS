/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

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

template <typename T>
void testing_copy_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_copy_strided_batched_fn = arg.fortran ? rocblas_copy_strided_batched<T, true>
                                                       : rocblas_copy_strided_batched<T, false>;

    rocblas_int    N           = 100;
    rocblas_int    incx        = 1;
    rocblas_int    incy        = 1;
    rocblas_stride stride_x    = incx * N;
    rocblas_stride stride_y    = incy * N;
    rocblas_int    batch_count = 5;

    rocblas_local_handle handle{arg};

    size_t size_x = stride_x * batch_count;
    size_t size_y = stride_y * batch_count;

    // Allocate device memory
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_copy_strided_batched_fn(
                              handle, N, nullptr, incx, stride_x, dy, incy, stride_y, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_copy_strided_batched_fn(
                              handle, N, dx, incx, stride_x, nullptr, incy, stride_y, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_copy_strided_batched_fn(
                              nullptr, N, dx, incx, stride_x, dy, incy, stride_y, batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_copy_strided_batched(const Arguments& arg)
{
    auto rocblas_copy_strided_batched_fn = arg.fortran ? rocblas_copy_strided_batched<T, true>
                                                       : rocblas_copy_strided_batched<T, false>;

    rocblas_int          N           = arg.N;
    rocblas_int          incx        = arg.incx;
    rocblas_int          incy        = arg.incy;
    rocblas_int          stride_x    = arg.stride_x;
    rocblas_int          stride_y    = arg.stride_y;
    rocblas_int          batch_count = arg.batch_count;
    rocblas_local_handle handle{arg};
    rocblas_int          abs_incy = incy >= 0 ? incy : -incy;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_copy_strided_batched_fn(
                handle, N, nullptr, incx, stride_x, nullptr, incy, stride_y, batch_count),
            rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_strided_batch_vector<T> hx(N, incx ? incx : 1, stride_x, batch_count);
    host_strided_batch_vector<T> hy(N, incy ? incy : 1, stride_y, batch_count);
    host_strided_batch_vector<T> hy_gold(N, incy ? incy : 1, stride_y, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());

    // Allocate device memory
    device_strided_batch_vector<T> dx(N, incx ? incx : 1, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy ? incy : 1, stride_y, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);

    hy_gold.copy_from(hy);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_copy_strided_batched_fn(
            handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_copy<T>(N, hx[b], incx, hy_gold[b], incy);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, stride_y, hy_gold, hy, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', 1, N, abs_incy, stride_y, hy_gold, hy, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_copy_strided_batched_fn(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_copy_strided_batched_fn(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            ArgumentLogging::NA_value,
            copy_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error);
    }
}
