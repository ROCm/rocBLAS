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

template <typename T>
void testing_copy_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_copy_strided_batched_fn
        = FORTRAN ? rocblas_copy_strided_batched<T, true> : rocblas_copy_strided_batched<T, false>;

    rocblas_int    N           = 100;
    rocblas_int    incx        = 1;
    rocblas_int    incy        = 1;
    rocblas_stride stride_x    = incx * N;
    rocblas_stride stride_y    = incy * N;
    rocblas_int    batch_count = 5;

    rocblas_local_handle handle;

    size_t size_x = stride_x * batch_count;
    size_t size_y = stride_y * batch_count;

    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
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
    const bool FORTRAN = arg.fortran;
    auto       rocblas_copy_strided_batched_fn
        = FORTRAN ? rocblas_copy_strided_batched<T, true> : rocblas_copy_strided_batched<T, false>;

    rocblas_int          N           = arg.N;
    rocblas_int          incx        = arg.incx;
    rocblas_int          incy        = arg.incy;
    rocblas_int          stride_x    = arg.stride_x;
    rocblas_int          stride_y    = arg.stride_y;
    rocblas_int          batch_count = arg.batch_count;
    rocblas_local_handle handle;
    rocblas_int          abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int          abs_incy = incy >= 0 ? incy : -incy;

    size_t size_x = size_t(stride_x) * size_t(batch_count);
    size_t size_y = size_t(stride_y) * size_t(batch_count);
    if(!size_x)
        size_x = batch_count;
    if(!size_y)
        size_y = batch_count;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_copy_strided_batched_fn(
                handle, N, nullptr, incx, stride_x, nullptr, incy, stride_y, batch_count),
            rocblas_status_success);
        return;
    }

    // allocate memory on device
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<T> hy_gold(size_y);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, abs_incx, stride_x, batch_count);
    rocblas_init<T>(hy, 1, N, abs_incy, stride_y, batch_count);

    // copy_strided_batched vector is easy in STL; hy_gold = hx: save a copy_strided_batched in hy_gold which will be output of CPU
    // BLAS
    hy_gold = hy;

    // copy_strided_batched data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_copy_strided_batched_fn(
            handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(hy, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_copy<T>(N, hx + b * stride_x, incx, hy_gold + b * stride_y, incy);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

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

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_copy_strided_batched_fn(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count);
        }

        gpu_time_used = get_time_us() - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            copy_gflop_count<T>(N),
            copy_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error);
    }
}
