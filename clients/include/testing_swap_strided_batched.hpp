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
void testing_swap_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_swap_strided_batched_fn
        = FORTRAN ? rocblas_swap_strided_batched<T, true> : rocblas_swap_strided_batched<T, false>;

    rocblas_int    N           = 100;
    rocblas_int    incx        = 1;
    rocblas_int    incy        = 1;
    rocblas_stride stridex     = 1;
    rocblas_stride stridey     = 1;
    rocblas_int    batch_count = 5;

    static const size_t safe_size = 100; //  arbitrarily set to 100

    rocblas_local_handle handle;

    // allocate memory on device
    device_vector<T> dx(safe_size);
    device_vector<T> dy(safe_size);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched_fn(
                              handle, N, nullptr, incx, stridex, dy, incy, stridey, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched_fn(
                              handle, N, dx, incx, stridex, nullptr, incy, stridey, batch_count),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_swap_strided_batched_fn(
                              nullptr, N, dx, incx, stridex, dy, incy, stridey, batch_count),
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
    rocblas_stride stridex     = arg.stride_x;
    rocblas_stride stridey     = arg.stride_y;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_swap_strided_batched_fn(
                handle, N, nullptr, incx, stridex, nullptr, incy, stridey, batch_count),
            rocblas_status_success);
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    size_t size_x = (size_t)(stridex >= 0 ? stridex : -stridex);
    size_t size_y = (size_t)(stridey >= 0 ? stridey : -stridey);
    // not testing non-standard strides
    size_x = std::max(size_x, N * abs_incx);
    size_y = std::max(size_y, N * abs_incy);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x * batch_count);
    host_vector<T> hy(size_y * batch_count);
    host_vector<T> hx_gold(size_x * batch_count);
    host_vector<T> hy_gold(size_y * batch_count);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, abs_incx, size_x, batch_count);
    rocblas_init<T>(hy, 1, N, abs_incy, size_y, batch_count);

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
            handle, N, dx, incx, stridex, dy, incy, stridey, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(hx, dx, dataSizeX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy, dy, dataSizeY, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_swap<T>(N, hx_gold + i * stridex, incx, hy_gold + i * stridey, incy);
        }
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incx, stridex, hx_gold, hx, batch_count);
            unit_check_general<T>(1, N, abs_incy, stridey, hy_gold, hy, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', 1, N, abs_incx, stridex, hx_gold, hx, batch_count);
            rocblas_error
                = norm_check_general<T>('F', 1, N, abs_incy, stridey, hy_gold, hy, batch_count);
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
                handle, N, dx, incx, stridex, dy, incy, stridey, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap_strided_batched_fn(
                handle, N, dx, incx, stridex, dy, incy, stridey, batch_count);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocblas_cout << "N,incx,incy,stride_x,stride_y,batch_count,rocblas-us" << std::endl;
        rocblas_cout << N << "," << incx << "," << incy << "," << stridex << "," << stridey << ","
                     << batch_count << "," << gpu_time_used << std::endl;
    }
}
