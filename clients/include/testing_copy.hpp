/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "utility.hpp"
#include "rocblas.hpp"
#include "cblas_interface.hpp"
#include "norm.hpp"
#include "unit.hpp"

template <typename T>
void testing_copy_bad_arg(const Arguments& arg)
{
    rocblas_int N                 = 100;
    rocblas_int incx              = 1;
    rocblas_int incy              = 1;
    static const size_t safe_size = 100; //  arbitrarily set to 100

    rocblas_local_handle handle;
    device_vector<T> dx(safe_size);
    device_vector<T> dy(safe_size);
    if(!dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_copy<T>(handle, N, nullptr, incx, dy, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_copy<T>(handle, N, dx, incx, nullptr, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_copy<T>(nullptr, N, dx, incx, dy, incy),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_copy(const Arguments& arg)
{
    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;
    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N <= 0)
    {
        static const size_t safe_size = 100; //  arbitrarily set to 100
        device_vector<T> dx(safe_size);
        device_vector<T> dy(safe_size);
        if(!dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_copy<T>(handle, N, dx, incx, dy, incy));
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x        = N * static_cast<size_t>(abs_incx);
    size_t size_y        = N * static_cast<size_t>(abs_incy);

    // allocate memory on device
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    if(!dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<T> hy_gold(size_y);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    // BLAS
    hy_gold = hy;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_copy<T>(handle, N, dx, incx, dy, incy));
        CHECK_HIP_ERROR(hipMemcpy(hy, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_copy<T>(N, hx, incx, hy_gold, incy);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_copy<T>(handle, N, dx, incx, dy, incy);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_copy<T>(handle, N, dx, incx, dy, incy);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,incy,rocblas-us";

        if(arg.norm_check)
            std::cout << ",CPU-us,error";

        std::cout << std::endl;

        std::cout << N << "," << incx << "," << incy << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cpu_time_used << "," << rocblas_error;

        std::cout << std::endl;
    }
}
