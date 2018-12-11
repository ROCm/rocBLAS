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
#include "flops.hpp"

/* ============================================================================================ */
template <typename T>
void testing_axpy_bad_arg(const Arguments& arg)
{
    rocblas_int N                 = 100;
    rocblas_int incx              = 1;
    rocblas_int incy              = 1;
    static const size_t safe_size = 100;
    float alpha_float             = 0.6;
    T alpha;

    if(std::is_same<T, rocblas_half>::value)
        alpha = float_to_half(alpha_float);
    else
        alpha = alpha_float;

    rocblas_local_handle handle;
    device_vector<T> dx(safe_size);
    device_vector<T> dy(safe_size);
    if(!dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_axpy<T>(handle, N, &alpha, nullptr, incx, dy, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_axpy<T>(handle, N, &alpha, dx, incx, nullptr, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_axpy<T>(handle, N, nullptr, dx, incx, dy, incy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_axpy<T>(nullptr, N, &alpha, dx, incx, dy, incy),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_axpy(const Arguments& arg)
{
    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;
    rocblas_int incy = arg.incy;
    T h_alpha;
    if(std::is_same<T, rocblas_half>::value)
        h_alpha = float_to_half(arg.alpha);
    else
        h_alpha = arg.alpha;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(N <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T> dx(safe_size);
        device_vector<T> dy(safe_size);
        if(!dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_axpy<T>(handle, N, &h_alpha, dx, incx, dy, incy));
        return;
    }

    rocblas_int abs_incx = incx > 0 ? incx : -incx;
    rocblas_int abs_incy = incy > 0 ? incy : -incy;
    size_t size_x        = N * static_cast<size_t>(abs_incx);
    size_t size_y        = N * static_cast<size_t>(abs_incy);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x);
    host_vector<T> hy_1(size_y);
    host_vector<T> hy_2(size_y);
    host_vector<T> hy_gold(size_y);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy_1, 1, N, abs_incy);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    // BLAS
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    device_vector<T> dx(size_x);
    device_vector<T> dy_1(size_y);
    device_vector<T> dy_2(size_y);
    device_vector<T> d_alpha(1);
    if(!dx || !dy_1 || !dy_2 || !d_alpha)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        // ROCBLAS pointer mode host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_axpy<T>(handle, N, &h_alpha, dx, incx, dy_1, incy));

        // ROCBLAS pointer mode device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_axpy<T>(handle, N, d_alpha, dx, incx, dy_2, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * size_y, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_axpy<T>(N, h_alpha, hx, incx, hy_gold, incy);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = axpy_gflop_count<T>(N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_1);
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_1);
            rocblas_error_2 = norm_check_general<T>('F', 1, N, abs_incy, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_axpy<T>(handle, N, &h_alpha, dx, incx, dy_1, incy);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_axpy<T>(handle, N, &h_alpha, dx, incx, dy_1, incy);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = axpy_gflop_count<T>(N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (3.0 * N) * sizeof(T) / gpu_time_used / 1e3;

        std::cout << "N,alpha,incx,incy,rocblas-Gflops,rocblas-GB/s,rocblas-us";

        if(arg.norm_check)
            std::cout << "CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << N << "," << h_alpha << "," << incx << "," << incy << "," << rocblas_gflops
                  << "," << rocblas_bandwidth << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << ',' << rocblas_error_1 << ',' << rocblas_error_2;

        std::cout << std::endl;
    }
}
