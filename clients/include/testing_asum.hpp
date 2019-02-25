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
#include "unit.hpp"
#include <complex.h>

template <typename T1, typename T2 = T1>
void testing_asum_bad_arg(const Arguments& arg)
{
    rocblas_int N                 = 100;
    rocblas_int incx              = 1;
    static const size_t safe_size = 100;
    T2 rocblas_result             = 10;
    T2* h_rocblas_result          = &rocblas_result;

    rocblas_local_handle handle;
    device_vector<T1> dx(safe_size);
    if(!dx)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS((rocblas_asum<T1, T2>(handle, N, nullptr, incx, h_rocblas_result)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_asum<T1, T2>(handle, N, dx, incx, nullptr)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_asum<T1, T2>(nullptr, N, dx, incx, h_rocblas_result)),
                          rocblas_status_invalid_handle);
}

template <typename T1, typename T2 = T1>
void testing_asum(const Arguments& arg)
{
    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;

    T2 rocblas_result_1;
    T2 rocblas_result_2;
    T2 cpu_result;
    double rocblas_error_1;
    double rocblas_error_2;
    rocblas_local_handle handle;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T1> dx(safe_size);
        device_vector<T2> d_rocblas_result_2(1);
        if(!dx || !d_rocblas_result_2)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_asum<T1, T2>(handle, N, dx, incx, d_rocblas_result_2)));
        return;
    }

    size_t size_x = N * static_cast<size_t>(incx);

    // allocate memory on device
    device_vector<T1> dx(size_x);
    device_vector<T2> d_rocblas_result_2(1);
    if(!dx || !d_rocblas_result_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    host_vector<T1> hx(size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T1>(hx, 1, N, incx);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T1) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_asum<T1, T2>(handle, N, dx, incx, &rocblas_result_1)));

        // GPU BLAS rocblas_pointer_mode_device
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T1) * size_x, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_asum<T1, T2>(handle, N, dx, incx, d_rocblas_result_2)));
        CHECK_HIP_ERROR(
            hipMemcpy(&rocblas_result_2, d_rocblas_result_2, sizeof(T1), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_asum<T1, T2>(N, hx, incx, &cpu_result);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T2>(1, 1, 1, &cpu_result, &rocblas_result_1);
            unit_check_general<T2>(1, 1, 1, &cpu_result, &rocblas_result_2);
        }

        if(arg.norm_check)
        {
            printf("cpu=%e, gpu_host_ptr,=%e, gup_dev_ptr=%e\n",
                   cpu_result,
                   rocblas_result_1,
                   rocblas_result_2);
            rocblas_error_1 = fabs((cpu_result - rocblas_result_1) / cpu_result);
            rocblas_error_2 = fabs((cpu_result - rocblas_result_2) / cpu_result);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_asum<T1, T2>(handle, N, dx, incx, &rocblas_result_1);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_asum<T1, T2>(handle, N, dx, incx, &rocblas_result_1);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,rocblas(us)";

        if(arg.norm_check)
            std::cout << ",CPU(us),error_host_ptr,error_dev_ptr";

        std::cout << std::endl;
        std::cout << N << "," << incx << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cpu_time_used << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}
