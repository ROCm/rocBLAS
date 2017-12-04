/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "rocblas.hpp"
#include "rocblas_test_unique_ptr.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "arg_check.h"
#include <complex.h>

using namespace std;

template <typename T1, typename T2>
void testing_asum_bad_arg()
{
    rocblas_int N         = 100;
    rocblas_int incx      = 1;
    rocblas_int safe_size = 100;
    T2 rocblas_result     = 10;
    T2* h_rocblas_result;
    h_rocblas_result = &rocblas_result;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T1) * safe_size),
                                         rocblas_test::device_free};
    T1* dx = (T1*)dx_managed.get();
    if(!dx)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing for (nullptr == dx)
    {
        T1* dx_null = nullptr;

        status = rocblas_asum<T1, T2>(handle, N, dx_null, incx, h_rocblas_result);

        verify_rocblas_status_invalid_pointer(status, "Error: dx is nullptr");
    }
    // testing for (nullptr == d_rocblas_result)
    {
        T2* h_rocblas_result_null = nullptr;

        status = rocblas_asum<T1, T2>(handle, N, dx, incx, h_rocblas_result_null);

        verify_rocblas_status_invalid_pointer(status, "Error: result is nullptr");
    }
    // testing for ( nullptr == handle )
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_asum<T1, T2>(handle_null, N, dx, incx, h_rocblas_result);

        verify_rocblas_status_invalid_handle(status);
    }
}

template <typename T1, typename T2>
rocblas_status testing_asum(Arguments argus)
{
    rocblas_int N         = argus.N;
    rocblas_int incx      = argus.incx;
    rocblas_int safe_size = 100; // arbitrarily set to 100

    T2 rocblas_result_1;
    T2 rocblas_result_2;
    T2 cpu_result;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T1) * safe_size),
                                             rocblas_test::device_free};
        auto d_rocblas_result_managed_2 =
            rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T2)), rocblas_test::device_free};
        T1* dx                 = (T1*)dx_managed.get();
        T2* d_rocblas_result_2 = (T2*)d_rocblas_result_managed_2.get();
        if(!dx || !d_rocblas_result_2)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_asum<T1, T2>(handle, N, dx, incx, d_rocblas_result_2);

        asum_arg_check(status, d_rocblas_result_2);

        return status;
    }

    rocblas_int size_x = N * incx;

    // allocate memory on device
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T1) * size_x),
                                         rocblas_test::device_free};
    auto d_rocblas_result_managed_2 =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T2)), rocblas_test::device_free};
    T1* dx                 = (T1*)dx_managed.get();
    T2* d_rocblas_result_2 = (T2*)d_rocblas_result_managed_2.get();
    if(!dx || !d_rocblas_result_2)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    vector<T1> hx(size_x);

    // Initial Data on CPU
    srand(1);
    rocblas_init<T1>(hx, 1, N, incx);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T1) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;

    if(argus.unit_check || argus.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_asum<T1, T2>(handle, N, dx, incx, &rocblas_result_1)));

        // GPU BLAS rocblas_pointer_mode_device
        CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T1) * size_x, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_asum<T1, T2>(handle, N, dx, incx, d_rocblas_result_2)));
        CHECK_HIP_ERROR(
            hipMemcpy(&rocblas_result_2, d_rocblas_result_2, sizeof(T1), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_asum<T1, T2>(N, hx.data(), incx, &cpu_result);
        cpu_time_used = get_time_us() - cpu_time_used;

        if(argus.unit_check)
        {
            unit_check_general<T2>(1, 1, 1, &cpu_result, &rocblas_result_1);
            unit_check_general<T2>(1, 1, 1, &cpu_result, &rocblas_result_2);
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            printf("cpu=%e, gpu_host_ptr,=%e, gup_dev_ptr=%e\n",
                   cpu_result,
                   rocblas_result_1,
                   rocblas_result_2);
            rocblas_error_1 = fabs((cpu_result - rocblas_result_1) / cpu_result);
            rocblas_error_2 = fabs((cpu_result - rocblas_result_2) / cpu_result);
        }
    }

    if(argus.timing)
    {
        int number_timing_iterations = 1;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_timing_iterations; iter++)
        {
            rocblas_asum<T1, T2>(handle, N, dx, incx, &rocblas_result_1);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_timing_iterations;

        cout << "N,rocblas(us)";

        if(argus.norm_check)
            cout << ",CPU(us),error_host_ptr,error_dev_ptr";

        cout << endl;
        cout << N << "," << gpu_time_used;

        if(argus.norm_check)
            cout << "," << cpu_time_used << "," << rocblas_error_1 << "," << rocblas_error_2;

        cout << endl;
    }

    return rocblas_status_success;
}
