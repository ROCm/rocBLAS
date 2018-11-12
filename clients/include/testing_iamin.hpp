/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "rocblas.hpp"
#include "arg_check.h"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include <complex.h>

using namespace std;

template <typename T>
void testing_iamin_bad_arg()
{
    rocblas_int N         = 100;
    rocblas_int incx      = 1;
    rocblas_int safe_size = 100;

    rocblas_status status;

    rocblas_local_handle handle;
    device_vector<T> dx(safe_size);
    if(!dx)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocblas_int h_rocblas_result;

    // testing for (nullptr == dx)
    {
        T* dx_null = nullptr;

        status = rocblas_iamin<T>(handle, N, dx_null, incx, &h_rocblas_result);

        verify_rocblas_status_invalid_pointer(status, "Error: x is nullptr");
    }
    // testing for (nullptr == h_rocblas_result)
    {
        rocblas_int* h_rocblas_result_null = nullptr;

        status = rocblas_iamin<T>(handle, N, dx, incx, h_rocblas_result_null);

        verify_rocblas_status_invalid_pointer(status, "Error: result is nullptr");
    }
    // testing for (nullptr == handle)
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_iamin<T>(handle_null, N, dx, incx, &h_rocblas_result);

        verify_rocblas_status_invalid_handle(status);
    }
}

template <typename T>
rocblas_status testing_iamin(Arguments argus)
{
    rocblas_int N         = argus.N;
    rocblas_int incx      = argus.incx;
    rocblas_int safe_size = 100; // arbritrarily set to 100

    rocblas_int h_rocblas_result_1;
    rocblas_int h_rocblas_result_2;
    rocblas_int cpu_result;

    rocblas_int rocblas_error_1;
    rocblas_int rocblas_error_2;

    rocblas_status status;
    rocblas_local_handle handle;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        device_vector<T> dx(safe_size);
        if(!dx)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_iamin<T>(handle, N, dx, incx, &h_rocblas_result_1);

        iamax_iamin_arg_check(status, &h_rocblas_result_1);

        return status;
    }

    rocblas_int size_x = N * incx;

    // allocate memory on device
    device_vector<T> dx(size_x);
    device_vector<rocblas_int> d_rocblas_result(1);
    if(!dx || !d_rocblas_result)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x);
    host_vector<T> hx_negated(size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, incx);

    for(int i = 0; i < N; i++)
    {
        hx_negated[i] = hx[i];
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;

    if(argus.unit_check || argus.norm_check)
    {
        // GPU BLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_iamin<T>(handle, N, dx, incx, &h_rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_iamin<T>(handle, N, dx, incx, d_rocblas_result));
        CHECK_HIP_ERROR(hipMemcpy(
            &h_rocblas_result_2, d_rocblas_result, sizeof(rocblas_int), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_iamax<T>(N, hx_negated, incx, &cpu_result);

        cpu_time_used = get_time_us() - cpu_time_used;
        cpu_result += 1; // make index 1 based as in Fortran BLAS, not 0 based as in CBLAS

        if(argus.unit_check)
        {
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &h_rocblas_result_1);
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &h_rocblas_result_2);
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            rocblas_error_1 = h_rocblas_result_1 - cpu_result;
            rocblas_error_2 = h_rocblas_result_2 - cpu_result;
            verify_equal<rocblas_int>(h_rocblas_result_1, cpu_result, "iamin result check");
            verify_equal<rocblas_int>(h_rocblas_result_2, cpu_result, "iamin result check");
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_iamin<T>(handle, N, dx, incx, d_rocblas_result);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_iamin<T>(handle, N, dx, incx, d_rocblas_result);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        cout << "N,incx,rocblas-us";

        if(argus.norm_check)
            cout << ",cpu_time_used,rocblas_error_host_ptr,rocblas_error_dev_ptr";

        cout << endl;

        cout << (int)N << "," << incx << "," << gpu_time_used;

        if(argus.norm_check)
            cout << "," << cpu_time_used << "," << rocblas_error_1 << "," << rocblas_error_2;

        cout << endl;
    }

    return rocblas_status_success;
}
