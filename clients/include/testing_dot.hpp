/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "rocblas.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include <complex.h>

using namespace std;

template <typename T>
rocblas_status testing_dot_bad_arg()
{
    rocblas_int N         = 100;
    rocblas_int incx      = 1;
    rocblas_int incy      = 1;
    rocblas_int safe_size = 100; //  arbitrarily set to 100

    rocblas_status status;

    rocblas_local_handle handle;
    device_vector<T> dx(safe_size);
    device_vector<T> dy(safe_size);
    device_vector<T> d_rocblas_result(1);
    if(!dx || !dy || !d_rocblas_result)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // test if (nullptr == dx)
    {
        T* dx_null = nullptr;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_dot<T>(handle, N, dx_null, incx, dy, incy, d_rocblas_result);

        verify_rocblas_status_invalid_pointer(status, "Error: x, y, or result is nullptr");
    }
    // test if (nullptr == dy)
    {
        T* dy_null = nullptr;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_dot<T>(handle, N, dx, incx, dy_null, incy, d_rocblas_result);

        verify_rocblas_status_invalid_pointer(status, "Error: x, y, or result is nullptr");
    }
    // test if (nullptr == d_rocblas_result)
    {
        T* d_rocblas_result_null = nullptr;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_dot<T>(handle, N, dx, incx, dy, incy, d_rocblas_result_null);

        verify_rocblas_status_invalid_pointer(status, "Error: x, y, or result is nullptr");
    }
    // test if ( nullptr == handle )
    {
        rocblas_handle handle_null = nullptr;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_dot<T>(handle_null, N, dx, incx, dy, incy, d_rocblas_result);

        verify_rocblas_status_invalid_handle(status);
    }
    return rocblas_status_success;
}

template <typename T>
rocblas_status testing_dot(Arguments argus)
{
    rocblas_int N         = argus.N;
    rocblas_int incx      = argus.incx;
    rocblas_int incy      = argus.incy;
    rocblas_int safe_size = 100; // arbitrarily set to 100

    T cpu_result;
    T rocblas_result_1;
    T rocblas_result_2;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_status status;

    rocblas_local_handle handle;

    // check to prevent undefined memmory allocation error
    if(N <= 0)
    {
        device_vector<T> dx(safe_size);
        device_vector<T> dy(safe_size);
        device_vector<T> d_rocblas_result(1);
        if(!dx || !dy || !d_rocblas_result)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        status = rocblas_dot<T>(handle, N, dx, incx, dy, incy, d_rocblas_result);

        nrm2_dot_arg_check<T>(status, d_rocblas_result);

        return status;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    rocblas_int size_x   = N * abs_incx;
    rocblas_int size_y   = N * abs_incy;

    // allocate memory on device
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> d_rocblas_result_2(1);
    if(!dx || !dy || !d_rocblas_result_2)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;

    if(argus.unit_check || argus.norm_check)
    {
        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_dot<T>(handle, N, dx, incx, dy, incy, &rocblas_result_1));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_dot<T>(handle, N, dx, incx, dy, incy, d_rocblas_result_2));
        CHECK_HIP_ERROR(
            hipMemcpy(&rocblas_result_2, d_rocblas_result_2, sizeof(T), hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_dot<T>(N, hx, incx, hy, incy, &cpu_result);
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = axpy_gflop_count<T>(N) / cpu_time_used * 1e6 * 1;

        if(argus.unit_check)
        {
            unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_1);
            unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_2);
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            printf("cpu=%f, gpu_host_ptr=%f, gpu_device_ptr=%f\n",
                   cpu_result,
                   rocblas_result_1,
                   rocblas_result_2);
            rocblas_error_1 = fabs((cpu_result - rocblas_result_1) / cpu_result);
            rocblas_error_2 = fabs((cpu_result - rocblas_result_2) / cpu_result);
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_dot<T>(handle, N, dx, incx, dy, incy, &rocblas_result_1);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_dot<T>(handle, N, dx, incx, dy, incy, &rocblas_result_1);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = dot_gflop_count<T>(N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (2.0 * N) * sizeof(T) / gpu_time_used / 1e3;

        cout << "N,incx,incy,rocblas-Gflops,rocblas-GB/s,rocblas-us";

        if(argus.norm_check)
            cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        cout << endl;
        cout << N << "," << incx << "," << incy << "," << rocblas_gflops << "," << rocblas_bandwidth
             << "," << gpu_time_used;

        if(argus.norm_check)
            cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        cout << endl;
    }

    return rocblas_status_success;
}
