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
#include "arg_check.h"
#include <complex.h>

using namespace std;

template <typename T>
void testing_scal_bad_arg()
{
    rocblas_int N    = 100;
    rocblas_int incx = 1;
    T alpha          = 0.6;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int size_x = N * incx;

    // allocate memory on device
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),
                                         rocblas_test::device_free};
    T* dx = (T*)dx_managed.get();
    if(!dx)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // test if (nullptr == dx)
    {
        T* dx_null = nullptr;

        status = rocblas_scal<T>(handle, N, &alpha, dx_null, incx);

        verify_rocblas_status_invalid_pointer(status, "Error: x is nullptr");
    }
    // test if (nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = rocblas_scal<T>(handle, N, d_alpha_null, dx, incx);

        verify_rocblas_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // test if (nullptr == handle)
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_scal<T>(handle_null, N, &alpha, dx, incx);

        verify_rocblas_status_invalid_handle(status);
    }
    return;
}

template <typename T>
rocblas_status testing_scal(Arguments argus)
{
    rocblas_int N         = argus.N;
    rocblas_int incx      = argus.incx;
    T h_alpha             = argus.alpha;
    rocblas_int safe_size = 100; // arbitrarily set to 100

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_status status;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || incx <= 0)
    {
        auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        T* dx = (T*)dx_managed.get();
        if(!dx)
        {
            verify_rocblas_status_success(rocblas_status_memory_error, "!dx");
            return rocblas_status_memory_error;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_scal<T>(handle, N, &h_alpha, dx, incx));

        return rocblas_status_success;
    }

    rocblas_int size_x = N * incx;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx_1(size_x);
    vector<T> hx_2(size_x);
    vector<T> hy_gold(size_x);

    // Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx_1, 1, N, incx);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    // BLAS
    hx_2    = hx_1;
    hy_gold = hx_1;

    // allocate memory on device
    auto dx_1_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),
                                           rocblas_test::device_free};
    auto dx_2_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),
                                           rocblas_test::device_free};
    auto d_alpha_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    T* dx_1    = (T*)dx_1_managed.get();
    T* dx_2    = (T*)dx_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    if(!dx_1 || !dx_2 || !d_alpha)
    {
        verify_rocblas_status_success(rocblas_status_memory_error, "!dx || !d_alpha");
        return rocblas_status_memory_error;
    }

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx_1, hx_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    CHECK_HIP_ERROR(hipMemcpy(dx_1, hx_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dx_2, hx_2.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))
        CHECK_ROCBLAS_ERROR(rocblas_scal<T>(handle, N, &h_alpha, dx_1, incx));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device))
        CHECK_ROCBLAS_ERROR(rocblas_scal<T>(handle, N, d_alpha, dx_2, incx));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx_1.data(), dx_1, sizeof(T) * N * incx, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hx_2.data(), dx_2, sizeof(T) * N * incx, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_scal<T>(N, h_alpha, hy_gold.data(), incx);
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = axpy_gflop_count<T>(N) / cpu_time_used * 1e6 * 1;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, incx, hy_gold.data(), hx_1.data());
            unit_check_general<T>(1, N, incx, hy_gold.data(), hx_2.data());
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, N, incx, hy_gold.data(), hx_1.data());
            rocblas_error_2 = norm_check_general<T>('F', 1, N, incx, hy_gold.data(), hx_2.data());
        }

    } // end of if unit/norm check

    if(argus.timing)
    {
        int number_timing_iterations = 1;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_timing_iterations; iter++)
        {
            rocblas_scal<T>(handle, N, &h_alpha, dx_1, incx);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_timing_iterations;
        rocblas_gflops    = axpy_gflop_count<T>(N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (2.0 * N) * sizeof(T) / gpu_time_used / 1e3;

        cout << "N,rocblas-Gflops,rocblas-GB/s,rocblas-us";

        if(argus.norm_check)
            cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";

        cout << endl;

        cout << N << "," << rocblas_gflops << "," << rocblas_bandwidth << "," << gpu_time_used;

        if(argus.norm_check)
            cout << cblas_gflops << ',' << rocblas_error_1 << ',' << rocblas_error_2;

        cout << endl;
    }

    return rocblas_status_success;
}
