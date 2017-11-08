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
#include "flops.h"
#include <complex.h>

using namespace std;

/* ============================================================================================ */
template<typename T>
void testing_axpy_bad_arg()
{
    rocblas_int N = 100;
    rocblas_int incx = 1;
    rocblas_int incy = 1;
    T alpha = 0.6;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    rocblas_int size_x = N * abs_incx;
    rocblas_int size_y = N * abs_incy;

    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),rocblas_test::device_free};
    auto dy_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_y),rocblas_test::device_free};
    T* dx = (T*) dx_managed.get();
    T* dy = (T*) dy_managed.get();
    if (!dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing for (nullptr == dx)
    {
        T *dx_null = nullptr;

        status = rocblas_axpy<T>(handle, N, &alpha, dx_null, incx, dy, incy);

        verify_rocblas_status_invalid_pointer(status,"Error: x is nullptr");
    }
    // testing for (nullptr == dy)
    {
        T *dy_null = nullptr;

        status = rocblas_axpy<T>(handle, N, &alpha, dx, incx, dy_null, incy);

        verify_rocblas_status_invalid_pointer(status,"Error: y is nullptr");
    }
    // testing for (nullptr == d_alpha)
    {
        T *d_alpha_null = nullptr;

        status = rocblas_axpy<T>(handle, N, d_alpha_null, dx, incx, dy, incy);

        verify_rocblas_status_invalid_pointer(status,"Error: y is nullptr");
    }
    // testing (nullptr == handle)
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_axpy<T>(handle_null, N, &alpha, dx, incx, dy, incy);

        verify_rocblas_status_invalid_handle(status);
    }
    return;
}

template<typename T>
rocblas_status testing_axpy(Arguments argus)
{
    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;
    T h_alpha = argus.alpha;
    rocblas_int safe_size = 100;   // arbitrarily set to 100

    std::unique_ptr<rocblas_test::handle_struct> test_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = test_handle->handle;

    //argument sanity check before allocating invalid memory
    if ( N <= 0 )
    {
        auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        auto dy_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        T* dx = (T*) dx_managed.get();
        T* dy = (T*) dy_managed.get();
        if (!dx || !dy)
        {
            verify_rocblas_status_success(rocblas_status_memory_error, "!dx || !dy");
            return rocblas_status_memory_error;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_axpy<T>(handle, N, &h_alpha, dx, incx, dy, incy));

        return rocblas_status_success;
    }

    rocblas_int abs_incx = incx > 0 ? incx : -incx;
    rocblas_int abs_incy = incy > 0 ? incy : -incy;
    rocblas_int size_x = N * abs_incx;
    rocblas_int size_y = N * abs_incy;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(size_x);
    vector<T> hy_1(size_y);
    vector<T> hy_2(size_y);
    vector<T> hy_gold(size_y);

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy_1, 1, N, abs_incy);

    //copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU BLAS
    hy_2 = hy_1;
    hy_gold = hy_1;

    //allocate memory on device
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),rocblas_test::device_free};
    auto dy_1_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_y),rocblas_test::device_free};
    auto dy_2_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_y),rocblas_test::device_free};
    auto d_alpha_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)),rocblas_test::device_free};
    T* dx = (T*) dx_managed.get();
    T* dy_1 = (T*) dy_1_managed.get();
    T* dy_2 = (T*) dy_2_managed.get();
    T* d_alpha = (T*) d_alpha_managed.get();
    if (!dx || !dy_1 || !dy_2 || !d_alpha)
    {
        verify_rocblas_status_success(rocblas_status_memory_error, "!dx || !dy_1 || !dy_2 || !d_alpha");
        return rocblas_status_memory_error;
    }

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T)*size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if (argus.timing)
    {
        int number_timing_iterations = 1;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        gpu_time_used = get_time_us();// in microseconds

        for(int iter=0; iter < number_timing_iterations; iter++)
        {
            rocblas_axpy<T>(handle, N, &h_alpha, dx, incx, dy_1, incy);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_timing_iterations;
        rocblas_gflops = axpy_gflop_count<T> (N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (3.0 * N) * sizeof(T)/ gpu_time_used / 1e3;
    }

    if (argus.unit_check || argus.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T)*size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T)*size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_axpy<T>(handle, N, &h_alpha, dx, incx, dy_1, incy));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_axpy<T>(handle, N, d_alpha, dx, incx, dy_2, incy));

        //copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T)*size_y, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T)*size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_axpy<T>(N, h_alpha, hx.data(), incx, hy_gold.data(), incy);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops = axpy_gflop_count<T> (N) / cpu_time_used * 1e6 * 1;

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold.data(), hy_1.data());
            unit_check_general<T>(1, N, abs_incy, hy_gold.data(), hy_2.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, N, abs_incy, hy_gold.data(), hy_1.data());
            rocblas_error_2 = norm_check_general<T>('F', 1, N, abs_incy, hy_gold.data(), hy_2.data());
        }
    }// end of if unit/norm check

    if(argus.timing)
    {
        //only norm_check return an norm error, unit check won't return anything
        cout << "N,rocblas-Gflops,rocblas-GB/s,rocblas-us";
        if(argus.norm_check)
        {
            cout << "CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr" ;
        }
        cout << endl;
        
        cout << N << "," << rocblas_gflops << "," << rocblas_bandwidth << "," << gpu_time_used;
       
        if(argus.norm_check)
        {
            cout << "," << cblas_gflops << ',' << rocblas_error_1 << ',' << rocblas_error_2;
        }
        cout << endl;
    }

    return rocblas_status_success;
}
