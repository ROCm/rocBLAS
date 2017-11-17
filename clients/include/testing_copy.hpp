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

template<typename T>
void testing_copy_bad_arg()
{
    rocblas_int N = 100;
    rocblas_int incx = 1;
    rocblas_int incy = 1;
    rocblas_int safe_size = 100;  //  arbitrarily set to 100

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
    auto dy_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
    T* dx = (T*) dx_managed.get();
    T* dy = (T*) dy_managed.get();
    if (!dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // check if (nullptr == dx)
    {
        T *dx_null = nullptr;

        status = rocblas_copy<T>(handle, N, dx_null, incx, dy, incy);

        verify_rocblas_status_invalid_pointer(status,"Error: x, y, is nullptr");
    }
    // check if (nullptr == dy )
    {
        T *dy_null = nullptr;

        status = rocblas_copy<T>(handle, N, dx, incx, dy_null, incy);

        verify_rocblas_status_invalid_pointer(status,"Error: x, y, is nullptr");
    }
    // check if ( nullptr == handle )
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_copy<T>(handle_null, N, dx, incx, dy, incy);

        verify_rocblas_status_invalid_handle(status);
    }

    return;
}

template<typename T>
rocblas_status testing_copy(Arguments argus)
{
    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;
    rocblas_int safe_size = 100;  //  arbitrarily set to 100

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    //argument sanity check before allocating invalid memory
    if ( N <= 0 )
    {
        auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        auto dy_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        T* dx = (T*) dx_managed.get();
        T* dy = (T*) dy_managed.get();
        if (!dx || !dy)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_copy<T>(handle, N, dx, incx, dy, incy);

        return status;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    rocblas_int size_x = N * abs_incx;
    rocblas_int size_y = N * abs_incy;

    //allocate memory on device
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x ),rocblas_test::device_free};
    auto dy_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_y ),rocblas_test::device_free};
    T* dx = (T*) dx_managed.get();
    T* dy = (T*) dy_managed.get();
    if (!dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(size_x);
    vector<T> hy(size_y);
    vector<T> hy_gold(size_y);

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    //copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU BLAS
    hy_gold = hy;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T)*size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(argus.unit_check || argus.norm_check)
    {
        // GPU BLAS
        CHECK_ROCBLAS_ERROR(rocblas_copy<T>(handle, N, dx, incx, dy, incy));
        CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T)*size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_copy<T>( N, hx.data(), incx, hy_gold.data(), incy);
        cpu_time_used = get_time_us() - cpu_time_used;


        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold.data(), hy.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold.data(), hy.data());
        }
    }

    if(argus.timing)
    {
        int number_timing_iterations = 1;

        gpu_time_used = get_time_us();// in microseconds

        for (int iter = 0; iter < number_timing_iterations; iter++)
        {
            status = rocblas_copy<T>(handle, N, dx, incx, dy, incy);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_timing_iterations;

        cout << "N,rocblas-us";

        if (argus.norm_check) cout << ",CPU-us,error";

        cout << endl;

        cout << N << "," << gpu_time_used;

        if (argus.norm_check) cout << "," << cpu_time_used << "," << rocblas_error;

        cout << endl;
    }

    return rocblas_status_success;
}
