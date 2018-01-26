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
void testing_swap_bad_arg()
{
    rocblas_int N         = 100;
    rocblas_int incx      = 1;
    rocblas_int incy      = 1;
    rocblas_int safe_size = 100; //  arbitrarily set to 100

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // allocate memory on device
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                         rocblas_test::device_free};
    auto dy_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                         rocblas_test::device_free};
    T* dx = (T*)dx_managed.get();
    T* dy = (T*)dy_managed.get();
    if(!dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // test if (nullptr == dx)
    {
        T* dx_null = nullptr;

        status = rocblas_swap<T>(handle, N, dx_null, incx, dy, incy);

        verify_rocblas_status_invalid_pointer(status, "Error: x, y, is nullptr");
    }
    // test if (nullptr == dy)
    {
        T* dy_null = nullptr;

        status = rocblas_swap<T>(handle, N, dx, incx, dy_null, incy);

        verify_rocblas_status_invalid_pointer(status, "Error: x, y, is nullptr");
    }
    // test if (nullptr == handle)
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_swap<T>(handle_null, N, dx, incx, dy, incy);

        verify_rocblas_status_invalid_handle(status);
    }
}

template <typename T>
rocblas_status testing_swap(Arguments argus)
{
    rocblas_int N         = argus.N;
    rocblas_int incx      = argus.incx;
    rocblas_int incy      = argus.incy;
    rocblas_int safe_size = 100; //  arbitrarily set to 100

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_status status;

    // argument sanity check before allocating invalid memory
    if(N <= 0)
    {
        auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dy_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        T* dx = (T*)dx_managed.get();
        T* dy = (T*)dy_managed.get();
        if(!dx || !dy)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_swap<T>(handle, N, dx, incx, dy, incy);

        return status;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    rocblas_int size_x   = N * abs_incx;
    rocblas_int size_y   = N * abs_incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(size_x);
    vector<T> hy(size_y);
    vector<T> hx_gold(size_x);
    vector<T> hy_gold(size_y);

    // Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, abs_incx);
    // make hy different to hx
    for(int i = 0; i < N; i++)
    {
        hy[i * abs_incy] = hx[i * abs_incx] + 1.0;
    };

    // swap vector is easy in STL; hy_gold = hx: save a swap in hy_gold which will be output of CPU
    // BLAS
    hx_gold = hx;
    hy_gold = hy;

    // allocate memory on device
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),
                                         rocblas_test::device_free};
    auto dy_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_y),
                                         rocblas_test::device_free};
    T* dx = (T*)dx_managed.get();
    T* dy = (T*)dy_managed.get();
    if(!dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    if(argus.unit_check || argus.norm_check)
    {
        // GPU BLAS
        CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_swap<T>(handle, N, dx, incx, dy, incy));
        CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        cblas_swap<T>(N, hx_gold.data(), incx, hy_gold.data(), incy);
        cpu_time_used = get_time_us() - cpu_time_used;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, abs_incx, hx_gold.data(), hx.data());
            unit_check_general<T>(1, N, abs_incy, hy_gold.data(), hy.data());
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incx, hx_gold.data(), hx.data());
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold.data(), hy.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_swap<T>(handle, N, dx, incx, dy, incy);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_swap<T>(handle, N, dx, incx, dy, incy);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        cout << "N,incx,incy,rocblas-us";
        cout << endl;
        cout << N << "," << incx << "," << incy << "," << gpu_time_used;
        cout << endl;
    }

    return rocblas_status_success;
}
