/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
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
#include "flops.h"
#include <complex.h>

using namespace std;

#define CLEANUP()                                                  \
do {                                                               \
        if (dx)           CHECK_HIP_ERROR(hipFree(dx));            \
        if (dy)           CHECK_HIP_ERROR(hipFree(dy));            \
        if (dy_devptr)    CHECK_HIP_ERROR(hipFree(dy_devptr));     \
        if (alpha_devptr) CHECK_HIP_ERROR(hipFree(alpha_devptr));  \
        rocblas_destroy_handle(handle);                            \
} while (0)

/* ============================================================================================ */
template<typename T>
void testing_axpy_bad_arg()
{
    rocblas_int N = 100;
    rocblas_int incx = 1;
    rocblas_int incy = 1;
    T alpha = 0.6;

    T *dx, *dy;

    rocblas_handle handle;
    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) 
    {
        rocblas_destroy_handle(handle);                            \
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    rocblas_int sizeX = N * abs_incx;
    rocblas_int sizeY = N * abs_incy;

    vector<T> hx(sizeX);
    vector<T> hy(sizeY);

    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, sizeY * sizeof(T)));

    srand(1);
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    //copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T)*sizeY, hipMemcpyHostToDevice));

    // testing for (nullptr == dx)
    {
        T *dx_null = nullptr;

        status = rocblas_axpy<T>(handle,
                    N,
                    &alpha,
                    dx_null, incx,
                    dy, incy);

        verify_rocblas_status_invalid_pointer(status,"Error: x is nullptr");
    }
    // testing for (nullptr == dy)
    {
        T *dy_null = nullptr;

        status = rocblas_axpy<T>(handle,
                    N,
                    &alpha,
                    dx, incx,
                    dy_null, incy);

        verify_rocblas_status_invalid_pointer(status,"Error: y is nullptr");
    }
    // testing (nullptr == handle)
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_axpy<T>(handle_null,
                    N,
                    &alpha,
                    dx, incx,
                    dy, incy);

        verify_rocblas_status_invalid_handle(status);
    }
    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    rocblas_destroy_handle(handle);
    return;
}

template<typename T>
rocblas_status testing_axpy(Arguments argus)
{
    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;
    T alpha = argus.alpha;

    T *dx = NULL, *dy = NULL, *dy_devptr = NULL, *alpha_devptr = NULL;

    rocblas_handle handle;
    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) {
        CLEANUP();
        return status;
    }

    //argument sanity check before allocating invalid memory
    if ( N <= 0 )
    {
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));  // 100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&dy, 100 * sizeof(T)));

        status = rocblas_axpy<T>(handle,
                    N,
                    &alpha,
                    dx, incx,
                    dy, incy);

        verify_rocblas_status_success(status, "rocblas_axpy, N <=0");

        CLEANUP();
        return status;
    }

    rocblas_int abs_incx = incx > 0 ? incx : -incx;
    rocblas_int abs_incy = incy > 0 ? incy : -incy;
    rocblas_int sizeX = N * abs_incx;
    rocblas_int sizeY = N * abs_incy;

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, sizeY * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy_devptr, sizeY * sizeof(T)));

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hy(sizeY);
    vector<T> hy_devptr(sizeY);
    vector<T> hy_gold(sizeY);

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    //copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU BLAS
    hy_devptr = hy;
    hy_gold = hy;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T)*sizeY, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_devptr, hy_devptr.data(), sizeof(T)*sizeY, hipMemcpyHostToDevice));

    if (nullptr == dx || nullptr == dy)
    {
        status = rocblas_axpy<T>(handle,
                    N,
                    &alpha,
                    dx, incx,
                    dy, incy);

        verify_rocblas_status_invalid_pointer(status,"Error: x, y, is nullptr");

        CLEANUP();
        return status;
    }
    if (nullptr == dy_devptr)
    {
        status = rocblas_axpy<T>(handle,
                    N,
                    &alpha,
                    dx, incx,
                    dy_devptr, incy);

        verify_rocblas_status_invalid_pointer(status,"Error: x, y, is nullptr");

        CLEANUP();
        return status;
    }
    else if (nullptr == handle)
    {
        status = rocblas_axpy<T>(handle,
                    N,
                    &alpha,
                    dx, incx,
                    dy, incy);

        verify_rocblas_status_invalid_handle(status);

        CLEANUP();
        return status;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error = 0.0;

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }
        status = rocblas_axpy<T>(handle,
                    N,
                    &alpha,
                    dx, incx,
                    dy, incy);

    if (status != rocblas_status_success) {
        CLEANUP();
        return status;
    }

    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = axpy_gflop_count<T> (N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (3.0 * N) * sizeof(T)/ gpu_time_used / 1e3;
    }

    CHECK_HIP_ERROR(hipMalloc(&alpha_devptr, sizeof(T)));
    CHECK_HIP_ERROR(hipMemcpy(alpha_devptr, &alpha, sizeof(T), hipMemcpyHostToDevice));

    status = rocblas_axpy<T>(handle,
                    N,
                    alpha_devptr,
                    dx, incx,
                    dy_devptr, incy);

    if (status != rocblas_status_success) {
        CLEANUP();
        return status;
    }
    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T)*sizeY, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy_devptr.data(), dy_devptr, sizeof(T)*sizeY, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
     /* =====================================================================
                 CPU BLAS
     =================================================================== */
        if(argus.timing)
        {
            cpu_time_used = get_time_us();
        }

        cblas_axpy<T>(N,
                    alpha,
                    hx.data(), incx,
                    hy_gold.data(), incy);

        if(argus.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = axpy_gflop_count<T> (N) / cpu_time_used * 1e6 * 1;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold.data(), hy.data());
            unit_check_general<T>(1, N, abs_incy, hy_gold.data(), hy_devptr.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold.data(), hy.data());
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold.data(), hy_devptr.data());
        }
    }// end of if unit/norm check

    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything
        cout << "N, rocblas-Gflops, rocblas-GB/s, rocblas-us";
        if(argus.norm_check){
            cout << "CPU-Gflops, norm-error" ;
        }
        cout << endl;
        
        cout << N << ", " << rocblas_gflops << ", " << rocblas_bandwidth << ", " << gpu_time_used;
       
        if(argus.norm_check){
            cout << cblas_gflops << ',';
            cout << rocblas_error;
        }
     cout << endl;
     }

    CLEANUP();    
    return rocblas_status_success;
}
