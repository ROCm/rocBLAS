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
#include <complex.h>

using namespace std;

/* ============================================================================================ */

template<typename T>
void testing_swap_bad_arg()
{
    rocblas_int N = 100;
    rocblas_int incx = 1;
    rocblas_int incy = 1;

    rocblas_handle handle;
    T *dx, *dy;

    rocblas_status status;
    status = rocblas_create_handle(&handle);

    if(status != rocblas_status_success) {
        printf("ERROR: rocblas_create_handle status = %d\n",status);
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    rocblas_int sizeX = N * abs_incx;
    rocblas_int sizeY = N * abs_incy;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hy(sizeY);
    vector<T> hx_gold(sizeX);
    vector<T> hy_gold(sizeY);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, sizeY * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, abs_incx);
    // make hy different to hx
    for (int i = 0; i < N; i++){ hy[i * abs_incy] = hx[i * abs_incx] + 1.0; };

    //swap vector is easy in STL; hy_gold = hx: save a swap in hy_gold which will be output of CPU BLAS
    hx_gold = hx;
    hy_gold = hy;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T)*sizeY, hipMemcpyHostToDevice));

    // test if (nullptr == dx)
    {
        T *dx_null = nullptr;
        status = rocblas_swap<T>(handle,
                    N,
                    dx_null, incx,
                    dy, incy);

        pointer_check(status,"Error: x, y, is nullptr");
    }
    // test if (nullptr == dy)
    {
        T *dy_null = nullptr;
        status = rocblas_swap<T>(handle,
                    N,
                    dx, incx,
                    dy_null, incy);

        pointer_check(status,"Error: x, y, is nullptr");
    }
    // test if (nullptr == handle)
    {
        rocblas_handle handle_null = nullptr;
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));  // 100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&dy, 100 * sizeof(T)));

        status = rocblas_swap<T>(handle_null,
                    N,
                    dx, incx,
                    dy, incy);

        handle_check(status);
    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    rocblas_destroy_handle(handle);
    return;
}

template<typename T>
rocblas_status testing_swap(Arguments argus)
{
    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;

    rocblas_handle handle;
    T *dx, *dy;

    rocblas_status status;
    status = rocblas_create_handle(&handle);

    if(status != rocblas_status_success) {
        printf("ERROR: rocblas_create_handle status = %d\n",status);
        return status;
    }

    //argument sanity check before allocating invalid memory
    if ( N <= 0 )
    {
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));  // 100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&dy, 100 * sizeof(T)));

        status = rocblas_swap<T>(handle,
                    N,
                    dx, incx,
                    dy, incy);

        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        rocblas_destroy_handle(handle);

        return status;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    rocblas_int sizeX = N * abs_incx;
    rocblas_int sizeY = N * abs_incy;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hy(sizeY);
    vector<T> hx_gold(sizeX);
    vector<T> hy_gold(sizeY);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, sizeY * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, abs_incx);
    // make hy different to hx
    for (int i = 0; i < N; i++){ hy[i * abs_incy] = hx[i * abs_incx] + 1.0; };

    //swap vector is easy in STL; hy_gold = hx: save a swap in hy_gold which will be output of CPU BLAS
    hx_gold = hx;
    hy_gold = hy;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T)*sizeY, hipMemcpyHostToDevice));

    if ( nullptr == dx || nullptr == dy )
    {
        status = rocblas_swap<T>(handle,
                    N,
                    dx, incx,
                    dy, incy);

        pointer_check(status,"Error: x, y, is nullptr");

        rocblas_destroy_handle(handle);

        return status;
    }
    else if ( nullptr == handle )
    {
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));  // 100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&dy, 100 * sizeof(T)));

        status = rocblas_swap<T>(handle,
                    N,
                    dx, incx,
                    dy, incy);

        handle_check(status);

        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        rocblas_destroy_handle(handle);

        return status;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }
        status = rocblas_swap<T>(handle,
                    N,
                    dx, incx,
                    dy, incy);

    if (status != rocblas_status_success) {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        rocblas_destroy_handle(handle);
        return status;
    }

    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
    }

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T)*sizeX, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T)*sizeY, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check){

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        cblas_swap<T>( N,
                       hx_gold.data(), incx,
                       hy_gold.data(), incy);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(1, N, abs_incx, hx_gold.data(), hx.data());
            unit_check_general<T>(1, N, abs_incy, hy_gold.data(), hy.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incx, hx_gold.data(), hx.data());
            rocblas_error = norm_check_general<T>('F', 1, N, abs_incy, hy_gold.data(), hy.data());
        }

    }// end of if unit/norm check


    BLAS_1_RESULT_PRINT

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
