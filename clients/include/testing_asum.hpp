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

template<typename T1, typename T2>
void testing_asum_bad_arg()
{
    rocblas_int N = 100;
    rocblas_int incx = 1;

    rocblas_handle handle;
    T1 *dx;
    T2 *d_rocblas_result;

    rocblas_status status;
    status = rocblas_create_handle(&handle);

    if(status != rocblas_status_success) {
        printf("ERROR: rocblas_create_handle status = %d\n",status);
    }

    rocblas_int sizeX = N * incx;

    vector<T1> hx(sizeX);
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T1)));
    CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(T2)));
    srand(1);
    rocblas_init<T1>(hx, 1, N, incx);
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T1)*N*incx, hipMemcpyHostToDevice));

    // testing for (nullptr == dx)
    {
        T1 *dx_null = nullptr;

        status = rocblas_asum<T1, T2>(handle,
                        N,
                        dx_null, incx,
                        d_rocblas_result);

        pointer_check(status,"Error: dx is nullptr");
    }
    // testing for (nullptr == d_rocblas_result)
    {
        T2 *d_rocblas_result_null = nullptr;

        status = rocblas_asum<T1, T2>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result_null);

        pointer_check(status,"Error: result is nullptr");
    }
    // testing for ( nullptr == handle )
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_asum<T1, T2>(handle_null,
                        N,
                        dx, incx,
                        d_rocblas_result);

        handle_check(status);
    }
}


template<typename T1, typename T2>
rocblas_status testing_asum(Arguments argus)
{
    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;

    rocblas_handle handle;
    T1 *dx;
    T2 *d_rocblas_result;
    T2 rocblas_result;
    T2 cpu_result;

    rocblas_status status;
    status = rocblas_create_handle(&handle);

    if(status != rocblas_status_success) {
        printf("ERROR: rocblas_create_handle status = %d\n",status);
        return status;
    }

    //check to prevent undefined memory allocation error
    if( N <= 0 || incx <= 0 ){
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T1)));  // 100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(T2)));

        status = rocblas_asum<T1, T2>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);

        asum_arg_check(status, d_rocblas_result);

        return status;
    }

    rocblas_int sizeX = N * incx;

    //Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    vector<T1> hx(sizeX);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T1)));
    CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(T2)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T1>(hx, 1, N, incx);

    //copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T1)*N*incx, hipMemcpyHostToDevice));

    if ( nullptr == dx || nullptr == d_rocblas_result)
    {
        status = rocblas_asum<T1, T2>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);

        pointer_check(status,"Error: dx or result is nullptr");

        return status;
    }
    else if ( nullptr == handle )
    {
        status = rocblas_asum<T1, T2>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);

        handle_check(status);

        return status;
    }

    rocblas_int device_pointer = 1;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
     //rocblas_asum accept both dev/host pointer for the scalar
    if(device_pointer){
        status = rocblas_asum<T1, T2>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);
    }
    else{
        status = rocblas_asum<T1, T2>(handle,
                        N,
                        dx, incx,
                        &rocblas_result);
    }

    if (status != rocblas_status_success) {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(d_rocblas_result));
        rocblas_destroy_handle(handle);
        return status;
    }

    if(device_pointer)    CHECK_HIP_ERROR(hipMemcpy(&rocblas_result, d_rocblas_result, sizeof(T1), hipMemcpyDeviceToHost));

    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
    }


    if(argus.unit_check || argus.norm_check){

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        cblas_asum<T1, T2>(N,
                    hx.data(), incx,
                    &cpu_result);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
        }


        if(argus.unit_check){
            unit_check_general<T2>(1, 1, 1, &cpu_result, &rocblas_result);
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            printf("cpu=%e, gpu=%e\n", cpu_result, rocblas_result);
            rocblas_error = fabs((cpu_result - rocblas_result)/cpu_result);
        }

    }// end of if unit/norm check


    BLAS_1_RESULT_PRINT


    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(d_rocblas_result));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
