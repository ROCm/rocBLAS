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

/* ============================================================================================ */
template<typename T>
void testing_iamax_bad_arg()
{
    rocblas_int N = 100;
    rocblas_int incx = 1;
    rocblas_int *d_rocblas_result;

    rocblas_handle handle;
    T *dx;

    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) {
        rocblas_destroy_handle(handle);
        return;
    }

    rocblas_int sizeX = N * incx;

    vector<T> hx(sizeX);
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(rocblas_int)));
    srand(1);
    rocblas_init<T>(hx, 1, N, incx);
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice));

    // testing for (nullptr == dx)
    {
        T *dx_null = nullptr;

        status = rocblas_iamax<T>(handle,
                        N,
                        dx_null, incx,
                        d_rocblas_result);

        verify_rocblas_status_invalid_pointer(status,"Error: x is nullptr");
    }
    // testing for (nullptr == d_rocblas_result)
    {
        rocblas_int *d_rocblas_result_null = nullptr;

        status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result_null);

        verify_rocblas_status_invalid_pointer(status,"Error: result is nullptr");
    }
    // testing for (nullptr == handle)
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_iamax<T>(handle_null,
                        N,
                        dx, incx,
                        d_rocblas_result);

        verify_rocblas_status_invalid_handle(status);
    }
    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(d_rocblas_result));
    rocblas_destroy_handle(handle);
}

template<typename T>
rocblas_status testing_iamax(Arguments argus)
{
    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int *d_rocblas_result;
    rocblas_int rocblas_result;
    rocblas_int cpu_result;

    rocblas_handle handle;
    T *dx;

    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) {
        rocblas_destroy_handle(handle);
        return status;
    }

    //check to prevent undefined memory allocation error
    if( N <= 0 || incx <= 0 )
    {
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));  // 100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(rocblas_int)));

        status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);

        iamax_arg_check(status, d_rocblas_result);

        return status;
    }

    rocblas_int sizeX = N * incx;

    //Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(rocblas_int)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, incx);

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice));

    if ( nullptr == dx || nullptr == d_rocblas_result)
    {
        status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);

        verify_rocblas_status_invalid_pointer(status,"Error: x or result is nullptr");

        return status;
    }
    else if ( nullptr == handle )
    {
        status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);

        verify_rocblas_status_invalid_handle(status);

        return status;
    }

    rocblas_int device_pointer = 1;

    double gpu_time_used, cpu_time_used;

    if(argus.timing){
        printf("Idamax: N    rocblas(us)     CPU(us)     error\n");
    }

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
     //rocblas_iamax accept both dev/host pointer for the scalar
    if(device_pointer){
        status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);
    }
    else{
        status = rocblas_iamax<T>(handle,
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

    if(device_pointer)    CHECK_HIP_ERROR(hipMemcpy(&rocblas_result, d_rocblas_result, sizeof(rocblas_int), hipMemcpyDeviceToHost));

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

        cblas_iamax<T>(N,
                    hx.data(), incx,
                    &cpu_result);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
        }


        if(argus.unit_check){
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &rocblas_result);
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            printf("The maximum index cpu=%d, gpu=%d\n", cpu_result, rocblas_result);
        }

    }// end of if unit/norm check


    if(argus.timing){
        printf("    %d    %8.2f    %8.2f     ---     \n", (int)N, gpu_time_used, cpu_time_used);
    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(d_rocblas_result));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
