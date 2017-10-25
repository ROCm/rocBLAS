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
    rocblas_int rocblas_h_result;
    rocblas_int rocblas_d_result;
    rocblas_int cpu_result;

    rocblas_handle handle;
    T *dx;

    rocblas_status h_status;
    rocblas_status d_status;
    h_status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(h_status,"ERROR: rocblas_create_handle");

    if(h_status != rocblas_status_success) {
        rocblas_destroy_handle(handle);
        return h_status;
    }

    //check to prevent undefined memory allocation error
    if( N <= 0 || incx <= 0 )
    {
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));  // 100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(rocblas_int)));

        d_status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);

        iamax_arg_check(d_status, d_rocblas_result);

        return d_status;
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
        d_status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);

        verify_rocblas_status_invalid_pointer(d_status,"Error: x or result is nullptr");

        return d_status;
    }
    else if ( nullptr == handle )
    {
        d_status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);

        verify_rocblas_status_invalid_handle(d_status);

        return d_status;
    }

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
                 GPU BLAS
     =================================================================== */
     //rocblas_iamax accept both dev/host pointer for the scalar
    {
        d_status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);
    }
    {
        h_status = rocblas_iamax<T>(handle,
                        N,
                        dx, incx,
                        &rocblas_h_result);
    }

    if ((h_status != rocblas_status_success) || (d_status != rocblas_status_success)) {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(d_rocblas_result));
        rocblas_destroy_handle(handle);
        if (h_status != rocblas_status_success) return h_status;
        if (d_status != rocblas_status_success) return d_status;
    }

    CHECK_HIP_ERROR(hipMemcpy(&rocblas_d_result, d_rocblas_result, sizeof(rocblas_int), hipMemcpyDeviceToHost));

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

        // make index 1 based as in Fortran BLAS, not 0 based as in CBLAS
        cpu_result += 1;

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
        }


        if(argus.unit_check){
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &rocblas_h_result);
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &rocblas_d_result);
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            printf("----------------------- The maximum index cpu=%d, gpu=%d\n", cpu_result, rocblas_h_result);
            printf("----------------------- The maximum index cpu=%d, gpu=%d\n", cpu_result, rocblas_d_result);
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
