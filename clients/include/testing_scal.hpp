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
void testing_scal_bad_arg()
{
    rocblas_int N = 100;
    rocblas_int incx = 1;
    T alpha = 0.6;

    T *dx;

    rocblas_handle handle;
    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) {
        rocblas_destroy_handle(handle);
        return;
    }

    rocblas_int sizeX = N * incx;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hz(sizeX);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, incx);

    //copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    //copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice));

    // test if (nullptr == dx)
    {
        T *dx_null = nullptr;
        status = rocblas_scal<T>(handle,
                    N,
                    &alpha,
                    dx_null, incx);

        verify_rocblas_status_invalid_pointer(status,"Error: x or alpha is nullptr");
    }
    // test if (nullptr == handle)
    {
        rocblas_handle handle_null = nullptr;
        status = rocblas_scal<T>(handle_null,
                    N,
                    &alpha,
                    dx, incx);

        verify_rocblas_status_invalid_handle(status);
    }

    CHECK_HIP_ERROR(hipFree(dx));
    rocblas_destroy_handle(handle);
    return;
}

template<typename T>
rocblas_status testing_scal(Arguments argus)
{
    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    T alpha = argus.alpha;

    T *dx;

    rocblas_handle handle;
    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) {
        rocblas_destroy_handle(handle);
        return status;
    }

    //argument sanity check before allocating invalid memory
    if (N <= 0 || incx <= 0)
    {
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));  // 100 is arbitary

        status = rocblas_scal<T>(handle,
                    N,
                    &alpha,
                    dx, incx);

        verify_rocblas_status_success(status,"ERROR rocblas_scal: (N <= 0 || incx <= 0)");

        return status;
    }

    rocblas_int sizeX = N * incx;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hz(sizeX);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, incx);

    //copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    //copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice));

    if (nullptr == dx)
    {
        status = rocblas_scal<T>(handle,
                    N,
                    &alpha,
                    dx, incx);

        verify_rocblas_status_invalid_pointer(status,"Error: x or alpha is nullptr");

        return status;
    }
    else if (nullptr == handle)
    {
        status = rocblas_scal<T>(handle,
                    N,
                    &alpha,
                    dx, incx);

        verify_rocblas_status_invalid_handle(status);

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

    status = rocblas_scal<T>(handle,
                    N,
                    &alpha,
                    dx, incx);
    if (status != rocblas_status_success) {
        CHECK_HIP_ERROR(hipFree(dx));
        rocblas_destroy_handle(handle);
        return status;
    }

    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
    }

        //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T)*N*incx, hipMemcpyDeviceToHost));


    if(argus.unit_check || argus.norm_check){

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        cblas_scal<T>(
                     N,
                     alpha,
                     hz.data(), incx);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
        }


        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(1, N, incx, hz.data(), hx.data());
        }

        //for(int i=0; i<10; i++){
        //    printf("CPU[%d]=%f, GPU[%d]=%f\n", i, hz[i], i, hx[i]);
        //}

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            rocblas_error = norm_check_general<T>('F', 1, N, incx, hz.data(), hx.data());
        }

    }// end of if unit/norm check


    BLAS_1_RESULT_PRINT

    CHECK_HIP_ERROR(hipFree(dx));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
