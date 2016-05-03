/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "rocblas.h"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include <complex.h>

using namespace std;

/* ============================================================================================ */

template<typename T>
rocblas_status testing_dot(Arguments argus)
{

    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;

    rocblas_status status = rocblas_success;

    //argument sanity check, quick return if input parameters are invalid before allocating invalid memory
    if ( N < 0 ){
        status = rocblas_invalid_dim;
        return status;
    }
    else if ( incx < 0 ){
        status = rocblas_invalid_incx;
        return status;
    }
    else if ( incy < 0 ){
        status = rocblas_invalid_incy;
        return status;
    }
    if (status != rocblas_success) {
        return status;
    }


    rocblas_int sizeX = N * incx;
    rocblas_int sizeY = N * incy;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hy(sizeY);

    T cpu_result, rocblas_result;
    T *dx, *dy, *d_rocblas_result;
    rocblas_int device_pointer = 1;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    rocblas_handle handle;
    rocblas_create(&handle);

    //allocate memory on device
    CHECK_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_ERROR(hipMalloc(&dy, sizeY * sizeof(T)));
    CHECK_ERROR(hipMalloc(&d_rocblas_result, sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, incx);
    rocblas_init<T>(hy, 1, N, incy);

    //copy data from CPU to device, does not work for incx != 1

    rocblas_set_vector(N, sizeof(T), hx.data(), incx, dx, incx);
    rocblas_set_vector(N, sizeof(T), hy.data(), incy, dy, incy);


    if(argus.timing){
        printf("dot     N    rocblas    (ms)     CPU (ms)     error\n");
    }


    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_ms();// in miliseconds
    }

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
     //rocblas_dot accept both dev/host pointer for the scalar
     if(device_pointer){
        status = rocblas_dot<T>(handle,
                        N,
                        dx, incx,
                        dy, incy, d_rocblas_result);
    }
    else{
        status = rocblas_dot<T>(handle,
                        N,
                        dx, incx,
                        dy, incy, &rocblas_result);
    }

    if (status != rocblas_success) {
        CHECK_ERROR(hipFree(dx));
        CHECK_ERROR(hipFree(dy));
        CHECK_ERROR(hipFree(d_rocblas_result));
        rocblas_destroy(handle);
        return status;
    }

    if(device_pointer)    CHECK_ERROR(hipMemcpy(&rocblas_result, d_rocblas_result, sizeof(T), hipMemcpyDeviceToHost));

    if(argus.timing){
        gpu_time_used = get_time_ms() - gpu_time_used;
    }

    rocblas_get_vector(N, sizeof(T), dx, incx, hx.data(), incx);

    if(argus.unit_check || argus.norm_check){

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
         if(argus.timing){
            cpu_time_used = get_time_ms();
        }

        cblas_dot<T>(N,
                    hx.data(), incx,
                    hy.data(), incy, &cpu_result);

        if(argus.timing){
            cpu_time_used = get_time_ms() - cpu_time_used;
        }


        if(argus.unit_check){
            unit_check_general<T>(1, 1, incx, &cpu_result, &rocblas_result);
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            printf("cpu=%f, gpu=%f\n", cpu_result, rocblas_result);
            rocblas_error = fabs((cpu_result - rocblas_result)/cpu_result);
        }

    }// end of if unit/norm check


    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything, only return the real part, imag part does not make sense
        if(argus.norm_check){
            printf("    %d    %8.2f    %8.2f     %8.2e \n", (int)N, gpu_time_used, cpu_time_used, rocblas_error);
        }
        else{
            printf("    %d    %8.2f    %8.2f     ---     \n", (int)N, gpu_time_used, cpu_time_used);
        }
    }

    CHECK_ERROR(hipFree(dx));
    CHECK_ERROR(hipFree(dy));
    CHECK_ERROR(hipFree(d_rocblas_result));
    rocblas_destroy(handle);
    return rocblas_success;
}
