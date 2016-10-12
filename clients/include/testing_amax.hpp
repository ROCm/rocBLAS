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
#include <complex.h>

using namespace std;

/* ============================================================================================ */

template<typename T>
rocblas_status testing_amax(Arguments argus)
{

    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;

    rocblas_status status = rocblas_status_success;

    //check to prevent undefined memory allocation error
    if( N < 0 || incx < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }

    rocblas_int sizeX = N * incx;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);

    T *dx;
    rocblas_int *d_rocblas_result;
    rocblas_int cpu_result, rocblas_result;

    rocblas_int device_pointer = 1;

    double gpu_time_used, cpu_time_used;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(rocblas_int)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, incx);

    //copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice));

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
     //rocblas_amax accept both dev/host pointer for the scalar
    if(device_pointer){
        status = rocblas_amax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);
    }
    else{
        status = rocblas_amax<T>(handle,
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

        cblas_amax<T>(N,
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
