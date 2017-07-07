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
rocblas_status testing_dot(Arguments argus)
{

    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;

    T *dx, *dy, *d_rocblas_result;
    rocblas_status status = rocblas_status_success;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    //argument sanity check, quick return if input parameters are invalid before allocating invalid memory
    if ( N <= 0 ){
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));    // 100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&dy, 100 * sizeof(T)));
        CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(T)));

        status = rocblas_dot<T>(handle,
                        N,
                        dx, incx,
                        dy, incy, d_rocblas_result);

        nrm2_dot_arg_check(status, d_rocblas_result);

        return status;
    }
//TODO: dot must change to give standard behavior for incx < 0 and incy < 0
    else if(incx < 0 || incy < 0){
         return rocblas_status_invalid_size;
    }
    else if ( nullptr == dx || nullptr == dy || nullptr == d_rocblas_result)
    {
        status = rocblas_dot<T>(handle,
                        N,
                        dx, incx,
                        dy, incy, d_rocblas_result);

        pointer_check(status,"Error: x, y, or result is nullptr");

        return status;
    }
    else if ( nullptr == handle )
    {
        status = rocblas_dot<T>(handle,
                        N,
                        dx, incx,
                        dy, incy, d_rocblas_result);

        handle_check(status);

        return status;
    }

    rocblas_int sizeX = N * incx;
    rocblas_int sizeY = N * incy;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hy(sizeY);

    T cpu_result, rocblas_result;
    rocblas_int device_pointer = 1;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, sizeY * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, incx);
    rocblas_init<T>(hy, 1, N, incy);

    //copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T)*N*incy, hipMemcpyHostToDevice));


    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
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

    if (status != rocblas_status_success) {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        CHECK_HIP_ERROR(hipFree(d_rocblas_result));
        rocblas_destroy_handle(handle);
        return status;
    }

    if(device_pointer)    CHECK_HIP_ERROR(hipMemcpy(&rocblas_result, d_rocblas_result, sizeof(T), hipMemcpyDeviceToHost));

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

        cblas_dot<T>(N,
                    hx.data(), incx,
                    hy.data(), incy, &cpu_result);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
        }


        if(argus.unit_check){
            unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result);
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            printf("cpu=%f, gpu=%f\n", cpu_result, rocblas_result);
            rocblas_error = fabs((cpu_result - rocblas_result)/cpu_result);
        }

    }// end of if unit/norm check

    BLAS_1_RESULT_PRINT


    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    CHECK_HIP_ERROR(hipFree(d_rocblas_result));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
