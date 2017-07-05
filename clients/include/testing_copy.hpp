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
rocblas_status testing_copy(Arguments argus)
{

    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;
    T *dx, *dy;

    rocblas_status status = rocblas_status_success;
    rocblas_handle handle;

    rocblas_create_handle(&handle);

    //argument sanity check before allocating invalid memory
    if ( N <= 0 )
    {
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));  // 100 is arbitary

        status = rocblas_copy<T>(handle,
                    N,
                    dx, incx,
                    dy, incy);

        return status;
    }
    else if ( N < 0 || incx <= 0 || incy <= 0 )
    {
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));  // 100 is arbitary

        status = rocblas_copy<T>(handle,
                    N,
                    dx, incx,
                    dy, incy);

        return status;
    }
    else if ( nullptr == dx || nullptr == dy )
    {
        status = rocblas_copy<T>(handle,
                    N,
                    dx, incx,
                    dy, incy);

        pointer_check(status,"Error: x, y, is nullptr");

        return status;
    }
    else if ( nullptr == handle )
    {
        status = rocblas_copy<T>(handle,
                    N,
                    dx, incx,
                    dy, incy);

        handle_check(status);

        return status;
    }

    rocblas_int sizeX = N * incx;
    rocblas_int sizeY = N * incy;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hy(sizeY);
    vector<T> hy_gold(sizeY);

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, sizeY * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, incx);
    rocblas_init<T>(hy, 1, N, incy);

    //copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU BLAS
    hy_gold = hy;

    //copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T)*N*incy, hipMemcpyHostToDevice));


    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

        status = rocblas_copy<T>(handle,
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
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T)*N*incy, hipMemcpyDeviceToHost));


    if(argus.unit_check || argus.norm_check){

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        cblas_copy<T>( N,
                       hx.data(),      incx,
                       hy_gold.data(), incy);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
        }


        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(1, N, incy, hy_gold.data(), hy.data());
        }

        //for(int i=0; i<10; i++){
        //    printf("CPU[%d]=%f, GPU[%d]=%f\n", i, hy_gold[i], i, hx[i]);
        //}

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            rocblas_error = norm_check_general<T>('F', 1, N, incy, hy_gold.data(), hy.data());
        }

    }// end of if unit/norm check


    BLAS_1_RESULT_PRINT

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
