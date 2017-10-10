/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
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
rocblas_status testing_bandwidth(Arguments argus)
{

    rocblas_int N = 26e7;

    rocblas_int incx = 1;

    rocblas_status status = rocblas_status_success;

    rocblas_int sizeX = N * incx;
    T alpha = 1.0;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);
    vector<T> hz(sizeX);

    if( N > hx.max_size()){
        printf("max_size of a std::vector is %lu, please reduce the input size \n", hx.max_size());
        return status;
    }


    T *dx, *dy;

    double gpu_time_used, gpu_bandwidth;

    rocblas_handle handle;
    rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) {
        rocblas_destroy_handle(handle);
        return status;
    }

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, sizeX * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, N, incx);

    //hz = hx;


    //copy data from CPU to device,
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice));

    printf("Bandwidth     MByte    GPU (GB/s)    Time (us) \n");

    /* =====================================================================
         Bandwidth
    =================================================================== */
    for(size_t size = 2000; size <= 16000; size += 1000){ 
    //for(size_t size = 1e6; size <= N; size += 1e7){

        CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*size*size*incx, hipMemcpyHostToDevice));

        gpu_time_used = get_time_us();// in microseconds

        hipMemcpy(dy, dx, sizeof(T)*size*size*incx, hipMemcpyDeviceToDevice);

        gpu_time_used = get_time_us() - gpu_time_used;

        gpu_bandwidth = 2 * size * size * sizeof(T) / (1e3 * gpu_time_used); //in GB/s

        printf("              %6.2f     %8.2f  %8.2f        \n", (int)size*(size)*sizeof(T)/1e6, gpu_bandwidth, gpu_time_used);

    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));

    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
