/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "rocblas.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

using namespace std;


/* ============================================================================================ */

template<typename T>
rocblas_status testing_set_get_vector(Arguments argus)
{
    rocblas_int M = argus.M;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;
    rocblas_int incd = argus.incd;

    rocblas_status status     = rocblas_status_success;
    rocblas_status status_set = rocblas_status_success;
    rocblas_status status_get = rocblas_status_success;

    //argument sanity check, quick return if input parameters are invalid before allocating invalid memory
    if ( M < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( incx <= 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( incy <= 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( incd <= 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }

    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hx(M * incx);
    vector<T> hy(M * incy);
    vector<T> hy_ref(M * incy);

    T *db;

    double gpu_time_used, cpu_time_used;
    double rocblas_bandwidth, cpu_bandwidth;
    double rocblas_error = 0.0;

    rocblas_handle handle;

    rocblas_create_handle(&handle);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&db, M * incd * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hx, 1, M, incx);
    rocblas_init<T>(hy, 1, M, incy);
    hy_ref = hy;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

    for (int iter = 0; iter < 1; iter++){

        status_set = rocblas_set_vector(
                     M,
                     sizeof(T),
                     (void *) hx.data(), incx,
                     (void *) db, incd);

        status_get = rocblas_get_vector(
                     M,
                     sizeof(T),
                     (void *) db, incd,
                     (void *) hy.data(), incy);

        if (status_set != rocblas_status_success) {
            CHECK_HIP_ERROR(hipFree(db));
            rocblas_destroy_handle(handle);
            return status;
        }

        if (status_get != rocblas_status_success) {
            CHECK_HIP_ERROR(hipFree(db));
            rocblas_destroy_handle(handle);
            return status;
        }
    }
    if (argus.timing) {
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_bandwidth = (M * sizeof(T))/ gpu_time_used / 1e3;
    }

    if (argus.unit_check || argus.norm_check) {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if (argus.timing) {
            cpu_time_used = get_time_us();
        }

        // reference calculation
        for (int i = 0; i < M; i++) {
            hy_ref[i*incy] = hx[i*incx];
        }

        if (argus.timing) {
            cpu_time_used = get_time_us() - cpu_time_used;
            cpu_bandwidth = (M * sizeof(T))/ cpu_time_used / 1e3;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if (argus.unit_check) {
            unit_check_general<T>(1, M, incy, hy.data(), hy_ref.data());
        }


        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if (argus.norm_check) {
            rocblas_error = norm_check_general<T>('F', 1, M, incy, hy.data(), hy_ref.data());
        }
    }

    if(argus.timing) {
        //only norm_check return an norm error, unit check won't return anything
        cout << "M, incx, incy, incd, rocblas-GB/s, ";
        if(argus.norm_check){
            cout << "CPU-GB/s" ;
        }
        cout << endl;

        cout << "GGG,"<< M << ',' << incx <<',' << incy <<','<< incd << ',' << rocblas_bandwidth << ','  ;

        if(argus.norm_check){
            cout << cpu_bandwidth ;
        }
        cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(db));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
