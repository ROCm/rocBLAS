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
#include "arg_check.h"
#include "flops.h"

using namespace std;


/* ============================================================================================ */

template<typename T>
rocblas_status testing_set_get_matrix(Arguments argus)
{
    rocblas_int rows = argus.rows;
    rocblas_int cols = argus.cols;
    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;

    rocblas_status status_set = rocblas_status_success;
    rocblas_status status_get = rocblas_status_success;

    T *dc;

    rocblas_handle handle;
    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) {
        rocblas_destroy_handle(handle);
        return status;
    }

    void *void_a, *void_b;

    //argument sanity check, quick return if input parameters are invalid before allocating invalid memory
    if ( rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || ldc <= 0 ){

        vector<T> ha(100);
        vector<T> hb(100);
        vector<T> hb_ref(100);
        vector<T> hc(100);

        CHECK_HIP_ERROR(hipMalloc(&dc, cols * ldc * sizeof(T)));

        status_set = rocblas_set_matrix(rows, cols, sizeof(T), (void *) ha.data(), lda, (void *) dc, ldc);
        status_get = rocblas_get_matrix(rows, cols, sizeof(T), (void *) dc, ldc, (void *) hb.data(), ldb);

        set_get_matrix_arg_check(status_set, rows, cols, lda, ldb, ldc);
        set_get_matrix_arg_check(status_get, rows, cols, lda, ldb, ldc);

        return status_set;
    }
    else if ( nullptr == handle )
    {
        vector<T> ha(100);
        vector<T> hb(100);
        vector<T> hb_ref(100);
        vector<T> hc(100);

        status_set = rocblas_set_matrix(rows, cols, sizeof(T), (void *) ha.data(), lda, (void *) dc, ldc);
        status_get = rocblas_get_matrix(rows, cols, sizeof(T), (void *) dc, ldc, (void *) hb.data(), ldb);

        verify_rocblas_status_invalid_handle(status_set);
        verify_rocblas_status_invalid_handle(status_get);

        return status_set;
    }

    if ( rows < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( cols < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( lda <= 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( ldb <= 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( ldc <= 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }

    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> ha(cols * lda);
    vector<T> hb(cols * ldb);
    vector<T> hb_ref(cols * ldb);
    vector<T> hc(cols * ldc);

    double gpu_time_used, cpu_time_used;
    double rocblas_bandwidth, cpu_bandwidth;
    double rocblas_error = 0.0;

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dc, cols * ldc * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(ha, rows, cols, lda);
    rocblas_init<T>(hb, rows, cols, ldb);
    hb_ref = hb;
    for(int i = 0; i < cols * ldc; i++){hc[i] = 100+i;};
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(T) * ldc * cols, hipMemcpyHostToDevice));
    for(int i = 0; i < cols * ldc; i++){hc[i] = 99.0;};

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

    for (int iter = 0; iter < 1; iter++){

        status_set = rocblas_set_matrix(rows, cols, sizeof(T), (void *) ha.data(), lda, (void *) dc, ldc);
        status_get = rocblas_get_matrix(rows, cols, sizeof(T), (void *) dc, ldc, (void *) hb.data(), ldb);
        if (status_set != rocblas_status_success || status_get != rocblas_status_success) {
            CHECK_HIP_ERROR(hipFree(dc));
            rocblas_destroy_handle(handle);
            return status;
        }
    }
    if (argus.timing) {
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_bandwidth = (rows*cols * sizeof(T))/ gpu_time_used / 1e3;
    }

    if (argus.unit_check || argus.norm_check) {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if (argus.timing) {
            cpu_time_used = get_time_us();
        }

        // reference calculation
        for (int i1 = 0; i1 < rows; i1++) {
            for (int i2 = 0; i2 < cols; i2++) {
                hb_ref[i1 + i2 * ldb ] = ha[i1 + i2 * lda];
            }
        }

        if (argus.timing) {
            cpu_time_used = get_time_us() - cpu_time_used;
            cpu_bandwidth = (rows * cols * sizeof(T))/ cpu_time_used / 1e3;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if (argus.unit_check) {
            unit_check_general<T>(rows, cols, ldb, hb.data(), hb_ref.data());
        }


        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if (argus.norm_check) {
            rocblas_error = norm_check_general<T>('F', rows, cols, ldb, hb.data(), hb_ref.data());
        }
    }

    if(argus.timing) {
        //only norm_check return an norm error, unit check won't return anything
        cout << "rows, cols, lda, ldb, rocblas-GB/s, ";
        if(argus.norm_check){
            cout << "CPU-GB/s" ;
        }
        cout << endl;

        cout << "GGG,"<< rows << ',' << cols <<',' << lda <<','<< ldb << ',' << rocblas_bandwidth << ','  ;

        if(argus.norm_check){
            cout << cpu_bandwidth ;
        }
        cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(dc));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
