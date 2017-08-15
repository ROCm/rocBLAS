/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

#include "rocblas.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "arg_check.h"
#include "flops.h"
#include <typeinfo>

using namespace std;

template<typename T>
void transpose_reference(rocblas_int m, rocblas_int n, T *A, rocblas_int lda, T *B, rocblas_int ldb, rocblas_int batch_size)
{
    //transpose per batch
    for(rocblas_int b = 0; b < batch_size; b++)
    {
        for(rocblas_int i = 0; i < m; i++)
        {
            #pragma unroll
            for(rocblas_int j = 0; j < n; j++)
            {
                B[b*m*ldb + j + i*ldb] = A[b*n*lda + i + j*lda];
            }
        }
    }

}

template<typename T>
rocblas_status testing_transpose(Arguments argus)
{

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;

    T *dA, *dB;

    rocblas_int  A_size, B_size;

    double gpu_time_used;
    double rocblas_bandwidth;

    double rocblas_error = 0.0;

    rocblas_handle handle;
    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) 
    {
        rocblas_destroy_handle(handle);
        return status;
    }

    A_size = lda * N; 
    B_size = ldb * M;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hB_copy(B_size);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, M, N, lda);

    //copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));


    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

    //library interface
    status = rocblas_transpose<T>(handle, 
                    M, N,
                    dA, lda,
                    dB, ldb);
      
    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_bandwidth = 2* sizeof(T) * M * N / (1e3 * gpu_time_used) ;
    }

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hB.data(), dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {

     /* =====================================================================
                 CPU Implementation
     =================================================================== */
        if(status != rocblas_status_invalid_size) //only valid size, compare with cblas
        {
            transpose_reference<T>(
                     M, N,
                     hA.data(), lda,
                     hB_copy.data(), ldb,
                     1);
        }

        #ifndef NDEBUG
        print_matrix(hB_copy, hB, min(N,3), min(M,3), ldb);
        #endif

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, M, ldb, hB_copy.data(), hB.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', N, M, ldb, hB_copy.data(), hB.data());
        }

    }// end of if unit/norm check

    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything
        cout << "Routine, M, N, lda, ldb, rocblas-GB/s, ";
        if(argus.norm_check){
            cout << "norm-error" ;
        }
        cout << endl;

        cout << "Transpose, " << M << ',' << N <<',' << lda <<',' << ldb <<','<< rocblas_bandwidth << ','  ;

        if(argus.norm_check){
            cout << rocblas_error;
        }

        cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));

    rocblas_destroy_handle(handle);
    return status;
}


