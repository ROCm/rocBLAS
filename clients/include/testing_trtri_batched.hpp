/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "rocblas.h"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

using namespace std;


/* ============================================================================================ */

template<typename T>
rocblas_status testing_trtri_batched(Arguments argus)
{

    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int batch_count = argus.batch_count;

    rocblas_int A_size = lda * N * batch_count;
    rocblas_int bsa = lda * N;

    rocblas_status status = rocblas_status_success;

    //argument sanity check, quick return if input parameters are invalid before allocating invalid memory
    if ( N < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( lda < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( batch_count < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }

    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hB(A_size);
    vector<T> hA;

    //Initial Data on CPU
    srand(1);
    vector<T> hA_sub(bsa);
    for(size_t i=0;i<batch_count;i++){
        rocblas_init_symmetric<T>(hA_sub, N, lda);
        for(int j=0;j<bsa;j++){
            hA.push_back(hA_sub[j]);
        }
    }
    hB = hA;

    T *dA, *dinvA;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;

    rocblas_handle handle;

    char char_uplo = argus.uplo_option;
    char char_diag = argus.diag_option;

    //char_uplo = 'U';
    rocblas_fill uplo = char2rocblas_fill(char_uplo);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_create_handle(&handle);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dinvA, A_size * sizeof(T)));

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*A_size,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dinvA, hA.data(), sizeof(T)*A_size,  hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }


    status = rocblas_trtri_batched_workspace<T>(handle,
            uplo, diag,
            N,
            dA, lda, bsa,
            dinvA, lda, bsa,
            batch_count);

    if (status != rocblas_status_success) {
        CHECK_HIP_ERROR(hipFree(dA));
        CHECK_HIP_ERROR(hipFree(dinvA));
        rocblas_destroy_handle(handle);
        return status;
    }

    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = trtri_gflop_count<T> (N) / gpu_time_used * 1e6 ;
    }

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA.data(), dinvA, sizeof(T)*A_size, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check){
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        for(size_t i=0;i<batch_count;i++){
            rocblas_int info = cblas_trtri<T>(
                    char_uplo, char_diag,
                    N,
                    hB.data() + i * bsa, lda);
            if(info != 0) printf("error in cblas_trtri\n");
        }
        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = trtri_gflop_count<T>(N) / cpu_time_used * 1e6;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(N, N*batch_count, lda, hB.data(), hA.data());
        }

        for(int i=0; i<32; i++){
            printf("CPU[%d]=%f, GPU[%d]=%f\n", i, hB[i], i, hA[i]);
        }
        //if enable norm check, norm check is invasive

        if(argus.norm_check){
            for(size_t i=0;i<batch_count;i++){
                rocblas_error = fmax(rocblas_error,
                                     norm_check_symmetric<T>('F', char_uplo, N, lda, hB.data()+i*bsa, hA.data()+i*bsa));
                //printf("error=%f, %lu\n", rocblas_error, i);
            }
        }
    } // end of norm_check

    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything
            cout << "batch, N, lda, rocblas-Gflops (us) ";
            if(argus.norm_check){
                cout << "CPU-Gflops(us), norm-error" ;
            }
            cout << endl;

            cout << batch_count << ',' << N <<',' << lda <<','<< rocblas_gflops << "(" << gpu_time_used  << "),";

            if(argus.norm_check){
                cout << cblas_gflops << "(" << cpu_time_used << "),";
                cout << rocblas_error;
            }

            cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dinvA));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
