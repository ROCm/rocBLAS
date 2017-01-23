/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <sys/time.h>
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
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template<typename T>
rocblas_status testing_gemm_batched(Arguments argus)
{

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;
    rocblas_int batch_count = argus.batch_count;

    //check here to prevent undefined memory allocation error
    if( M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 || batch_count < 0){
        return rocblas_status_invalid_size;
    }

    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int  A_size, B_size, C_size, A_row, A_col, B_row, B_col;
    rocblas_int  bsa, bsb, bsc; // batch size A, B, C
    T alpha = argus.alpha;
    T beta = argus.beta;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error = 0.0;
    rocblas_handle handle;
    rocblas_status status = rocblas_status_success;
    rocblas_create_handle(&handle);

    if(transA == rocblas_operation_none){
        A_row =  M; A_col = K;
    }
    else{
        A_row = K; A_col = M;
    }

    if(transB == rocblas_operation_none){
        B_row =  K; B_col = N;
    }
    else{
        B_row = N; B_col = K;
    }

    bsa = lda * A_col; bsb = ldb * B_col; bsc = ldc * N;
    A_size = bsa * batch_count;
    B_size = bsb * batch_count;
    C_size = bsc * batch_count;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hC(C_size);
    vector<T> hC_copy(C_size);

    T *dA, *dB, *dC;

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dC, C_size * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, A_row, A_col * batch_count, lda);
    rocblas_init<T>(hB, B_row, B_col * batch_count, ldb);
    rocblas_init<T>(hC, M, N * batch_count, ldc);

    //copy vector is easy in STL; hz = hx: save a copy in hC_copy which will be output of CPU BLAS
    hC_copy = hC;

    //copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*A_size,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T)*B_size,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T)*C_size,  hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

    //library interface
    status = rocblas_gemm_batched<T>(handle, transA, transB,
                    M, N, K,
                    &alpha, dA, lda, bsa,
                    dB, ldb, bsb,
                    &beta, dC, ldc, bsc, batch_count);


 //    sleep(1);
    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<T> (M, N, K) * batch_count / gpu_time_used * 1e6;
    }

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(T)*C_size,      hipMemcpyDeviceToHost));


    if(argus.unit_check || argus.norm_check){

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        for(rocblas_int i=0;i<batch_count; i++){
            cblas_gemm<T>(
                     transA, transB, M, N, K,
                     alpha, hA.data() + bsa * i, lda,
                     hB.data() + bsb * i, ldb,
                     beta, hC_copy.data() + bsc * i, ldc);
        }

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = gemm_gflop_count<T>(M, N, K) * batch_count / cpu_time_used * 1e6;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(M, N*batch_count, lda, hC_copy.data(), hC.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            rocblas_error = norm_check_general<T>('F', M, N*batch_count, lda, hC_copy.data(), hC.data());
        }

    }// end of if unit/norm check

    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything,
            cout << "Batch_Count, M, N, K, lda, ldb, ldc, rocblas-Gflops (us) ";
            if(argus.norm_check){
                cout << "CPU-Gflops(us), norm-error" ;
            }
            cout << endl;

            cout << "GG," << batch_count << M <<','<< N <<',' << K <<',' << lda <<','<< ldb <<',' << ldc <<',' << rocblas_gflops << "(" << gpu_time_used << "),";

            if(argus.norm_check){
                cout << cblas_gflops << "(" << cpu_time_used << "),";
                cout << rocblas_error;
            }

            cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));

    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
