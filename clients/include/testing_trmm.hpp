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
rocblas_status testing_trmm(Arguments argus)
{

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;

    char char_side = argus.side_option;
    char char_uplo = argus.uplo_option;
    char char_transA = argus.transA_option;
    char char_diag = argus.diag_option;
    T alpha = argus.alpha;

    rocblas_side side = char2rocblas_side(char_side);
    rocblas_fill uplo = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_int K = (side == rocblas_side_left ? M : N);
    rocblas_int A_size = lda * K;
    rocblas_int B_size = ldb * N;

    rocblas_status status = rocblas_status_success;

    //check here to prevent undefined memory allocation error
    if( M < 0 || N < 0 || lda < 0 || ldb < 0){
        return rocblas_status_invalid_size;
    }
    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hC(B_size);
    vector<T> hB_copy(B_size);

    T *dA, *dB, *dC;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;

    rocblas_handle handle;

    rocblas_create_handle(&handle);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dC, B_size * sizeof(T))); //dB and dC are exact the same size

    //Initial Data on CPU
    srand(1);
    rocblas_init_symmetric<T>(hA, K, lda);
    rocblas_init<T>(hB, M, N, ldb);
    hB_copy = hB;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*A_size,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T)*B_size,  hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

/*
    status = rocblas_trmm<T>(handle,
            side, uplo,
            transA, diag,
            M, N,
            &alpha,
            dA,lda,
            dB,ldb,
            dC,ldc);
*/

    if (status != rocblas_status_success) {
        CHECK_HIP_ERROR(hipFree(dA));
        CHECK_HIP_ERROR(hipFree(dB));
        CHECK_HIP_ERROR(hipFree(dC));
        rocblas_destroy_handle(handle);
        return status;
    }

    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = trmm_gflop_count<T> (M, N, K) / gpu_time_used * 1e6 ;
    }

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(T)*B_size, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check){
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        cblas_trmm<T>(
                side, uplo,
                transA, diag,
                M, N, alpha,
                hA.data(), lda,
                hB_copy.data(), ldb);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = trmm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(M, N, ldb, hB_copy.data(), hC.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            rocblas_error = norm_check_general<T>('F', M, N, ldb, hB_copy.data(), hC.data());
        }
    }

    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything
            cout << "M, N, lda, rocblas-Gflops (us) ";
            if(argus.norm_check){
                cout << "CPU-Gflops(us), norm-error" ;
            }
            cout << endl;

            cout << M <<','<< N <<',' << lda <<','<< rocblas_gflops << "(" << gpu_time_used  << "),";

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
