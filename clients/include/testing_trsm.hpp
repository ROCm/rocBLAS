/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>    // std::numeric_limits<T>::epsilon();
#include <cmath>     // std::abs

#include "rocblas.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "arg_check.h"
#include "flops.h"

#define FORWARD_TOLERANCE 40
#define BACKWARD_TOLERANCE 10

using namespace std;

template <typename T>
void printMatrix(const char* name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda) {
    printf("---------- %s ----------\n", name);
    for( int i = 0; i < m; i++) {
        for( int j = 0; j < n; j++) {
            printf("%f ",A[i + j * lda]);
        }
        printf("\n");
    }
}

/* ============================================================================================ */

template<typename T>
rocblas_status testing_trsm(Arguments argus)
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

    T *dA, *dB;

    rocblas_side side = char2rocblas_side(char_side);
    rocblas_fill uplo = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_int K = (side == rocblas_side_left ? M : N);
    rocblas_int A_size = lda * K;
    rocblas_int B_size = ldb * N;

    rocblas_handle handle;
    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success)
    {
        rocblas_destroy_handle(handle);
        return status;
    }

    //check here to prevent undefined memory allocation error
    if( M < 0 || N < 0 || lda < K || ldb < M)
    {
        CHECK_HIP_ERROR(hipMalloc(&dA, 100 * sizeof(T)));
        CHECK_HIP_ERROR(hipMalloc(&dB, 100 * sizeof(T)));

        status = rocblas_trsm<T>(handle,
            side, uplo,
            transA, diag,
            M, N,
            &alpha,
            dA,lda,
            dB,ldb);

        trsm_arg_check(status, M, N, lda, ldb);

        return status;
    }
    else if (nullptr == dA || nullptr == dB )
    {
        status = rocblas_trsm<T>(handle,
            side, uplo,
            transA, diag,
            M, N,
            &alpha,
            dA,lda,
            dB,ldb);

        verify_rocblas_status_invalid_pointer(status, "ERROR: A or B or C is nullptr");

        return status;
    }
    else if (nullptr == handle )
    {
        status = rocblas_trsm<T>(handle,
            side, uplo,
            transA, diag,
            M, N,
            &alpha,
            dA,lda,
            dB,ldb);

        verify_rocblas_status_invalid_handle(status);

        return status;
    }
    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hC(A_size);
    vector<T> hB(B_size);
    vector<T> hB_copy(B_size);
    vector<T> hX(B_size);
    
    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;
    T forward_tolerance =  FORWARD_TOLERANCE;
    T backward_tolerance =  BACKWARD_TOLERANCE;
    T eps = std::numeric_limits<T>::epsilon();


    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));

//  Random lower triangular matrices have condition number
//  that grows exponentially with matrix size. Random full
//  matrices have condition that grows linearly with 
//  matrix size. To generate a lower triangular matrix
//  with condition number that grows with matrix size 
//  start with full random matrix A. Calculate B <- A A^T. 
//  Make B strictly diagonal dominant. Next use Cholesky 
//  factorization to calculate L L^T = B. These L factors 
//  should have condition number approximately equal to
//  the condition number of the original matrix.

//  initialize full random matrix hA with all entries in [1, 10]
    rocblas_init<T>(hA, K, K, lda);

// printMatrix<T>("initialized hA", hA.data(), 4, 4, lda);

//  //pad untouched area into zero
    for(int i = K; i < lda; i++)
    {
        for(int j = 0; j < K; j++)
        {
            hA[i+j*lda] = 0.0;
        }
    }

//  calculate hC = hA * hA ^ T
    cblas_gemm(rocblas_operation_none, rocblas_operation_transpose, 
        K, K, K, (T)1.0, hA.data(), lda, hA.data(), lda, (T)0.0, hC.data(), lda);

//  copy hC into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < K; i++)
    {
        T t = 0.0;
        for(int j = 0; j < K; j++)
        {
            hA[i + j * lda] = hC[i + j * lda];
            t += hC[i + j * lda] > 0 ? hC[i + j * lda] : -hC[i + j * lda];
        }
        hA[i+i*lda] = t;
    }

//  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf(char_uplo, K, hA.data(), lda);

//  make unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        for(int i = 0; i < K; i++)
        {
            T diag = hA[i + i * lda];
            for(int j = 0; j < K; j++)
            {
                hA[i+j*lda] = hA[i+j*lda] / diag;
            }
        }
    }

    //Initial hB, hX on CPU
    rocblas_init<T>(hB, M, N, ldb);
    //pad untouched area into zero
    for(int i=M;i<ldb;i++)
    {
        for(int j=0;j<N;j++)
        {
            hB[i+j*ldb] = 0.0;
        }
    }    
    hX = hB;//original solution hX

    //Calculate hB = hA*hX;
    cblas_trmm<T>(
                side, uplo,
                transA, diag,
                M, N, 1.0/alpha,
                (const T*)hA.data(), lda,
                hB.data(), ldb);
    
    hB_copy = hB;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*A_size,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T)*B_size,  hipMemcpyHostToDevice));


    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us();// in microseconds
    }

    status = rocblas_trsm<T>(handle,
            side, uplo,
            transA, diag,
            M, N,
            &alpha,
            dA,lda,
            dB,ldb);

    if(argus.timing)
    {
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = trsm_gflop_count<T> (M, N, K) / gpu_time_used * 1e6 ;
    }

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hB.data(), dB, sizeof(T)*B_size, hipMemcpyDeviceToHost));


    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(argus.timing)
        {
            cpu_time_used = get_time_us();
        }

        cblas_trsm<T>(
                side, uplo,
                transA, diag,
                M, N, alpha,
                (const T*)hA.data(), lda,
                hB_copy.data(), ldb);

        if(argus.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = trsm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;
        }

        // Forward Error Check
        //      error is the one norm of the scaled error for each column

        T max_error = 0.0;
        for (int i = 0; i < N; i++)
        {
            T error = 0.0;
            for (int j = 0; j < M; j++)
            {
                if(hB_copy[j + i*ldb] != 0)
                {
                    error += std::abs((hB_copy[j + i*ldb] - hB[j + i*ldb]) / hB_copy[j + i*ldb]); 
                }
                else
                {
                    error += std::abs(hB[j + i*ldb]);
                }
            }
            max_error = max_error > error ? max_error : error;
        }
        trsm_forward_error_check<T>(max_error, M, forward_tolerance, eps);
    }


    if(argus.timing)
    {
        //only norm_check return an norm error, unit check won't return anything
            cout << "M, N, lda, ldb, side, uplo, transA, diag, rocblas-Gflops (us) ";
            if(argus.norm_check)
            {
                cout << "CPU-Gflops(us), norm-error" ;
            }
            cout << endl;

            cout << M << ',' << N <<',' << lda <<','<< ldb <<',' << char_side << ',' << char_uplo << ',' 
                 << char_transA << ','  << char_diag << ',' <<
                 rocblas_gflops << "(" << gpu_time_used  << "),";

            if(argus.norm_check)
            {
                cout << cblas_gflops << "(" << cpu_time_used << "),";
                cout << rocblas_error;
            }

            cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
