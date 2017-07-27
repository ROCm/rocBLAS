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

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 20

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

    T *dA, *dXorB;

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
        CHECK_HIP_ERROR(hipMalloc(&dXorB, 100 * sizeof(T)));

        status = rocblas_trsm<T>(handle,
            side, uplo,
            transA, diag,
            M, N,
            &alpha,
            dA,lda,
            dXorB,ldb);

        trsm_arg_check(status, M, N, lda, ldb);

        CHECK_HIP_ERROR(hipFree(dA));
        CHECK_HIP_ERROR(hipFree(dXorB));

        return status;
    }
    else if (nullptr == dA || nullptr == dXorB )
    {
        status = rocblas_trsm<T>(handle,
            side, uplo,
            transA, diag,
            M, N,
            &alpha,
            dA,lda,
            dXorB,ldb);

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
            dXorB,ldb);

        verify_rocblas_status_invalid_handle(status);

        return status;
    }
    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> AAT(A_size);
    vector<T> hB(B_size);
    vector<T> hX(B_size);
    vector<T> hXorB(B_size);
    vector<T> cpuXorB(B_size);
    
    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;
    T error_eps_multiplier =  ERROR_EPS_MULTIPLIER;
    T residual_eps_multiplier =  RESIDUAL_EPS_MULTIPLIER;
    T eps = std::numeric_limits<T>::epsilon();


    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dXorB, B_size * sizeof(T)));

//  Random lower triangular matrices have condition number
//  that grows exponentially with matrix size. Random full
//  matrices have condition that grows linearly with 
//  matrix size. 
//
//  We want a triangular matrix with condition number that grows 
//  lineary with matrix size. We start with full random matrix A. 
//  Calculate symmetric AAT <- A A^T. Make AAT strictly diagonal 
//  dominant. A strictly diagonal dominant matrix is SPD so we 
//  can use Cholesky to calculate L L^T = AAT. These L factors 
//  should have condition number approximately equal to
//  the condition number of the original matrix A.

//  initialize full random matrix hA with all entries in [1, 10]
    rocblas_init<T>(hA, K, K, lda);

//  //pad untouched area into zero
    for(int i = K; i < lda; i++)
    {
        for(int j = 0; j < K; j++)
        {
            hA[i+j*lda] = 0.0;
        }
    }

//  calculate AAT = hA * hA ^ T
    cblas_gemm(rocblas_operation_none, rocblas_operation_transpose, 
        K, K, K, (T)1.0, hA.data(), lda, hA.data(), lda, (T)0.0, AAT.data(), lda);

//  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < K; i++)
    {
        T t = 0.0;
        for(int j = 0; j < K; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += AAT[i + j * lda] > 0 ? AAT[i + j * lda] : -AAT[i + j * lda];
        }
        hA[i+i*lda] = t;
    }

//  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf(char_uplo, K, hA.data(), lda);

//  make unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        if('L' == char_uplo || 'l' == char_uplo)
        {
            for(int i = 0; i < K; i++)
            {
                T diag = hA[i + i * lda];
                for(int j = 0; j <= i; j++)
                {
                    hA[i+j*lda] = hA[i+j*lda] / diag;
                }
            }
        }
        else
        {
            for(int j = 0; j < K; j++)
            {
                T diag = hA[j + j * lda];
                for(int i = 0; i <= j; i++)
                {
                    hA[i+j*lda] = hA[i+j*lda] / diag;
                }
            }
        }
    }

    //Initial hX
    rocblas_init<T>(hX, M, N, ldb);
    //pad untouched area into zero
    for(int i=M;i<ldb;i++)
    {
        for(int j=0;j<N;j++)
        {
            hX[i+j*ldb] = 0.0;
        }
    }    
    hB = hX;

    // Calculate hB = hA*hX;
    cblas_trmm<T>(
                side, uplo,
                transA, diag,
                M, N, 1.0/alpha,
                (const T*)hA.data(), lda,
                hB.data(), ldb);
    
    hXorB = hB;          // hXorB <- B
    cpuXorB = hB;        // cpuXorB <- B

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*A_size,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB.data(), sizeof(T)*B_size,  hipMemcpyHostToDevice));

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
            dXorB,ldb);     // dXorB <- A^(-1) B

    if(argus.timing)
    {
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = trsm_gflop_count<T> (M, N, K) / gpu_time_used * 1e6 ;
    }

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hXorB.data(), dXorB, sizeof(T)*B_size, hipMemcpyDeviceToHost));

    T max_err = 0.0;
    T max_res = 0.0;
    if(argus.unit_check || argus.norm_check)
    {
        // Error Check
        // hXorB contains calculated X, so error is hX - hXorB 

        // err is the one norm of the scaled error for a single column
        // max_err is the maximum of err for all columns
        for (int i = 0; i < N; i++)
        {
            T err = 0.0;
            for (int j = 0; j < M; j++)
            {
                if(hX[j + i*ldb] != 0)
                {
                    err += std::abs((hX[j + i*ldb] - hXorB[j + i*ldb]) / hX[j + i*ldb]); 
                }
                else
                {
                    err += std::abs(hXorB[j + i*ldb]);
                }
            }
            max_err = max_err > err ? max_err : err;
        }
        trsm_err_res_check<T>(max_err, M, error_eps_multiplier, eps);


        // Residual Check
        cblas_trmm<T>(
                side, uplo,
                transA, diag,
                M, N, 1.0/alpha,
                (const T*)hA.data(), lda,
                hXorB.data(), ldb);        // hXorB <- hA * (A^(-1) B) ;

        // hXorB contains A * (calculated X), so residual = A * (calculated X) - B
        //                                                = hXorB - hB 
        // res is the one norm of the scaled residual for each column
        for (int i = 0; i < N; i++)
        {
            T res = 0.0;
            for (int j = 0; j < M; j++)
            {
                if(hB[j + i*ldb] != 0)
                {
                    res += std::abs((hXorB[j + i*ldb] - hB[j + i*ldb]) / hB[j + i*ldb]); 
                }
                else
                {
                    res += std::abs(hXorB[j + i*ldb]);
                }
            }
            max_res = max_res > res ? max_res : res;
        }

        trsm_err_res_check<T>(max_res, M, residual_eps_multiplier, eps);
    }

    if(argus.timing)
    {
        cpu_time_used = get_time_us();

        cblas_trsm<T>(
                side, uplo,
                transA, diag,
                M, N, alpha,
                (const T*)hA.data(), lda,
                cpuXorB.data(), ldb);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops = trsm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;

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
            cout << max_err;
        }

        cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dXorB));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
