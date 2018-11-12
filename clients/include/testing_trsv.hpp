/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits> // std::numeric_limits<T>::epsilon();
#include <cmath>  // std::abs

#include "rocblas.hpp"
#include "arg_check.h"
#include "rocblas_test_unique_ptr.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 20

using namespace std;

template <typename T>
void printMatrix(const char* name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    printf("---------- %s ----------\n", name);
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%f ", A[i + j * lda]);
        }
        printf("\n");
    }
}

template <typename T>
rocblas_status testing_trsm(Arguments argus)
{
    rocblas_int M   = argus.M;
    rocblas_int N   = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;

    char char_side   = argus.side_option;
    char char_uplo   = argus.uplo_option;
    char char_transA = argus.transA_option;
    char char_diag   = argus.diag_option;
    T alpha_h        = argus.alpha;

    rocblas_int safe_size = 100; // arbitrarily set to 100

    rocblas_side side        = char2rocblas_side(char_side);
    rocblas_fill uplo        = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal diag    = char2rocblas_diagonal(char_diag);

    rocblas_int K      = side == rocblas_side_left ? M : N;
    rocblas_int size_A = lda * K;
    rocblas_int size_B = ldb * N;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M)
    {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dXorB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                                rocblas_test::device_free};
        T* dA    = (T*)dA_managed.get();
        T* dXorB = (T*)dXorB_managed.get();
        if(!dA || !dXorB)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        status =
            rocblas_trsm<T>(handle, side, uplo, transA, diag, M, N, &alpha_h, dA, lda, dXorB, ldb);

        trsm_arg_check(status, M, N, lda, ldb);

        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> AAT(size_A);
    vector<T> hB(size_B);
    vector<T> hX(size_B);
    vector<T> hXorB_1(size_B);
    vector<T> hXorB_2(size_B);
    vector<T> cpuXorB(size_B);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;
    T error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    T residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    T eps                     = std::numeric_limits<T>::epsilon();

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                         rocblas_test::device_free};
    auto dXorB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),
                                            rocblas_test::device_free};
    auto alpha_d_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    T* dA      = (T*)dA_managed.get();
    T* dXorB   = (T*)dXorB_managed.get();
    T* alpha_d = (T*)alpha_d_managed.get();
    if(!dA || !dXorB || !alpha_d)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

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

    //  pad untouched area into zero
    for(int i = K; i < lda; i++)
    {
        for(int j = 0; j < K; j++)
        {
            hA[i + j * lda] = 0.0;
        }
    }

    //  calculate AAT = hA * hA ^ T
    cblas_gemm(rocblas_operation_none,
               rocblas_operation_transpose,
               K,
               K,
               K,
               (T)1.0,
               hA.data(),
               lda,
               hA.data(),
               lda,
               (T)0.0,
               AAT.data(),
               lda);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < K; i++)
    {
        T t = 0.0;
        for(int j = 0; j < K; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += AAT[i + j * lda] > 0 ? AAT[i + j * lda] : -AAT[i + j * lda];
        }
        hA[i + i * lda] = t;
    }

    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf(char_uplo, K, hA.data(), lda);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        if('L' == char_uplo || 'l' == char_uplo)
        {
            for(int i = 0; i < K; i++)
            {
                T diag = hA[i + i * lda];
                for(int j = 0; j <= i; j++)
                {
                    hA[i + j * lda] = hA[i + j * lda] / diag;
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
                    hA[i + j * lda] = hA[i + j * lda] / diag;
                }
            }
        }
    }

    // Initial hX
    rocblas_init<T>(hX, M, N, ldb);
    // pad untouched area into zero
    for(int i = M; i < ldb; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hX[i + j * ldb] = 0.0;
        }
    }
    hB = hX;

    // Calculate hB = hA*hX;
    cblas_trmm<T>(
        side, uplo, transA, diag, M, N, 1.0 / alpha_h, (const T*)hA.data(), lda, hB.data(), ldb);

    hXorB_1 = hB; // hXorB <- B
    hXorB_2 = hB; // hXorB <- B
    cpuXorB = hB; // cpuXorB <- B

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB_1.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));

    T max_err_1 = 0.0;
    T max_err_2 = 0.0;
    T max_res_1 = 0.0;
    T max_res_2 = 0.0;
    if(argus.unit_check || argus.norm_check)
    {
        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(
            hipMemcpy(dXorB, hXorB_1.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(
            rocblas_trsm<T>(handle, side, uplo, transA, diag, M, N, &alpha_h, dA, lda, dXorB, ldb));

        CHECK_HIP_ERROR(
            hipMemcpy(hXorB_1.data(), dXorB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(
            hipMemcpy(dXorB, hXorB_2.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(alpha_d, &alpha_h, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(
            rocblas_trsm<T>(handle, side, uplo, transA, diag, M, N, alpha_d, dA, lda, dXorB, ldb));

        CHECK_HIP_ERROR(
            hipMemcpy(hXorB_2.data(), dXorB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        // Error Check
        // hXorB contains calculated X, so error is hX - hXorB

        // err is the one norm of the scaled error for a single column
        // max_err is the maximum of err for all columns
        for(int i = 0; i < N; i++)
        {
            T err_1 = 0.0;
            T err_2 = 0.0;
            for(int j = 0; j < M; j++)
            {
                if(hX[j + i * ldb] != 0)
                {
                    err_1 += std::abs((hX[j + i * ldb] - hXorB_1[j + i * ldb]) / hX[j + i * ldb]);
                    err_2 += std::abs((hX[j + i * ldb] - hXorB_2[j + i * ldb]) / hX[j + i * ldb]);
                }
                else
                {
                    err_1 += std::abs(hXorB_1[j + i * ldb]);
                    err_2 += std::abs(hXorB_2[j + i * ldb]);
                }
            }
            max_err_1 = max_err_1 > err_1 ? max_err_1 : err_1;
            max_err_2 = max_err_2 > err_2 ? max_err_2 : err_2;
        }
        trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

        // Residual Check
        // hXorB <- hA * (A^(-1) B) ;
        cblas_trmm<T>(side,
                      uplo,
                      transA,
                      diag,
                      M,
                      N,
                      1.0 / alpha_h,
                      (const T*)hA.data(),
                      lda,
                      hXorB_1.data(),
                      ldb);
        cblas_trmm<T>(side,
                      uplo,
                      transA,
                      diag,
                      M,
                      N,
                      1.0 / alpha_h,
                      (const T*)hA.data(),
                      lda,
                      hXorB_2.data(),
                      ldb);

        // hXorB contains A * (calculated X), so residual = A * (calculated X) - B
        //                                                = hXorB - hB
        // res is the one norm of the scaled residual for each column
        for(int i = 0; i < N; i++)
        {
            T res_1 = 0.0;
            T res_2 = 0.0;
            for(int j = 0; j < M; j++)
            {
                if(hB[j + i * ldb] != 0)
                {
                    res_1 += std::abs((hXorB_1[j + i * ldb] - hB[j + i * ldb]) / hB[j + i * ldb]);
                    res_2 += std::abs((hXorB_2[j + i * ldb] - hB[j + i * ldb]) / hB[j + i * ldb]);
                }
                else
                {
                    res_1 += std::abs(hXorB_1[j + i * ldb]);
                    res_2 += std::abs(hXorB_2[j + i * ldb]);
                }
            }
            max_res_1 = max_res_1 > res_1 ? max_res_1 : res_1;
            max_res_2 = max_res_2 > res_2 ? max_res_2 : res_2;
        }
        trsm_err_res_check<T>(max_res_1, M, residual_eps_multiplier, eps);
        trsm_err_res_check<T>(max_res_2, M, residual_eps_multiplier, eps);
    }

    if(argus.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(
            hipMemcpy(dXorB, hXorB_1.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        gpu_time_used = get_time_us(); // in microseconds

        CHECK_ROCBLAS_ERROR(
            rocblas_trsm<T>(handle, side, uplo, transA, diag, M, N, &alpha_h, dA, lda, dXorB, ldb));

        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = trsm_gflop_count<T>(M, N, K) / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        cblas_trsm<T>(
            side, uplo, transA, diag, M, N, alpha_h, (const T*)hA.data(), lda, cpuXorB.data(), ldb);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = trsm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        cout << "M,N,lda,ldb,side,uplo,transA,diag,rocblas-Gflops,us";

        if(argus.norm_check)
            cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        cout << endl;

        cout << M << ',' << N << ',' << lda << ',' << ldb << ',' << char_side << ',' << char_uplo
             << ',' << char_transA << ',' << char_diag << ',' << rocblas_gflops << ","
             << gpu_time_used;

        if(argus.norm_check)
            cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                 << max_err_2;

        cout << endl;
    }
    return rocblas_status_success;
}
