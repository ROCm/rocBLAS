/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits> // std::numeric_limits<T>::epsilon();
#include <cmath>  // std::abs
#include <string>

#include "rocblas.hpp"
#include "arg_check.h"
#include "rocblas_test_unique_ptr.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

// this is for the single precision case, which is not very stable
#define ERROR_EPS_MULTIPLIER 2000

using namespace std;

template <typename T>
void printMatrix(const string name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    cout << "---------- " << name << " ----------" << endl;
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
rocblas_status testing_potf2(Arguments argus)
{
    
    rocblas_int N   = argus.N;
    rocblas_int lda = argus.lda;

    char char_uplo   = argus.uplo_option;

    rocblas_int safe_size = 100; // arbitrarily set to 100

    rocblas_fill uplo        = char2rocblas_fill(char_uplo);

    rocblas_int size_A = lda * N;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check here to prevent undefined memory allocation error
    if(N < 0 || lda < N)
    {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        T* dA    = (T*)dA_managed.get();
        if(!dA)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        status =
            roclapack_potf2<T>(handle, uplo, N, dA, lda);

        potf2_arg_check(status, N);

        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> AAT(size_A);

    double gpu_time_used, cpu_time_used;
    T error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    T eps                     = std::numeric_limits<T>::epsilon();

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                         rocblas_test::device_free};
    T* dA      = (T*)dA_managed.get();
    if(!dA)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //  Random lower triangular matrices are not positive-definite as required
    //  by the Cholesky decomposition
    //
    //  We start with full random matrix A. Calculate symmetric AAT <- A A^T.
    //  Make AAT strictly diagonal dominant. A strictly diagonal dominant matrix
    //  is SPD so we can use Cholesky to calculate L L^T = AAT.

    //  initialize full random matrix hA with all entries in [1, 10]
    rocblas_init<T>(hA, N, N, lda);

    //  pad untouched area into zero
    for(int i = N; i < lda; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hA[i + j * lda] = 0.0;
        }
    }
    
    // put it into [0, 1]
    for(int i = N; i < lda; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hA[i + j * lda] = (hA[i + j * lda]-1.0)/10.0;
        }
    }

    //  calculate AAT = hA * hA ^ T
    cblas_gemm(rocblas_operation_none,
               rocblas_operation_transpose,
               N,
               N,
               N,
               (T)1.0,
               hA.data(),
               lda,
               hA.data(),
               lda,
               (T)0.0,
               AAT.data(),
               lda);

    //  copy AAT into hA, make hA positive-definite
    for(int i = 0; i < N; i++)
    {
        T t = 0.0;
        for(int j = 0; j < N; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            //t += AAT[i + j * lda] > 0 ? AAT[i + j * lda] : -AAT[i + j * lda];
        }
        hA[i + i * lda] += 1;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

    T max_err_1 = 0.0;
    if(argus.unit_check || argus.norm_check)
    {
        // calculate dXorB <- A^(-1) B rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        
        CHECK_ROCBLAS_ERROR(
            roclapack_potf2<T>(handle, uplo, N, dA, lda));

        CHECK_HIP_ERROR(
            hipMemcpy(AAT.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));
        
        cblas_potf2<T>(uplo, N, hA.data(), lda);

        // Error Check
        // AAT contains calculated decomposition, so error is hA - AAT
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                AAT[i + j*lda] = abs(AAT[i + j*lda] - hA[i + j*lda]);
            }
        }
        
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                max_err_1 = max_err_1 > AAT[j + i * lda] ? max_err_1 : AAT[j + i * lda];
            }
        }
        potf2_err_res_check<T>(max_err_1, N, error_eps_multiplier, eps);
    }
    
    if(argus.timing)
    {
        // GPU rocBLAS
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        gpu_time_used = get_time_us(); // in microseconds

        CHECK_ROCBLAS_ERROR(
            roclapack_potf2<T>(handle, uplo, N, dA, lda));

        gpu_time_used  = get_time_us() - gpu_time_used;

        // CPU cblas
        cpu_time_used = get_time_us();

        cblas_potf2<T>(uplo, N, hA.data(), lda);

        cpu_time_used = get_time_us() - cpu_time_used;

        // only norm_check return an norm error, unit check won't return anything
        cout << "N,lda,uplo,us,us";

        if(argus.norm_check)
            cout << ",norm_error_host_ptr";

        cout << endl;

        cout << N << ',' << lda << ',' << char_uplo << ',' << gpu_time_used << ',' << cpu_time_used;

        if(argus.norm_check)
            cout << "," << max_err_1;

        cout << endl;
    }
    return rocblas_status_success;
}
