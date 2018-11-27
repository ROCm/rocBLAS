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

// template <typename T>
// void printMatrix(const char* name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
// {
//     printf("---------- %s ----------\n", name);
//     for(int i = 0; i < m; i++)
//     {
//         for(int j = 0; j < n; j++)
//         {
//             printf("%f ", A[i* lda + j ]);
//         }
//         printf("\n");
//     }
// }

template <typename T>
rocblas_status testing_trsv(Arguments argus)
{
    rocblas_int M   = argus.M;
    rocblas_int lda = argus.lda;
    rocblas_int incx         = argus.incx;
    char char_uplo   = argus.uplo_option;
    char char_transA = argus.transA_option;
    char char_diag   = argus.diag_option;

    rocblas_int safe_size = 100; // arbitrarily set to 100

    rocblas_fill uplo        = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal diag    = char2rocblas_diagonal(char_diag);

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // check here to prevent undefined memory allocation error
    if(M < 0 || lda < M || 0 == incx)
    {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dx_or_b_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                                rocblas_test::device_free};
        T* dA    = (T*)dA_managed.get();
        T* dx_or_b = (T*)dx_or_b_managed.get();
        if(!dA || !dx_or_b)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        status =
            rocblas_trsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);

        trsv_arg_check(status, M, lda, incx);

        return status;
    }

    rocblas_int size_A = lda * M;
    rocblas_int size_x, abs_incx;
    rocblas_int abs_incy;

    abs_incx = incx >= 0 ? incx : -incx;
    size_x = M * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> AAT(size_A);
    vector<T> hb(size_x);
    vector<T> hx(size_x);
    vector<T> hx_or_b_1(size_x);
    vector<T> hx_or_b_2(size_x);
    vector<T> cpu_x_or_b(size_x);
    vector<T> my_cpu_x_or_b(size_x);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;
    T error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    T residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    T eps                     = std::numeric_limits<T>::epsilon();

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                         rocblas_test::device_free};
    auto dxorb_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),
                                            rocblas_test::device_free};

    T* dA      = (T*)dA_managed.get();
    T* dx_or_b   = (T*)dxorb_managed.get();

    rocblas_init<T>(hA, M, M, lda);

        //  calculate AAT = hA * hA ^ T
    cblas_gemm(rocblas_operation_none,
               rocblas_operation_transpose,
               M,
               M,
               M,
               (T)1.0,
               hA.data(),
               lda,
               hA.data(),
               lda,
               (T)0.0,
               AAT.data(),
               lda); 

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < M; i++)
    {
        T t = 0.0;
        for(int j = 0; j < M; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += AAT[i + j * lda] > 0 ? AAT[i + j * lda] : -AAT[i + j * lda];
        }
        hA[i + i * lda] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf(char_uplo, M, hA.data(), lda);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        if('L' == char_uplo || 'l' == char_uplo)
        {
            for(int i = 0; i < M; i++)
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
            for(int j = 0; j < M; j++)
            {
                T diag = hA[j + j * lda];
                for(int i = 0; i <= j; i++)
                {
                    hA[i + j * lda] = hA[i + j * lda] / diag;
                }
            }
        }
    }

    rocblas_init<T>(hx, 1, M, abs_incx); 
    hb = hx;

    // Calculate hb = hA*hx;
    cblas_trmv<T>(
        uplo, transA, diag, M, (const T*)hA.data(), lda, hb.data(), incx);
    cpu_x_or_b = hb; // cpuXorB <- B
    hx_or_b_1 = hb;
    hx_or_b_2 = hb;
    my_cpu_x_or_b= hb;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));
    T max_err_1 = 0.0;
    T max_err_2 = 0.0;
    T max_res_1 = 0.0;
    T max_res_2 = 0.0;
    if(argus.unit_check || argus.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_trsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx));
        CHECK_HIP_ERROR(
            hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_2.data(), sizeof(T) * size_x, hipMemcpyHostToDevice))
        CHECK_ROCBLAS_ERROR(rocblas_trsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx));
        CHECK_HIP_ERROR(hipMemcpy(hx_or_b_2.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        T err_1 = 0.0;
        T err_2 = 0.0;
        for(int i = 0; i < M; i++)
        {
            if(hx[i * abs_incx] != 0)
            {
                err_1 += std::abs((hx[i * abs_incx] - hx_or_b_1[i * abs_incx]) / hx[i * abs_incx]);
                err_2 += std::abs((hx[i * abs_incx] - hx_or_b_2[i * abs_incx]) / hx[i * abs_incx]);
            }
            else
            {
                err_1 += std::abs(hx_or_b_1[i * abs_incx]);
                err_2 += std::abs(hx_or_b_2[i * abs_incx]);
            }
        }
        max_err_1 = max_err_1 > err_1 ? max_err_1 : err_1;
        max_err_2 = max_err_2 > err_2 ? max_err_2 : err_2;
        trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

        cblas_trmv<T>(uplo,
                    transA,
                    diag,
                    M,
                    (const T*)hA.data(),
                    lda,
                    hx_or_b_1.data(),
                    incx);
        cblas_trmv<T>(uplo,
                    transA,
                    diag,
                    M,
                    (const T*)hA.data(),
                    lda,
                    hx_or_b_2.data(),
                    incx);
        // hx_or_b contains A * (calculated X), so residual = A * (calculated x) - b
        //                                                  = hx_or_b - hb
        // res is the one norm of the scaled residual for each column

        T res_1 = 0.0;
        T res_2 = 0.0;
        for(int i = 0; i < M; i++)
        {
            if(hb[i * abs_incx] != 0)
            {
                res_1 += std::abs((hx_or_b_1[i * abs_incx] - hb[i * abs_incx]) / hb[i * abs_incx]);
                res_2 += std::abs((hx_or_b_2[i * abs_incx] - hb[i * abs_incx]) / hb[i * abs_incx]);
            }
            else
            {
                res_1 += std::abs(hx_or_b_1[i * abs_incx]);
                res_2 += std::abs(hx_or_b_2[i * abs_incx]);
            }
        }
        max_res_1 = max_res_1 > res_1 ? max_res_1 : res_1;
        max_res_2 = max_res_2 > res_2 ? max_res_2 : res_2;

        trsm_err_res_check<T>(max_res_1, M, residual_eps_multiplier, eps);
        trsm_err_res_check<T>(max_res_2, M, residual_eps_multiplier, eps);
    }

    if(argus.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(
            hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        gpu_time_used = get_time_us(); // in microseconds

        CHECK_ROCBLAS_ERROR(
            rocblas_trsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx));

        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = trsv_gflop_count<T>(M) / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        cblas_trsv<T>(
            uplo, transA, diag, M, (const T*)hA.data(), lda, cpu_x_or_b.data(), incx);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = trsv_gflop_count<T>(M) / cpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        cout << "M,lda,incx,uplo,transA,diag,rocblas-Gflops,us";

        if(argus.norm_check)
            cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        cout << endl;

        cout << M  << ',' << lda << ',' << incx  << ',' << char_uplo
             << ',' << char_transA << ',' << char_diag << ',' << rocblas_gflops << ","
             << gpu_time_used;

        if(argus.norm_check)
            cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                 << max_err_2;

        cout << endl;
    }

    return rocblas_status_success;
}
