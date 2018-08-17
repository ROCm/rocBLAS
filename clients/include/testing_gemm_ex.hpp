/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

#include "rocblas.hpp"
#include "arg_check.h"
#include "rocblas_test_unique_ptr.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template <typename T>
rocblas_status testing_gemm_ex_template(rocblas_operation transA,
                                        rocblas_operation transB,
                                        rocblas_int M,
                                        rocblas_int N,
                                        rocblas_int K,
                                        float alpha_float,
                                        rocblas_int lda,
                                        rocblas_int ldb,
                                        float beta_float,
                                        rocblas_int ldc,
                                        rocblas_int ldd,
                                        rocblas_int norm_check,
                                        rocblas_int unit_check,
                                        rocblas_int timing,
                                        int number_hot_calls)
{
    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
    uint32_t kernel_index  = 0;
    uint32_t flags         = 0;

    T h_alpha;
    T h_beta;
    rocblas_precision a_type;
    rocblas_precision b_type;
    rocblas_precision c_type;
    rocblas_precision d_type;
    rocblas_precision compute_type;

    if(is_same<T, rocblas_half>::value)
    {
        h_alpha      = float_to_half(alpha_float);
        h_beta       = float_to_half(beta_float);
        a_type       = rocblas_precision_half;
        b_type       = rocblas_precision_half;
        c_type       = rocblas_precision_half;
        d_type       = rocblas_precision_half;
        compute_type = rocblas_precision_half;
    }
    else if(is_same<T, float>::value)
    {
        h_alpha      = static_cast<T>(alpha_float);
        h_beta       = static_cast<T>(beta_float);
        a_type       = rocblas_precision_single;
        b_type       = rocblas_precision_single;
        c_type       = rocblas_precision_single;
        d_type       = rocblas_precision_single;
        compute_type = rocblas_precision_single;
    }
    else if(is_same<T, double>::value)
    {
        h_alpha      = static_cast<T>(alpha_float);
        h_beta       = static_cast<T>(beta_float);
        a_type       = rocblas_precision_double;
        b_type       = rocblas_precision_double;
        c_type       = rocblas_precision_double;
        d_type       = rocblas_precision_double;
        compute_type = rocblas_precision_double;
    }
    else
    {
        return rocblas_status_not_implemented;
    }

    const size_t safe_size = 100;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error = 0.0;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M)
    {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        T* dA = (T*)dA_managed.get();
        T* dB = (T*)dB_managed.get();
        T* dC = (T*)dC_managed.get();
        T* dD = (T*)dD_managed.get();
        if(!dA || !dB || !dC || !dD)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_gemm_ex(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 &alpha_float,
                                 dA,
                                 a_type,
                                 lda,
                                 dB,
                                 b_type,
                                 ldb,
                                 &beta_float,
                                 dC,
                                 c_type,
                                 ldc,
                                 dD,
                                 d_type,
                                 ldd,
                                 compute_type,
                                 algo,
                                 kernel_index,
                                 flags);

        gemm_arg_check(status, M, N, K, lda, ldb, ldc);

        return status;
    }

    const size_t size_A = static_cast<size_t>(lda) * static_cast<size_t>(A_col);
    const size_t size_B = static_cast<size_t>(ldb) * static_cast<size_t>(B_col);
    const size_t size_C = static_cast<size_t>(ldc) * static_cast<size_t>(N);
    const size_t size_D = static_cast<size_t>(ldd) * static_cast<size_t>(N);

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                         rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),
                                         rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_C),
                                         rocblas_test::device_free};
    auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_D),
                                         rocblas_test::device_free};
    auto d_alpha_float_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(float)), rocblas_test::device_free};
    auto d_beta_float_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(float)), rocblas_test::device_free};
    T* dA                = (T*)dA_managed.get();
    T* dB                = (T*)dB_managed.get();
    T* dC                = (T*)dC_managed.get();
    T* dD                = (T*)dD_managed.get();
    float* d_alpha_float = (float*)d_alpha_float_managed.get();
    float* d_beta_float  = (float*)d_beta_float_managed.get();
    if(!dA || !dB || !dC || !dD || !d_alpha_float || !d_beta_float)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hB(size_B);
    vector<T> hC(size_C);
    vector<T> hD_1(size_D);
    vector<T> hD_2(size_D);
    vector<T> hD_gold(size_D);

    // Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, A_row, A_col, lda);
    rocblas_init_alternating_sign<T>(hB, B_row, B_col, ldb);
    rocblas_init<T>(hC, M, N, ldc);
    rocblas_init<T>(hD_1, M, N, ldd);
    /*
        if(is_same<T, rocblas_half>::value)
        {
            std::cout << "----A-------------------------------------------" << std::endl;
            for(int i = 0; i < size_A; i++){ cout << half_to_float(hA[i]) << "  "; }
            std::cout << std::endl << "-----B------------------------------------------" <<
       std::endl;
            for(int i = 0; i < size_B; i++){ cout << half_to_float(hB[i]) << "  "; }
            std::cout << std::endl << "-----C------------------------------------------" <<
       std::endl;
            for(int i = 0; i < size_C; i++){ cout << half_to_float(hC[i]) << "  "; }
            std::cout << std::endl << "-----D------------------------------------------" <<
       std::endl;
            for(int i = 0; i < size_D; i++){ cout << half_to_float(hD_1[i]) << "  "; }
            std::cout << std::endl << "------------------------------------------------" <<
       std::endl;
        }
        else
        {
            std::cout << "----A-------------------------------------------" << std::endl;
            for(int i = 0; i < size_A; i++){ cout << hA[i] << "  "; }
            std::cout << std::endl << "-----B------------------------------------------" <<
       std::endl;
            for(int i = 0; i < size_B; i++){ cout << hB[i] << "  "; }
            std::cout << std::endl << "-----C------------------------------------------" <<
       std::endl;
            for(int i = 0; i < size_C; i++){ cout << hC[i] << "  "; }
            std::cout << std::endl << "-----D------------------------------------------" <<
       std::endl;
            for(int i = 0; i < size_D; i++){ cout << hD_1[i] << "  "; }
            std::cout << std::endl << "------------------------------------------------" <<
       std::endl;
        }
    */
    hD_2    = hD_1;
    hD_gold = hD_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T) * size_C, hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_HIP_ERROR(hipMemcpy(dD, hD_1.data(), sizeof(T) * size_D, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            &alpha_float,
                                            dA,
                                            a_type,
                                            lda,
                                            dB,
                                            b_type,
                                            ldb,
                                            &beta_float,
                                            dC,
                                            c_type,
                                            ldc,
                                            dD,
                                            d_type,
                                            ldd,
                                            compute_type,
                                            algo,
                                            kernel_index,
                                            flags));

        CHECK_HIP_ERROR(hipMemcpy(hD_1.data(), dD, sizeof(T) * size_D, hipMemcpyDeviceToHost));
        /*
                std::cout << std::endl << "-----hD_1---------------------------------------" <<
        std::endl;
                if(is_same<T, rocblas_half>::value)
                {
                    for(int i = 0; i < size_D; i++){ cout << half_to_float(hD_1[i]) << "  "; }
        //          for(int i = 0; i < size_C; i++){ cout << std::hex << hD_1[i] << "  "; }
                }
                else
                {
                    for(int i = 0; i < size_D; i++){ cout << hD_1[i] << "  "; }
                }
                std::cout << std::endl << "------------------------------------------------" <<
        std::endl;
        */
        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(hipMemcpy(dD, hD_2.data(), sizeof(T) * size_D, hipMemcpyHostToDevice));

        CHECK_HIP_ERROR(
            hipMemcpy(d_alpha_float, &alpha_float, sizeof(float), hipMemcpyHostToDevice));

        CHECK_HIP_ERROR(hipMemcpy(d_beta_float, &beta_float, sizeof(float), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            d_alpha_float,
                                            dA,
                                            a_type,
                                            lda,
                                            dB,
                                            b_type,
                                            ldb,
                                            d_beta_float,
                                            dC,
                                            c_type,
                                            ldc,
                                            dD,
                                            d_type,
                                            ldd,
                                            compute_type,
                                            algo,
                                            kernel_index,
                                            flags));

        CHECK_HIP_ERROR(hipMemcpy(hD_2.data(), dD, sizeof(T) * size_D, hipMemcpyDeviceToHost));

        // CPU BLAS
        // copy C matrix into D matrix
        for(int i2 = 0; i2 < N; i2++)
        {
            for(int i1 = 0; i1 < M; i1++)
            {
                hD_gold[i1 + i2 * ldd] = hC[i1 + i2 * ldc];
            }
        }
        cpu_time_used = get_time_us();

        cblas_gemm<T>(transA,
                      transB,
                      M,
                      N,
                      K,
                      h_alpha,
                      hA.data(),
                      lda,
                      hB.data(),
                      ldb,
                      h_beta,
                      hD_gold.data(),
                      ldd);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;
/*
   std::cout << std::endl << "---gold---gold---gold---------------------------" << std::endl;
   if(is_same<T, rocblas_half>::value)
   {
       for(int i = 0; i < size_D; i++){ std::cout << half_to_float(hD_gold[i]) << "  "; }
//     for(int i = 0; i < size_D; i++){ std::cout << std::hex << hD_gold[i] << "  "; }
   }
   else
   {
       for(int i = 0; i < size_D; i++){ std::cout << hD_gold[i] << "  "; }
   }
   std::cout << std::endl << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
*/
#ifndef NDEBUG
// print_matrix(hC_gold, hC, min(M, 3), min(N, 3), ldc);
#endif

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(unit_check)
        {
            unit_check_general<T>(M, N, ldd, hD_gold.data(), hD_1.data());
            unit_check_general<T>(M, N, ldd, hD_gold.data(), hD_2.data());
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched
        // in compilation time
        if(norm_check)
        {
            rocblas_error = norm_check_general<T>('F', M, N, ldd, hD_gold.data(), hD_1.data());
            rocblas_error = norm_check_general<T>('F', M, N, ldd, hD_gold.data(), hD_2.data());
        }
    }

    if(timing)
    {
        int number_cold_calls = 2;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_gemm<T>(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
        }

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm<T>(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
        }
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<T>(M, N, K) * number_hot_calls / gpu_time_used * 1e6;

        cout << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,us";

        if(unit_check || norm_check)
            cout << ",CPU-Gflops(us),norm-error";

        cout << endl;

        cout << transA << "," << transB << "," << M << "," << N << "," << K << "," << h_alpha << ","
             << lda << "," << ldb << "," << h_beta << "," << ldc << "," << rocblas_gflops << ","
             << gpu_time_used / number_hot_calls;

        if(unit_check || norm_check)
        {
            cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;
        }

        cout << endl;
    }
    return status;
}

// template <typename T>
rocblas_status testing_gemm_ex(Arguments argus)
{
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;
    rocblas_int ldd = argus.ldd;

    rocblas_precision a_type       = argus.a_type;
    rocblas_precision b_type       = argus.b_type;
    rocblas_precision c_type       = argus.c_type;
    rocblas_precision d_type       = argus.d_type;
    rocblas_precision compute_type = argus.compute_type;

    float alpha = argus.alpha;
    float beta  = argus.beta;

    rocblas_int norm_check = argus.norm_check;
    rocblas_int unit_check = argus.unit_check;
    rocblas_int timing     = argus.timing;
    int number_hot_calls   = argus.iters;

    if(a_type == rocblas_precision_half && b_type == rocblas_precision_half &&
       c_type == rocblas_precision_half && d_type == rocblas_precision_half &&
       compute_type == rocblas_precision_half)
    {
        return testing_gemm_ex_template<rocblas_half>(transA,
                                                      transB,
                                                      M,
                                                      N,
                                                      K,
                                                      alpha,
                                                      lda,
                                                      ldb,
                                                      beta,
                                                      ldc,
                                                      ldd,
                                                      norm_check,
                                                      unit_check,
                                                      timing,
                                                      number_hot_calls);
    }
    else if(a_type == rocblas_precision_single && b_type == rocblas_precision_single &&
            c_type == rocblas_precision_single && d_type == rocblas_precision_single &&
            compute_type == rocblas_precision_single)
    {
        return testing_gemm_ex_template<float>(transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               alpha,
                                               lda,
                                               ldb,
                                               beta,
                                               ldc,
                                               ldd,
                                               norm_check,
                                               unit_check,
                                               timing,
                                               number_hot_calls);
    }
    else if(a_type == rocblas_precision_double && b_type == rocblas_precision_double &&
            c_type == rocblas_precision_double && d_type == rocblas_precision_double &&
            compute_type == rocblas_precision_double)
    {
        return testing_gemm_ex_template<double>(transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                alpha,
                                                lda,
                                                ldb,
                                                beta,
                                                ldc,
                                                ldd,
                                                norm_check,
                                                unit_check,
                                                timing,
                                                number_hot_calls);
    }
    else
    {
        return rocblas_status_not_implemented;
    }
}
