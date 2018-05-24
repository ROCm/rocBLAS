/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

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

template <typename T>
rocblas_status testing_gemm_strided_batched(Arguments argus)
{
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    T h_alpha = argus.alpha;
    T h_beta  = argus.beta;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;

    rocblas_int stride_a    = argus.stride_a;
    rocblas_int stride_b    = argus.stride_b;
    rocblas_int stride_c    = argus.stride_c;
    rocblas_int batch_count = argus.batch_count;

    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int safe_size = 100; // arbitrarily set to 100

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 || batch_count <= 0 ||
       stride_a <= 0 || stride_b <= 0 || stride_c <= 0)
    {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        T* dA = (T*)dA_managed.get();
        T* dB = (T*)dB_managed.get();
        T* dC = (T*)dC_managed.get();
        if(!dA || !dB || !dC)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_gemm_strided_batched<T>(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 &h_alpha,
                                                 dA,
                                                 lda,
                                                 stride_a,
                                                 dB,
                                                 ldb,
                                                 stride_b,
                                                 &h_beta,
                                                 dC,
                                                 ldc,
                                                 stride_c,
                                                 batch_count);

        gemm_strided_batched_arg_check(
            status, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count);

        return status;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error = 0.0;

    size_t size_one_a = transA == rocblas_operation_none
                            ? static_cast<size_t>(K) * static_cast<size_t>(lda)
                            : static_cast<size_t>(M) * static_cast<size_t>(lda);
    size_t size_one_b = transB == rocblas_operation_none
                            ? static_cast<size_t>(N) * static_cast<size_t>(ldb)
                            : static_cast<size_t>(K) * static_cast<size_t>(ldb);
    size_t size_one_c = N * ldc;

    size_t size_a =
        size_one_a + static_cast<size_t>(stride_a) * static_cast<size_t>(batch_count - 1);
    size_t size_b =
        size_one_b + static_cast<size_t>(stride_b) * static_cast<size_t>(batch_count - 1);
    size_t size_c =
        size_one_c + static_cast<size_t>(stride_c) * static_cast<size_t>(batch_count - 1);

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_a),
                                         rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_b),
                                         rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_c),
                                         rocblas_test::device_free};
    auto d_alpha_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    auto d_beta_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    T* dA      = (T*)dA_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* dC      = (T*)dC_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();
    if((!dA && (size_a != 0)) || (!dB && (size_b != 0)) || (!dC && (size_c != 0)) || !d_alpha ||
       !d_beta)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hA(size_a);
    vector<T> hB(size_b);
    vector<T> hC_1(size_c);
    vector<T> hC_2(size_c);
    vector<T> hC_gold(size_c);

    // Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, A_row, A_col * batch_count, lda);
    rocblas_init<T>(hB, B_row, B_col * batch_count, ldb);
    rocblas_init<T>(hC_1, M, N * batch_count, ldc);
    hC_2    = hC_1;
    hC_gold = hC_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * size_b, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_1.data(), sizeof(T) * size_c, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched<T>(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            &h_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            &h_beta,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC, sizeof(T) * size_c, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_2.data(), sizeof(T) * size_c, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched<T>(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            d_beta,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC, sizeof(T) * size_c, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(rocblas_int i = 0; i < batch_count; i++)
        {
            cblas_gemm<T>(transA,
                          transB,
                          M,
                          N,
                          K,
                          h_alpha,
                          hA.data() + stride_a * i,
                          lda,
                          hB.data() + stride_b * i,
                          ldb,
                          h_beta,
                          hC_gold.data() + stride_c * i,
                          ldc);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<T>(M, N, K) * batch_count / cpu_time_used * 1e6;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold.data(), hC_1.data());
            unit_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold.data(), hC_2.data());
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            rocblas_error =
                norm_check_general<T>('F', M, N * batch_count, lda, hC_gold.data(), hC_1.data());
            rocblas_error =
                norm_check_general<T>('F', M, N * batch_count, lda, hC_gold.data(), hC_2.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_gemm_strided_batched<T>(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            &h_alpha,
                                            dA,
                                            lda,
                                            stride_a,
                                            dB,
                                            ldb,
                                            stride_b,
                                            &h_beta,
                                            dC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_strided_batched<T>(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            &h_alpha,
                                            dA,
                                            lda,
                                            stride_a,
                                            dB,
                                            ldb,
                                            stride_b,
                                            &h_beta,
                                            dC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }

        gpu_time_used  = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops = gemm_gflop_count<T>(M, N, K) * batch_count / gpu_time_used * 1e6;

        cout << "transA,transB,M,N,K,alpha,lda,stride_a,ldb,stride_b,beta,ldc,stride_c,Batch_Count,"
                "rocblas-Gflops,"
                "us";

        if(argus.norm_check)
            cout << ",CPU-Gflops,us,norm-error";

        cout << endl;

        cout << argus.transA_option << "," << argus.transB_option << "," << M << "," << N << ","
             << K << "," << h_alpha << "," << lda << "," << stride_a << "," << ldb << ","
             << stride_b << "," << h_beta << "," << ldc << "," << stride_c << "," << batch_count
             << "," << rocblas_gflops << "," << gpu_time_used;

        if(argus.norm_check)
            cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        cout << endl;
    }

    return rocblas_status_success;
}
