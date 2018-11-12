/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <mutex>
#include <condition_variable>

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

std::mutex memcpy_mutex;

/* ============================================================================================ */

template <typename T>
rocblas_status testing_gemm_parallel(Arguments const& argus,
                                     // std::shared_future<void> & start_rocblas,
                                     std::condition_variable& cv,
                                     int& waiting_threads,
                                     int total_threads)
{
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;

    T h_alpha;
    T h_beta;
    if(is_same<T, rocblas_half>::value)
    {
        float alpha_float = argus.alpha;
        float beta_float  = argus.beta;

        h_alpha = float_to_half(alpha_float);
        h_beta  = float_to_half(beta_float);
    }
    else
    {
        h_alpha = argus.alpha;
        h_beta  = argus.beta;
    }

    const size_t safe_size = 100;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    double rocblas_error = 0.0;

    rocblas_status status = rocblas_status_success;

    std::unique_lock<std::mutex> lock(memcpy_mutex);

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    const auto size_A = static_cast<size_t>(lda) * static_cast<size_t>(A_col);
    const auto size_B = static_cast<size_t>(ldb) * static_cast<size_t>(B_col);
    const auto size_C = static_cast<size_t>(ldc) * static_cast<size_t>(N);

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                         rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),
                                         rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_C),
                                         rocblas_test::device_free};
    auto d_alpha_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    auto d_beta_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    auto dA      = static_cast<T*>(dA_managed.get());
    auto dB      = static_cast<T*>(dB_managed.get());
    auto dC      = static_cast<T*>(dC_managed.get());
    auto d_alpha = static_cast<T*>(d_alpha_managed.get());
    auto d_beta  = static_cast<T*>(d_beta_managed.get());
    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hB(size_B);
    vector<T> hC_1(size_C);
    vector<T> hC_2(size_C);
    vector<T> hC_gold(size_C);

    // Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, A_row, A_col, lda);
    rocblas_init_alternating_sign<T>(hB, B_row, B_col, ldb);
    rocblas_init<T>(hC_1, M, N, ldc);

    //  std::cout << "------------------------------------------------" << std::endl;
    //  for(int i = 0; i < size_A; i++){ cout << half_to_float(hA[i]) << "  "; }
    //  std::cout << std::endl << "------------------------------------------------" << std::endl;
    //  for(int i = 0; i < size_B; i++){ cout << half_to_float(hB[i]) << "  "; }
    //  std::cout << std::endl << "------------------------------------------------" << std::endl;
    //  for(int i = 0; i < size_C; i++){ cout << half_to_float(hC_1[i]) << "  "; }
    //  std::cout << std::endl << "------------------------------------------------" << std::endl;

    hC_2    = hC_1;
    hC_gold = hC_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));

    // ROCBLAS rocblas_pointer_mode_host
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    CHECK_HIP_ERROR(hipMemcpy(dC, hC_1.data(), sizeof(T) * size_C, hipMemcpyHostToDevice));

    waiting_threads++;
    cv.notify_all();
    while(waiting_threads < total_threads)
        cv.wait(lock);

    lock.unlock();

    CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
        handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
        handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
    // CPU BLAS
    if(argus.timing)
    {
        cpu_time_used = get_time_us();
    }

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
                  hC_gold.data(),
                  ldc);

    if(argus.timing)
    {
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;
    }

    lock.lock();

    CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(hipMemcpy(dC, hC_2.data(), sizeof(T) * size_C, hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

//  std::cout << std::endl << "---gold---gold---gold---------------------------" << std::endl;
//  for(int i = 0; i < size_C; i++){ std::cout << half_to_float(hC_gold[i]) << "  "; }
//  std::cout << std::endl << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
#ifndef NDEBUG
// print_matrix(hC_gold, hC, min(M, 3), min(N, 3), ldc);
#endif

    // enable unit check, notice unit check is not invasive, but norm check is,
    // unit check and norm check can not be interchanged their order
    if(argus.unit_check)
    {
        unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_1.data());
        unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_2.data());
    }

    // if enable norm check, norm check is invasive
    // any typeinfo(T) will not work here, because template deduction is matched
    // in compilation time
    if(argus.norm_check)
    {
        double error_hst_ptr =
            fabs(norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_1.data()));
        double error_dev_ptr =
            fabs(norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_2.data()));
        rocblas_error = error_hst_ptr > error_dev_ptr ? error_hst_ptr : error_dev_ptr;
    }

    return status;
}
