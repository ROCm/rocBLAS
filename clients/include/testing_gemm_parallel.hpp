/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <mutex>
#include <condition_variable>

#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "utility.hpp"
#include "cblas_interface.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "flops.hpp"

std::mutex memcpy_mutex;

/* ============================================================================================ */

template <typename T>
void testing_gemm_parallel(const Arguments& arg,
                           // std::shared_future<void> & start_rocblas,
                           std::condition_variable& cv,
                           int& waiting_threads,
                           int total_threads)
{
    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T h_alpha;
    T h_beta;
    if(std::is_same<T, rocblas_half>{})
    {
        h_alpha = float_to_half(arg.alpha);
        h_beta  = float_to_half(arg.beta);
    }
    else
    {
        h_alpha = arg.alpha;
        h_beta  = arg.beta;
    }

    double rocblas_error = 0.0;

    std::unique_lock<std::mutex> lock(memcpy_mutex);

    rocblas_local_handle handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        static const size_t safe_size = 100;
        device_vector<T> dA(safe_size);
        device_vector<T> dB(safe_size);
        device_vector<T> dC(safe_size);
        if(!dA || !dB || !dC)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemm<T>(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc),
            rocblas_status_invalid_size);
        return;
    }

    const auto size_A = static_cast<size_t>(lda) * static_cast<size_t>(A_col);
    const auto size_B = static_cast<size_t>(ldb) * static_cast<size_t>(B_col);
    const auto size_C = static_cast<size_t>(ldc) * static_cast<size_t>(N);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> dC(size_C);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hC_1(size_C);
    host_vector<T> hC_2(size_C);
    host_vector<T> hC_gold(size_C);

    // Initial Data on CPU
    rocblas_seedrand();
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
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));

    // ROCBLAS rocblas_pointer_mode_host
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    CHECK_HIP_ERROR(hipMemcpy(dC, hC_1, sizeof(T) * size_C, hipMemcpyHostToDevice));

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

    cblas_gemm<T, T>(transA, transB, M, N, K, h_alpha, hA, lda, hB, ldb, h_beta, hC_gold, ldc);

    lock.lock();

    CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_2, sizeof(T) * size_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(hC_2, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

    if(arg.unit_check)
    {
        unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_1.data());
        unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_2.data());
    }

    if(arg.norm_check)
    {
        double error_hst_ptr =
            fabs(norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_1.data()));
        double error_dev_ptr =
            fabs(norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_2.data()));
        rocblas_error = error_hst_ptr > error_dev_ptr ? error_hst_ptr : error_dev_ptr;
    }
}
