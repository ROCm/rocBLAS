/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_gemm_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_fn = arg.fortran ? rocblas_gemm<T, true> : rocblas_gemm<T, false>;

        const rocblas_int M = 100;
        const rocblas_int N = 100;
        const rocblas_int K = 100;

        const rocblas_int lda = 100;
        const rocblas_int ldb = 100;
        const rocblas_int ldc = 100;

        device_vector<T> alpha_d(1), beta_d(1), zero_d(1), one_d(1);
        const T          alpha_h(1), beta_h(1), zero_h(0), one_h(1);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* zero  = &zero_h;
        const T* one   = &one_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
        }

        const size_t safe_size = 100;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // allocate memory on device
        device_vector<T> dA(safe_size);
        device_vector<T> dB(safe_size);
        device_vector<T> dC(safe_size);
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemm_fn(
                handle, transA, transB, M, N, K, alpha, nullptr, lda, dB, ldb, beta, dC, ldc),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemm_fn(
                handle, transA, transB, M, N, K, alpha, dA, lda, nullptr, ldb, beta, dC, ldc),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemm_fn(
                handle, transA, transB, M, N, K, alpha, dA, lda, dB, ldb, beta, nullptr, ldc),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemm_fn(
                handle, transA, transB, M, N, K, nullptr, dA, lda, dB, ldb, beta, dC, ldc),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemm_fn(
                handle, transA, transB, M, N, K, alpha, dA, lda, dB, ldb, nullptr, dC, ldc),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemm_fn(
                nullptr, transA, transB, M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc),
            rocblas_status_invalid_handle);

        // If M==0, then all pointers can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle,
                                              transA,
                                              transB,
                                              0,
                                              N,
                                              K,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              nullptr,
                                              ldc),
                              rocblas_status_success);

        // If N==0, then all pointers can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle,
                                              transA,
                                              transB,
                                              M,
                                              0,
                                              K,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              nullptr,
                                              ldc),
                              rocblas_status_success);

        // If K==0, then alpha, A and B can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle,
                                              transA,
                                              transB,
                                              M,
                                              N,
                                              0,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              beta,
                                              dC,
                                              ldc),
                              rocblas_status_success);

        // If alpha==0, then A and B can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(
            rocblas_gemm_fn(
                handle, transA, transB, M, N, K, zero, nullptr, lda, nullptr, ldb, beta, dC, ldc),
            rocblas_status_success);

        // If alpha==0 && beta==1, then A, B and C can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle,
                                              transA,
                                              transB,
                                              M,
                                              N,
                                              K,
                                              zero,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              one,
                                              nullptr,
                                              ldc),
                              rocblas_status_success);
    }
}

template <typename T>
void testing_gemm(const Arguments& arg)
{
    auto rocblas_gemm_fn = arg.fortran ? rocblas_gemm<T, true> : rocblas_gemm<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used      = 0.0;
    double               rocblas_error = 0.0;
    bool                 HMM           = arg.HMM;
    rocblas_local_handle handle{arg};

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // check here to prevent undefined memory allocation error
    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M;
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle,
                                              transA,
                                              transB,
                                              M,
                                              N,
                                              K,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              nullptr,
                                              ldc),
                              rocblas_status_invalid_size);

        return;
    }

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        device_vector<T> dA(1);
        device_vector<T> dB(1);
        device_vector<T> dC(1);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));
        return;
    }
#endif

    const size_t size_A      = size_t(lda) * A_col;
    const size_t size_B      = size_t(ldb) * B_col;
    const size_t size_C      = size_t(ldc) * N;
    const size_t size_C_copy = arg.unit_check || arg.norm_check ? size_C : 0;

    // allocate memory on device
    device_vector<T> dA(size_A, 1, HMM);
    device_vector<T> dB(size_B, 1, HMM);
    device_vector<T> dC(size_C, 1, HMM);
    device_vector<T> d_alpha(1, 1, HMM);
    device_vector<T> d_beta(1, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hC_1(size_C);
    host_vector<T> hC_gold(size_C_copy);

    // Initialize data on host memory
    rocblas_init_matrix(hA,
                        arg,
                        A_row,
                        A_col,
                        lda,
                        0,
                        1,
                        rocblas_client_alpha_sets_nan,
                        rocblas_client_general_matrix,
                        true);
    rocblas_init_matrix(hB,
                        arg,
                        B_row,
                        B_col,
                        ldb,
                        0,
                        1,
                        rocblas_client_alpha_sets_nan,
                        rocblas_client_general_matrix,
                        false,
                        true);
    rocblas_init_matrix(
        hC_1, arg, M, N, ldc, 0, 1, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    if(size_C_copy)
    {
        hC_gold = hC_1;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    CHECK_HIP_ERROR(dC.transfer_from(hC_1));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // gold has copy of hC1
        CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        cblas_gemm<T>(transA, transB, M, N, K, h_alpha, hA, lda, hB, ldb, h_beta, hC_gold, ldc);

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        //releasing already used host memory
        hA = host_vector<T>();
        hB = host_vector<T>();

        // check host error and norm
        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<T>;
                near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
            }
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1));
            rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
        }

        // fetch device mode GPU results
        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<T>;
                near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
            }
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1));
            rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_fn(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_K, e_alpha, e_lda, e_beta, e_ldb, e_ldc>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gemm_gflop_count<T>(M, N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error);
    }
}
