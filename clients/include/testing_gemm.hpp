/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
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
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_gemm_fn = FORTRAN ? rocblas_gemm<T, true> : rocblas_gemm<T, false>;

    const rocblas_int M = 100;
    const rocblas_int N = 100;
    const rocblas_int K = 100;

    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;
    const rocblas_int ldc = 100;

    const T alpha = 1.0;
    const T beta  = 1.0;

    const size_t safe_size = 100;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_local_handle handle;

    // allocate memory on device
    device_vector<T> dA(safe_size);
    device_vector<T> dB(safe_size);
    device_vector<T> dC(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &alpha, nullptr, lda, dB, ldb, &beta, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &alpha, dA, lda, nullptr, ldb, &beta, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, nullptr, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemm_fn(handle, transA, transB, M, N, K, nullptr, dA, lda, dB, ldb, &beta, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, nullptr, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemm_fn(nullptr, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc),
        rocblas_status_invalid_handle);

    // if k == 0 we can test that A and B can both be nullptr without issue.
    EXPECT_ROCBLAS_STATUS(
        rocblas_gemm_fn(
            handle, transA, transB, M, N, 0, &alpha, nullptr, lda, nullptr, ldb, &beta, dC, ldc),
        rocblas_status_success);
}

template <typename T>
void testing_gemm(const Arguments& arg)
{
    const bool FORTRAN         = arg.fortran;
    auto       rocblas_gemm_fn = FORTRAN ? rocblas_gemm<T, true> : rocblas_gemm<T, false>;

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

    double               gpu_time_used, cpu_time_used;
    double               rocblas_gflops, cblas_gflops;
    double               rocblas_error = 0.0;
    rocblas_local_handle handle;

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

    const auto size_A      = size_t(lda) * size_t(A_col);
    const auto size_B      = size_t(ldb) * size_t(B_col);
    const auto size_C      = size_t(ldc) * size_t(N);
    const auto size_C_copy = arg.unit_check || arg.norm_check ? size_C : 0;

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> dC(size_C);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hC_1(size_C);
    host_vector<T> hC_2(size_C_copy);
    host_vector<T> hC_gold(size_C_copy);

    // Initial Data on CPU
    if(arg.initialization == rocblas_initialization_random_int)
    {
        rocblas_seedrand();
        rocblas_init<T>(hA, A_row, A_col, lda);
        rocblas_init_alternating_sign<T>(hB, B_row, B_col, ldb);
        if(rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai))
            rocblas_init_nan<T>(hC_1, M, N, ldc);
        else
            rocblas_init<T>(hC_1, M, N, ldc);
    }
    else if(arg.initialization == rocblas_initialization_trig_float)
    {
        rocblas_init_sin<T>(hA, A_row, A_col, lda);
        rocblas_init_cos<T>(hB, B_row, B_col, ldb);
        if(rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai))
            rocblas_init_nan<T>(hC_1, M, N, ldc);
        else
            rocblas_init_sin<T>(hC_1, M, N, ldc);
    }
    else if(arg.initialization == rocblas_initialization_hpl)
    {
        rocblas_seedrand();
        rocblas_init_hpl<T>(hA, A_row, A_col, lda);
        rocblas_init_hpl<T>(hB, B_row, B_col, ldb);
        if(rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai))
            rocblas_init_nan<T>(hC_1, M, N, ldc);
        else
            rocblas_init_hpl<T>(hC_1, M, N, ldc);
    }

    if(size_C_copy)
    {
        hC_2    = hC_1;
        hC_gold = hC_1;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_1, sizeof(T) * size_C, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_2, sizeof(T) * size_C, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        // CPU BLAS
        if(arg.timing)
        {
            cpu_time_used = get_time_us();
        }

        cblas_gemm<T>(transA, transB, M, N, K, h_alpha, hA, lda, hB, ldb, h_beta, hC_gold, ldc);

        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = gemm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;
        }

        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<T>;
                near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
                near_check_general<T>(M, N, ldc, hC_gold, hC_2, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
                unit_check_general<T>(M, N, ldc, hC_gold, hC_2);
            }
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1));
            auto err2     = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2));
            rocblas_error = err1 > err2 ? err1 : err2;
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

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_fn(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
        }
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<T>(M, N, K) * number_hot_calls / gpu_time_used * 1e6;

        rocblas_cout << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,us";

        if(arg.unit_check || arg.norm_check)
            rocblas_cout << ",CPU-Gflops,us,norm-error";

        rocblas_cout << std::endl;

        rocblas_cout << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << ","
                     << arg.get_alpha<T>() << "," << lda << "," << ldb << "," << arg.get_beta<T>()
                     << "," << ldc << "," << rocblas_gflops << ","
                     << gpu_time_used / number_hot_calls;

        if(arg.unit_check || arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        rocblas_cout << std::endl;
    }
}
