/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "rocblas_datatype2string.hpp"
#include "utility.hpp"
#include "rocblas.hpp"
#include "cblas_interface.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "near.hpp"
#include "flops.hpp"

template <typename T>
void testing_gemm_strided_batched(const Arguments& arg)
{
    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    T h_alpha;
    T h_beta;
    if(std::is_same<T, rocblas_half>{})
    {
        h_alpha = float_to_half(arg.alpha);
        h_beta  = rocblas_isnan(arg.beta) ? 0 : float_to_half(arg.beta);
    }
    else
    {
        h_alpha = arg.alpha;
        h_beta  = rocblas_isnan(arg.beta) ? 0 : arg.beta;
    }

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    rocblas_int stride_a    = arg.stride_a;
    rocblas_int stride_b    = arg.stride_b;
    rocblas_int stride_c    = arg.stride_c;
    rocblas_int batch_count = arg.batch_count;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_local_handle handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // Early exit
    if(!M || !N || !batch_count)
        return;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T> dA(safe_size);
        device_vector<T> dB(safe_size);
        device_vector<T> dC(safe_size);
        if(!dA || !dB || !dC)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched<T>(handle,
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
                                                              batch_count),
                              rocblas_status_invalid_size);
        return;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    double rocblas_error = 0.0;

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
    device_vector<T> dA(size_a);
    device_vector<T> dB(size_b);
    device_vector<T> dC(size_c);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    if((!dA && size_a) || (!dB && size_b) || (!dC && size_c) || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hA(size_a);
    host_vector<T> hB(size_b);
    host_vector<T> hC_1(size_c);
    host_vector<T> hC_2(size_c);
    host_vector<T> hC_gold(size_c);

    // Initial Data on CPU
    rocblas_seedrand();

    rocblas_init<T>(hA, A_row, A_col, lda, stride_a, batch_count);
    rocblas_init_alternating_sign<T>(hB, B_row, B_col, ldb, stride_b, batch_count);
    if(rocblas_isnan(arg.beta))
        rocblas_init_nan<T>(hC_1, M, N, ldc, stride_c, batch_count);
    else
        rocblas_init<T>(hC_1, M, N, ldc, stride_c, batch_count);

    hC_2    = hC_1;
    hC_gold = hC_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_b, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_1, sizeof(T) * size_c, hipMemcpyHostToDevice));

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

        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_c, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_2, sizeof(T) * size_c, hipMemcpyHostToDevice));
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

        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC, sizeof(T) * size_c, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(rocblas_int i = 0; i < batch_count; i++)
        {
            cblas_gemm<T, T>(transA,
                             transB,
                             M,
                             N,
                             K,
                             h_alpha,
                             hA + stride_a * i,
                             lda,
                             hB + stride_b * i,
                             ldb,
                             h_beta,
                             hC_gold + stride_c * i,
                             ldc);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<T>(M, N, K) * batch_count / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<T>;
                near_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold, hC_1, tol);
                near_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold, hC_2, tol);
            }
            else
            {
                unit_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold, hC_1);
                unit_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold, hC_2);
            }
        }

        if(arg.norm_check)
        {
            double error_hst_ptr =
                fabs(norm_check_general<T>('F', M, N, ldc, stride_c, batch_count, hC_gold, hC_1));
            double error_dev_ptr =
                fabs(norm_check_general<T>('F', M, N, ldc, stride_c, batch_count, hC_gold, hC_2));
            rocblas_error = error_hst_ptr > error_dev_ptr ? error_hst_ptr : error_dev_ptr;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
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

        std::cout
            << "transA,transB,M,N,K,alpha,lda,stride_a,ldb,stride_b,beta,ldc,stride_c,Batch_Count,"
               "rocblas-Gflops,"
               "us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,us,norm-error";

        std::cout << std::endl;

        std::cout << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << ","
                  << (std::is_same<T, rocblas_half>{} ? half_to_float(h_alpha) : h_alpha) << ","
                  << lda << "," << stride_a << "," << ldb << "," << stride_b << ","
                  << (std::is_same<T, rocblas_half>{} ? half_to_float(h_beta) : h_beta) << ","
                  << ldc << "," << stride_c << "," << batch_count << "," << rocblas_gflops << ","
                  << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        std::cout << std::endl;
    }
}
