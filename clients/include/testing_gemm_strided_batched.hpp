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
void testing_gemm_strided_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_gemm_strided_batched_fn
        = FORTRAN ? rocblas_gemm_strided_batched<T, true> : rocblas_gemm_strided_batched<T, false>;

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

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

    // check here to prevent undefined memory allocation error
    // Note: K==0 is not an early exit, since C must still be multiplied by beta
    bool invalid_size
        = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              K,
                                                              nullptr,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              ldb,
                                                              stride_b,
                                                              nullptr,
                                                              nullptr,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    double rocblas_error = 0.0;

    size_t size_one_a
        = transA == rocblas_operation_none ? size_t(K) * size_t(lda) : size_t(M) * size_t(lda);
    size_t size_one_b
        = transB == rocblas_operation_none ? size_t(N) * size_t(ldb) : size_t(K) * size_t(ldb);
    size_t size_one_c = N * ldc;

    size_t     size_a      = size_one_a + size_t(stride_a) * size_t(batch_count - 1);
    size_t     size_b      = size_one_b + size_t(stride_b) * size_t(batch_count - 1);
    size_t     size_c      = size_one_c + size_t(stride_c) * size_t(batch_count - 1);
    const auto size_c_copy = arg.unit_check || arg.norm_check ? size_c : 0;

    // allocate memory on device
    device_vector<T> dA(size_a);
    device_vector<T> dB(size_b);
    device_vector<T> dC(size_c);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hA(size_a);
    host_vector<T> hB(size_b);
    host_vector<T> hC_1(size_c);
    host_vector<T> hC_2(size_c_copy);
    host_vector<T> hC_gold(size_c_copy);

    // Initial Data on CPU
    rocblas_seedrand();

    rocblas_init<T>(hA, A_row, A_col, lda, stride_a, batch_count);
    rocblas_init_alternating_sign<T>(hB, B_row, B_col, ldb, stride_b, batch_count);
    if(rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai))
        rocblas_init_nan<T>(hC_1, M, N, ldc, stride_c, batch_count);
    else
        rocblas_init<T>(hC_1, M, N, ldc, stride_c, batch_count);

    if(size_c_copy)
    {
        hC_2    = hC_1;
        hC_gold = hC_1;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_b, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_1, sizeof(T) * size_c, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_fn(handle,
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

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_fn(handle,
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
            cblas_gemm<T>(transA,
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
                near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_1, batch_count, tol);
                near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_2, batch_count, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
                unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_2, batch_count);
            }
        }

        if(arg.norm_check)
        {
            double error_hst_ptr = std::abs(
                norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC_1, batch_count));
            double error_dev_ptr = std::abs(
                norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC_2, batch_count));
            rocblas_error = error_hst_ptr > error_dev_ptr ? error_hst_ptr : error_dev_ptr;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_fn(handle,
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
            rocblas_gemm_strided_batched_fn(handle,
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

        rocblas_cout
            << "transA,transB,M,N,K,alpha,lda,stride_a,ldb,stride_b,beta,ldc,stride_c,Batch_Count,"
               "rocblas-Gflops,"
               "us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,us,norm-error";

        rocblas_cout << std::endl;

        rocblas_cout << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << ","
                     << arg.get_alpha<T>() << "," << lda << "," << stride_a << "," << ldb << ","
                     << stride_b << "," << arg.get_beta<T>() << "," << ldc << "," << stride_c << ","
                     << batch_count << "," << rocblas_gflops << "," << gpu_time_used;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        rocblas_cout << std::endl;
    }
}
