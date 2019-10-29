/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
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
void testing_gemm_batched(const Arguments& arg)
{
    rocblas_local_handle handle;
    rocblas_int          M           = arg.M;
    rocblas_int          N           = arg.N;
    rocblas_int          K           = arg.K;
    T                    h_alpha     = arg.alpha;
    T                    h_beta      = rocblas_isnan(arg.beta) ? 0 : arg.beta;
    rocblas_int          lda         = arg.lda;
    rocblas_int          ldb         = arg.ldb;
    rocblas_int          ldc         = arg.ldc;
    rocblas_int          batch_count = arg.batch_count;
    rocblas_operation    transA      = char2rocblas_operation(arg.transA);
    rocblas_operation    transB      = char2rocblas_operation(arg.transB);
    rocblas_int          A_row       = transA == rocblas_operation_none ? M : K;
    rocblas_int          A_col       = transA == rocblas_operation_none ? K : M;
    rocblas_int          B_row       = transB == rocblas_operation_none ? K : N;
    rocblas_int          B_col       = transB == rocblas_operation_none ? N : K;

    // check here to prevent undefined memory allocation error
    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta.
    if(M <= 0 || N <= 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count <= 0)
    {
        rocblas_int safe_size = 100;
        rocblas_int num_batch = batch_count < 0 ? 1 : batch_count;

        device_vector<T*, 0, T> dA(1);
        device_vector<T*, 0, T> dB(1);
        device_vector<T*, 0, T> dC(1);

        if(!dA || !dB || !dC)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched<T>(handle,
                                                      transA,
                                                      transB,
                                                      M,
                                                      N,
                                                      K,
                                                      &h_alpha,
                                                      dA,
                                                      lda,
                                                      dB,
                                                      ldb,
                                                      &h_beta,
                                                      dC,
                                                      ldc,
                                                      batch_count),
                              !M || !N || !batch_count ? rocblas_status_success
                                                       : rocblas_status_invalid_size);

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

    size_t size_a = size_one_a;
    size_t size_b = size_one_b;
    size_t size_c = size_one_c;

    // allocate memory on device
    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dB(batch_count);
    device_vector<T*, 0, T> dC(batch_count);
    device_vector<T>        d_alpha(1);
    device_vector<T>        d_beta(1);

    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hA[batch_count];
    host_vector<T> hB[batch_count];
    host_vector<T> hC_1[batch_count];
    host_vector<T> hC_2[batch_count];
    host_vector<T> hC_gold[batch_count];

    for(int i = 0; i < batch_count; i++)
    {
        hA[i]      = host_vector<T>(size_a);
        hB[i]      = host_vector<T>(size_b);
        hC_1[i]    = host_vector<T>(size_c);
        hC_2[i]    = host_vector<T>(size_c);
        hC_gold[i] = host_vector<T>(size_c);
    }

    device_batch_vector<T> Av(batch_count, size_a);
    device_batch_vector<T> Bv(batch_count, size_b);
    device_batch_vector<T> Cv(batch_count, size_c);

    int last = batch_count - 1;
    if((!Av[last] && size_a) || (!Bv[last] && size_b) || (!Cv[last] && size_c))
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    for(int i = 0; i < batch_count; i++)
    {
        rocblas_init<T>(hA[i], A_row, A_col, lda);

        rocblas_init_alternating_sign<T>(hB[i], B_row, B_col, ldb);

        if(rocblas_isnan(arg.beta))
            rocblas_init_nan<T>(hC_1[i], M, N, ldc);
        else
            rocblas_init<T>(hC_1[i], M, N, ldc);

        hC_2[i]    = hC_1[i];
        hC_gold[i] = hC_1[i];
    }

    // 1. Use intermediate arrays to access device memory from host
    for(int i = 0; i < batch_count; i++)
    {
        CHECK_HIP_ERROR(hipMemcpy(Av[i], hA[i], sizeof(T) * size_a, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(Bv[i], hB[i], sizeof(T) * size_b, hipMemcpyHostToDevice));
    }
    // 2. Copy intermediate arrays into device arrays
    CHECK_HIP_ERROR(hipMemcpy(dA, Av, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, Bv, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, Cv, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < batch_count; i++)
        {
            CHECK_HIP_ERROR(hipMemcpy(Cv[i], hC_1[i], sizeof(T) * size_c, hipMemcpyHostToDevice));
        }
        CHECK_ROCBLAS_ERROR((rocblas_gemm_batched<T>(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     &h_alpha,
                                                     dA,
                                                     lda,
                                                     dB,
                                                     ldb,
                                                     &h_beta,
                                                     dC,
                                                     ldc,
                                                     batch_count)));

        for(int i = 0; i < batch_count; i++)
        {
            CHECK_HIP_ERROR(hipMemcpy(hC_1[i], Cv[i], sizeof(T) * size_c, hipMemcpyDeviceToHost));
        }

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int i = 0; i < batch_count; i++)
        {
            CHECK_HIP_ERROR(hipMemcpy(Cv[i], hC_2[i], sizeof(T) * size_c, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
        for(int i = 0; i < batch_count; i++)
        {
            CHECK_HIP_ERROR(hipMemcpy(Cv[i], hC_2[i], sizeof(T) * size_c, hipMemcpyHostToDevice));
        }

        CHECK_ROCBLAS_ERROR((rocblas_gemm_batched<T>(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     d_alpha,
                                                     dA,
                                                     lda,
                                                     dB,
                                                     ldb,
                                                     d_beta,
                                                     dC,
                                                     ldc,
                                                     batch_count)));

        for(int i = 0; i < batch_count; i++)
        {
            CHECK_HIP_ERROR(hipMemcpy(hC_2[i], Cv[i], sizeof(T) * size_c, hipMemcpyDeviceToHost));
        }
        // CPU BLAS
        cpu_time_used = get_time_us();
        for(rocblas_int i = 0; i < batch_count; i++)
        {
            cblas_gemm<T, T>(
                transA, transB, M, N, K, h_alpha, hA[i], lda, hB[i], ldb, h_beta, hC_gold[i], ldc);
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
                near_check_general<T>(M, N, batch_count, ldc, hC_gold, hC_1, tol);
                near_check_general<T>(M, N, batch_count, ldc, hC_gold, hC_2, tol);
            }
            else
            {
                unit_check_general<T>(M, N, batch_count, ldc, hC_gold, hC_1);
                unit_check_general<T>(M, N, batch_count, ldc, hC_gold, hC_2);
            }
        }

        if(arg.norm_check)
        {
            double error_hst_ptr
                = std::abs(norm_check_general<T>('F', M, N, ldc, batch_count, hC_gold, hC_1));
            double error_dev_ptr
                = std::abs(norm_check_general<T>('F', M, N, ldc, batch_count, hC_gold, hC_2));
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
            CHECK_ROCBLAS_ERROR((rocblas_gemm_batched<T>(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         &h_alpha,
                                                         dA,
                                                         lda,
                                                         dB,
                                                         ldb,
                                                         &h_beta,
                                                         dC,
                                                         ldc,
                                                         batch_count)));
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_batched<T>(handle,
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    K,
                                    &h_alpha,
                                    dA,
                                    lda,
                                    Bv,
                                    ldb,
                                    &h_beta,
                                    Cv,
                                    ldc,
                                    batch_count);
        }

        gpu_time_used  = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops = gemm_gflop_count<T>(M, N, K) * batch_count / gpu_time_used * 1e6;

        std::cout << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,Batch_Count,"
                     "rocblas-Gflops,"
                     "us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,us,norm-error";

        std::cout << std::endl;

        std::cout << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << ","
                  << arg.get_alpha<T>() << "," << lda << "," << ldb << "," << arg.get_beta<T>()
                  << "," << ldc << "," << batch_count << "," << rocblas_gflops << ","
                  << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        std::cout << std::endl;
    }
}

template <typename T>
void testing_gemm_batched_bad_arg(const Arguments& arg)
{
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
    rocblas_int          batch_count = 5;

    // allocate memory on device
    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dB(batch_count);
    device_vector<T*, 0, T> dC(batch_count);

    if(!dA || !dB || !dC)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched<T>(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  nullptr,
                                                  lda,
                                                  dB,
                                                  ldb,
                                                  &beta,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched<T>(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA,
                                                  lda,
                                                  nullptr,
                                                  ldb,
                                                  &beta,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched<T>(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA,
                                                  lda,
                                                  dB,
                                                  ldb,
                                                  &beta,
                                                  nullptr,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched<T>(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  dA,
                                                  lda,
                                                  dB,
                                                  ldb,
                                                  &beta,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched<T>(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA,
                                                  lda,
                                                  dB,
                                                  ldb,
                                                  nullptr,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched<T>(nullptr,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA,
                                                  lda,
                                                  dB,
                                                  ldb,
                                                  &beta,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_handle);
}
