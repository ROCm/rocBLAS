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
void testing_gemm_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_gemm_batched_fn
        = FORTRAN ? rocblas_gemm_batched<T, true> : rocblas_gemm_batched<T, false>;

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
    bool invalid_size
        = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_fn(handle,
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
                                                      ldc,
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

    size_t     size_a      = size_one_a;
    size_t     size_b      = size_one_b;
    size_t     size_c      = size_one_c;
    const auto size_c_copy = arg.unit_check || arg.norm_check ? size_c : 0;

    // allocate memory on device
    device_batch_vector<T> dA(size_a, 1, batch_count);
    device_batch_vector<T> dB(size_b, 1, batch_count);
    device_batch_vector<T> dC(size_c, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hA(size_a, 1, batch_count);
    host_batch_vector<T> hB(size_b, 1, batch_count);
    host_batch_vector<T> hC_1(size_c, 1, batch_count);
    host_batch_vector<T> hC_2(size_c_copy, 1, batch_count);
    host_batch_vector<T> hC_gold(size_c_copy, 1, batch_count);
    host_vector<T>       halpha(1);
    host_vector<T>       hbeta(1);
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Initial Data on CPU
    rocblas_init(hA, true);
    for(int i = 0; i < batch_count; i++)
    {
        rocblas_init_alternating_sign<T>(hB[i], B_row, B_col, ldb);
    }
    if(rocblas_isnan(arg.beta))
        rocblas_init_nan(hC_1, false);
    else
        rocblas_init(hC_1, false);

    if(size_c_copy)
    {
        hC_2.copy_from(hC_1);
        hC_gold.copy_from(hC_1);
    }

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_1));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR((rocblas_gemm_batched_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     &h_alpha,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dB.ptr_on_device(),
                                                     ldb,
                                                     &h_beta,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count)));

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dC.transfer_from(hC_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));

        CHECK_ROCBLAS_ERROR((rocblas_gemm_batched_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     d_alpha,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dB.ptr_on_device(),
                                                     ldb,
                                                     d_beta,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count)));

        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(rocblas_int i = 0; i < batch_count; i++)
        {
            cblas_gemm<T>(
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
                near_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count, tol);
                near_check_general<T>(M, N, ldc, hC_gold, hC_2, batch_count, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
                unit_check_general<T>(M, N, ldc, hC_gold, hC_2, batch_count);
            }
        }

        if(arg.norm_check)
        {
            double error_hst_ptr
                = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count));
            double error_dev_ptr
                = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2, batch_count));
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
            CHECK_ROCBLAS_ERROR((rocblas_gemm_batched_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         &h_alpha,
                                                         dA.ptr_on_device(),
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         ldb,
                                                         &h_beta,
                                                         dC.ptr_on_device(),
                                                         ldc,
                                                         batch_count)));
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_batched_fn(handle,
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    K,
                                    &h_alpha,
                                    dA.ptr_on_device(),
                                    lda,
                                    dB.ptr_on_device(),
                                    ldb,
                                    &h_beta,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }

        gpu_time_used  = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops = gemm_gflop_count<T>(M, N, K) * batch_count / gpu_time_used * 1e6;

        rocblas_cout << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,Batch_Count,"
                        "rocblas-Gflops,"
                        "us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,us,norm-error";

        rocblas_cout << std::endl;

        rocblas_cout << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << ","
                     << arg.get_alpha<T>() << "," << lda << "," << ldb << "," << arg.get_beta<T>()
                     << "," << ldc << "," << batch_count << "," << rocblas_gflops << ","
                     << gpu_time_used;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        rocblas_cout << std::endl;
    }
}

template <typename T>
void testing_gemm_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_gemm_batched_fn
        = FORTRAN ? rocblas_gemm_batched<T, true> : rocblas_gemm_batched<T, false>;

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
    device_batch_vector<T> dA(safe_size, 1, batch_count);
    device_batch_vector<T> dB(safe_size, 1, batch_count);
    device_batch_vector<T> dC(safe_size, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  nullptr,
                                                  lda,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  &beta,
                                                  dC.ptr_on_device(),
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  nullptr,
                                                  ldb,
                                                  &beta,
                                                  dC.ptr_on_device(),
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  &beta,
                                                  nullptr,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  &beta,
                                                  dC.ptr_on_device(),
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  nullptr,
                                                  dC.ptr_on_device(),
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_fn(nullptr,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  &beta,
                                                  dC.ptr_on_device(),
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_handle);
}
