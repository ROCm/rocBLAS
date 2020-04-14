/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "bytes.hpp"
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
void testing_syrk_strided_batched_bad_arg(const Arguments& arg)
{
    rocblas_local_handle    handle;
    const rocblas_fill      uplo        = rocblas_fill_upper;
    const rocblas_operation transA      = rocblas_operation_none;
    const rocblas_int       N           = 100;
    const rocblas_int       K           = 100;
    const rocblas_int       lda         = 100;
    const rocblas_int       ldc         = 100;
    const T                 alpha       = 1.0;
    const T                 beta        = 1.0;
    rocblas_stride          strideA     = 1;
    rocblas_stride          strideC     = 1;
    rocblas_int             batch_count = 2;

    const size_t safe_size = 100;
    // allocate memory on device
    device_vector<T> dA(batch_count);
    device_vector<T> dC(batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_syrk_strided_batched<T>(nullptr,
                                                          uplo,
                                                          transA,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          &beta,
                                                          dC,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(rocblas_syrk_strided_batched<T>(handle,
                                                          rocblas_fill_full,
                                                          transA,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          &beta,
                                                          dC,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_syrk_strided_batched<T>(handle,
                                                          uplo,
                                                          rocblas_operation_conjugate_transpose,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          &beta,
                                                          dC,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_syrk_strided_batched<T>(handle,
                                                          uplo,
                                                          transA,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          &beta,
                                                          dC,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_syrk_strided_batched<T>(handle,
                                                          uplo,
                                                          transA,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          nullptr,
                                                          lda,
                                                          strideA,
                                                          &beta,
                                                          dC,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_syrk_strided_batched<T>(handle,
                                                          uplo,
                                                          transA,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          nullptr,
                                                          dC,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_syrk_strided_batched<T>(handle,
                                                          uplo,
                                                          transA,
                                                          N,
                                                          K,
                                                          &alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          &beta,
                                                          nullptr,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocblas_syrk_strided_batched<T>(handle,
                                                          uplo,
                                                          transA,
                                                          0,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          lda,
                                                          strideA,
                                                          nullptr,
                                                          nullptr,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                          rocblas_status_success);
}

template <typename T>
void testing_syrk_strided_batched(const Arguments& arg)
{
    rocblas_local_handle handle;
    rocblas_fill         uplo        = char2rocblas_fill(arg.uplo);
    rocblas_operation    transA      = char2rocblas_operation(arg.transA);
    rocblas_int          N           = arg.N;
    rocblas_int          K           = arg.K;
    rocblas_int          lda         = arg.lda;
    rocblas_int          ldc         = arg.ldc;
    T                    alpha       = arg.get_alpha<T>();
    T                    beta        = arg.get_beta<T>();
    rocblas_stride       strideA     = arg.stride_a;
    rocblas_stride       strideC     = arg.stride_c;
    rocblas_int          batch_count = arg.batch_count;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;

    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalidSize = N < 0 || K < 0 || ldc < N || (transA == rocblas_operation_none && lda < N)
                       || (transA != rocblas_operation_none && lda < K) || batch_count < 0;
    if(N == 0 || batch_count == 0 || invalidSize)
    {
        // ensure invalid sizes checked before pointer check
        EXPECT_ROCBLAS_STATUS(rocblas_syrk_strided_batched<T>(handle,
                                                              uplo,
                                                              transA,
                                                              N,
                                                              K,
                                                              nullptr,
                                                              nullptr,
                                                              lda,
                                                              strideA,
                                                              nullptr,
                                                              nullptr,
                                                              ldc,
                                                              strideC,
                                                              batch_count),
                              invalidSize ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    strideA = std::max(strideA,
                       rocblas_stride(size_t(lda) * (transA == rocblas_operation_none ? K : N)));
    strideC = std::max(strideC, rocblas_stride(size_t(ldc) * N));

    size_t size_A = strideA * batch_count;
    size_t size_C = strideC * batch_count;

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dC(size_C);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> h_alpha(1);
    host_vector<T> h_beta(1);
    host_vector<T> hA(size_A);
    host_vector<T> hC_1(size_C);
    host_vector<T> hC_2(size_C);
    host_vector<T> hC_gold(size_C);

    CHECK_HIP_ERROR(h_alpha.memcheck());
    CHECK_HIP_ERROR(h_beta.memcheck());
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_2.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;
    rocblas_seedrand();
    rocblas_init<T>(hA);
    rocblas_init<T>(hC_1);

    hC_2    = hC_1;
    hC_gold = hC_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    if(arg.unit_check || arg.norm_check)
    {
        // host alpha/beta
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dC.transfer_from(hC_1));

        CHECK_ROCBLAS_ERROR(rocblas_syrk_strided_batched<T>(handle,
                                                            uplo,
                                                            transA,
                                                            N,
                                                            K,
                                                            &h_alpha[0],
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            &h_beta[0],
                                                            dC,
                                                            ldc,
                                                            strideC,
                                                            batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // device alpha/beta
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));

        CHECK_ROCBLAS_ERROR(rocblas_syrk_strided_batched<T>(handle,
                                                            uplo,
                                                            transA,
                                                            N,
                                                            K,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            d_beta,
                                                            dC,
                                                            ldc,
                                                            strideC,
                                                            batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        // CPU BLAS
        if(arg.timing)
        {
            cpu_time_used = get_time_us();
        }

        // cpu reference
        for(int i = 0; i < batch_count; i++)
        {
            cblas_syrk<T>(uplo,
                          transA,
                          N,
                          K,
                          h_alpha[0],
                          hA + i * strideA,
                          lda,
                          h_beta[0],
                          hC_gold + i * strideC,
                          ldc);
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = batch_count * syrk_gflop_count<T>(N, K) / cpu_time_used * 1e6;
        }

        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_float_complex>{}
               || std::is_same<T, rocblas_double_complex>{})
            {
                const double tol = K * sum_error_tolerance<T>;
                near_check_general<T>(N, N, batch_count, ldc, strideC, hC_gold, hC_1, tol);
                near_check_general<T>(N, N, batch_count, ldc, strideC, hC_gold, hC_2, tol);
            }
            else
            {
                unit_check_general<T>(N, N, batch_count, ldc, strideC, hC_gold, hC_1);
                unit_check_general<T>(N, N, batch_count, ldc, strideC, hC_gold, hC_2);
            }
        }

        if(arg.norm_check)
        {
            auto err1 = std::abs(
                norm_check_general<T>('F', N, N, ldc, strideC, batch_count, hC_gold, hC_1));
            auto err2 = std::abs(
                norm_check_general<T>('F', N, N, ldc, strideC, batch_count, hC_gold, hC_2));
            rocblas_error = err1 > err2 ? err1 : err2;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_syrk_strided_batched<T>(handle,
                                            uplo,
                                            transA,
                                            N,
                                            K,
                                            h_alpha,
                                            dA,
                                            lda,
                                            strideA,
                                            h_beta,
                                            dC,
                                            ldc,
                                            strideC,
                                            batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_syrk_strided_batched<T>(handle,
                                            uplo,
                                            transA,
                                            N,
                                            K,
                                            h_alpha,
                                            dA,
                                            lda,
                                            strideA,
                                            h_beta,
                                            dC,
                                            ldc,
                                            strideC,
                                            batch_count);
        }
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops
            = batch_count * syrk_gflop_count<T>(N, K) * number_hot_calls / gpu_time_used * 1e6;

        std::cout << "uplo,transA,N,K,alpha,lda,strideA,beta,ldc,strideC,rocblas-Gflops,us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,us,norm-error";

        std::cout << std::endl;

        std::cout << arg.uplo << "," << arg.transA << "," << N << "," << K << ","
                  << arg.get_alpha<T>() << "," << lda << "," << strideA << "," << arg.get_beta<T>()
                  << "," << ldc << "," << strideC << "," << rocblas_gflops << ","
                  << gpu_time_used / number_hot_calls;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        std::cout << std::endl;
    }
}
