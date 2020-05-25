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

template <typename T, bool HERM>
void testing_symm_hemm_strided_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN    = arg.fortran;
    auto       rocblas_fn = HERM ? (FORTRAN ? rocblas_hemm_strided_batched<T, true>
                                      : rocblas_hemm_strided_batched<T, false>)
                           : (FORTRAN ? rocblas_symm_strided_batched<T, true>
                                      : rocblas_symm_strided_batched<T, false>);

    rocblas_local_handle handle;
    const rocblas_side   side        = rocblas_side_left;
    const rocblas_fill   uplo        = rocblas_fill_upper;
    const rocblas_int    M           = 100;
    const rocblas_int    N           = 100;
    const rocblas_int    lda         = 100;
    const rocblas_int    ldb         = 100;
    const rocblas_int    ldc         = 100;
    const T              alpha       = 1.0;
    const T              beta        = 1.0;
    rocblas_stride       strideA     = 1;
    rocblas_stride       strideB     = 1;
    rocblas_stride       strideC     = 1;
    rocblas_int          batch_count = 2;

    const size_t safe_size = 100;
    // allocate memory on device
    device_strided_batch_vector<T> dA(1, 1, strideA, batch_count);
    device_strided_batch_vector<T> dB(1, 1, strideB, batch_count);
    device_strided_batch_vector<T> dC(1, 1, strideC, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_fn(nullptr,
                                     side,
                                     uplo,
                                     M,
                                     N,
                                     &alpha,
                                     dA,
                                     lda,
                                     strideA,
                                     dB,
                                     ldb,
                                     strideB,
                                     &beta,
                                     dC,
                                     ldc,
                                     strideC,
                                     batch_count),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                     rocblas_side_both,
                                     uplo,
                                     M,
                                     N,
                                     &alpha,
                                     dA,
                                     lda,
                                     strideA,
                                     dB,
                                     ldb,
                                     strideB,
                                     &beta,
                                     dC,
                                     ldc,
                                     strideC,
                                     batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                     side,
                                     rocblas_fill_full,
                                     M,
                                     N,
                                     &alpha,
                                     dA,
                                     lda,
                                     strideA,
                                     dB,
                                     ldb,
                                     strideB,
                                     &beta,
                                     dC,
                                     ldc,
                                     strideC,
                                     batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                     side,
                                     uplo,
                                     M,
                                     N,
                                     nullptr,
                                     dA,
                                     lda,
                                     strideA,
                                     dB,
                                     ldb,
                                     strideB,
                                     &beta,
                                     dC,
                                     ldc,
                                     strideC,
                                     batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                     side,
                                     uplo,
                                     M,
                                     N,
                                     &alpha,
                                     nullptr,
                                     lda,
                                     strideA,
                                     dB,
                                     ldb,
                                     strideB,
                                     &beta,
                                     dC,
                                     ldc,
                                     strideC,
                                     batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                     side,
                                     uplo,
                                     M,
                                     N,
                                     &alpha,
                                     dA,
                                     lda,
                                     strideA,
                                     nullptr,
                                     ldb,
                                     strideB,
                                     &beta,
                                     dC,
                                     ldc,
                                     strideC,
                                     batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                     side,
                                     uplo,
                                     M,
                                     N,
                                     &alpha,
                                     dA,
                                     lda,
                                     strideA,
                                     dB,
                                     ldb,
                                     strideB,
                                     nullptr,
                                     dC,
                                     ldc,
                                     strideC,
                                     batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                     side,
                                     uplo,
                                     M,
                                     N,
                                     &alpha,
                                     dA,
                                     lda,
                                     strideA,
                                     dB,
                                     ldb,
                                     strideB,
                                     &beta,
                                     nullptr,
                                     ldc,
                                     strideC,
                                     batch_count),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                     side,
                                     uplo,
                                     0,
                                     N,
                                     nullptr,
                                     nullptr,
                                     lda,
                                     strideA,
                                     nullptr,
                                     ldb,
                                     strideB,
                                     nullptr,
                                     nullptr,
                                     ldc,
                                     strideC,
                                     batch_count),
                          rocblas_status_success);
}

template <typename T, bool HERM>
void testing_symm_hemm_strided_batched(const Arguments& arg)
{
    auto rocblas_fn     = HERM ? rocblas_hemm_strided_batched<T> : rocblas_symm_strided_batched<T>;
    auto gflop_count_fn = HERM ? hemm_gflop_count<T> : symm_gflop_count<T>;

    rocblas_local_handle handle;
    rocblas_side         side        = char2rocblas_side(arg.side);
    rocblas_fill         uplo        = char2rocblas_fill(arg.uplo);
    rocblas_int          M           = arg.M;
    rocblas_int          N           = arg.N;
    rocblas_int          lda         = arg.lda;
    rocblas_int          ldb         = arg.ldb;
    rocblas_int          ldc         = arg.ldc;
    T                    alpha       = arg.get_alpha<T>();
    T                    beta        = arg.get_beta<T>();
    rocblas_stride       strideA     = arg.stride_a;
    rocblas_stride       strideB     = arg.stride_b;
    rocblas_stride       strideC     = arg.stride_c;
    rocblas_int          batch_count = arg.batch_count;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;

    // Note: N==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = batch_count < 0 || M < 0 || N < 0 || ldc < M || ldb < M
                        || (side == rocblas_side_left && (lda < M))
                        || (side != rocblas_side_left && (lda < N));
    if(M == 0 || N == 0 || batch_count == 0 || invalid_size)
    {
        // ensure invalid sizes checked before pointer check
        EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                         side,
                                         uplo,
                                         M,
                                         N,
                                         nullptr,
                                         nullptr,
                                         lda,
                                         strideA,
                                         nullptr,
                                         ldb,
                                         strideB,
                                         nullptr,
                                         nullptr,
                                         ldc,
                                         strideC,
                                         batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    size_t cols = (side == rocblas_side_left ? std::max(M, 1) : std::max(N, 1));
    strideA     = std::max(strideA, rocblas_stride(lda * cols));
    strideB     = std::max(strideB, rocblas_stride(size_t(ldb) * N));
    strideC     = std::max(strideC, rocblas_stride(size_t(ldc) * N));

    size_t sizeA = strideA;
    size_t sizeB = strideB;
    size_t sizeC = strideC;

    // allocate memory on device
    device_strided_batch_vector<T> dA(sizeA, 1, strideA, batch_count);
    device_strided_batch_vector<T> dB(sizeB, 1, strideB, batch_count);
    device_strided_batch_vector<T> dC(sizeC, 1, strideC, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory

    host_strided_batch_vector<T> hA(sizeA, 1, strideA, batch_count);
    host_strided_batch_vector<T> hB(sizeB, 1, strideB, batch_count);
    host_strided_batch_vector<T> hC_1(sizeC, 1, strideC, batch_count);
    host_strided_batch_vector<T> hC_2(sizeC, 1, strideC, batch_count);
    host_strided_batch_vector<T> hC_gold(sizeC, 1, strideC, batch_count);
    host_vector<T>               h_alpha(1);
    host_vector<T>               h_beta(1);
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_2.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());
    CHECK_HIP_ERROR(h_alpha.memcheck());
    CHECK_HIP_ERROR(h_beta.memcheck());

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;
    rocblas_seedrand();
    rocblas_init<T>(hA);
    rocblas_init<T>(hB);
    rocblas_init<T>(hC_1);

    hC_2.copy_from(hC_1);
    hC_gold.copy_from(hC_1);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    if(arg.unit_check || arg.norm_check)
    {
        // host alpha/beta
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dC.transfer_from(hC_1));

        CHECK_ROCBLAS_ERROR(rocblas_fn(handle,
                                       side,
                                       uplo,
                                       M,
                                       N,
                                       &h_alpha[0],
                                       dA,
                                       lda,
                                       strideA,
                                       dB,
                                       ldb,
                                       strideB,
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

        CHECK_ROCBLAS_ERROR(rocblas_fn(handle,
                                       side,
                                       uplo,
                                       M,
                                       N,
                                       d_alpha,
                                       dA,
                                       lda,
                                       strideA,
                                       dB,
                                       ldb,
                                       strideB,
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
            if(HERM)
            {
                cblas_hemm<T>(
                    side, uplo, M, N, h_alpha, hA[i], lda, hB[i], ldb, h_beta, hC_gold[i], ldc);
            }
            else
            {
                cblas_symm<T>(side,
                              uplo,
                              M,
                              N,
                              h_alpha[0],
                              hA[i],
                              lda,
                              hB[i],
                              ldb,
                              h_beta[0],
                              hC_gold[i],
                              ldc);
            }
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = batch_count * gflop_count_fn(side, M, N) / cpu_time_used * 1e6;
        }

        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_float_complex>{}
               || std::is_same<T, rocblas_double_complex>{})
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(M, N, ldc, strideC, hC_gold, hC_1, batch_count, tol);
                near_check_general<T>(M, N, ldc, strideC, hC_gold, hC_2, batch_count, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldc, strideC, hC_gold, hC_1, batch_count);
                unit_check_general<T>(M, N, ldc, strideC, hC_gold, hC_2, batch_count);
            }
        }

        if(arg.norm_check)
        {
            auto err1 = std::abs(
                norm_check_general<T>('F', M, N, ldc, strideC, hC_gold, hC_1, batch_count));
            auto err2 = std::abs(
                norm_check_general<T>('F', M, N, ldc, strideC, hC_gold, hC_2, batch_count));
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
            rocblas_fn(handle,
                       side,
                       uplo,
                       M,
                       N,
                       h_alpha,
                       dA,
                       lda,
                       strideA,
                       dB,
                       ldb,
                       strideB,
                       h_beta,
                       dC,
                       ldc,
                       strideC,
                       batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_fn(handle,
                       side,
                       uplo,
                       M,
                       N,
                       h_alpha,
                       dA,
                       lda,
                       strideA,
                       dB,
                       ldb,
                       strideB,
                       h_beta,
                       dC,
                       ldc,
                       strideC,
                       batch_count);
        }
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops
            = batch_count * gflop_count_fn(side, M, N) * number_hot_calls / gpu_time_used * 1e6;

        rocblas_cout << "side,uplo,M,N,alpha,lda,strideA,ldb,strideB,beta,ldc,strideC,batch_count,"
                        "rocblas-Gflops,us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,us,norm-error";

        rocblas_cout << std::endl;

        rocblas_cout << arg.side << "," << arg.uplo << "," << M << "," << N << ","
                     << arg.get_alpha<T>() << "," << lda << "," << strideA << "," << ldb << ","
                     << strideB << "," << arg.get_beta<T>() << "," << ldc << "," << strideC << ","
                     << batch_count << "," << rocblas_gflops << ","
                     << gpu_time_used / number_hot_calls;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        rocblas_cout << std::endl;
    }
}
