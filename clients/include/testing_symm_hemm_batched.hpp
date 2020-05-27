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
void testing_symm_hemm_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_fn
        = HERM ? (FORTRAN ? rocblas_hemm_batched<T, true> : rocblas_hemm_batched<T, false>)
               : (FORTRAN ? rocblas_symm_batched<T, true> : rocblas_symm_batched<T, false>);

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
    rocblas_int          batch_count = 2;

    const size_t safe_size = 100;
    // allocate memory on device
    device_batch_vector<T> dA(safe_size, 1, batch_count);
    device_batch_vector<T> dB(safe_size, 1, batch_count);
    device_batch_vector<T> dC(safe_size, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(
            nullptr, side, uplo, M, N, &alpha, dA, lda, dB, ldb, &beta, dC, ldc, batch_count),
        rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(rocblas_fn(handle,
                                     rocblas_side_both,
                                     uplo,
                                     M,
                                     N,
                                     &alpha,
                                     dA,
                                     lda,
                                     dB,
                                     ldb,
                                     &beta,
                                     dC,
                                     ldc,
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
                                     dB,
                                     ldb,
                                     &beta,
                                     dC,
                                     ldc,
                                     batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(
            handle, side, uplo, M, N, nullptr, dA, lda, dB, ldb, &beta, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(
            handle, side, uplo, M, N, &alpha, nullptr, lda, dB, ldb, &beta, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(
            handle, side, uplo, M, N, &alpha, dA, lda, nullptr, ldb, &beta, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(
            handle, side, uplo, M, N, &alpha, dA, lda, dB, ldb, nullptr, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_fn(
            handle, side, uplo, M, N, &alpha, dA, lda, dB, ldb, &beta, nullptr, ldc, batch_count),
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
                                     nullptr,
                                     ldb,
                                     nullptr,
                                     nullptr,
                                     ldc,
                                     batch_count),
                          rocblas_status_success);
}

template <typename T, bool HERM>
void testing_symm_hemm_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_fn
        = HERM ? (FORTRAN ? rocblas_hemm_batched<T, true> : rocblas_hemm_batched<T, false>)
               : (FORTRAN ? rocblas_symm_batched<T, true> : rocblas_symm_batched<T, false>);
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
                                         nullptr,
                                         ldb,
                                         nullptr,
                                         nullptr,
                                         ldc,
                                         batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    size_t     cols   = (side == rocblas_side_left ? std::max(M, 1) : std::max(N, 1));
    const auto size_A = size_t(lda) * cols;
    const auto size_B = size_t(ldb) * N;
    const auto size_C = size_t(ldc) * N;

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dB(size_B, 1, batch_count);
    device_batch_vector<T> dC(size_C, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>       h_alpha(1);
    host_vector<T>       h_beta(1);
    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> hB(size_B, 1, batch_count);
    host_batch_vector<T> hC_1(size_C, 1, batch_count);
    host_batch_vector<T> hC_2(size_C, 1, batch_count);
    host_batch_vector<T> hC_gold(size_C, 1, batch_count);
    CHECK_HIP_ERROR(h_alpha.memcheck());
    CHECK_HIP_ERROR(h_beta.memcheck());
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_2.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

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
                                       dA.ptr_on_device(),
                                       lda,
                                       dB.ptr_on_device(),
                                       ldb,
                                       &h_beta[0],
                                       dC.ptr_on_device(),
                                       ldc,
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
                                       dA.ptr_on_device(),
                                       lda,
                                       dB.ptr_on_device(),
                                       ldb,
                                       d_beta,
                                       dC.ptr_on_device(),
                                       ldc,
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
            auto err1 = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count));
            auto err2 = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2, batch_count));
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
                       dA.ptr_on_device(),
                       lda,
                       dB.ptr_on_device(),
                       ldb,
                       h_beta,
                       dC.ptr_on_device(),
                       ldc,
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
                       dA.ptr_on_device(),
                       lda,
                       dB.ptr_on_device(),
                       ldb,
                       h_beta,
                       dC.ptr_on_device(),
                       ldc,
                       batch_count);
        }
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops
            = batch_count * gflop_count_fn(side, M, N) * number_hot_calls / gpu_time_used * 1e6;

        rocblas_cout << "side,uplo,M,N,alpha,lda,ldb,beta,ldc,batch_count,rocblas-Gflops,us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,us,norm-error";

        rocblas_cout << std::endl;

        rocblas_cout << arg.side << "," << arg.uplo << "," << M << "," << N << ","
                     << arg.get_alpha<T>() << "," << lda << "," << ldb << "," << arg.get_beta<T>()
                     << "," << ldc << "," << batch_count << "," << rocblas_gflops << ","
                     << gpu_time_used / number_hot_calls;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        rocblas_cout << std::endl;
    }
}
