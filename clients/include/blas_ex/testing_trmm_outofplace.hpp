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
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_trmm_outofplace_bad_arg(const Arguments& arg)
{
    auto rocblas_trmm_outofplace_fn
        = arg.fortran ? rocblas_trmm_outofplace<T, true> : rocblas_trmm_outofplace<T, false>;

    const rocblas_int M   = 100;
    const rocblas_int N   = 100;
    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;
    const rocblas_int ldc = 100;

    const T alpha = 1.0;
    const T zero  = 0.0;

    const rocblas_side      side   = rocblas_side_left;
    const rocblas_fill      uplo   = rocblas_fill_upper;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    rocblas_int K = side == rocblas_side_left ? M : N;

    // Allocate device memory
    device_matrix<T> dA(K, K, lda);
    device_matrix<T> dB(M, N, ldb);
    device_matrix<T> dC(M, N, ldc);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    // check for invalid enum
    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, rocblas_side_both, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb, dC, ldc),
        rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_fn(handle,
                                                     side,
                                                     (rocblas_fill)rocblas_side_both,
                                                     transA,
                                                     diag,
                                                     M,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     lda,
                                                     dB,
                                                     ldb,
                                                     dC,
                                                     ldc),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_fn(handle,
                                                     side,
                                                     uplo,
                                                     (rocblas_operation)rocblas_side_both,
                                                     diag,
                                                     M,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     lda,
                                                     dB,
                                                     ldb,
                                                     dC,
                                                     ldc),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_fn(handle,
                                                     side,
                                                     uplo,
                                                     transA,
                                                     (rocblas_diagonal)rocblas_side_both,
                                                     M,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     lda,
                                                     dB,
                                                     ldb,
                                                     dC,
                                                     ldc),
                          rocblas_status_invalid_value);

    // check for invalid size
    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, -1, N, &alpha, dA, lda, dB, ldb, dC, ldc),
        rocblas_status_invalid_size);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, -1, &alpha, dA, lda, dB, ldb, dC, ldc),
        rocblas_status_invalid_size);

    // check for invalid leading dimension
    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, M - 1, dC, ldc),
        rocblas_status_invalid_size);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb, dC, M - 1),
        rocblas_status_invalid_size);

    EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_fn(handle,
                                                     rocblas_side_left,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     M,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     M - 1,
                                                     dB,
                                                     ldb,
                                                     dC,
                                                     ldc),
                          rocblas_status_invalid_size);

    EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_fn(handle,
                                                     rocblas_side_right,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     M,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     N - 1,
                                                     dB,
                                                     ldb,
                                                     dC,
                                                     ldc),
                          rocblas_status_invalid_size);

    // check that nullpointer gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            nullptr, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb, dC, ldc),
        rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, N, nullptr, dA, lda, dB, ldb, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, N, &alpha, nullptr, lda, dB, ldb, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, nullptr, ldb, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb, nullptr, ldc),
        rocblas_status_invalid_pointer);

    // quick return: If alpha==0, then A and B can be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, N, &zero, nullptr, lda, nullptr, ldb, dC, ldc),
        rocblas_status_success);

    // quick return: If M==0, then all pointers can be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_fn(handle,
                                                     side,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     0,
                                                     N,
                                                     nullptr,
                                                     nullptr,
                                                     lda,
                                                     nullptr,
                                                     ldb,
                                                     nullptr,
                                                     ldc),
                          rocblas_status_success);

    // quick return: If N==0, then all pointers can be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_fn(handle,
                                                     side,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     M,
                                                     0,
                                                     nullptr,
                                                     nullptr,
                                                     lda,
                                                     nullptr,
                                                     ldb,
                                                     nullptr,
                                                     ldc),
                          rocblas_status_success);
}

template <typename T>
void testing_trmm_outofplace(const Arguments& arg)
{
    auto rocblas_trmm_outofplace_fn
        = arg.fortran ? rocblas_trmm_outofplace<T, true> : rocblas_trmm_outofplace<T, false>;

    rocblas_int M   = arg.M;
    rocblas_int N   = arg.N;
    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    char char_side   = arg.side;
    char char_uplo   = arg.uplo;
    char char_transA = arg.transA;
    char char_diag   = arg.diag;
    T    h_alpha_T   = arg.get_alpha<T>();

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_int K      = side == rocblas_side_left ? M : N;
    size_t      size_A = lda * size_t(K);
    size_t      size_B = ldb * size_t(N);
    size_t      size_C = ldc * size_t(N);

    rocblas_local_handle handle{arg};

    // ensure invalid sizes and quick return checked before pointer check
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M || ldc < M;
    if(M == 0 || N == 0 || invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_trmm_outofplace_fn(handle,
                                                         side,
                                                         uplo,
                                                         transA,
                                                         diag,
                                                         M,
                                                         N,
                                                         nullptr,
                                                         nullptr,
                                                         lda,
                                                         nullptr,
                                                         ldb,
                                                         nullptr,
                                                         ldc),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(K, K, lda);
    host_matrix<T> hB(M, N, ldb);
    host_matrix<T> hB_gold(M, N, ldb);
    host_matrix<T> hC(M, N, ldc);
    host_matrix<T> hC_1(M, N, ldc);
    host_matrix<T> hC_2(M, N, ldc);
    host_matrix<T> hC_gold(M, N, ldc);

    // Allocate device memory
    device_matrix<T> dA(K, K, lda);
    device_matrix<T> dB(M, N, ldb);
    device_matrix<T> dC(M, N, ldc);
    device_vector<T> alpha_d(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(alpha_d.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix);

    hB_gold = hB;
    hC_1    = hC; // hXorC <- C
    hC_2    = hC; // hXorC <- C
    hC_gold = hC; // hC_gold <- C

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double rocblas_error          = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dC.transfer_from(hC_1));

        CHECK_ROCBLAS_ERROR(rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, N, &h_alpha_T, dA, lda, dB, ldb, dC, ldc));

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // calculate dB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC_2));
        CHECK_HIP_ERROR(hipMemcpy(alpha_d, &h_alpha_T, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_trmm_outofplace_fn(
            handle, side, uplo, transA, diag, M, N, alpha_d, dA, lda, dB, ldb, dC, ldc));

        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        // CPU BLAS
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        cblas_trmm<T>(side, uplo, transA, diag, M, N, h_alpha_T, hA, lda, hB_gold, ldb);

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // copy B matrix into C matrix
        copy_matrix_with_different_leading_dimensions(hB_gold, hC_gold);

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
            CHECK_ROCBLAS_ERROR(rocblas_trmm_outofplace_fn(
                handle, side, uplo, transA, diag, M, N, &h_alpha_T, dA, lda, dB, ldb, dC, ldc));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_trmm_outofplace_fn(
                handle, side, uplo, transA, diag, M, N, &h_alpha_T, dA, lda, dB, ldb, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side, e_uplo, e_transA, e_diag, e_M, e_N, e_alpha, e_lda, e_ldb, e_ldc>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         trmm_gflop_count<T>(M, N, side),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error);
    }
}
