/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "cblas_interface.hpp"
#include "flops.hpp"
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

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 40

template <typename T>
void testing_trsm_bad_arg(const Arguments& arg)
{
    auto rocblas_trsm_fn = arg.fortran ? rocblas_trsm<T, true> : rocblas_trsm<T, false>;

    const rocblas_int M   = 100;
    const rocblas_int N   = 100;
    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;

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

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    EXPECT_ROCBLAS_STATUS(
        rocblas_trsm_fn(handle, side, uplo, transA, diag, M, N, &alpha, nullptr, lda, dB, ldb),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trsm_fn(handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, nullptr, ldb),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trsm_fn(handle, side, uplo, transA, diag, M, N, nullptr, dA, lda, dB, ldb),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trsm_fn(nullptr, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb),
        rocblas_status_invalid_handle);

    // If M==0, then all pointers can be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_trsm_fn(
            handle, side, uplo, transA, diag, 0, N, nullptr, nullptr, lda, nullptr, ldb),
        rocblas_status_success);

    // If N==0, then all pointers can be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_trsm_fn(
            handle, side, uplo, transA, diag, M, 0, nullptr, nullptr, lda, nullptr, ldb),
        rocblas_status_success);

    // If alpha==0, then A can be nullptr without error
    EXPECT_ROCBLAS_STATUS(
        rocblas_trsm_fn(handle, side, uplo, transA, diag, M, N, &zero, nullptr, lda, dB, ldb),
        rocblas_status_success);
}

template <typename T>
void testing_trsm(const Arguments& arg)
{
    auto rocblas_trsm_fn = arg.fortran ? rocblas_trsm<T, true> : rocblas_trsm<T, false>;

    rocblas_int M   = arg.M;
    rocblas_int N   = arg.N;
    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;

    char char_side   = arg.side;
    char char_uplo   = arg.uplo;
    char char_transA = arg.transA;
    char char_diag   = arg.diag;
    T    alpha_h     = arg.get_alpha<T>();

    bool HMM = arg.HMM;

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_int K = side == rocblas_side_left ? M : N;

    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M;
    if(invalid_size)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        EXPECT_ROCBLAS_STATUS(
            rocblas_trsm_fn(
                handle, side, uplo, transA, diag, M, N, nullptr, nullptr, lda, nullptr, ldb),
            rocblas_status_invalid_size);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(K, K, lda);
    host_matrix<T> hAAT(K, K, lda);
    host_matrix<T> hB(M, N, ldb);
    host_matrix<T> hX(M, N, ldb);
    host_matrix<T> hXorB_1(M, N, ldb);
    host_matrix<T> hXorB_2(M, N, ldb);
    host_matrix<T> cpuXorB(M, N, ldb);

    // Allocate device memory
    device_matrix<T> dA(K, K, lda, HMM);
    device_matrix<T> dXorB(M, N, ldb, HMM);
    device_vector<T> alpha_d(1, 1, HMM);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dXorB.memcheck());
    CHECK_DEVICE_ALLOCATION(alpha_d.memcheck());

    //  Random lower triangular matrices have condition number
    //  that grows exponentially with matrix size. Random full
    //  matrices have condition that grows linearly with
    //  matrix size.
    //
    //  We want a triangular matrix with condition number that grows
    //  lineary with matrix size. We start with full random matrix A.
    //  Calculate symmetric hAAT <- A A^T. Make hAAT strictly diagonal
    //  dominant. A strictly diagonal dominant matrix is SPD so we
    //  can use Cholesky to calculate L L^T = hAAT. These L factors
    //  should have condition number approximately equal to
    //  the condition number of the original matrix A.

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_matrix(
        hX, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, false, true);

    //  calculate hAAT = hA * hA ^ T or hAAT = hA * hA ^ H if complex
    cblas_gemm<T>(rocblas_operation_none,
                  rocblas_operation_conjugate_transpose,
                  K,
                  K,
                  K,
                  T(1.0),
                  hA,
                  lda,
                  hA,
                  lda,
                  T(0.0),
                  hAAT,
                  lda);

    //  copy hAAT into hA, make hA strictly diagonal dominant, and therefore SPD
    copy_hAAT_to_hA<T>((T*)hAAT, (T*)hA, K, size_t(lda));

    //  calculate Cholesky factorization of SPD (or Hermitian if complex) matrix hA
    cblas_potrf<T>(char_uplo, K, hA, lda);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(diag == rocblas_diagonal_unit)
    {
        make_unit_diagonal(uplo, (T*)hA, lda, K);
    }

    hB = hX;

    // Calculate hB = hA*hX;
    cblas_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA, lda, hB, ldb);
    hXorB_1 = hB; // hXorB <- B
    hXorB_2 = hB; // hXorB <- B
    cpuXorB = hB; // cpuXorB <- B

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used  = 0.0;
    double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();
    double max_err_1               = 0.0;
    double max_err_2               = 0.0;

    if(!ROCBLAS_REALLOC_ON_DEMAND)
    {
        // Compute size
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(
            rocblas_trsm_fn(handle, side, uplo, transA, diag, M, N, &alpha_h, dA, lda, dXorB, ldb));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        // Allocate memory
        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

        CHECK_ROCBLAS_ERROR(
            rocblas_trsm_fn(handle, side, uplo, transA, diag, M, N, &alpha_h, dA, lda, dXorB, ldb));

        CHECK_HIP_ERROR(hXorB_1.transfer_from(dXorB));

        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_2));
        CHECK_HIP_ERROR(hipMemcpy(alpha_d, &alpha_h, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(
            rocblas_trsm_fn(handle, side, uplo, transA, diag, M, N, alpha_d, dA, lda, dXorB, ldb));

        CHECK_HIP_ERROR(hXorB_2.transfer_from(dXorB));

        if(alpha_h == 0)
        {
            // expecting 0 output, set hX == 0
            rocblas_init_zero((T*)hX, M, N, ldb);

            if(arg.unit_check)
            {
                unit_check_general<T>(M, N, ldb, hX, hXorB_1);
                unit_check_general<T>(M, N, ldb, hX, hXorB_2);
            }

            if(arg.norm_check)
            {
                max_err_1 = std::abs(norm_check_general<T>('F', M, N, ldb, hX, hXorB_1));
                max_err_2 = std::abs(norm_check_general<T>('F', M, N, ldb, hX, hXorB_2));
            }
        }
        else
        {
            //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
            // calculate vector-induced-norm 1 of matrix E
            max_err_1 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hX, hXorB_1));
            max_err_2 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hX, hXorB_2));

            //unit test
            trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

            // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
            cblas_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA, lda, hXorB_1, ldb);
            cblas_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA, lda, hXorB_2, ldb);

            max_err_1 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hXorB_1, hB));
            max_err_2 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hXorB_2, hB));

            //unit test
            trsm_err_res_check<T>(max_err_1, M, residual_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, residual_eps_multiplier, eps);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        // GPU rocBLAS
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_trsm_fn(
                handle, side, uplo, transA, diag, M, N, &alpha_h, dA, lda, dXorB, ldb));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_trsm_fn(
                handle, side, uplo, transA, diag, M, N, &alpha_h, dA, lda, dXorB, ldb));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // CPU cblas
        cpu_time_used = get_time_us_no_sync();

        cblas_trsm<T>(side, uplo, transA, diag, M, N, alpha_h, hA, lda, cpuXorB, ldb);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        ArgumentModel<e_side, e_uplo, e_transA, e_diag, e_M, e_N, e_alpha, e_lda, e_ldb>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         trsm_gflop_count<T>(M, N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         max_err_1,
                         max_err_2);
    }
}
