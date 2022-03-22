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
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 40
#define TRSM_BLOCK 128

template <typename T>
void testing_trsm_strided_batched_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_trsm_strided_batched_ex_fn
        = arg.fortran ? rocblas_trsm_strided_batched_ex_fortran : rocblas_trsm_strided_batched_ex;

    const rocblas_int M           = 100;
    const rocblas_int N           = 100;
    const rocblas_int lda         = 100;
    const rocblas_int ldb         = 100;
    const rocblas_int batch_count = 5;
    const T           alpha       = 1.0;
    const T           zero        = 0.0;

    const rocblas_side      side   = rocblas_side_left;
    const rocblas_fill      uplo   = rocblas_fill_upper;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    rocblas_int          K          = side == rocblas_side_left ? M : N;
    const rocblas_stride stride_a   = lda * K;
    const rocblas_stride stride_b   = ldb * N;
    const rocblas_stride strideInvA = TRSM_BLOCK * K;
    size_t               size_A     = batch_count * stride_a;
    size_t               size_B     = batch_count * stride_b;
    size_t               sizeinvA   = batch_count * strideInvA;

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> dInvA(sizeinvA);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dInvA.memcheck());

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             N,
                                                             &alpha,
                                                             nullptr,
                                                             lda,
                                                             stride_a,
                                                             dB,
                                                             ldb,
                                                             stride_b,
                                                             batch_count,
                                                             dInvA,
                                                             sizeinvA,
                                                             strideInvA,
                                                             rocblas_datatype_f32_r),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             N,
                                                             &alpha,
                                                             dA,
                                                             lda,
                                                             stride_a,
                                                             nullptr,
                                                             ldb,
                                                             stride_b,
                                                             batch_count,
                                                             dInvA,
                                                             sizeinvA,
                                                             strideInvA,
                                                             rocblas_datatype_f32_r),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             N,
                                                             nullptr,
                                                             dA,
                                                             lda,
                                                             stride_a,
                                                             dB,
                                                             ldb,
                                                             stride_b,
                                                             batch_count,
                                                             dInvA,
                                                             sizeinvA,
                                                             strideInvA,
                                                             rocblas_datatype_f32_r),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(nullptr,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             N,
                                                             &alpha,
                                                             dA,
                                                             lda,
                                                             stride_a,
                                                             dB,
                                                             ldb,
                                                             stride_b,
                                                             batch_count,
                                                             dInvA,
                                                             sizeinvA,
                                                             strideInvA,
                                                             rocblas_datatype_f32_r),
                          rocblas_status_invalid_handle);

    // When batch_count==0, all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             N,
                                                             nullptr,
                                                             nullptr,
                                                             lda,
                                                             stride_a,
                                                             nullptr,
                                                             ldb,
                                                             stride_b,
                                                             0,
                                                             dInvA,
                                                             sizeinvA,
                                                             strideInvA,
                                                             rocblas_datatype_f32_r),
                          rocblas_status_success);

    // When M==0, all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             0,
                                                             N,
                                                             nullptr,
                                                             nullptr,
                                                             lda,
                                                             stride_a,
                                                             nullptr,
                                                             ldb,
                                                             stride_b,
                                                             batch_count,
                                                             dInvA,
                                                             sizeinvA,
                                                             strideInvA,
                                                             rocblas_datatype_f32_r),
                          rocblas_status_success);

    // When N==0, all pointers may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             lda,
                                                             stride_a,
                                                             nullptr,
                                                             ldb,
                                                             stride_b,
                                                             batch_count,
                                                             dInvA,
                                                             sizeinvA,
                                                             strideInvA,
                                                             rocblas_datatype_f32_r),
                          rocblas_status_success);

    // When alpha==0, A may be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             N,
                                                             &zero,
                                                             nullptr,
                                                             lda,
                                                             stride_a,
                                                             dB,
                                                             ldb,
                                                             stride_b,
                                                             batch_count,
                                                             dInvA,
                                                             sizeinvA,
                                                             strideInvA,
                                                             rocblas_datatype_f32_r),
                          rocblas_status_success);

    // Unsupported datatype
    EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                             side,
                                                             uplo,
                                                             transA,
                                                             diag,
                                                             M,
                                                             N,
                                                             &zero,
                                                             nullptr,
                                                             lda,
                                                             stride_a,
                                                             dB,
                                                             ldb,
                                                             stride_b,
                                                             batch_count,
                                                             dInvA,
                                                             sizeinvA,
                                                             strideInvA,
                                                             rocblas_datatype_bf16_r),
                          rocblas_status_not_implemented);
}

template <typename T>
void testing_trsm_strided_batched_ex(const Arguments& arg)
{
    auto rocblas_trsm_strided_batched_ex_fn
        = arg.fortran ? rocblas_trsm_strided_batched_ex_fortran : rocblas_trsm_strided_batched_ex;

    rocblas_int M   = arg.M;
    rocblas_int N   = arg.N;
    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;

    char        char_side   = arg.side;
    char        char_uplo   = arg.uplo;
    char        char_transA = arg.transA;
    char        char_diag   = arg.diag;
    T           alpha_h     = arg.alpha;
    rocblas_int stride_A    = arg.stride_a;
    rocblas_int stride_B    = arg.stride_b;
    rocblas_int batch_count = arg.batch_count;

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_int K           = side == rocblas_side_left ? M : N;
    size_t      size_A      = lda * size_t(K) + stride_A * (batch_count - 1);
    size_t      size_B      = ldb * size_t(N) + stride_B * (batch_count - 1);
    rocblas_int stride_invA = TRSM_BLOCK * K;
    size_t      size_invA   = stride_invA * batch_count;

    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0;
    if(invalid_size || batch_count == 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 N,
                                                                 nullptr,
                                                                 nullptr,
                                                                 lda,
                                                                 stride_A,
                                                                 nullptr,
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 nullptr,
                                                                 TRSM_BLOCK * K,
                                                                 stride_invA,
                                                                 arg.compute_type),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hX(size_B);
    host_vector<T> hXorB_1(size_B);
    host_vector<T> hXorB_2(size_B);
    host_vector<T> cpuXorB(size_B);

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used  = 0.0;
    double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dXorB(size_B);
    device_vector<T> alpha_d(1);
    device_vector<T> dinvA(size_invA);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dXorB.memcheck());
    CHECK_DEVICE_ALLOCATION(alpha_d.memcheck());
    CHECK_DEVICE_ALLOCATION(dinvA.memcheck());

    //  Random lower triangular matrices have condition number
    //  that grows exponentially with matrix size. Random full
    //  matrices have condition that grows linearly with
    //  matrix size.
    //
    //  We want a triangular matrix with condition number that grows
    //  lineary with matrix size. We start with full random matrix A.
    //  Calculate symmetric AAT <- A A^T. Make AAT strictly diagonal
    //  dominant. A strictly diagonal dominant matrix is SPD so we
    //  can use Cholesky to calculate L L^T = AAT. These L factors
    //  should have condition number approximately equal to
    //  the condition number of the original matrix A.

    // Initialize data on host memory
    rocblas_init_matrix(hA,
                        arg,
                        K,
                        K,
                        lda,
                        stride_A,
                        batch_count,
                        rocblas_client_never_set_nan,
                        rocblas_client_triangular_matrix,
                        true);

    rocblas_init_matrix(hX,
                        arg,
                        M,
                        N,
                        ldb,
                        stride_B,
                        batch_count,
                        rocblas_client_never_set_nan,
                        rocblas_client_general_matrix,
                        false,
                        true);
    hB = hX;

    //  Initialize full random matrix hA with all entries in [1, 10]
    rocblas_init<T>(hA, K, K, lda, stride_A, batch_count);

    //  pad untouched area into zero
    for(int b = 0; b < batch_count; b++)
    {
        //  calculate AAT = hA * hA ^ T or AAT = hA * hA ^ H if complex
        cblas_gemm<T>(rocblas_operation_none,
                      rocblas_operation_conjugate_transpose,
                      K,
                      K,
                      K,
                      T(1.0),
                      hA + stride_A * b,
                      lda,
                      hA + stride_A * b,
                      lda,
                      T(0.0),
                      AAT + stride_A * b,
                      lda);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < K; i++)
        {
            T           t    = 0.0;
            rocblas_int idx2 = stride_A * b + i + i * lda;
            for(int j = 0; j < K; j++)
            {
                rocblas_int idx1 = stride_A * b + i + j * lda;
                hA[idx1]         = AAT[idx1];
                t += rocblas_abs(AAT[idx1]);
            }
            hA[idx2] = t;
        }

        //  calculate Cholesky factorization of SPD (or Hermitian if complex) matrix hA
        cblas_potrf<T>(char_uplo, K, hA + b * stride_A, lda);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(char_diag == 'U' || char_diag == 'u')
        {
            if('L' == char_uplo || 'l' == char_uplo)
            {
                for(int i = 0; i < K; i++)
                {
                    T diag = hA[stride_A * b + i + i * lda];
                    for(int j = 0; j <= i; j++)
                        hA[stride_A * b + i + j * lda] = hA[stride_A * b + i + j * lda] / diag;
                }
            }
            else
            {
                for(int j = 0; j < K; j++)
                {
                    T diag = hA[stride_A * b + j + j * lda];
                    for(int i = 0; i <= j; i++)
                        hA[stride_A * b + i + j * lda] = hA[stride_A * b + i + j * lda] / diag;
                }
            }
        }
    }

    // Calculate hB = hA*hX;
    for(int b = 0; b < batch_count; b++)
        cblas_trmm<T>(side,
                      uplo,
                      transA,
                      diag,
                      M,
                      N,
                      1.0 / alpha_h,
                      hA + b * stride_A,
                      lda,
                      hB + b * stride_B,
                      ldb);

    hXorB_1 = hB; // hXorB <- B
    hXorB_2 = hB; // hXorB <- B
    cpuXorB = hB; // cpuXorB <- B

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB_1, sizeof(T) * size_B, hipMemcpyHostToDevice));

    rocblas_int sub_stride_A    = TRSM_BLOCK * lda + TRSM_BLOCK;
    rocblas_int sub_stride_invA = TRSM_BLOCK * TRSM_BLOCK;

    int blocks = K / TRSM_BLOCK;

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;

    if(!ROCBLAS_REALLOC_ON_DEMAND)
    {
        // Compute size
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));

        if(arg.unit_check || arg.norm_check)
        {
            for(int b = 0; b < batch_count; b++)
            {
                if(blocks > 0)
                {
                    CHECK_ALLOC_QUERY(rocblas_trtri_strided_batched<T>(handle,
                                                                       uplo,
                                                                       diag,
                                                                       TRSM_BLOCK,
                                                                       dA + b * stride_A,
                                                                       lda,
                                                                       sub_stride_A,
                                                                       dinvA + b * stride_invA,
                                                                       TRSM_BLOCK,
                                                                       sub_stride_invA,
                                                                       blocks));
                }

                if(K % TRSM_BLOCK != 0 || blocks == 0)
                {
                    CHECK_ALLOC_QUERY(rocblas_trtri_strided_batched<T>(
                        handle,
                        uplo,
                        diag,
                        K - TRSM_BLOCK * blocks,
                        dA + sub_stride_A * blocks + b * stride_A,
                        lda,
                        sub_stride_A,
                        dinvA + sub_stride_invA * blocks + b * stride_invA,
                        TRSM_BLOCK,
                        sub_stride_invA,
                        1));
                }
            }

            CHECK_ALLOC_QUERY(rocblas_trsm_strided_batched_ex_fn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 N,
                                                                 &alpha_h,
                                                                 dA,
                                                                 lda,
                                                                 stride_A,
                                                                 dXorB,
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 dinvA,
                                                                 size_invA,
                                                                 stride_invA,
                                                                 arg.compute_type));
        }

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        // Allocate memory
        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB_1, sizeof(T) * size_B, hipMemcpyHostToDevice));

        hipStream_t rocblas_stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

        for(int b = 0; b < batch_count; b++)
        {
            if(blocks > 0)
            {
                CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched<T>(handle,
                                                                     uplo,
                                                                     diag,
                                                                     TRSM_BLOCK,
                                                                     dA + b * stride_A,
                                                                     lda,
                                                                     sub_stride_A,
                                                                     dinvA + b * stride_invA,
                                                                     TRSM_BLOCK,
                                                                     sub_stride_invA,
                                                                     blocks));
            }

            if(K % TRSM_BLOCK != 0 || blocks == 0)
            {
                CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched<T>(
                    handle,
                    uplo,
                    diag,
                    K - TRSM_BLOCK * blocks,
                    dA + sub_stride_A * blocks + b * stride_A,
                    lda,
                    sub_stride_A,
                    dinvA + sub_stride_invA * blocks + b * stride_invA,
                    TRSM_BLOCK,
                    sub_stride_invA,
                    1));
            }
        }

        CHECK_ROCBLAS_ERROR(rocblas_trsm_strided_batched_ex_fn(handle,
                                                               side,
                                                               uplo,
                                                               transA,
                                                               diag,
                                                               M,
                                                               N,
                                                               &alpha_h,
                                                               dA,
                                                               lda,
                                                               stride_A,
                                                               dXorB,
                                                               ldb,
                                                               stride_B,
                                                               batch_count,
                                                               dinvA,
                                                               size_invA,
                                                               stride_invA,
                                                               arg.compute_type));

        CHECK_HIP_ERROR(hipMemcpy(hXorB_1, dXorB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB_2, sizeof(T) * size_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(alpha_d, &alpha_h, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_trsm_strided_batched_ex_fn(handle,
                                                               side,
                                                               uplo,
                                                               transA,
                                                               diag,
                                                               M,
                                                               N,
                                                               alpha_d,
                                                               dA,
                                                               lda,
                                                               stride_A,
                                                               dXorB,
                                                               ldb,
                                                               stride_B,
                                                               batch_count,
                                                               dinvA,
                                                               size_invA,
                                                               stride_invA,
                                                               arg.compute_type));

        CHECK_HIP_ERROR(hipMemcpy(hXorB_2, dXorB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate vector-induced-norm 1 of matrix E
        for(int b = 0; b < batch_count; b++)
        {
            max_err_1 = rocblas_abs(
                matrix_norm_1<T>(M, N, ldb, &hX[b * stride_B], &hXorB_1[b * stride_B]));
            max_err_2 = rocblas_abs(
                matrix_norm_1<T>(M, N, ldb, &hX[b * stride_B], &hXorB_2[b * stride_B]));

            //unit check
            trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

            // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
            cblas_trmm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          1.0 / alpha_h,
                          hA + stride_A * b,
                          lda,
                          hXorB_1 + stride_B * b,
                          ldb);
            cblas_trmm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          1.0 / alpha_h,
                          hA + stride_A * b,
                          lda,
                          hXorB_2 + stride_B * b,
                          ldb);

            // calculate vector-induced-norm 1 of matrix res
            max_err_1 = rocblas_abs(
                matrix_norm_1<T>(M, N, ldb, &hXorB_1[b * stride_B], &hB[b * stride_B]));
            max_err_2 = rocblas_abs(
                matrix_norm_1<T>(M, N, ldb, &hXorB_2[b * stride_B], &hB[b * stride_B]));

            //unit test
            trsm_err_res_check<T>(max_err_1, M, residual_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, residual_eps_multiplier, eps);
        }
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB_1, sizeof(T) * size_B, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        CHECK_ROCBLAS_ERROR(rocblas_trsm_strided_batched_ex_fn(handle,
                                                               side,
                                                               uplo,
                                                               transA,
                                                               diag,
                                                               M,
                                                               N,
                                                               &alpha_h,
                                                               dA,
                                                               lda,
                                                               stride_A,
                                                               dXorB,
                                                               ldb,
                                                               stride_B,
                                                               batch_count,
                                                               dinvA,
                                                               size_invA,
                                                               stride_invA,
                                                               arg.compute_type));

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // CPU cblas
        cpu_time_used = get_time_us_no_sync();

        for(int b = 0; b < batch_count; b++)
            cblas_trsm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          alpha_h,
                          hA + b * stride_A,
                          lda,
                          cpuXorB + b * stride_B,
                          ldb);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        ArgumentModel<e_side,
                      e_uplo,
                      e_transA,
                      e_diag,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_ldb,
                      e_stride_b,
                      e_batch_count>{}
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
