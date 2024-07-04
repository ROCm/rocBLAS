/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "cblas_interface.hpp"
#include "client_utility.hpp"
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

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 40
#define TRSM_BLOCK 128

template <typename T>
void testing_trsm_strided_batched_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_trsm_strided_batched_ex_fn = arg.api & c_API_FORTRAN
                                                  ? rocblas_trsm_strided_batched_ex_fortran
                                                  : rocblas_trsm_strided_batched_ex;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_int M           = 100;
        const rocblas_int N           = 100;
        const rocblas_int lda         = 100;
        const rocblas_int ldb         = 100;
        const rocblas_int batch_count = 2;

        device_vector<T> alpha_d(1), zero_d(1);

        const T alpha_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        const rocblas_side      side   = rocblas_side_left;
        const rocblas_fill      uplo   = rocblas_fill_upper;
        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

        rocblas_int          K          = side == rocblas_side_left ? M : N;
        const rocblas_stride stride_A   = lda * K;
        const rocblas_stride stride_B   = ldb * N;
        const rocblas_stride strideInvA = TRSM_BLOCK * K;
        size_t               sizeinvA   = batch_count * strideInvA;

        // Allocate device memory
        device_strided_batch_matrix<T> dA(K, K, lda, stride_A, batch_count);
        device_strided_batch_matrix<T> dB(M, N, ldb, stride_B, batch_count);
        device_strided_batch_matrix<T> dinvA(TRSM_BLOCK, TRSM_BLOCK, K, strideInvA, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dinvA.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(nullptr,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 N,
                                                                 alpha,
                                                                 dA,
                                                                 lda,
                                                                 stride_A,
                                                                 dB,
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 dinvA,
                                                                 sizeinvA,
                                                                 strideInvA,
                                                                 rocblas_datatype_f32_r),
                              rocblas_status_invalid_handle);

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
                                                                 stride_A,
                                                                 dB,
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 dinvA,
                                                                 sizeinvA,
                                                                 strideInvA,
                                                                 rocblas_datatype_f32_r),
                              rocblas_status_invalid_pointer);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                                     side,
                                                                     uplo,
                                                                     transA,
                                                                     diag,
                                                                     M,
                                                                     N,
                                                                     alpha,
                                                                     nullptr,
                                                                     lda,
                                                                     stride_A,
                                                                     dB,
                                                                     ldb,
                                                                     stride_B,
                                                                     batch_count,
                                                                     dinvA,
                                                                     sizeinvA,
                                                                     strideInvA,
                                                                     rocblas_datatype_f32_r),
                                  rocblas_status_invalid_pointer);
        }

        EXPECT_ROCBLAS_STATUS(rocblas_trsm_strided_batched_ex_fn(handle,
                                                                 side,
                                                                 uplo,
                                                                 transA,
                                                                 diag,
                                                                 M,
                                                                 N,
                                                                 alpha,
                                                                 dA,
                                                                 lda,
                                                                 stride_A,
                                                                 nullptr,
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 dinvA,
                                                                 sizeinvA,
                                                                 strideInvA,
                                                                 rocblas_datatype_f32_r),
                              rocblas_status_invalid_pointer);

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
                                                                 stride_A,
                                                                 nullptr,
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 dinvA,
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
                                                                 stride_A,
                                                                 nullptr,
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 dinvA,
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
                                                                 zero,
                                                                 nullptr,
                                                                 lda,
                                                                 stride_A,
                                                                 dB,
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 dinvA,
                                                                 sizeinvA,
                                                                 strideInvA,
                                                                 rocblas_datatype_f32_r),
                              rocblas_status_success);

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
                                                                 stride_A,
                                                                 nullptr,
                                                                 ldb,
                                                                 stride_B,
                                                                 0,
                                                                 dinvA,
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
                                                                 stride_A,
                                                                 dB,
                                                                 ldb,
                                                                 stride_B,
                                                                 batch_count,
                                                                 dinvA,
                                                                 sizeinvA,
                                                                 strideInvA,
                                                                 rocblas_datatype_bf16_r),
                              rocblas_status_not_implemented);
    }
}

template <typename T>
void testing_trsm_strided_batched_ex(const Arguments& arg)
{
    auto rocblas_trsm_strided_batched_ex_fn = arg.api & c_API_FORTRAN
                                                  ? rocblas_trsm_strided_batched_ex_fortran
                                                  : rocblas_trsm_strided_batched_ex;

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

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(K, K, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hAAT(K, K, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hB(M, N, ldb, stride_B, batch_count);
    host_strided_batch_matrix<T> hX(M, N, ldb, stride_B, batch_count);
    host_strided_batch_matrix<T> hXorB_1(M, N, ldb, stride_B, batch_count);
    host_strided_batch_matrix<T> hXorB_2(M, N, ldb, stride_B, batch_count);
    host_strided_batch_matrix<T> cpuXorB(M, N, ldb, stride_B, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAAT.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hX.memcheck());
    CHECK_HIP_ERROR(hXorB_1.memcheck());
    CHECK_HIP_ERROR(hXorB_2.memcheck());
    CHECK_HIP_ERROR(cpuXorB.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(K, K, lda, stride_A, batch_count);
    device_strided_batch_matrix<T> dXorB(M, N, ldb, stride_B, batch_count);
    device_strided_batch_matrix<T> dinvA(TRSM_BLOCK, TRSM_BLOCK, K, stride_invA, batch_count);
    device_vector<T>               alpha_d(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dXorB.memcheck());
    CHECK_DEVICE_ALLOCATION(alpha_d.memcheck());
    CHECK_DEVICE_ALLOCATION(dinvA.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(hA,
                        arg,
                        rocblas_client_never_set_nan,
                        rocblas_client_diagonally_dominant_triangular_matrix,
                        true);
    rocblas_init_matrix(
        hX, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, false, true);

    hB.copy_from(hX);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(diag == rocblas_diagonal_unit)
    {
        for(int b = 0; b < batch_count; b++)
        {
            make_unit_diagonal(uplo, (T*)hA[b], lda, K);
        }
    }

    // Calculate hB = hA*hX;
    for(int b = 0; b < batch_count; b++)
        ref_trmm<T>(side,
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

    hXorB_1.copy_from(hB);
    hXorB_2.copy_from(hB);
    cpuXorB.copy_from(hB);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

    rocblas_int sub_stride_A    = TRSM_BLOCK * lda + TRSM_BLOCK;
    rocblas_int sub_stride_invA = TRSM_BLOCK * TRSM_BLOCK;

    int blocks = K / TRSM_BLOCK;

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;
    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used  = 0.0;
    double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

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
                                                                       dA[b],
                                                                       lda,
                                                                       sub_stride_A,
                                                                       dinvA[b],
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
                        (T*)dA + sub_stride_A * blocks + b * stride_A,
                        lda,
                        sub_stride_A,
                        (T*)dinvA + sub_stride_invA * blocks + b * stride_invA,
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
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

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
                                                                     dA[b],
                                                                     lda,
                                                                     sub_stride_A,
                                                                     dinvA[b],
                                                                     TRSM_BLOCK,
                                                                     sub_stride_invA,
                                                                     blocks));
            }

            if(K % TRSM_BLOCK != 0 || blocks == 0)
            {
                int remainder = K - TRSM_BLOCK * blocks;
                if(remainder)
                    CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched<T>(
                        handle,
                        uplo,
                        diag,
                        remainder,
                        (T*)dA + sub_stride_A * blocks + b * stride_A,
                        lda,
                        sub_stride_A,
                        (T*)dinvA + sub_stride_invA * blocks + b * stride_invA,
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

        CHECK_HIP_ERROR(hXorB_1.transfer_from(dXorB));

        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_2));
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

        CHECK_HIP_ERROR(hXorB_2.transfer_from(dXorB));

        if(arg.repeatability_check)
        {
            host_strided_batch_matrix<T> hXorB_copy(M, N, ldb, stride_B, batch_count);
            CHECK_HIP_ERROR(hXorB_copy.memcheck());

            for(int i = 0; i < arg.iters; i++)
            {
                CHECK_HIP_ERROR(dXorB.transfer_from(hB));
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

                CHECK_HIP_ERROR(hXorB_copy.transfer_from(dXorB));
                unit_check_general<T>(M, N, ldb, stride_B, hXorB_2, hXorB_copy, batch_count);
            }
            return;
        }

        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate vector-induced-norm 1 of matrix E
        for(int b = 0; b < batch_count; b++)
        {
            max_err_1 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hX[b], hXorB_1[b]));
            max_err_2 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hX[b], hXorB_2[b]));

            //unit check
            trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

            // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
            ref_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA[b], lda, hXorB_1[b], ldb);
            ref_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA[b], lda, hXorB_2[b], ldb);

            // calculate vector-induced-norm 1 of matrix res
            max_err_1 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hXorB_1[b], hB[b]));
            max_err_2 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hXorB_2[b], hB[b]));

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
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
        {
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
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // CPU cblas
        cpu_time_used = get_time_us_no_sync();

        for(int b = 0; b < batch_count; b++)
            ref_trsm<T>(side, uplo, transA, diag, M, N, alpha_h, hA[b], lda, cpuXorB[b], ldb);

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
