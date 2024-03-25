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
#include "type_dispatch.hpp"
#include "unit.hpp"
#include "utility.hpp"

#define DEBUG_PRINT 0

/* ============================================================================================ */
template <typename Ti, typename To, typename Tc>
void testing_gemm_strided_batched_ex_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_strided_batched_ex_fn = arg.api == FORTRAN
                                                      ? rocblas_gemm_strided_batched_ex_fortran
                                                      : rocblas_gemm_strided_batched_ex;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const rocblas_int M = 100;
        const rocblas_int N = 100;
        const rocblas_int K = 101;

        const rocblas_int lda = 101;
        const rocblas_int ldb = 101;
        const rocblas_int ldc = 101;
        const rocblas_int ldd = 101;

        const rocblas_int stride_a = 100 * 100;
        const rocblas_int stride_b = 100 * 100;
        const rocblas_int stride_c = 100 * 100;
        const rocblas_int stride_d = 100 * 100;

        const rocblas_int batch_count = 1;

        const rocblas_datatype a_type       = rocblas_type2datatype<Ti>();
        const rocblas_datatype b_type       = rocblas_type2datatype<Ti>();
        const rocblas_datatype c_type       = rocblas_type2datatype<To>();
        const rocblas_datatype d_type       = rocblas_type2datatype<To>();
        const rocblas_datatype compute_type = rocblas_type2datatype<Tc>();

        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
        int32_t           solution_index = 0;
        rocblas_int       flags          = 0;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        device_vector<Tc> alpha_d(1), beta_d(1), zero_d(1);

        const Tc alpha_h(1), beta_h(2), zero_h(0);

        const Tc* alpha = &alpha_h;
        const Tc* beta  = &beta_h;
        const Tc* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        rocblas_int A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
        rocblas_int A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
        rocblas_int B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
        rocblas_int B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

        // Allocate device memory
        device_strided_batch_matrix<Ti> dA(A_row, A_col, lda, stride_a, batch_count);
        device_strided_batch_matrix<Ti> dB(B_row, B_col, ldb, stride_b, batch_count);
        device_strided_batch_matrix<To> dC(M, N, ldc, stride_c, batch_count);
        device_strided_batch_matrix<To> dD(M, N, ldd, stride_d, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());
        CHECK_DEVICE_ALLOCATION(dD.memcheck());

        // clang-format off

// check for invalid enum
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, (rocblas_operation) rocblas_side_both, transB, M, N, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_value);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, (rocblas_operation) rocblas_side_both, M, N, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_value);

// check for invalid size
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, -1, N, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, -1, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, -1, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, -1, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

// check for invalid leading dimension
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, nullptr,
nullptr, a_type, M-1, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, K-1, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, nullptr,
nullptr, a_type, K-1, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, N-1, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, M-1, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, M-1, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

// check that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(nullptr, transA, transB, M, N, K, alpha,
dA, a_type, lda, stride_a, dB, b_type, ldb, stride_b, beta, dC, c_type, ldc, stride_c,
dD, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_handle);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, nullptr,
dA, a_type, lda, stride_a, dB, b_type, ldb, stride_b, beta, dC, c_type, ldc, stride_c,
dD, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, alpha,
nullptr, a_type, lda, stride_a, dB, b_type, ldb, stride_b, beta, dC, c_type, ldc, stride_c,
dD, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, beta, dC, c_type, ldc, stride_c,
dD, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, stride_a, dB, b_type, ldb, stride_b, nullptr, dC, c_type, ldc, stride_c,
dD, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, stride_a, dB, b_type, ldb, stride_b, beta, nullptr, c_type, ldc, stride_c,
dD, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_pointer);

EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, stride_a, dB, b_type, ldb, stride_b, beta, dC, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_pointer);


EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, stride_a, dB, b_type, ldb, stride_b, beta, dC, c_type, ldc, stride_c, dC, // aliased C
d_type, ldc + 1, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_invalid_size);

// If batch_count==0, then all pointers can be nullptr without issue
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c, nullptr,
d_type, ldd, stride_d, 0, compute_type, algo, solution_index, flags), rocblas_status_success);

// If M==0, then all pointers can be nullptr without issue
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, 0, N, K, nullptr,
dA, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_success);

// If N==0, then all pointers can be nullptr without issue
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, 0, K, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, nullptr, nullptr, c_type, ldc, stride_c,
nullptr, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_success);

// the following tests still output to D

// If K==0, then alpha, A and B can both be nullptr without issue.
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, 0, nullptr,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, beta, dC, c_type, ldc, stride_c,
dD, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_success);

// If alpha==0, then A and B can both be nullptr without issue.
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, zero,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, beta, dC, c_type, ldc, stride_c,
dD, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_success);

// alpha==0 && beta==1 must still copy C to D so no quick return

// If alpha==0 && beta==0 then A, B and C can be nullptr without issue.
EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle, transA, transB, M, N, K, zero,
nullptr, a_type, lda, stride_a, nullptr, b_type, ldb, stride_b, zero, nullptr, c_type, ldc, stride_c,
dD, d_type, ldd, stride_d, batch_count, compute_type, algo, solution_index, flags), rocblas_status_success);

        // clang-format on
    }
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_strided_batched_ex(const Arguments& arg)
{
    auto rocblas_gemm_strided_batched_ex_fn = arg.api == FORTRAN
                                                  ? rocblas_gemm_strided_batched_ex_fortran
                                                  : rocblas_gemm_strided_batched_ex;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    bool alpha_isnan = arg.alpha_isnan<Tc>();
    bool beta_isnan  = arg.beta_isnan<Tc>();
    if(!std::is_same_v<
           To,
           float> && !std::is_same_v<To, double> && !std::is_same_v<To, rocblas_half> && !rocblas_is_complex<To> && (alpha_isnan || beta_isnan))
        return; // Exclude integers or other types which don't support NaN

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used      = 0.0;
    double               rocblas_error = 0.0;
    rocblas_local_handle handle{arg};
    auto                 transA = char2rocblas_operation(arg.transA);
    auto                 transB = char2rocblas_operation(arg.transB);
    int                  M = arg.M, N = arg.N, K = arg.K;
    int                  lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    // dropping sign bit as test strides are positive, and no int64 host_strided_batch_matrix operator[]
    size_t stride_a = arg.stride_a, stride_b = arg.stride_b;
    size_t stride_c = arg.stride_c, stride_d = arg.stride_d;
    auto   A_row       = transA == rocblas_operation_none ? M : std::max(K, 1);
    auto   A_col       = transA == rocblas_operation_none ? std::max(K, 1) : M;
    auto   B_row       = transB == rocblas_operation_none ? std::max(K, 1) : N;
    auto   B_col       = transB == rocblas_operation_none ? N : std::max(K, 1);
    int    batch_count = arg.batch_count;
    auto   d_type      = arg.d_type;

    rocblas_math_mode math_mode = rocblas_math_mode(arg.math_mode);
    CHECK_ROCBLAS_ERROR(rocblas_set_math_mode(handle, math_mode));
    CHECK_ROCBLAS_ERROR(rocblas_get_math_mode(handle, &math_mode));

    // check for invalid sizes
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M
                        || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex_fn(handle,
                                                                 transA,
                                                                 transB,
                                                                 M,
                                                                 N,
                                                                 K,
                                                                 nullptr,
                                                                 nullptr,
                                                                 arg.a_type,
                                                                 lda,
                                                                 stride_a,
                                                                 nullptr,
                                                                 arg.b_type,
                                                                 ldb,
                                                                 stride_b,
                                                                 nullptr,
                                                                 nullptr,
                                                                 arg.c_type,
                                                                 ldc,
                                                                 stride_c,
                                                                 nullptr,
                                                                 arg.d_type,
                                                                 ldd,
                                                                 stride_d,
                                                                 batch_count,
                                                                 arg.compute_type,
                                                                 algo,
                                                                 solution_index,
                                                                 flags),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        device_strided_batch_matrix<Ti> dA(1, 1, 1, 1, 1);
        device_strided_batch_matrix<Ti> dB(1, 1, 1, 1, 1);
        device_strided_batch_matrix<To> dC(1, 1, 1, 1, 1);
        device_strided_batch_matrix<To> dD(1, 1, 1, 1, 1);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_fn(handle,
                                                               transA,
                                                               transB,
                                                               M,
                                                               N,
                                                               K,
                                                               &h_alpha_Tc,
                                                               dA,
                                                               arg.a_type,
                                                               lda,
                                                               stride_a,
                                                               dB,
                                                               arg.b_type,
                                                               ldb,
                                                               stride_b,
                                                               &h_beta_Tc,
                                                               dC,
                                                               arg.c_type,
                                                               ldc,
                                                               stride_c,
                                                               dD,
                                                               arg.d_type,
                                                               ldd,
                                                               stride_d,
                                                               batch_count,
                                                               arg.compute_type,
                                                               algo,
                                                               solution_index,
                                                               flags));
        return;
    }
#endif
    // update after invalid checks
    if(!arg.outofplace)
    {
        ldd      = ldc;
        stride_d = stride_c;
        d_type   = arg.c_type;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    using To_hpa = std::conditional_t<std::is_same_v<To, rocblas_bfloat16>, float, To>;
    host_strided_batch_matrix<Ti>     hA(A_row, A_col, lda, stride_a, batch_count);
    host_strided_batch_matrix<Ti>     hB(B_row, B_col, ldb, stride_b, batch_count);
    host_strided_batch_matrix<To>     hC(M, N, ldc, stride_c, batch_count);
    host_strided_batch_matrix<To>     hD_1(M, N, ldd, stride_d, batch_count);
    host_strided_batch_matrix<To>     hD_2(M, N, ldd, stride_d, batch_count);
    host_strided_batch_matrix<To_hpa> hD_gold(M, N, ldd, stride_d, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());
    CHECK_HIP_ERROR(hD_1.memcheck());
    CHECK_HIP_ERROR(hD_2.memcheck());
    CHECK_HIP_ERROR(hD_gold.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<Ti> dA(A_row, A_col, lda, stride_a, batch_count);
    device_strided_batch_matrix<Ti> dB(B_row, B_col, ldb, stride_b, batch_count);
    // if C!=D, allocate C and D normally
    // if C==D, allocate C big enough for the larger of C and D; D points to C
    device_strided_batch_matrix<To> dC(M, N, ldc, stride_c, batch_count);
    device_strided_batch_matrix<To> dD
        = (arg.outofplace) ? device_strided_batch_matrix<To>(M, N, ldd, stride_d, batch_count)
                           : device_strided_batch_matrix<To>(0, 1, 1, 1, 1);
    device_strided_batch_matrix<To>& dDref = (arg.outofplace) ? dD : dC;
    device_vector<Tc>                d_alpha_Tc(1);
    device_vector<Tc>                d_beta_Tc(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha_Tc.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta_Tc.memcheck());

    bool alt       = (rocblas_gemm_flags_fp16_alt_impl & flags);
    bool alt_round = (rocblas_gemm_flags_fp16_alt_impl_rnz & flags);

    // Initialize data on host memory
    rocblas_init_matrix<Ti>(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix<Ti, true>(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix<To, true>(
        hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    rocblas_init_nan<To>(hD_1, M, N, ldd, stride_d, batch_count);
    rocblas_init_nan<To_hpa>(hD_gold, M, N, ldd, stride_d, batch_count);

    hD_2.copy_from(hD_1);

#if 0 // Copied from testing_gemm_ex.hpp
    if(std::is_same_v<To, rocblas_half> && std::is_same_v<Tc, float>)
    {
        // half precision IEEE has max and lowest values 65504 and -65504,
        // foat precision IEEE has max and lowest values 3.403e+38 and -3.403e+38
        // the following will overflow to inf in half arithmetic,
        // but it will equal zero in float arithmetic   65504 * 2 - 65504 * 2
        //
        // set matrix A and matrix B upper left block to values below to cause
        // inf overflow with 16 bit arithmetic, but no overflow for 32 bit arithmetic
        //
        // 65500 65500             2   -2
        // 65500 65500            -2    2
        //
        const rocblas_half ieee_half_near_max(65504.0 - 4.0);
        const rocblas_half positive_two      (2.0);
        const rocblas_half negative_two      (-2.0);
        Ti* A = (Ti*)hA;
        Ti* B = (Ti*)hB;
        if(M >= 2 && N >= 2 && K >= 2)
        {
            A[0]       = ieee_half_near_max;
            A[1]       = ieee_half_near_max;
            A[lda]     = ieee_half_near_max;
            A[lda + 1] = ieee_half_near_max;
            B[0]       = positive_two;
            B[1]       = negative_two;
            B[ldb]     = negative_two;
            B[ldb + 1] = positive_two;
        }
    }
#endif

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_fn(handle,
                                                               transA,
                                                               transB,
                                                               M,
                                                               N,
                                                               K,
                                                               &h_alpha_Tc,
                                                               dA,
                                                               arg.a_type,
                                                               lda,
                                                               stride_a,
                                                               dB,
                                                               arg.b_type,
                                                               ldb,
                                                               stride_b,
                                                               &h_beta_Tc,
                                                               dC,
                                                               arg.c_type,
                                                               ldc,
                                                               stride_c,
                                                               dDref,
                                                               d_type,
                                                               ldd,
                                                               stride_d,
                                                               batch_count,
                                                               arg.compute_type,
                                                               algo,
                                                               solution_index,
                                                               flags));
        handle.post_test(arg);
        // copy output from device to CPU
        CHECK_HIP_ERROR(hD_1.transfer_from(dDref));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_fn(handle,
                                                               transA,
                                                               transB,
                                                               M,
                                                               N,
                                                               K,
                                                               d_alpha_Tc,
                                                               dA,
                                                               arg.a_type,
                                                               lda,
                                                               stride_a,
                                                               dB,
                                                               arg.b_type,
                                                               ldb,
                                                               stride_b,
                                                               d_beta_Tc,
                                                               dC,
                                                               arg.c_type,
                                                               ldc,
                                                               stride_c,
                                                               dDref,
                                                               d_type,
                                                               ldd,
                                                               stride_d,
                                                               batch_count,
                                                               arg.compute_type,
                                                               algo,
                                                               solution_index,
                                                               flags));

        CHECK_HIP_ERROR(hD_2.transfer_from(dDref));

        // copy C matrix into D matrix
        copy_matrix_with_different_leading_dimensions(hC, hD_gold);

        // For the xf32 xdl math op, cast type of A/B from float to xfloat32 .
        if(std::is_same<Ti, float>{} && math_mode == rocblas_xf32_xdl_math_op)
        {
            type_to_xdl_math_op_type<rocblas_xfloat32, float>(hA.data(), hA.nmemb());
            type_to_xdl_math_op_type<rocblas_xfloat32, float>(hB.data(), hB.nmemb());
        }

        cpu_time_used = get_time_us_no_sync();

        // CPU BLAS
        for(rocblas_int b = 0; b < batch_count; b++)
        {
            ref_gemm<Ti, To_hpa>(
                transA,
                transB,
                M,
                N,
                K,
                h_alpha_Tc,
                hA[b],
                lda,
                hB[b],
                ldb,
                h_beta_Tc,
                hD_gold[b],
                ldd,
                alt ? (alt_round ? rocblas_bfloat16::rocblas_truncate_t::rocblas_round_near_zero
                                 : rocblas_bfloat16::rocblas_truncate_t::rocblas_truncate)
                    : rocblas_bfloat16::rocblas_truncate_t::rocblas_round_near_even);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            if((rocblas_handle(handle)->getArchMajor() == 11) && (sizeof(Ti) == 2))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<Tc, Ti, To>;
                near_check_general<To, To_hpa>(
                    M, N, ldd, stride_d, hD_gold, hD_1, batch_count, tol);
                near_check_general<To, To_hpa>(
                    M, N, ldd, stride_d, hD_gold, hD_2, batch_count, tol);
            }
            else if(std::is_same_v<Tc, rocblas_half> && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<Tc>;
                near_check_general<To, To_hpa>(
                    M, N, ldd, stride_d, hD_gold, hD_1, batch_count, tol);
                near_check_general<To, To_hpa>(
                    M, N, ldd, stride_d, hD_gold, hD_2, batch_count, tol);
            }
            else
            {
                unit_check_general<To, To_hpa>(M, N, ldd, stride_d, hD_gold, hD_1, batch_count);
                unit_check_general<To, To_hpa>(M, N, ldd, stride_d, hD_gold, hD_2, batch_count);
            }
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general('F', hD_gold, hD_1));
            auto err2     = std::abs(norm_check_general('F', hD_gold, hD_2));
            rocblas_error = err1 > err2 ? err1 : err2;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_fn(handle,
                                                                   transA,
                                                                   transB,
                                                                   M,
                                                                   N,
                                                                   K,
                                                                   &h_alpha_Tc,
                                                                   dA,
                                                                   arg.a_type,
                                                                   lda,
                                                                   stride_a,
                                                                   dB,
                                                                   arg.b_type,
                                                                   ldb,
                                                                   stride_b,
                                                                   &h_beta_Tc,
                                                                   dC,
                                                                   arg.c_type,
                                                                   ldc,
                                                                   stride_c,
                                                                   dDref,
                                                                   d_type,
                                                                   ldd,
                                                                   stride_d,
                                                                   batch_count,
                                                                   arg.compute_type,
                                                                   algo,
                                                                   solution_index,
                                                                   flags));
        }

        int         number_hot_calls = arg.iters;
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_strided_batched_ex_fn(handle,
                                               transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               &h_alpha_Tc,
                                               dA,
                                               arg.a_type,
                                               lda,
                                               stride_a,
                                               dB,
                                               arg.b_type,
                                               ldb,
                                               stride_b,
                                               &h_beta_Tc,
                                               dC,
                                               arg.c_type,
                                               ldc,
                                               stride_c,
                                               dDref,
                                               d_type,
                                               ldd,
                                               stride_d,
                                               batch_count,
                                               arg.compute_type,
                                               algo,
                                               solution_index,
                                               flags);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_beta,
                      e_ldb,
                      e_stride_b,
                      e_ldc,
                      e_stride_c,
                      e_ldd,
                      e_stride_d,
                      e_batch_count>{}
            .log_args<To>(rocblas_cout,
                          arg,
                          gpu_time_used,
                          gemm_gflop_count<Tc>(M, N, K),
                          ArgumentLogging::NA_value,
                          cpu_time_used,
                          rocblas_error);
    }
}
