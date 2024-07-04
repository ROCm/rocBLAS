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

#define ROCBLAS_BETA_FEATURES_API
#include "../../library/src/include/handle.hpp"
#include "client_utility.hpp"
#include "rocblas.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"

template <typename Ti, typename To, typename Tc>
void testing_gemm_batched_ex_get_solutions(const Arguments& arg)
{
    rocblas_gemm_algo algo = rocblas_gemm_algo_solution_index;
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

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
    auto                 A_row       = transA == rocblas_operation_none ? M : std::max(K, 1);
    auto                 A_col       = transA == rocblas_operation_none ? std::max(K, 1) : M;
    auto                 B_row       = transB == rocblas_operation_none ? std::max(K, 1) : N;
    auto                 B_col       = transB == rocblas_operation_none ? N : std::max(K, 1);
    int                  batch_count = arg.batch_count;
    auto                 d_type      = arg.d_type;

    // Quick-return or error sizes
    // Note: K==0 is not an early exit, since we still must multiply C by beta
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M
                        || batch_count < 0;

    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex(handle,
                                                      transA,
                                                      transB,
                                                      M,
                                                      N,
                                                      K,
                                                      &h_alpha_Tc,
                                                      nullptr,
                                                      arg.a_type,
                                                      lda,
                                                      nullptr,
                                                      arg.b_type,
                                                      ldb,
                                                      nullptr,
                                                      nullptr,
                                                      arg.c_type,
                                                      ldc,
                                                      nullptr,
                                                      arg.d_type,
                                                      ldd,
                                                      batch_count,
                                                      arg.compute_type,
                                                      algo,
                                                      solution_index,
                                                      flags),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // update after invalid checks
    if(!arg.outofplace)
    {
        // c alias of d must be identical descriptors
        ldd    = ldc;
        d_type = arg.c_type;
    }

    const size_t size_one_a
        = transA == rocblas_operation_none ? size_t(K) * size_t(lda) : size_t(M) * size_t(lda);
    const size_t size_one_b
        = transB == rocblas_operation_none ? size_t(N) * size_t(ldb) : size_t(K) * size_t(ldb);
    const size_t size_one_c = N * ldc;
    const size_t size_one_d = N * ldd;
    const size_t size_a     = size_one_a;
    const size_t size_b     = size_one_b;
    const size_t size_c     = size_one_c;
    const size_t size_d     = size_one_d;

    // Allocate device memory
    device_batch_matrix<Ti> dA(A_row, A_col, lda, batch_count);
    device_batch_matrix<Ti> dB(B_row, B_col, ldb, batch_count);
    // if C!=D, allocate C and D normally
    // if C==D, allocate C big enough for the larger of C and D; D points to C
    device_batch_matrix<To>  dC(M, N, ldc, batch_count);
    device_batch_matrix<To>  dD = (arg.outofplace) ? device_batch_matrix<To>(M, N, ldd, batch_count)
                                                   : device_batch_matrix<To>(0, 1, 1, 1);
    device_batch_matrix<To>& dDref = (arg.outofplace) ? dD : dC;
    device_vector<Tc>        d_alpha_Tc(1);
    device_vector<Tc>        d_beta_Tc(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha_Tc.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta_Tc.memcheck());

#define GEMM_B_EX_ARGS                                                                        \
    handle, transA, transB, M, N, K, &h_alpha_Tc, dA.ptr_on_device(), arg.a_type, lda,        \
        dB.ptr_on_device(), arg.b_type, ldb, &h_beta_Tc, dC.ptr_on_device(), arg.c_type, ldc, \
        dD.ptr_on_device(), arg.d_type, ldd, batch_count, arg.compute_type, algo
#define rocblas_gemm_batched_exM(...) rocblas_gemm_batched_ex(__VA_ARGS__)

    // Get number of solutions
    rocblas_int size;
    CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_get_solutions(
        GEMM_B_EX_ARGS, rocblas_gemm_flags_none, NULL, &size));

    rocblas_int              size_large = size * 2;
    std::vector<rocblas_int> ary(size_large, -1);

    if(size >= 2)
    {
        rocblas_int size_small = size / 2;
        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_get_solutions(
            GEMM_B_EX_ARGS, rocblas_gemm_flags_none, ary.data(), &size_small));
        EXPECT_EQ(ary[size_small], -1);
    }

    CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_get_solutions(
        GEMM_B_EX_ARGS, rocblas_gemm_flags_none, ary.data(), &size));
    EXPECT_EQ(ary[size], -1);

    CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_get_solutions(
        GEMM_B_EX_ARGS, rocblas_gemm_flags_none, ary.data(), &size_large));
    EXPECT_EQ(ary[size], -1);

    for(auto sol : ary)
    {
        CHECK_ROCBLAS_ERROR(
            rocblas_gemm_batched_exM(GEMM_B_EX_ARGS, sol, rocblas_gemm_flags_check_solution_index));
    }

    // Testing 0 and negative values work (uses default solution)
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_batched_exM(GEMM_B_EX_ARGS, 0, rocblas_gemm_flags_check_solution_index));
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_batched_exM(GEMM_B_EX_ARGS, -1, rocblas_gemm_flags_check_solution_index));

    rocblas_int max = -1;
    for(auto sol : ary)
    {
        if(sol > max)
            max = sol;
    }

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemm_batched_exM(GEMM_B_EX_ARGS, max + 1, rocblas_gemm_flags_none),
        rocblas_status_invalid_value);

    // Testing get solutions by type - should be superset of solutions that solve problem
    rocblas_int size_type;
    CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_get_solutions_by_type(handle,
                                                                      arg.a_type,
                                                                      arg.c_type,
                                                                      arg.compute_type,
                                                                      rocblas_gemm_flags_none,
                                                                      NULL,
                                                                      &size_type));

    std::vector<rocblas_int> ary_type(size_type);
    CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_get_solutions_by_type(handle,
                                                                      arg.a_type,
                                                                      arg.c_type,
                                                                      arg.compute_type,
                                                                      rocblas_gemm_flags_none,
                                                                      ary_type.data(),
                                                                      &size_type));

    std::vector<rocblas_int> valid_ary(ary.begin(), ary.begin() + size); // Trim off junk values
    std::sort(ary_type.begin(), ary_type.end());
    std::sort(valid_ary.begin(), valid_ary.end());

    bool ary_is_subset
        = std::includes(ary_type.begin(), ary_type.end(), valid_ary.begin(), valid_ary.end());
    EXPECT_TRUE(ary_is_subset);
}
