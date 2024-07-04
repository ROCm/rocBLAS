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
void testing_gemm_strided_batched_ex_get_solutions(const Arguments& arg)
{
    rocblas_gemm_algo algo = rocblas_gemm_algo_solution_index;
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

    // check for invalid sizes
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M
                        || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex(handle,
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

    // update after invalid checks
    if(!arg.outofplace)
    {
        ldd      = ldc;
        stride_d = stride_c;
        d_type   = arg.c_type;
    }

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

#define GEMM_SB_EX_ARGS                                                                          \
    handle, transA, transB, M, N, K, &h_alpha_Tc, dA, arg.a_type, lda, stride_a, dB, arg.b_type, \
        ldb, stride_b, &h_beta_Tc, dC, arg.c_type, ldc, stride_c, dDref, d_type, ldd, stride_d,  \
        batch_count, arg.compute_type, algo
#define rocblas_gemm_strided_batched_exM(...) rocblas_gemm_strided_batched_ex(__VA_ARGS__)

    // Get number of solutions
    rocblas_int size;
    CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_get_solutions(
        GEMM_SB_EX_ARGS, rocblas_gemm_flags_none, NULL, &size));

    rocblas_int              size_large = size * 2;
    std::vector<rocblas_int> ary(size_large, -1);

    if(size >= 2)
    {
        rocblas_int size_small = size / 2;
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_get_solutions(
            GEMM_SB_EX_ARGS, rocblas_gemm_flags_none, ary.data(), &size_small));
        EXPECT_EQ(ary[size_small], -1);
    }

    CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_get_solutions(
        GEMM_SB_EX_ARGS, rocblas_gemm_flags_none, ary.data(), &size));
    EXPECT_EQ(ary[size], -1);

    CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_get_solutions(
        GEMM_SB_EX_ARGS, rocblas_gemm_flags_none, ary.data(), &size_large));
    EXPECT_EQ(ary[size], -1);

    for(auto sol : ary)
    {
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_exM(
            GEMM_SB_EX_ARGS, sol, rocblas_gemm_flags_check_solution_index));
    }

    // Testing 0 and negative values work (uses default solution)
    CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_exM(
        GEMM_SB_EX_ARGS, 0, rocblas_gemm_flags_check_solution_index));
    CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_exM(
        GEMM_SB_EX_ARGS, -1, rocblas_gemm_flags_check_solution_index));

    rocblas_int max = -1;
    for(auto sol : ary)
    {
        if(sol > max)
            max = sol;
    }

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemm_strided_batched_exM(GEMM_SB_EX_ARGS, max + 1, rocblas_gemm_flags_none),
        rocblas_status_invalid_value);

    // Testing get solutions by type - should be superset of solutions that solve problem
    rocblas_int size_type;
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                              arg.a_type,
                                                              arg.c_type,
                                                              arg.compute_type,
                                                              rocblas_gemm_flags_none,
                                                              NULL,
                                                              &size_type));

    std::vector<rocblas_int> ary_type(size_type);
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions_by_type(handle,
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
