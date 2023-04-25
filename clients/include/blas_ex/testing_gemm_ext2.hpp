/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../library/src/include/handle.hpp"
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

template <typename Ti, typename To, typename Tc, typename To_hpa>
void reference_gemm_ext2(rocblas_int    M,
                         rocblas_int    N,
                         rocblas_int    K,
                         Tc             alpha,
                         const Ti*      A,
                         rocblas_stride row_stride_a,
                         rocblas_stride col_stride_a,
                         const Ti*      B,
                         rocblas_stride row_stride_b,
                         rocblas_stride col_stride_b,
                         Tc             beta,
                         const To*      C,
                         rocblas_stride row_stride_c,
                         rocblas_stride col_stride_c,
                         To_hpa*        D,
                         rocblas_stride row_stride_d,
                         rocblas_stride col_stride_d)
{
    for(rocblas_int row = 0; row < M; row++)
        for(rocblas_int col = 0; col < N; col++)
        {
            Tc t(0);
            if(alpha)
                for(rocblas_int k = 0; k < K; k++)
                    t += Tc(A[row * row_stride_a + k * col_stride_a])
                         * Tc(B[k * row_stride_b + col * col_stride_b]);
            D[row * row_stride_d + col * col_stride_d]
                = beta ? beta * C[row * row_stride_c + col * col_stride_c] + alpha * t : alpha * t;
        }
}

/* ============================================================================================ */
template <typename Ti, typename To, typename Tc>
void testing_gemm_ext2_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_ext2_fn
            = arg.api == FORTRAN ? rocblas_gemm_ext2_fortran : rocblas_gemm_ext2;

        const rocblas_int M = 100;
        const rocblas_int N = 100;
        const rocblas_int K = 101;

        const rocblas_int lda = 101;
        const rocblas_int ldb = 101;
        const rocblas_int ldc = 101;
        const rocblas_int ldd = 101;

        const rocblas_stride row_stride_a = 1, col_stride_a = lda;
        const rocblas_stride row_stride_b = 1, col_stride_b = ldb;
        const rocblas_stride row_stride_c = 0, col_stride_c = ldc;
        const rocblas_stride row_stride_d = 1, col_stride_d = ldd;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const rocblas_datatype a_type       = rocblas_type2datatype<Ti>();
        const rocblas_datatype b_type       = rocblas_type2datatype<Ti>();
        const rocblas_datatype c_type       = rocblas_type2datatype<To>();
        const rocblas_datatype d_type       = rocblas_type2datatype<To>();
        const rocblas_datatype compute_type = rocblas_type2datatype<Tc>();

        const rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
        int32_t                 solution_index = 0;
        rocblas_int             flags          = 0;

        static const size_t safe_size = 100;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        device_vector<Tc> alpha_d(1), beta_d(1), zero_d(1);
        const Tc          alpha_h(1), beta_h(1), zero_h(0);

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
        device_matrix<Ti> dA(A_row, A_col, col_stride_a);
        device_matrix<Ti> dB(B_row, B_col, col_stride_b);
        device_matrix<To> dC(M, N, col_stride_c);
        device_matrix<To> dD(M, N, col_stride_d);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());
        CHECK_DEVICE_ALLOCATION(dD.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   alpha,
                                                   nullptr,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   beta,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   alpha,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   nullptr,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   beta,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   alpha,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   beta,
                                                   nullptr,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   alpha,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   beta,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   nullptr,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   nullptr,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   beta,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   alpha,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   nullptr,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(nullptr,
                                                   M,
                                                   N,
                                                   K,
                                                   alpha,
                                                   dA,
                                                   a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   dB,
                                                   b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   beta,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   dC,
                                                   c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_handle);
    }
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_ext2(const Arguments& arg)
{
    auto rocblas_gemm_ext2_fn = arg.api == FORTRAN ? rocblas_gemm_ext2_fortran : rocblas_gemm_ext2;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    bool nantest = rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai);
    if(!std::is_same_v<
           To,
           float> && !std::is_same_v<To, double> && !std::is_same_v<To, rocblas_half> && !rocblas_is_complex<To> && nantest)
        return; // Exclude integers or other types which don't support NaN

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;

    rocblas_local_handle handle{arg};

    auto transA = char2rocblas_operation(arg.transA);
    auto transB = char2rocblas_operation(arg.transB);
    int  M = arg.M, N = arg.N, K = arg.K;
    int  lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;

    auto A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
    auto A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
    auto B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
    auto B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

    auto d_type = arg.d_type;

    // Row and column strides are based on transpose and leading dimensions
    rocblas_stride row_stride_a = transA == rocblas_operation_none ? 1 : lda;
    rocblas_stride col_stride_a = transA == rocblas_operation_none ? lda : 1;
    rocblas_stride row_stride_b = transB == rocblas_operation_none ? 1 : ldb;
    rocblas_stride col_stride_b = transB == rocblas_operation_none ? ldb : 1;
    rocblas_stride row_stride_c = 1;
    rocblas_stride col_stride_c = arg.ldc;
    rocblas_stride row_stride_d = 1;
    rocblas_stride col_stride_d = arg.ldd;

    // For now, rocblas_gemm_ext2 only supports row_stride_c == 0
    // We do not want to flood the Arguments structure with arbitrary strides,
    // So we artificially set row_stride_c = 0 here, leaving the other strides
    // the same as a normal GEMM
    row_stride_c = 0;

    // check if we are going to use int8x4 or int8 from arg flags
    bool pack_to_int8x4 = arg.flags & rocblas_gemm_flags_pack_int8x4;

    // check for invalid sizes
    bool invalid_size = M < 0 || N < 0 || K < 0;

    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ext2_fn(handle,
                                                   M,
                                                   N,
                                                   K,
                                                   nullptr,
                                                   nullptr,
                                                   arg.a_type,
                                                   row_stride_a,
                                                   col_stride_a,
                                                   nullptr,
                                                   arg.b_type,
                                                   row_stride_b,
                                                   col_stride_b,
                                                   nullptr,
                                                   nullptr,
                                                   arg.c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   nullptr,
                                                   arg.c_type,
                                                   row_stride_c,
                                                   col_stride_c,
                                                   arg.compute_type,
                                                   algo,
                                                   solution_index,
                                                   flags),
                              rocblas_status_invalid_size);
        return;
    }

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        device_vector<Ti> dA(1);
        device_vector<Ti> dB(1);
        device_vector<To> dC(1);
        device_vector<To> dD(1);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ext2_fn(handle,
                                                 M,
                                                 N,
                                                 K,
                                                 &h_alpha_Tc,
                                                 dA,
                                                 arg.a_type,
                                                 row_stride_a,
                                                 col_stride_a,
                                                 dB,
                                                 arg.b_type,
                                                 row_stride_b,
                                                 col_stride_b,
                                                 &h_beta_Tc,
                                                 dC,
                                                 arg.c_type,
                                                 row_stride_c,
                                                 col_stride_c,
                                                 dD,
                                                 arg.d_type,
                                                 row_stride_d,
                                                 col_stride_d,
                                                 arg.compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags));
        return;
    }
#endif

    // update after invalid checks
    if(!arg.c_noalias_d)
    {
        // c alias of d must be identical descriptors
        row_stride_d = row_stride_c;
        col_stride_d = col_stride_c;
        ldd          = ldc;
        d_type       = arg.c_type;
    }

    const size_t size_A = size_t(col_stride_a) * size_t(A_col);
    const size_t size_B = size_t(col_stride_b) * size_t(B_col);
    const size_t size_C = size_t(col_stride_c) * size_t(N);
    const size_t size_D = size_t(col_stride_d) * size_t(N);

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<Ti> hA(A_row, A_col, col_stride_a);
    host_matrix<Ti> hB(B_row, B_col, col_stride_b);
    host_matrix<To> hC(M, N, col_stride_c);
    host_matrix<To> hD_1(M, N, col_stride_d);
    host_matrix<To> hD_2(M, N, col_stride_d);
    using To_hpa = std::conditional_t<std::is_same_v<To, rocblas_bfloat16>, float, To>;
    host_matrix<To_hpa> hD_gold(M, N, col_stride_d);

    // Allocate device memory
    device_matrix<Ti> dA(A_row, A_col, col_stride_a);
    device_matrix<Ti> dB(B_row, B_col, col_stride_b);
    // if C!=D, allocate C and D normally
    // if C==D, allocate C normally and dummy D
    device_matrix<To> dC(M, N, col_stride_c);
    device_matrix<To> dD
        = (arg.c_noalias_d) ? device_matrix<To>(M, N, col_stride_d) : device_matrix<To>(0, 1, 1);
    device_matrix<To>& dDref = (arg.c_noalias_d) ? dD : dC;
    device_vector<Tc>  d_alpha_Tc(1);
    device_vector<Tc>  d_beta_Tc(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha_Tc.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta_Tc.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
    rocblas_init_matrix(hD_1, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);

    if(std::is_same_v<To, rocblas_half> && std::is_same_v<Tc, float>)
    {
        // half precision IEEE has max and lowest values 65504 and -65504,
        // float precision IEEE has max and lowest values 3.403e+38 and -3.403e+38
        // the following will overflow to inf in half arithmetic,
        // but it will equal zero in float arithmetic   65504 * 2 - 65504 * 2
        //
        // set matrix A and matrix B so reduction sum has 65504 * 2 - 65504 * 2
        //
        const rocblas_half ieee_half_near_max(65504.0 - 4.0);
        const rocblas_half positive_two(2.0);
        const rocblas_half negative_two(-2.0);
        if(M >= 2 && N >= 2 && K >= 2)
        {
            Ti* A = (Ti*)hA;
            Ti* B = (Ti*)hB;
            if(transA == rocblas_operation_none)
            {
                A[0]   = Ti(ieee_half_near_max);
                A[lda] = Ti(ieee_half_near_max);
            }
            else
            {
                A[0] = Ti(ieee_half_near_max);
                A[1] = Ti(ieee_half_near_max);
            }
            if(transB == rocblas_operation_none)
            {
                for(int j = 0; j < N; j++)
                {
                    B[j * ldb]     = j % 2 == 0 ? Ti(positive_two) : Ti(negative_two);
                    B[1 + j * ldb] = j % 2 == 0 ? Ti(negative_two) : Ti(positive_two);
                }
            }
            else
            {
                for(int j = 0; j < N; j++)
                {
                    B[j]       = j % 2 == 0 ? Ti(positive_two) : Ti(negative_two);
                    B[ldb + j] = j % 2 == 0 ? Ti(negative_two) : Ti(positive_two);
                }
            }
        }
    }

    hD_2 = hD_1;

    // copy data from CPU to device
    // do packing only when pack_to_int8x4=true (int8x4)
    // if int8x4 and A not transposed and valid case, pack A
    if(std::is_same_v<Ti, int8_t> && transA == rocblas_operation_none && pack_to_int8x4)
    {
        host_matrix<Ti> hA_packed(M, K, lda);

        rocblas_packInt8((Ti*)hA_packed, (Ti*)hA, M, K, lda);
        CHECK_HIP_ERROR(dA.transfer_from(hA_packed));
    }
    else
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }

    // do packing only when pack_to_int8x4=true (int8x4)
    // if int8x4 and B transposed and valid case, pack B
    if(std::is_same_v<Ti, int8_t> && transB != rocblas_operation_none && pack_to_int8x4)
    {
        host_matrix<Ti> hB_packed(N, K, ldb);

        rocblas_packInt8((Ti*)hB_packed, (Ti*)hB, N, K, ldb);
        CHECK_HIP_ERROR(dB.transfer_from(hB_packed));
    }
    else
    {
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }

    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ext2_fn(handle,
                                                 M,
                                                 N,
                                                 K,
                                                 &h_alpha_Tc,
                                                 dA,
                                                 arg.a_type,
                                                 row_stride_a,
                                                 col_stride_a,
                                                 dB,
                                                 arg.b_type,
                                                 row_stride_b,
                                                 col_stride_b,
                                                 &h_beta_Tc,
                                                 dC,
                                                 arg.c_type,
                                                 row_stride_c,
                                                 col_stride_c,
                                                 dDref,
                                                 d_type,
                                                 row_stride_d,
                                                 col_stride_d,
                                                 arg.compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags));

        CHECK_HIP_ERROR(hD_1.transfer_from(dDref));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        if(!arg.c_noalias_d)
        {
            CHECK_HIP_ERROR(dC.transfer_from(hC));
        }
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ext2_fn(handle,
                                                 M,
                                                 N,
                                                 K,
                                                 d_alpha_Tc,
                                                 dA,
                                                 arg.a_type,
                                                 row_stride_a,
                                                 col_stride_a,
                                                 dB,
                                                 arg.b_type,
                                                 row_stride_b,
                                                 col_stride_b,
                                                 d_beta_Tc,
                                                 dC,
                                                 arg.c_type,
                                                 row_stride_c,
                                                 col_stride_c,
                                                 dDref,
                                                 d_type,
                                                 row_stride_d,
                                                 col_stride_d,
                                                 arg.compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags));

        CHECK_HIP_ERROR(hD_2.transfer_from(dDref));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        reference_gemm_ext2<Ti, To, Tc, To_hpa>(M,
                                                N,
                                                K,
                                                h_alpha_Tc,
                                                hA,
                                                row_stride_a,
                                                col_stride_a,
                                                hB,
                                                row_stride_b,
                                                col_stride_b,
                                                h_beta_Tc,
                                                hC,
                                                row_stride_c,
                                                col_stride_c,
                                                hD_gold,
                                                row_stride_d,
                                                col_stride_d);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<To>(M, N, K) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            if((rocblas_handle(handle)->getArchMajor() == 11) && (sizeof(Ti) == 2))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<Tc, Ti, To>;
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, tol);
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, tol);
            }
            else if(std::is_same_v<Tc, rocblas_half> && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<Tc>;
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, tol);
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, tol);
            }
            else
            {
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1);
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2);
            }
        }

        if(arg.norm_check)
        {
            auto err1
                = std::abs(norm_check_general<To>('F', M, N, ldd, (To_hpa*)hD_gold, (To*)hD_1));
            auto err2
                = std::abs(norm_check_general<To>('F', M, N, ldd, (To_hpa*)hD_gold, (To*)hD_2));
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
            CHECK_ROCBLAS_ERROR(rocblas_gemm_ext2_fn(handle,
                                                     M,
                                                     N,
                                                     K,
                                                     &h_alpha_Tc,
                                                     dA,
                                                     arg.a_type,
                                                     row_stride_a,
                                                     col_stride_a,
                                                     dB,
                                                     arg.b_type,
                                                     row_stride_b,
                                                     col_stride_b,
                                                     &h_beta_Tc,
                                                     dC,
                                                     arg.c_type,
                                                     row_stride_c,
                                                     col_stride_c,
                                                     dDref,
                                                     d_type,
                                                     row_stride_d,
                                                     col_stride_d,
                                                     arg.compute_type,
                                                     algo,
                                                     solution_index,
                                                     flags));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_ext2_fn(handle,
                                 M,
                                 N,
                                 K,
                                 &h_alpha_Tc,
                                 dA,
                                 arg.a_type,
                                 row_stride_a,
                                 col_stride_a,
                                 dB,
                                 arg.b_type,
                                 row_stride_b,
                                 col_stride_b,
                                 &h_beta_Tc,
                                 dC,
                                 arg.c_type,
                                 row_stride_c,
                                 col_stride_c,
                                 dDref,
                                 d_type,
                                 row_stride_d,
                                 col_stride_d,
                                 arg.compute_type,
                                 algo,
                                 solution_index,
                                 flags);
        }
        gpu_time_used  = get_time_us_sync(stream) - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<Ti>(M, N, K) * number_hot_calls / gpu_time_used * 1e6;

        rocblas_cout << "M,N,K,alpha,row_stride_a,col_stride_a,row_stride_b,col_stride_b,beta,row_"
                        "stride_c,col_stride_c,row_stride_d,col_stride_d,rocblas-Gflops,us";

        if(arg.unit_check || arg.norm_check)
            rocblas_cout << ",CPU-Gflops(us),norm-error";

        rocblas_cout << std::endl;

        rocblas_cout << M << "," << N << "," << K << "," << arg.alpha << "," << row_stride_a << ","
                     << col_stride_a << "," << row_stride_b << "," << col_stride_b << ","
                     << arg.beta << "," << row_stride_c << "," << col_stride_c << ","
                     << row_stride_d << "," << col_stride_d << "," << rocblas_gflops << ","
                     << gpu_time_used / number_hot_calls;

        if(arg.unit_check || arg.norm_check)
        {
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;
        }

        rocblas_cout << std::endl;
    }
}
