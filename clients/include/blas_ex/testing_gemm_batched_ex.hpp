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
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "type_dispatch.hpp"
#include "unit.hpp"
#include "utility.hpp"

#define DEBUG_PRINT 0

/* ============================================================================================ */
template <typename Ti, typename To, typename Tc>
void testing_gemm_batched_ex_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_batched_ex_fn
            = arg.fortran ? rocblas_gemm_batched_ex_fortran : rocblas_gemm_batched_ex;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const rocblas_int M = 100;
        const rocblas_int N = 100;
        const rocblas_int K = 100;

        const rocblas_int lda = 100;
        const rocblas_int ldb = 100;
        const rocblas_int ldc = 100;
        const rocblas_int ldd = 100;

        const rocblas_int batch_count = 1;

        const rocblas_datatype a_type       = rocblas_type2datatype<Ti>();
        const rocblas_datatype b_type       = rocblas_type2datatype<Ti>();
        const rocblas_datatype c_type       = rocblas_type2datatype<To>();
        const rocblas_datatype d_type       = rocblas_type2datatype<To>();
        const rocblas_datatype compute_type = rocblas_type2datatype<Tc>();

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

        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
        int32_t           solution_index = 0;
        rocblas_int       flags          = 0;

        const size_t safe_size = N * ldd;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // allocate memory on device
        device_batch_vector<Ti> dA(safe_size, 1, batch_count);
        device_batch_vector<Ti> dB(safe_size, 1, batch_count);
        device_batch_vector<To> dC(safe_size, 1, batch_count);
        device_batch_vector<To> dD(safe_size, 1, batch_count);
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());
        CHECK_DEVICE_ALLOCATION(dD.memcheck());

        // host
        host_batch_vector<To> hC(safe_size, 1, batch_count);
        rocblas_seedrand();
        rocblas_init<To>(hC);
        dC.transfer_from(hC);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         alpha,
                                                         dA.ptr_on_device(),
                                                         a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         dC.ptr_on_device(), // aliased C
                                                         d_type,
                                                         ldc + 1,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         alpha,
                                                         nullptr,
                                                         a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         alpha,
                                                         dA.ptr_on_device(),
                                                         a_type,
                                                         lda,
                                                         nullptr,
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         alpha,
                                                         dA.ptr_on_device(),
                                                         a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         nullptr,
                                                         c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         alpha,
                                                         dA.ptr_on_device(),
                                                         a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         nullptr,
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         nullptr,
                                                         dA.ptr_on_device(),
                                                         a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         alpha,
                                                         dA.ptr_on_device(),
                                                         a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         b_type,
                                                         ldb,
                                                         nullptr,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(nullptr,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         alpha,
                                                         dA.ptr_on_device(),
                                                         a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_invalid_handle);

        // If batch_count==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         nullptr,
                                                         nullptr,
                                                         a_type,
                                                         lda,
                                                         nullptr,
                                                         b_type,
                                                         ldb,
                                                         nullptr,
                                                         nullptr,
                                                         c_type,
                                                         ldc,
                                                         nullptr,
                                                         d_type,
                                                         ldd,
                                                         0,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_success);

        // If M==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         0,
                                                         N,
                                                         K,
                                                         nullptr,
                                                         nullptr,
                                                         a_type,
                                                         lda,
                                                         nullptr,
                                                         b_type,
                                                         ldb,
                                                         nullptr,
                                                         nullptr,
                                                         c_type,
                                                         ldc,
                                                         nullptr,
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_success);

        // If N==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         0,
                                                         N,
                                                         K,
                                                         alpha,
                                                         dA.ptr_on_device(),
                                                         a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_success);

        /* TODO: LWPMLSE-171
        // the following tests still output to D

        // If K==0, then A and B can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         0,
                                                         alpha,
                                                         nullptr,
                                                         a_type,
                                                         lda,
                                                         nullptr,
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_success);


        // If alpha==0, then A and B can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         zero,
                                                         nullptr,
                                                         a_type,
                                                         lda,
                                                         nullptr,
                                                         b_type,
                                                         ldb,
                                                         beta,
                                                         dC.ptr_on_device(),
                                                         c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         d_type,
                                                         ldd,
                                                         batch_count,
                                                         compute_type,
                                                         algo,
                                                         solution_index,
                                                         flags),
                              rocblas_status_success);
*/
    }
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_batched_ex(const Arguments& arg)
{
    auto rocblas_gemm_batched_ex_fn
        = arg.fortran ? rocblas_gemm_batched_ex_fortran : rocblas_gemm_batched_ex;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
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
    auto                 M = arg.M, N = arg.N, K = arg.K;
    auto                 lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto                 A_row       = transA == rocblas_operation_none ? M : K;
    auto                 A_col       = transA == rocblas_operation_none ? K : M;
    auto                 B_row       = transB == rocblas_operation_none ? K : N;
    auto                 B_col       = transB == rocblas_operation_none ? N : K;
    auto                 batch_count = arg.batch_count;
    auto                 d_type      = arg.d_type;

    // Quick-return or error sizes
    // Note: K==0 is not an early exit, since we still must multiply C by beta
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M
                        || batch_count < 0;

    // size checking is only needed for int8x4
    bool pack_to_int8x4 = arg.flags & rocblas_gemm_flags_pack_int8x4;
    bool int8_invalid   = (pack_to_int8x4 && std::is_same<Ti, int8_t>{}
                         && (K % 4 != 0 || (transA != rocblas_operation_none && lda % 4 != 0)));

    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
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
    if(int8_invalid)
    {
        // This check is currently done below the invalid_pointer checks, so we can't pass in nullptrs.
        device_batch_vector<Ti> dA(1, 1, 1);
        device_batch_vector<Ti> dB(1, 1, 1);
        device_batch_vector<To> dC(1, 1, 1);
        device_batch_vector<To> dD(1, 1, 1);
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());
        CHECK_DEVICE_ALLOCATION(dD.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex_fn(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         &h_alpha_Tc,
                                                         dA.ptr_on_device(),
                                                         arg.a_type,
                                                         lda,
                                                         dB.ptr_on_device(),
                                                         arg.b_type,
                                                         ldb,
                                                         &h_beta_Tc,
                                                         dC.ptr_on_device(),
                                                         arg.c_type,
                                                         ldc,
                                                         dD.ptr_on_device(),
                                                         arg.d_type,
                                                         ldd,
                                                         batch_count,
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
        device_batch_vector<Ti> dA(1, 1, batch_count);
        device_batch_vector<Ti> dB(1, 1, batch_count);
        device_batch_vector<To> dC(1, 1, batch_count);
        device_batch_vector<To> dD(1, 1, batch_count);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_fn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       &h_alpha_Tc,
                                                       dA.ptr_on_device(),
                                                       arg.a_type,
                                                       lda,
                                                       dB.ptr_on_device(),
                                                       arg.b_type,
                                                       ldb,
                                                       &h_beta_Tc,
                                                       dC.ptr_on_device(),
                                                       arg.c_type,
                                                       ldc,
                                                       dD.ptr_on_device(),
                                                       arg.d_type,
                                                       ldd,
                                                       batch_count,
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

    // allocate memory on device
    device_batch_vector<Ti> dA(size_a, 1, batch_count);
    device_batch_vector<Ti> dB(size_b, 1, batch_count);

    // if C!=D, allocate C and D normally
    // if C==D, allocate C big enough for the larger of C and D; D points to C
    device_batch_vector<To> dC = device_batch_vector<To>(size_c, 1, batch_count);
    device_batch_vector<To> dD = (arg.c_noalias_d) ? device_batch_vector<To>(size_d, 1, batch_count)
                                                   : device_batch_vector<To>(0, 1, 0);
    device_batch_vector<To>& dDref = (arg.c_noalias_d) ? dD : dC;

    device_vector<Tc> d_alpha_Tc(1);
    device_vector<Tc> d_beta_Tc(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha_Tc.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta_Tc.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    using To_hpa = std::conditional_t<std::is_same<To, rocblas_bfloat16>{}, float, To>;
    host_batch_vector<Ti>     hA(size_a, 1, batch_count);
    host_batch_vector<Ti>     hB(size_b, 1, batch_count);
    host_batch_vector<To>     hC(size_c, 1, batch_count);
    host_batch_vector<To>     hD_1(size_d, 1, batch_count);
    host_batch_vector<To>     hD_2(size_d, 1, batch_count);
    host_batch_vector<To_hpa> hD_gold(size_d, 1, batch_count);

    // Initial Data on CPU
    rocblas_seedrand();
    for(int b = 0; b < batch_count; b++)
    {
        if(arg.alpha_isnan<Tc>())
        {
            rocblas_init_nan<Ti>(hA[b], A_row, A_col, lda);
            rocblas_init_nan<Ti>(hB[b], B_row, B_col, ldb);
        }
        else
        {
            if(arg.initialization == rocblas_initialization::rand_int)
            {
                rocblas_init<Ti>(hA[b], A_row, A_col, lda);
                rocblas_init_alternating_sign<Ti>(hB[b], B_row, B_col, ldb);
            }
            else if(arg.initialization == rocblas_initialization::trig_float)
            {
                rocblas_init_sin<Ti>(hA[b], A_row, A_col, lda);
                rocblas_init_cos<Ti>(hB[b], B_row, B_col, ldb);
            }
            else if(arg.initialization == rocblas_initialization::hpl)
            {
                rocblas_init_hpl<Ti>(hA[b], A_row, A_col, lda);
                rocblas_init_hpl<Ti>(hB[b], B_row, B_col, ldb);
            }
            else
            {
#ifdef GOOGLE_TEST
                FAIL() << "unknown initialization type";
                return;
#else
                rocblas_cerr << "unknown initialization type" << std::endl;
                rocblas_abort();
#endif
            }
        }

        if(arg.beta_isnan<Tc>())
        {
            rocblas_init_nan<To>(hC[b], M, N, ldc);
        }
        else
        {
            if(arg.initialization == rocblas_initialization::rand_int)
                rocblas_init<To>(hC[b], M, N, ldc);
            else if(arg.initialization == rocblas_initialization::trig_float)
                rocblas_init_sin<To>(hC[b], M, N, ldc);
            else if(arg.initialization == rocblas_initialization::hpl)
                rocblas_init_hpl<To>(hC[b], M, N, ldc);
        }

        rocblas_init_nan<To>(hD_1[b], M, N, ldd);
    }

    hD_2.copy_from(hD_1);
    for(int b = 0; b < batch_count; b++)
    {
        for(size_t i = 0; i < size_d; i++)
        {
            hD_gold[b][i] = hD_1[b][i];
        }
    }

#if 0 // Copied from testing_gemm_ex.hpp
    if(std::is_same<To, rocblas_half>{} && std::is_same<Tc, float>{})
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
        if(M >= 2 && N >= 2 && K >= 2)
        {
            hA[0]       = ieee_half_near_max;
            hA[1]       = ieee_half_near_max;
            hA[lda]     = ieee_half_near_max;
            hA[lda + 1] = ieee_half_near_max;
            hB[0]       = positive_two;
            hB[1]       = negative_two;
            hB[ldb]     = negative_two;
            hB[ldb + 1] = positive_two;
        }
    }
#endif

    // copy data from CPU to device
    if(std::is_same<Ti, int8_t>{} && transA == rocblas_operation_none && pack_to_int8x4)
    {
        host_batch_vector<Ti> hA_packed(size_a, 1, batch_count);
        hA_packed.copy_from(hA);

        for(int b = 0; b < batch_count; b++)
            rocblas_packInt8(hA_packed[b], hA[b], M, K, lda);

        CHECK_HIP_ERROR(dA.transfer_from(hA_packed));
    }
    else
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }

    if(std::is_same<Ti, int8_t>{} && transB != rocblas_operation_none && pack_to_int8x4)
    {
        host_batch_vector<Ti> hB_packed(size_b, 1, batch_count);
        hB_packed.copy_from(hB);

        for(int b = 0; b < batch_count; b++)
            rocblas_packInt8(hB_packed[b], hB[b], N, K, ldb);

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
        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_fn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       &h_alpha_Tc,
                                                       dA.ptr_on_device(),
                                                       arg.a_type,
                                                       lda,
                                                       dB.ptr_on_device(),
                                                       arg.b_type,
                                                       ldb,
                                                       &h_beta_Tc,
                                                       dC.ptr_on_device(),
                                                       arg.c_type,
                                                       ldc,
                                                       dDref.ptr_on_device(),
                                                       d_type,
                                                       ldd,
                                                       batch_count,
                                                       arg.compute_type,
                                                       algo,
                                                       solution_index,
                                                       flags));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hD_1.transfer_from(dDref));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_fn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       d_alpha_Tc,
                                                       dA.ptr_on_device(),
                                                       arg.a_type,
                                                       lda,
                                                       dB.ptr_on_device(),
                                                       arg.b_type,
                                                       ldb,
                                                       d_beta_Tc,
                                                       dC.ptr_on_device(),
                                                       arg.c_type,
                                                       ldc,
                                                       dDref.ptr_on_device(),
                                                       d_type,
                                                       ldd,
                                                       batch_count,
                                                       arg.compute_type,
                                                       algo,
                                                       solution_index,
                                                       flags));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hD_2.transfer_from(dDref));

        // CPU BLAS
        // copy C matrix into D matrix
        if(batch_count > 0 && N > 0 && M > 0)
            for(int b = 0; b < batch_count; b++)
                for(int i2 = 0; i2 < N; i2++)
                    for(int i1 = 0; i1 < M; i1++)
                    {
                        hD_gold[b][i1 + (i2 * ldd)] = hC[b][i1 + (i2 * ldc)];
                    }
        cpu_time_used = get_time_us_no_sync();

        for(rocblas_int b = 0; b < batch_count; b++)
        {
            cblas_gemm<Ti, To_hpa>(transA,
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
                                   ldd);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            if(std::is_same<Tc, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<Tc>;
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, batch_count, tol);
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, batch_count, tol);
            }
            else
            {
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, batch_count);
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, batch_count);
            }
        }

        if(arg.norm_check)
        {
            auto err1
                = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_1, batch_count));
            auto err2
                = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_2, batch_count));
            rocblas_error = err1 > err2 ? err1 : err2;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_fn(handle,
                                                           transA,
                                                           transB,
                                                           M,
                                                           N,
                                                           K,
                                                           &h_alpha_Tc,
                                                           dA.ptr_on_device(),
                                                           arg.a_type,
                                                           lda,
                                                           dB.ptr_on_device(),
                                                           arg.b_type,
                                                           ldb,
                                                           &h_beta_Tc,
                                                           dC.ptr_on_device(),
                                                           arg.c_type,
                                                           ldc,
                                                           dDref.ptr_on_device(),
                                                           d_type,
                                                           ldd,
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
            rocblas_gemm_batched_ex_fn(handle,
                                       transA,
                                       transB,
                                       M,
                                       N,
                                       K,
                                       &h_alpha_Tc,
                                       dA.ptr_on_device(),
                                       arg.a_type,
                                       lda,
                                       dB.ptr_on_device(),
                                       arg.b_type,
                                       ldb,
                                       &h_beta_Tc,
                                       dC.ptr_on_device(),
                                       arg.c_type,
                                       ldc,
                                       dDref.ptr_on_device(),
                                       d_type,
                                       ldd,
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
                      e_beta,
                      e_ldb,
                      e_ldc,
                      e_ldd,
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
