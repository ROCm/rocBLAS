/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
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
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

/* ============================================================================================ */
template <typename Ti, typename To, typename Tc>
void testing_gemm_ex_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_ex_fn = arg.fortran ? rocblas_gemm_ex_fortran : rocblas_gemm_ex;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const rocblas_int M = 100;
        const rocblas_int N = 100;
        const rocblas_int K = 100;

        const rocblas_int lda = 100;
        const rocblas_int ldb = 100;
        const rocblas_int ldc = 100;
        const rocblas_int ldd = 100;

        const rocblas_datatype a_type       = rocblas_datatype_f32_r;
        const rocblas_datatype b_type       = rocblas_datatype_f32_r;
        const rocblas_datatype c_type       = rocblas_datatype_f32_r;
        const rocblas_datatype d_type       = rocblas_datatype_f32_r;
        const rocblas_datatype compute_type = rocblas_datatype_f32_r;

        device_vector<float> alpha_d(1), beta_d(1), zero_d(1);
        const float          alpha_h(1), beta_h(1), zero_h(0);

        const float* alpha = &alpha_h;
        const float* beta  = &beta_h;
        const float* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        const rocblas_gemm_algo algo      = rocblas_gemm_algo_standard;
        static const size_t     safe_size = 100;

        int32_t     solution_index = 0;
        rocblas_int flags          = 0;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // allocate memory on device
        device_vector<float> dA(safe_size);
        device_vector<float> dB(safe_size);
        device_vector<float> dC(safe_size);
        device_vector<float> dD(safe_size);
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());
        CHECK_DEVICE_ALLOCATION(dD.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 nullptr,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 nullptr,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 nullptr,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 nullptr,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 nullptr,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 nullptr,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(nullptr,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,

                                                 ldd,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_invalid_handle);

        // If M==0, then all pointers can be nullptr without issue
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
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
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_success);

        // If N==0, then all pointers can be nullptr without issue
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 0,
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
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_success);

#if 0
        // TODO: Currently these tests fail
        // If K==0, then A and B can both be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
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
                                                 &beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_success);

        // If alpha==0, then A and B can both be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
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
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_success);
#endif
    }
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_ex(const Arguments& arg)
{
    auto rocblas_gemm_ex_fn = arg.fortran ? rocblas_gemm_ex_fortran : rocblas_gemm_ex;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    bool alpha_isnan = arg.alpha_isnan<Tc>();
    bool beta_isnan  = arg.beta_isnan<Tc>();
    if(!std::is_same<To, float>{} && !std::is_same<To, double>{}
       && !std::is_same<To, rocblas_half>{} && !is_complex<To> && (alpha_isnan || beta_isnan))
        return; // Exclude integers or other types which don't support NaN

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double rocblas_error          = 0.0;

    rocblas_local_handle handle{arg};
    auto                 transA = char2rocblas_operation(arg.transA);
    auto                 transB = char2rocblas_operation(arg.transB);
    auto                 M = arg.M, N = arg.N, K = arg.K;
    auto                 lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto                 A_row = transA == rocblas_operation_none ? M : K;
    auto                 A_col = transA == rocblas_operation_none ? K : M;
    auto                 B_row = transB == rocblas_operation_none ? K : N;
    auto                 B_col = transB == rocblas_operation_none ? N : K;

    // check for invalid sizes
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M;

    // size checking is only needed for int8x4
    bool pack_to_int8x4 = arg.flags & rocblas_gemm_flags_pack_int8x4;
    bool int8_invalid   = (pack_to_int8x4 && std::is_same<Ti, int8_t>{}
                         && (K % 4 != 0 || (transA != rocblas_operation_none && lda % 4 != 0)
                             || (transB == rocblas_operation_none && ldb % 4 != 0)));

    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 nullptr,
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
                                                 arg.compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags),
                              rocblas_status_invalid_size);
        return;
    }
    if(int8_invalid)
    {
        // This check is currently done below the invalid_pointer checks, so we can't pass in nullptrs.
        static const size_t safe_size = 100;
        device_vector<Ti>   dA(safe_size);
        device_vector<Ti>   dB(safe_size);
        device_vector<To>   dC(safe_size);
        device_vector<To>   dD(safe_size);
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());
        CHECK_DEVICE_ALLOCATION(dD.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 &h_alpha_Tc,
                                                 dA,
                                                 arg.a_type,
                                                 lda,
                                                 dB,
                                                 arg.b_type,
                                                 ldb,
                                                 &h_beta_Tc,
                                                 dC,
                                                 arg.c_type,
                                                 ldc,
                                                 dD,
                                                 arg.d_type,
                                                 ldd,
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
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_fn(handle,
                                               transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               &h_alpha_Tc,
                                               dA,
                                               arg.a_type,
                                               lda,
                                               dB,
                                               arg.b_type,
                                               ldb,
                                               &h_beta_Tc,
                                               dC,
                                               arg.c_type,
                                               ldc,
                                               dD,
                                               arg.d_type,
                                               ldd,
                                               arg.compute_type,
                                               algo,
                                               solution_index,
                                               flags));
        return;
    }
#endif

    const size_t size_A      = size_t(lda) * size_t(A_col);
    const size_t size_B      = size_t(ldb) * size_t(B_col);
    const size_t size_C      = size_t(ldc) * size_t(N);
    const size_t size_D      = size_t(ldd) * size_t(N);
    const size_t max_CD      = std::max(size_C, size_D);
    const size_t size_D_copy = arg.unit_check || arg.norm_check ? size_D : 0;

    // allocate memory on device
    device_vector<Ti> dA(size_A);
    device_vector<Ti> dB(size_B);

    // if C!=D, allocate C and D normally
    // if C==D, allocate C big enough for the larger of C and D; D points to C
    device_vector<To> dC
        = (arg.c_noalias_d) ? device_vector<To>(size_C) : device_vector<To>(max_CD);
    device_vector<To>  dD    = (arg.c_noalias_d) ? device_vector<To>(size_D) : device_vector<To>(0);
    device_vector<To>& dDref = (arg.c_noalias_d) ? dD : dC;

    device_vector<Tc> d_alpha_Tc(1);
    device_vector<Tc> d_beta_Tc(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha_Tc.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta_Tc.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti> hA(size_A);
    host_vector<Ti> hB(size_B);
    host_vector<To> hC(size_C);
    host_vector<To> hD_1(size_D_copy);
    using To_hpa = std::conditional_t<std::is_same<To, rocblas_bfloat16>{}, float, To>;
    host_vector<To_hpa> hD_gold(size_D_copy);

    bool alt = (rocblas_gemm_flags_fp16_alt_impl & flags);

    rocblas_seedrand();

    // Initial Data on CPU
    if(alpha_isnan)
    {
        rocblas_init_nan<Ti>(hA, A_row, A_col, lda);
        rocblas_init_nan<Ti>(hB, B_row, B_col, ldb);
    }
    else
    {
        if(arg.initialization == rocblas_initialization::rand_int)
        {
            rocblas_init<Ti>(hA, A_row, A_col, lda);
            rocblas_init_alternating_sign<Ti>(hB, B_row, B_col, ldb);
        }
        else if(arg.initialization == rocblas_initialization::trig_float)
        {
            rocblas_init_sin<Ti>(hA, A_row, A_col, lda);
            rocblas_init_cos<Ti>(hB, B_row, B_col, ldb);
        }
        else if(arg.initialization == rocblas_initialization::hpl)
        {
            rocblas_init_hpl<Ti>(hA, A_row, A_col, lda);
            rocblas_init_hpl<Ti>(hB, B_row, B_col, ldb);
        }
        else if(arg.initialization == rocblas_initialization::special)
        {
            rocblas_init_alt_impl_big<Ti>(hA, A_row, A_col, lda);
            rocblas_init_alt_impl_small<Ti>(hB, B_row, B_col, ldb);
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
    if(beta_isnan)
    {
        rocblas_init_nan<To>(hC, M, N, ldc);
    }
    else
    {
        if(arg.initialization == rocblas_initialization::rand_int)
            rocblas_init<To>(hC, M, N, ldc);
        else if(arg.initialization == rocblas_initialization::trig_float)
            rocblas_init_sin<To>(hC, M, N, ldc);
        else if(arg.initialization == rocblas_initialization::hpl)
            rocblas_init_hpl<To>(hC, M, N, ldc);
        else if(arg.initialization == rocblas_initialization::special)
            rocblas_init<To>(hC, M, N, ldc);
    }
    if(size_D_copy)
    {
        rocblas_init_nan<To>(hD_1, M, N, ldd);
        hD_gold = hD_1;
    }

    if(std::is_same<To, rocblas_half>{} && std::is_same<Tc, float>{}
       && arg.initialization != rocblas_initialization::special)
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
            if(transA == rocblas_operation_none)
            {
                hA[0]   = Ti(ieee_half_near_max);
                hA[lda] = Ti(ieee_half_near_max);
            }
            else
            {
                hA[0] = Ti(ieee_half_near_max);
                hA[1] = Ti(ieee_half_near_max);
            }
            if(transB == rocblas_operation_none)
            {
                for(int j = 0; j < N; j++)
                {
                    hB[j * ldb]     = j % 2 == 0 ? Ti(positive_two) : Ti(negative_two);
                    hB[1 + j * ldb] = j % 2 == 0 ? Ti(negative_two) : Ti(positive_two);
                }
            }
            else
            {
                for(int j = 0; j < N; j++)
                {
                    hB[j]       = j % 2 == 0 ? Ti(positive_two) : Ti(negative_two);
                    hB[ldb + j] = j % 2 == 0 ? Ti(negative_two) : Ti(positive_two);
                }
            }
        }
    }

    // copy data from CPU to device
    // do packing only when pack_to_int8x4=true (int8x4)
    // if int8x4 and A not transposed and valid case, pack A
    if(std::is_same<Ti, int8_t>{} && transA == rocblas_operation_none && pack_to_int8x4)
    {
        host_vector<Ti> hA_packed(hA);

        rocblas_packInt8(hA_packed, M, K, lda);
        CHECK_HIP_ERROR(hipMemcpy(dA, hA_packed, sizeof(Ti) * size_A, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ti) * size_A, hipMemcpyHostToDevice));
    }

    // do packing only when pack_to_int8x4=true (int8x4)
    // if int8x4 and B transposed and valid case, pack B
    if(std::is_same<Ti, int8_t>{} && transB != rocblas_operation_none && pack_to_int8x4)
    {
        host_vector<Ti> hB_packed(hB);

        rocblas_packInt8(hB_packed, N, K, ldb);
        CHECK_HIP_ERROR(hipMemcpy(dB, hB_packed, sizeof(Ti) * size_B, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(Ti) * size_B, hipMemcpyHostToDevice));
    }

    CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(To) * size_C, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_fn(handle,
                                               transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               &h_alpha_Tc,
                                               dA,
                                               arg.a_type,
                                               lda,
                                               dB,
                                               arg.b_type,
                                               ldb,
                                               &h_beta_Tc,
                                               dC,
                                               arg.c_type,
                                               ldc,
                                               dDref,
                                               arg.d_type,
                                               ldd,
                                               arg.compute_type,
                                               algo,
                                               solution_index,
                                               flags));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hD_1, dDref, sizeof(To) * size_D, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(To) * size_C, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_fn(handle,
                                               transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               d_alpha_Tc,
                                               dA,
                                               arg.a_type,
                                               lda,
                                               dB,
                                               arg.b_type,
                                               ldb,
                                               d_beta_Tc,
                                               dC,
                                               arg.c_type,
                                               ldc,
                                               dDref,
                                               arg.d_type,
                                               ldd,
                                               arg.compute_type,
                                               algo,
                                               solution_index,
                                               flags));

        // CPU BLAS
        // copy C matrix into D matrix
        for(int i2 = 0; i2 < N; i2++)
            for(int i1 = 0; i1 < M; i1++)
                hD_gold[i1 + i2 * ldd] = hC[i1 + i2 * ldc];

        cpu_time_used = get_time_us_no_sync();

        cblas_gemm<Ti, To_hpa, Tc>(
            transA, transB, M, N, K, h_alpha_Tc, hA, lda, hB, ldb, h_beta_Tc, hD_gold, ldd, alt);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        //releasing already used host memory
        hA = host_vector<Ti>();
        hB = host_vector<Ti>();
        hC = host_vector<To>();

        if(arg.unit_check)
        {
            if(std::is_same<Tc, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = sqrt(K) * sum_error_tolerance<Tc>;
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, tol);
            }
            else
            {
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1);
            }
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_1));
            rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
        }

        // fetch device mode GPU results
        CHECK_HIP_ERROR(hipMemcpy(hD_1, dDref, sizeof(To) * size_D, hipMemcpyDeviceToHost));

        if(arg.unit_check)
        {
            if(std::is_same<Tc, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<Tc>;
                near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, tol);
            }
            else
            {
                unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1);
            }
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_1));
            rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_fn(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   &h_alpha_Tc,
                                                   dA,
                                                   arg.a_type,
                                                   lda,
                                                   dB,
                                                   arg.b_type,
                                                   ldb,
                                                   &h_beta_Tc,
                                                   dC,
                                                   arg.c_type,
                                                   ldc,
                                                   dDref,
                                                   arg.d_type,
                                                   ldd,
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
            rocblas_gemm_ex_fn(handle,
                               transA,
                               transB,
                               M,
                               N,
                               K,
                               &h_alpha_Tc,
                               dA,
                               arg.a_type,
                               lda,
                               dB,
                               arg.b_type,
                               ldb,
                               &h_beta_Tc,
                               dC,
                               arg.c_type,
                               ldc,
                               dDref,
                               arg.d_type,
                               ldd,
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
