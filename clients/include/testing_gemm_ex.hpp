/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "handle.h"
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

    const float alpha_float = 1.0;
    const float beta_float  = 1.0;

    const rocblas_gemm_algo algo      = rocblas_gemm_algo_standard;
    static const size_t     safe_size = 100;

    int32_t              solution_index = 0;
    rocblas_int          flags          = 0;
    rocblas_local_handle handle;

    // allocate memory on device
    device_vector<float> dA(safe_size);
    device_vector<float> dB(safe_size);
    device_vector<float> dC(safe_size);
    device_vector<float> dD(safe_size);
    if(!dA || !dB || !dC || !dD)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex(handle,
                                          transA,
                                          transB,
                                          M,
                                          N,
                                          K,
                                          &alpha_float,
                                          nullptr,
                                          a_type,
                                          lda,
                                          dB,
                                          b_type,
                                          ldb,
                                          &beta_float,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex(handle,
                                          transA,
                                          transB,
                                          M,
                                          N,
                                          K,
                                          &alpha_float,
                                          dA,
                                          a_type,
                                          lda,
                                          nullptr,
                                          b_type,
                                          ldb,
                                          &beta_float,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex(handle,
                                          transA,
                                          transB,
                                          M,
                                          N,
                                          K,
                                          &alpha_float,
                                          dA,
                                          a_type,
                                          lda,
                                          dB,
                                          b_type,
                                          ldb,
                                          &beta_float,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex(handle,
                                          transA,
                                          transB,
                                          M,
                                          N,
                                          K,
                                          &alpha_float,
                                          dA,
                                          a_type,
                                          lda,
                                          dB,
                                          b_type,
                                          ldb,
                                          &beta_float,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex(handle,
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
                                          &beta_float,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex(handle,
                                          transA,
                                          transB,
                                          M,
                                          N,
                                          K,
                                          &alpha_float,
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

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex(nullptr,
                                          transA,
                                          transB,
                                          M,
                                          N,
                                          K,
                                          &alpha_float,
                                          dA,
                                          a_type,
                                          lda,
                                          dB,
                                          b_type,
                                          ldb,
                                          &beta_float,
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
}

namespace
{
    bool is_replacement_kernel(rocblas_operation transA,
                               rocblas_operation transB,
                               rocblas_int       m,
                               rocblas_int       n,
                               rocblas_int       k)
    {
        int arc = _rocblas_handle::device_arch_id();
        if(arc == 908 && transA == rocblas_operation_transpose && transB == rocblas_operation_none
           && ((m == 512 && n == 512 && k == 512) || (m == 1024 && n == 1024 && k == 1024)
               || (m == 2048 && n == 2048 && k == 2048) || (m == 4096 && n == 4096 && k == 4096)
               || (m == 960 && n == 1024 && k == 1024) || (m == 3840 && n == 4096 && k == 4096)))
            return true;
        return false;
    }

    template <typename Ti, typename To, typename Tc>
    void reference_gemm(rocblas_operation transA,
                        rocblas_operation transB,
                        rocblas_int       m,
                        rocblas_int       n,
                        rocblas_int       k,
                        Tc                alpha,
                        Ti*               A,
                        rocblas_int       lda,
                        Ti*               B,
                        rocblas_int       ldb,
                        Tc                beta,
                        To*               C,
                        rocblas_int       ldc)
    {
        cblas_gemm<Ti, To, Tc>(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template <>
    void reference_gemm(rocblas_operation transA,
                        rocblas_operation transB,
                        rocblas_int       m,
                        rocblas_int       n,
                        rocblas_int       k,
                        float             alpha,
                        rocblas_bfloat16* A,
                        rocblas_int       lda,
                        rocblas_bfloat16* B,
                        rocblas_int       ldb,
                        float             beta,
                        rocblas_bfloat16* C,
                        rocblas_int       ldc)
    {
        const size_t       size_C = size_t(ldc) * size_t(n);
        host_vector<float> C_float(size_C);
        for(int i = 0; i < size_C; ++i)
            C_float[i] = C[i];
        cblas_gemm<rocblas_bfloat16, float, float>(
            transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C_float, ldc);
        bool round = !is_replacement_kernel(transA, transB, m, n, k);
        for(int i = 0; i < size_C; ++i)
            C[i] = round ? rocblas_bfloat16(C_float[i]) : float_to_bfloat16_truncate(C_float[i]);
    }
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_ex(const Arguments& arg)
{
    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    bool nantest = rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai);
    if(!std::is_same<To, float>{} && !std::is_same<To, double>{}
       && !std::is_same<To, rocblas_half>{} && !is_complex<To> && nantest)
        return; // Exclude integers or other types which don't support NaN

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;

    rocblas_local_handle handle;
    auto                 transA = char2rocblas_operation(arg.transA);
    auto                 transB = char2rocblas_operation(arg.transB);
    auto                 M = arg.M, N = arg.N, K = arg.K;
    auto                 lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto                 A_row = transA == rocblas_operation_none ? M : K;
    auto                 A_col = transA == rocblas_operation_none ? K : M;
    auto                 B_row = transB == rocblas_operation_none ? K : N;
    auto                 B_col = transB == rocblas_operation_none ? N : K;

    // check for invalid sizes
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M
       || (std::is_same<Ti, int8_t>{}
           && (K % 4 != 0 || (transA != rocblas_operation_none && lda % 4 != 0)
               || (transB == rocblas_operation_none && ldb % 4 != 0))))
    {
        static const size_t safe_size = 100;
        device_vector<Ti>   dA(safe_size);
        device_vector<Ti>   dB(safe_size);
        device_vector<To>   dC(safe_size);
        device_vector<To>   dD(safe_size);
        if(!dA || !dB || !dC || !dD)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_ex(handle,
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

    const size_t size_A = size_t(lda) * size_t(A_col);
    const size_t size_B = size_t(ldb) * size_t(B_col);
    const size_t size_C = size_t(ldc) * size_t(N);
    const size_t size_D = size_t(ldd) * size_t(N);

    // allocate memory on device
    device_vector<Ti> dA(size_A);
    device_vector<Ti> dB(size_B);
    device_vector<To> dC(size_C);
    device_vector<To> dD(size_D);
    device_vector<Tc> d_alpha_Tc(1);
    device_vector<Tc> d_beta_Tc(1);
    if(!dA || !dB || !dC || !dD || !d_alpha_Tc || !d_beta_Tc)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti> hA(size_A);
    host_vector<Ti> hB(size_B);
    host_vector<To> hC(size_C);
    host_vector<To> hC_1(size_C);
    host_vector<To> hC_2(size_C);
    host_vector<To> hC_gold(size_C);
    host_vector<To> hD_1(size_D);
    host_vector<To> hD_2(size_D);
    host_vector<To> hD_gold(size_D);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<Ti>(hA, A_row, A_col, lda);
    rocblas_init_alternating_sign<Ti>(hB, B_row, B_col, ldb);
    if(nantest)
        rocblas_init_nan<To>(hC, M, N, ldc);
    else
        rocblas_init<To>(hC, M, N, ldc);
    rocblas_init<To>(hD_1, M, N, ldd);

    if(std::is_same<To, rocblas_half>{} && std::is_same<Tc, float>{})
    {
        // half precision IEEE has max and lowest values 65504 and -65504,
        // float precision IEEE has max and lowest values 3.403e+38 and -3.403e+38
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
        const rocblas_half positive_two(2.0);
        const rocblas_half negative_two(-2.0);
        if(M >= 2 && N >= 2 && K >= 2)
        {
            hA[0]       = Ti(ieee_half_near_max);
            hA[1]       = Ti(ieee_half_near_max);
            hA[lda]     = Ti(ieee_half_near_max);
            hA[lda + 1] = Ti(ieee_half_near_max);
            hB[0]       = Ti(positive_two);
            hB[1]       = Ti(negative_two);
            hB[ldb]     = Ti(negative_two);
            hB[ldb + 1] = Ti(positive_two);
        }
    }
    else if(std::is_same<Ti, rocblas_bfloat16>{} && std::is_same<Tc, float>{})
    {
        // half precision IEEE has max and lowest values 65504 and -65504,
        // float precision IEEE has max and lowest values 3.403e+38 and -3.403e+38
        // the following will overflow to inf in half arithmetic,
        // but it will equal zero in float arithmetic   65504 * 2 - 65504 * 2
        //
        // set matrix A and matrix B upper left block to values below to cause
        // inf overflow with 16 bit arithmetic, but no overflow for 32 bit arithmetic
        //
        // 65500 65500             2   -2
        // 65500 65500            -2    2
        //
        const float ieee_half_near_max = 65504.0f - 4.0f;
        const float positive_two       = 2.0f;
        const float negative_two       = -2.0f;
        if(M >= 2 && N >= 2 && K >= 2)
        {
            hA[0]       = Ti(ieee_half_near_max);
            hA[1]       = Ti(ieee_half_near_max);
            hA[lda]     = Ti(ieee_half_near_max);
            hA[lda + 1] = Ti(ieee_half_near_max);
            hB[0]       = Ti(positive_two);
            hB[1]       = Ti(negative_two);
            hB[ldb]     = Ti(negative_two);
            hB[ldb + 1] = Ti(positive_two);
        }
    }

    hD_2    = hD_1;
    hD_gold = hD_1;
    hC_gold = hC;

    // copy data from CPU to device
    // if int8 and A not transposed and valid case, pack A
    if(std::is_same<Ti, int8_t>{} && transA == rocblas_operation_none)
    {
        host_vector<Ti> hA_packed(hA);

        rocblas_packInt8(hA_packed, M, K, lda);
        CHECK_HIP_ERROR(hipMemcpy(dA, hA_packed, sizeof(Ti) * size_A, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ti) * size_A, hipMemcpyHostToDevice));
    }

    // if int8 and B transposed and valid case, pack B
    if(std::is_same<Ti, int8_t>{} && transB != rocblas_operation_none)
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
        CHECK_HIP_ERROR(hipMemcpy(dD, hD_1, sizeof(To) * size_D, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
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

        CHECK_HIP_ERROR(hipMemcpy(hD_1, dD, sizeof(To) * size_D, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(To) * size_C, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dD, hD_2, sizeof(To) * size_D, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
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
                                            dD,
                                            arg.d_type,
                                            ldd,
                                            arg.compute_type,
                                            algo,
                                            solution_index,
                                            flags));

        CHECK_HIP_ERROR(hipMemcpy(hD_2, dD, sizeof(To) * size_D, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC, sizeof(To) * size_C, hipMemcpyDeviceToHost));

        // CPU BLAS
        // copy C matrix into D matrix
        for(int i2 = 0; i2 < N; i2++)
        {
            for(int i1 = 0; i1 < M; i1++)
            {
                hD_gold[i1 + i2 * ldd] = hC[i1 + i2 * ldc];
            }
        }
        cpu_time_used = get_time_us();

        reference_gemm<Ti, To, Tc>(
            transA, transB, M, N, K, h_alpha_Tc, hA, lda, hB, ldb, h_beta_Tc, hD_gold, ldd);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<To>(M, N, K) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            if(std::is_same<Tc, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<Tc>;
                near_check_general<To>(M, N, ldd, hD_gold, hD_1, tol);
                near_check_general<To>(M, N, ldd, hD_gold, hD_2, tol);
                unit_check_general<To>(M, N, ldc, hC_gold, hC_1);
                unit_check_general<To>(M, N, ldc, hC_gold, hC_2);
            }
            else
            {
                unit_check_general<To>(M, N, ldd, hD_gold, hD_1);
                unit_check_general<To>(M, N, ldd, hD_gold, hD_2);
                unit_check_general<To>(M, N, ldc, hC_gold, hC_1);
                unit_check_general<To>(M, N, ldc, hC_gold, hC_2);
            }
        }

        if(arg.norm_check)
        {
            auto err1 = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_1));
            auto err2 = std::abs(norm_check_general<To>('F', M, N, ldd, hD_gold, hD_2));
            auto errD = err1 > err2 ? err1 : err2;

            auto err3 = std::abs(norm_check_general<To>('F', M, N, ldc, hC_gold, hC_1));
            auto err4 = std::abs(norm_check_general<To>('F', M, N, ldc, hC_gold, hC_2));
            auto errC = err3 > err4 ? err3 : err4;

            rocblas_error = errD > errC ? errD : errC;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
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
                                                dC,
                                                arg.c_type,
                                                ldc,
                                                arg.compute_type,
                                                algo,
                                                solution_index,
                                                flags));
        }

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_ex(handle,
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
                            dC,
                            arg.c_type,
                            ldc,
                            arg.compute_type,
                            algo,
                            solution_index,
                            flags);
        }
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<Ti>(M, N, K) * number_hot_calls / gpu_time_used * 1e6;

        std::cout << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,us";

        if(arg.unit_check || arg.norm_check)
            std::cout << ",CPU-Gflops(us),norm-error";

        std::cout << std::endl;

        std::cout << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
                  << M << "," << N << "," << K << "," << arg.alpha << "," << lda << "," << ldb
                  << "," << arg.beta << "," << ldc << "," << rocblas_gflops << ","
                  << gpu_time_used / number_hot_calls;

        if(arg.unit_check || arg.norm_check)
        {
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;
        }

        std::cout << std::endl;
    }
}
