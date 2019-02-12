/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "utility.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas.hpp"
#include "cblas_interface.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "near.hpp"
#include "flops.hpp"

#define DEBUG_PRINT false

/* ============================================================================================ */
template <typename Ti, typename To, typename Tc>
void testing_gemm_strided_batched_ex_bad_arg(const Arguments& arg)
{
    const rocblas_int M = 100;
    const rocblas_int N = 100;
    const rocblas_int K = 100;

    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;
    const rocblas_int ldc = 100;
    const rocblas_int ldd = 100;

    const rocblas_int stride_a = 100 * 100;
    const rocblas_int stride_b = 100 * 100;
    const rocblas_int stride_c = 100 * 100;
    const rocblas_int stride_d = 100 * 100;

    const rocblas_int batch_count = 1;

    rocblas_datatype a_type       = rocblas_datatype_f32_r;
    rocblas_datatype b_type       = rocblas_datatype_f32_r;
    rocblas_datatype c_type       = rocblas_datatype_f32_r;
    rocblas_datatype d_type       = rocblas_datatype_f32_r;
    rocblas_datatype compute_type = rocblas_datatype_f32_r;

    const float alpha_float = 1.0;
    const float beta_float  = 1.0;

    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
    int32_t solution_index = 0;
    rocblas_int flags      = 0;
    size_t workspace_size  = 0;
    void* workspace        = nullptr;

    const size_t safe_size = 100;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

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

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          &alpha_float,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          stride_a,
                                                          dB,
                                                          b_type,
                                                          ldb,
                                                          stride_b,
                                                          &beta_float,
                                                          dC,
                                                          c_type,
                                                          ldc,
                                                          stride_c,
                                                          dD,
                                                          d_type,
                                                          ldd,
                                                          stride_d,
                                                          batch_count,
                                                          compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags,
                                                          &workspace_size,
                                                          workspace),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          &alpha_float,
                                                          dA,
                                                          a_type,
                                                          lda,
                                                          stride_a,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          stride_b,
                                                          &beta_float,
                                                          dC,
                                                          c_type,
                                                          ldc,
                                                          stride_c,
                                                          dD,
                                                          d_type,
                                                          ldd,
                                                          stride_d,
                                                          batch_count,
                                                          compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags,
                                                          &workspace_size,
                                                          workspace),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          &alpha_float,
                                                          dA,
                                                          a_type,
                                                          lda,
                                                          stride_a,
                                                          dB,
                                                          b_type,
                                                          ldb,
                                                          stride_b,
                                                          &beta_float,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          stride_c,
                                                          dD,
                                                          d_type,
                                                          ldd,
                                                          stride_d,
                                                          batch_count,
                                                          compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags,
                                                          &workspace_size,
                                                          workspace),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          &alpha_float,
                                                          dA,
                                                          a_type,
                                                          lda,
                                                          stride_a,
                                                          dB,
                                                          b_type,
                                                          ldb,
                                                          stride_b,
                                                          &beta_float,
                                                          dC,
                                                          c_type,
                                                          ldc,
                                                          stride_c,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          stride_d,
                                                          batch_count,
                                                          compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags,
                                                          &workspace_size,
                                                          workspace),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          dA,
                                                          a_type,
                                                          lda,
                                                          stride_a,
                                                          dB,
                                                          b_type,
                                                          ldb,
                                                          stride_b,
                                                          &beta_float,
                                                          dC,
                                                          c_type,
                                                          ldc,
                                                          stride_c,
                                                          dD,
                                                          d_type,
                                                          ldd,
                                                          stride_d,
                                                          batch_count,
                                                          compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags,
                                                          &workspace_size,
                                                          workspace),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          &alpha_float,
                                                          dA,
                                                          a_type,
                                                          lda,
                                                          stride_a,
                                                          dB,
                                                          b_type,
                                                          ldb,
                                                          stride_b,
                                                          nullptr,
                                                          dC,
                                                          c_type,
                                                          ldc,
                                                          stride_c,
                                                          dD,
                                                          d_type,
                                                          ldd,
                                                          stride_d,
                                                          batch_count,
                                                          compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags,
                                                          &workspace_size,
                                                          workspace),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex(nullptr,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          &alpha_float,
                                                          dA,
                                                          a_type,
                                                          lda,
                                                          stride_a,
                                                          dB,
                                                          b_type,
                                                          ldb,
                                                          stride_b,
                                                          &beta_float,
                                                          dC,
                                                          c_type,
                                                          ldc,
                                                          stride_c,
                                                          dD,
                                                          d_type,
                                                          ldd,
                                                          stride_d,
                                                          batch_count,
                                                          compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags,
                                                          &workspace_size,
                                                          workspace),
                          rocblas_status_invalid_handle);
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_strided_batched_ex(const Arguments& arg)
{
    rocblas_gemm_algo algo = static_cast<rocblas_gemm_algo>(arg.algo);
    int32_t solution_index = arg.solution_index;
    uint32_t flags         = arg.flags;
    size_t workspace_size  = arg.workspace_size;
    void* workspace        = nullptr;

    To h_alpha_To, h_beta_To;
    if(std::is_same<To, rocblas_half>::value)
    {
        h_alpha_To = float_to_half(arg.alpha);
        h_beta_To  = float_to_half(arg.beta);
    }
    else if(std::is_same<To, float>::value || std::is_same<To, double>::value ||
            std::is_same<To, int32_t>::value)
    {
        h_alpha_To = static_cast<To>(arg.alpha);
        h_beta_To  = static_cast<To>(arg.beta);
    }
    else
    {
#ifdef GOOGLE_TEST
        ADD_FAILURE() << "Unimplemented types";
#else
        fputs("Error: Unimplemented types\n", stderr);
#endif
        return;
    }

    Tc h_alpha_Tc, h_beta_Tc;
    if(std::is_same<Tc, rocblas_half>::value)
    {
        h_alpha_Tc = float_to_half(arg.alpha);
        h_beta_Tc  = float_to_half(arg.beta);
    }
    else if(std::is_same<Tc, float>::value || std::is_same<Tc, double>::value ||
            std::is_same<Tc, int32_t>::value)
    {
        h_alpha_Tc = static_cast<Tc>(arg.alpha);
        h_beta_Tc  = static_cast<Tc>(arg.beta);
    }
    else
    {
#ifdef GOOGLE_TEST
        ADD_FAILURE() << "Unimplemented types";
#endif
        return;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;
    rocblas_local_handle handle;
    auto transA = char2rocblas_operation(arg.transA);
    auto transB = char2rocblas_operation(arg.transB);
    auto M = arg.M, N = arg.N, K = arg.K;
    auto lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto stride_a = arg.stride_a, stride_b = arg.stride_b;
    auto stride_c = arg.stride_c, stride_d = arg.stride_d;
    auto A_row       = transA == rocblas_operation_none ? M : K;
    auto A_col       = transA == rocblas_operation_none ? K : M;
    auto B_row       = transB == rocblas_operation_none ? K : N;
    auto B_col       = transB == rocblas_operation_none ? N : K;
    auto batch_count = arg.batch_count;

    // Early exit
    if(!M || !N || !batch_count)
        return;

    // check for invalid sizes
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M ||
       batch_count < 0 || (std::is_same<Ti, int8_t>::value &&
                           (K % 4 != 0 || (transA != rocblas_operation_none && lda % 4 != 0) ||
                            (transB == rocblas_operation_none && ldb % 4 != 0))))
    {
        static const size_t safe_size = 100;
        device_vector<Ti> dA(safe_size);
        device_vector<Ti> dB(safe_size);
        device_vector<To> dC(safe_size);
        device_vector<To> dD(safe_size);
        if(!dA || !dB || !dC || !dD)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_strided_batched_ex(handle,
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
                                                              flags,
                                                              &workspace_size,
                                                              workspace),
                              rocblas_status_invalid_size);
        return;
    }

    size_t size_one_a = transA == rocblas_operation_none
                            ? static_cast<size_t>(K) * static_cast<size_t>(lda)
                            : static_cast<size_t>(M) * static_cast<size_t>(lda);
    size_t size_one_b = transB == rocblas_operation_none
                            ? static_cast<size_t>(N) * static_cast<size_t>(ldb)
                            : static_cast<size_t>(K) * static_cast<size_t>(ldb);
    size_t size_one_c = N * ldc;
    size_t size_one_d = N * ldd;
    size_t size_a     = size_one_a;
    size_t size_b     = size_one_b;
    size_t size_c     = size_one_c;
    size_t size_d     = size_one_d;

    if(batch_count > 1)
    {
        size_a += static_cast<size_t>(stride_a) * static_cast<size_t>(batch_count - 1);
        size_b += static_cast<size_t>(stride_b) * static_cast<size_t>(batch_count - 1);
        size_c += static_cast<size_t>(stride_c) * static_cast<size_t>(batch_count - 1);
        size_d += static_cast<size_t>(stride_d) * static_cast<size_t>(batch_count - 1);
    }

    // allocate memory on device
    device_vector<Ti> dA(size_a);
    device_vector<Ti> dB(size_b);
    device_vector<To> dC(size_c);
    device_vector<To> dD(size_d);
    device_vector<Tc> d_alpha_Tc(1);
    device_vector<Tc> d_beta_Tc(1);
    if((!dA && size_a) || (!dB && size_b) || (!dC && size_c) || (!dD && size_d) || !d_alpha_Tc ||
       !d_beta_Tc)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Ti> hA(size_a);
    host_vector<Ti> hB(size_b);
    host_vector<To> hC(size_c);
    host_vector<To> hD_1(size_d);
    host_vector<To> hD_2(size_d);
    host_vector<To> hD_gold(size_d);

    // Initial Data on CPU
    rocblas_seedrand();

    rocblas_init<Ti>(hA, A_row, A_col, lda, stride_a, batch_count);
    rocblas_init_alternating_sign<Ti>(hB, B_row, B_col, ldb, stride_b, batch_count);
    rocblas_init<To>(hC, M, N, ldc, stride_c, batch_count);
    rocblas_init<To>(hD_1, M, N, ldd, stride_d, batch_count);

#if DEBUG_PRINT
    if(std::is_same<To, rocblas_half>::value)
    {
        std::cout << "----A-----------------" << std::endl;
        for(int i = 0; i < size_a; i++)
        {
            cout << half_to_float(hA[i]) << "  ";
        }
        std::cout << std::endl << "-----B-----------------" << std::endl;
        for(int i = 0; i < size_b; i++)
        {
            cout << half_to_float(hB[i]) << "  ";
        }
        std::cout << std::endl << "-----C-----------------" << std::endl;
        for(int i = 0; i < size_c; i++)
        {
            cout << half_to_float(hC[i]) << "  ";
        }
        std::cout << std::endl << "-----D-----------------" << std::endl;
        for(int i = 0; i < size_d; i++)
        {
            cout << half_to_float(hD_1[i]) << "  ";
        }
        std::cout << std::endl << "-----------------------" << std::endl;
    }
    else
    {
        std::cout << "----A-----------------" << std::endl;
        for(int i = 0; i < size_a; i++)
        {
            cout << hA[i] << "  ";
        }
        std::cout << std::endl << "-----B-----------------" << std::endl;
        for(int i = 0; i < size_b; i++)
        {
            cout << hB[i] << "  ";
        }
        std::cout << std::endl << "-----C-----------------" << std::endl;
        for(int i = 0; i < size_c; i++)
        {
            cout << hC[i] << "  ";
        }
        std::cout << std::endl << "-----D-----------------" << std::endl;
        for(int i = 0; i < size_d; i++)
        {
            cout << hD_1[i] << "  ";
        }
        std::cout << std::endl << "-----------------------" << std::endl;
    }
#endif

#if 0 // Copied from testing_gemm_ex.hpp
    if(std::is_same<To, rocblas_half>::value && std::is_same<Tc, float>::value)
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
        const rocblas_half ieee_half_near_max = float_to_half(65504.0 - 4.0);
        const rocblas_half positive_two       = float_to_half(2.0);
        const rocblas_half negative_two       = float_to_half(-2.0);
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

    hD_2    = hD_1;
    hD_gold = hD_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ti) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(Ti) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(To) * size_c, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_HIP_ERROR(hipMemcpy(dD, hD_1, sizeof(To) * size_d, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex(handle,
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
                                                            flags,
                                                            &workspace_size,
                                                            workspace));

        CHECK_HIP_ERROR(hipMemcpy(hD_1, dD, sizeof(To) * size_d, hipMemcpyDeviceToHost));

#if DEBUG_PRINT
        std::cout << std::endl << "-----hD_1---------------------------------------" << std::endl;
        if(std::is_same<To, rocblas_half>::value)
            for(int i = 0; i < size_d; i++)
                cout << half_to_float(hD_1[i]) << "  ";
        else
            for(int i = 0; i < size_d; i++)
                cout << hD_1[i] << "  ";
        std::cout << std::endl;
#endif

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dD, hD_2, sizeof(To) * size_d, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex(handle,
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
                                                            dD,
                                                            arg.d_type,
                                                            ldd,
                                                            stride_d,
                                                            batch_count,
                                                            arg.compute_type,
                                                            algo,
                                                            solution_index,
                                                            flags,
                                                            &workspace_size,
                                                            workspace));

        CHECK_HIP_ERROR(hipMemcpy(hD_2, dD, sizeof(To) * size_d, hipMemcpyDeviceToHost));

#if DEBUG_PRINT
        std::cout << std::endl << "-----hD_2---------------------------------------" << std::endl;
        if(std::is_same<To, rocblas_half>::value)
            for(int i = 0; i < size_d; i++)
                cout << half_to_float(hD_2[i]) << "  ";
        else
            for(int i = 0; i < size_d; i++)
                cout << hD_2[i] << "  ";
        std::cout << std::endl;
#endif

        // CPU BLAS
        // copy C matrix into D matrix
        if(batch_count > 0 && N > 0 && M > 0)
            for(int i3 = 0; i3 < batch_count; i3++)
                for(int i2 = 0; i2 < N; i2++)
                    for(int i1 = 0; i1 < M; i1++)
                    {
                        hD_gold[i1 + (i2 * ldd) + (i3 * stride_d)] =
                            hC[i1 + (i2 * ldc) + (i3 * stride_c)];
                    }
        cpu_time_used = get_time_us();

        for(rocblas_int i = 0; i < batch_count; i++)
        {
            cblas_gemm<Ti, To>(transA,
                               transB,
                               M,
                               N,
                               K,
                               h_alpha_To,
                               hA + stride_a * i,
                               lda,
                               hB + stride_b * i,
                               ldb,
                               h_beta_To,
                               hD_gold + stride_d * i,
                               ldd);
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<To>(M, N, K) * batch_count / cpu_time_used * 1e6;

#if DEBUG_PRINT
        std::cout << std::endl << "---gold---gold---gold---------------------" << std::endl;
        if(std::is_same<To, rocblas_half>::value)
            for(int i = 0; i < size_d; i++)
                std::cout << half_to_float(hD_gold[i]) << "  ";
        else
            for(int i = 0; i < size_d; i++)
                std::cout << hD_gold[i] << "  ";

        std::cout << std::endl << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
        for(int i3 = 0; i3 < batch_count; i3++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                for(int i1 = 0; i1 < M; i1++)
                {
                    if(hD_gold[i1 + (i2 * ldd) + (i3 * stride_d)] !=
                       hD_1[i1 + (i2 * ldd) + (i3 * stride_d)])
                    {
                        if(std::is_same<To, rocblas_half>::value)
                        {
                            std::cout
                                << "batch, i, j, hd_gold, hd_1= " << i3 << ", " << i2 << ", " << i1
                                << ", " << half_to_float(hD_gold[i1 + (i2 * ldd) + (i3 * stride_d)])
                                << ", " << half_to_float(hD_1[i1 + (i2 * ldd) + (i3 * stride_d)])
                                << ", " << std::endl;
                        }
                        else
                        {
                            std::cout << "batch, i, j, hd_gold, hd_1= " << i3 << ", " << i2 << ", "
                                      << i1 << ", " << hD_gold[i1 + (i2 * ldd) + (i3 * stride_d)]
                                      << ", " << hD_1[i1 + (i2 * ldd) + (i3 * stride_d)] << ", "
                                      << std::endl;
                        }
                    }
                }
            }
        }
#endif

        if(arg.unit_check)
        {
            if(std::is_same<Tc, rocblas_half>::value && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<Tc>;
                near_check_general<To>(M, N, batch_count, ldd, stride_d, hD_gold, hD_1, tol);
                near_check_general<To>(M, N, batch_count, ldd, stride_d, hD_gold, hD_2, tol);
            }
            else
            {
                unit_check_general<To>(M, N, batch_count, ldd, stride_d, hD_gold, hD_1);
                unit_check_general<To>(M, N, batch_count, ldd, stride_d, hD_gold, hD_2);
            }
        }

        if(arg.norm_check)
        {
            auto err1 =
                fabs(norm_check_general<To>('F', M, N, ldd, stride_d, batch_count, hD_gold, hD_1));
            auto err2 =
                fabs(norm_check_general<To>('F', M, N, ldd, stride_d, batch_count, hD_gold, hD_2));
            rocblas_error = err1 > err2 ? err1 : err2;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex(handle,
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
                                                                dC,
                                                                arg.c_type,
                                                                ldc,
                                                                stride_c,
                                                                batch_count,
                                                                arg.compute_type,
                                                                algo,
                                                                solution_index,
                                                                flags,
                                                                &workspace_size,
                                                                workspace));
        }

        int number_hot_calls = arg.iters;
        gpu_time_used        = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_strided_batched_ex(handle,
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
                                            dC,
                                            arg.c_type,
                                            ldc,
                                            stride_c,
                                            batch_count,
                                            arg.compute_type,
                                            algo,
                                            solution_index,
                                            flags,
                                            &workspace_size,
                                            workspace);
        }
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops =
            gemm_gflop_count<To>(M, N, K) * batch_count * number_hot_calls / gpu_time_used * 1e6;

        std::cout
            << "transA,transB,M,N,K,alpha,lda,stride_a,ldb,stride_b,beta,ldc,stride_c,ldd,stride_"
               "d,batch_count,rocblas-Gflops,us";

        if(arg.unit_check || arg.norm_check)
            std::cout << ",CPU-Gflops(us),norm-error";

        std::cout << std::endl;

        std::cout << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
                  << M << "," << N << "," << K << ","
                  << (std::is_same<To, rocblas_half>::value ? half_to_float(h_alpha_To)
                                                            : h_alpha_To)
                  << "," << lda << "," << stride_a << "," << ldb << "," << stride_b << ","
                  << (std::is_same<To, rocblas_half>::value ? half_to_float(h_beta_To) : h_beta_To)
                  << "," << ldc << "," << stride_c << "," << ldd << "," << stride_d << ","
                  << batch_count << "," << rocblas_gflops << ","
                  << gpu_time_used / number_hot_calls;

        if(arg.unit_check || arg.norm_check)
        {
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;
        }

        std::cout << std::endl;
    }
}
