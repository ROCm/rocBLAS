/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

#include "rocblas.hpp"
#include "arg_check.h"
#include "rocblas_test_unique_ptr.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"
#include <typeinfo>

#define DEBUG_PRINT false

using namespace std;

/* ============================================================================================ */
void testing_gemm_strided_batched_ex_bad_arg()
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
    rocblas_int solution_index;
    rocblas_int flags;
    size_t* workspace_size = 0;
    void* workspace;

    const size_t safe_size = 100;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(float) * safe_size),
                                         rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(float) * safe_size),
                                         rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(float) * safe_size),
                                         rocblas_test::device_free};
    auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(float) * safe_size),
                                         rocblas_test::device_free};
    float* dA = (float*)dA_managed.get();
    float* dB = (float*)dB_managed.get();
    float* dC = (float*)dC_managed.get();
    float* dD = (float*)dC_managed.get();
    if(!dA || !dB || !dC || !dD)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    {
        float* dA_null = nullptr;

        status = rocblas_gemm_strided_batched_ex(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 &alpha_float,
                                                 dA_null,
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
                                                 workspace_size,
                                                 workspace);

        verify_rocblas_status_invalid_pointer(status, "ERROR: A is nullptr");
    }
    {
        float* dB_null = nullptr;

        status = rocblas_gemm_strided_batched_ex(handle,
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
                                                 dB_null,
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
                                                 workspace_size,
                                                 workspace);

        verify_rocblas_status_invalid_pointer(status, "ERROR: B is nullptr");
    }
    {
        float* dC_null = nullptr;

        status = rocblas_gemm_strided_batched_ex(handle,
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
                                                 dC_null,
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
                                                 workspace_size,
                                                 workspace);

        verify_rocblas_status_invalid_pointer(status, "ERROR: C is nullptr");
    }
    {
        float* dD_null = nullptr;

        status = rocblas_gemm_strided_batched_ex(handle,
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
                                                 dD_null,
                                                 d_type,
                                                 ldd,
                                                 stride_d,
                                                 batch_count,
                                                 compute_type,
                                                 algo,
                                                 solution_index,
                                                 flags,
                                                 workspace_size,
                                                 workspace);

        verify_rocblas_status_invalid_pointer(status, "ERROR: D is nullptr");
    }
    {
        float* alpha_null = nullptr;

        status = rocblas_gemm_strided_batched_ex(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha_null,
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
                                                 workspace_size,
                                                 workspace);

        verify_rocblas_status_invalid_pointer(status, "ERROR: alpha is nullptr");
    }
    {
        float* beta_null = nullptr;

        status = rocblas_gemm_strided_batched_ex(handle,
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
                                                 beta_null,
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
                                                 workspace_size,
                                                 workspace);

        verify_rocblas_status_invalid_pointer(status, "ERROR: beta is nullptr");
    }
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_gemm_strided_batched_ex(handle_null,
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
                                                 workspace_size,
                                                 workspace);

        verify_rocblas_status_invalid_handle(status);
    }

    return;
}

template <typename Td, typename Tc>
rocblas_status testing_gemm_strided_batched_ex_template(rocblas_operation transA,
                                                        rocblas_operation transB,
                                                        rocblas_int M,
                                                        rocblas_int N,
                                                        rocblas_int K,
                                                        float alpha_float,
                                                        rocblas_int lda,
                                                        rocblas_int ldb,
                                                        float beta_float,
                                                        rocblas_int ldc,
                                                        rocblas_int ldd,
                                                        rocblas_int stride_a,
                                                        rocblas_int stride_b,
                                                        rocblas_int stride_c,
                                                        rocblas_int stride_d,
                                                        rocblas_int batch_count,
                                                        rocblas_int norm_check,
                                                        rocblas_int unit_check,
                                                        rocblas_int timing,
                                                        int number_hot_calls,
                                                        rocblas_datatype a_type,
                                                        rocblas_datatype b_type,
                                                        rocblas_datatype c_type,
                                                        rocblas_datatype d_type,
                                                        rocblas_datatype compute_type)
{
    rocblas_gemm_algo algo  = rocblas_gemm_algo_standard;
    uint32_t solution_index = 0;
    uint32_t flags          = 0;
    size_t* workspace_size  = 0;
    void* workspace;

    Td h_alpha_Td;
    Td h_beta_Td;

    if(is_same<Td, rocblas_half>::value)
    {
        h_alpha_Td = float_to_half(alpha_float);
        h_beta_Td  = float_to_half(beta_float);
    }
    else if(is_same<Td, float>::value || is_same<Td, double>::value)
    {
        h_alpha_Td = static_cast<Td>(alpha_float);
        h_beta_Td  = static_cast<Td>(beta_float);
    }
    else
    {
        return rocblas_status_not_implemented;
    }

    Tc h_alpha_Tc;
    Tc h_beta_Tc;

    if(is_same<Tc, rocblas_half>::value)
    {
        h_alpha_Tc = float_to_half(alpha_float);
        h_beta_Tc  = float_to_half(beta_float);
    }
    else if(is_same<Tc, float>::value || is_same<Tc, double>::value)
    {
        h_alpha_Tc = static_cast<Tc>(alpha_float);
        h_beta_Tc  = static_cast<Tc>(beta_float);
    }
    else
    {
        return rocblas_status_not_implemented;
    }

    const size_t safe_size = 100;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    double rocblas_error = 0.0;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M ||
       batch_count < 0)
    {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * safe_size),
                                             rocblas_test::device_free};
        auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * safe_size),
                                             rocblas_test::device_free};
        auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * safe_size),
                                             rocblas_test::device_free};
        auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * safe_size),
                                             rocblas_test::device_free};
        Td* dA = (Td*)dA_managed.get();
        Td* dB = (Td*)dB_managed.get();
        Td* dC = (Td*)dC_managed.get();
        Td* dD = (Td*)dD_managed.get();
        if(!dA || !dB || !dC || !dD)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_gemm_strided_batched_ex(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 &h_alpha_Tc,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 stride_a,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 stride_b,
                                                 &h_beta_Tc,
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
                                                 workspace_size,
                                                 workspace);

        gemm_strided_batched_arg_check(
            status, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count);

        return status;
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

    if(batch_count > 0)
    {
        size_a += static_cast<size_t>(stride_a) * static_cast<size_t>(batch_count - 1);
        size_b += static_cast<size_t>(stride_b) * static_cast<size_t>(batch_count - 1);
        size_c += static_cast<size_t>(stride_c) * static_cast<size_t>(batch_count - 1);
        size_d += static_cast<size_t>(stride_d) * static_cast<size_t>(batch_count - 1);
    }

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * size_a),
                                         rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * size_b),
                                         rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * size_c),
                                         rocblas_test::device_free};
    auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * size_d),
                                         rocblas_test::device_free};
    auto d_alpha_Tc_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Tc)), rocblas_test::device_free};
    auto d_beta_Tc_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Tc)), rocblas_test::device_free};
    Td* dA         = (Td*)dA_managed.get();
    Td* dB         = (Td*)dB_managed.get();
    Td* dC         = (Td*)dC_managed.get();
    Td* dD         = (Td*)dD_managed.get();
    Tc* d_alpha_Tc = (Tc*)d_alpha_Tc_managed.get();
    Tc* d_beta_Tc  = (Tc*)d_beta_Tc_managed.get();
    if((!dA && (size_a != 0)) || (!dB && (size_b != 0)) || (!dC && (size_c != 0)) ||
       (!dD && (size_d != 0)) || !d_alpha_Tc || !d_beta_Tc)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<Td> hA(size_a);
    vector<Td> hB(size_b);
    vector<Td> hC(size_c);
    vector<Td> hD_1(size_d);
    vector<Td> hD_2(size_d);
    vector<Td> hD_gold(size_d);

    // Initial Data on CPU
    srand(1);

    rocblas_init<Td>(hA, A_row, A_col, lda, stride_a, batch_count);
    rocblas_init_alternating_sign<Td>(hB, B_row, B_col, ldb, stride_b, batch_count);
    rocblas_init<Td>(hC, M, N, ldc, stride_c, batch_count);
    rocblas_init<Td>(hD_1, M, N, ldd, stride_d, batch_count);

#if DEBUG_PRINT
    if(is_same<Td, rocblas_half>::value)
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
    hD_2    = hD_1;
    hD_gold = hD_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(Td) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(Td) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(Td) * size_c, hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_HIP_ERROR(hipMemcpy(dD, hD_1.data(), sizeof(Td) * size_d, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            &h_alpha_Tc,
                                                            dA,
                                                            a_type,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            b_type,
                                                            ldb,
                                                            stride_b,
                                                            &h_beta_Tc,
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
                                                            workspace_size,
                                                            workspace));

        CHECK_HIP_ERROR(hipMemcpy(hD_1.data(), dD, sizeof(Td) * size_d, hipMemcpyDeviceToHost));
#if DEBUG_PRINT
        std::cout << std::endl << "-----hD_1---------------------------------------" << std::endl;
        if(is_same<Td, rocblas_half>::value)
        {
            for(int i = 0; i < size_d; i++)
            {
                cout << half_to_float(hD_1[i]) << "  ";
            }
        }
        else
        {
            for(int i = 0; i < size_d; i++)
            {
                cout << hD_1[i] << "  ";
            }
        }
        std::cout << std::endl;
#endif

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(hipMemcpy(dD, hD_2.data(), sizeof(Td) * size_d, hipMemcpyHostToDevice));

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
                                                            a_type,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            b_type,
                                                            ldb,
                                                            stride_b,
                                                            d_beta_Tc,
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
                                                            workspace_size,
                                                            workspace));

        CHECK_HIP_ERROR(hipMemcpy(hD_2.data(), dD, sizeof(Td) * size_d, hipMemcpyDeviceToHost));
#if DEBUG_PRINT
        std::cout << std::endl << "-----hD_2---------------------------------------" << std::endl;
        if(is_same<Td, rocblas_half>::value)
        {
            for(int i = 0; i < size_d; i++)
            {
                cout << half_to_float(hD_2[i]) << "  ";
            }
        }
        else
        {
            for(int i = 0; i < size_d; i++)
            {
                cout << hD_2[i] << "  ";
            }
        }
        std::cout << std::endl;
#endif

        // CPU BLAS
        // copy C matrix into D matrix
        if(batch_count > 0 && N > 0 && M > 0)
        {
            for(int i3 = 0; i3 < batch_count; i3++)
            {
                for(int i2 = 0; i2 < N; i2++)
                {
                    for(int i1 = 0; i1 < M; i1++)
                    {
                        hD_gold[i1 + (i2 * ldd) + (i3 * stride_d)] =
                            hC[i1 + (i2 * ldc) + (i3 * stride_c)];
                    }
                }
            }
        }
        cpu_time_used = get_time_us();

        for(rocblas_int i = 0; i < batch_count; i++)
        {
            cblas_gemm<Td>(transA,
                           transB,
                           M,
                           N,
                           K,
                           h_alpha_Td,
                           hA.data() + stride_a * i,
                           lda,
                           hB.data() + stride_b * i,
                           ldb,
                           h_beta_Td,
                           hD_gold.data() + stride_d * i,
                           ldd);
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemm_gflop_count<Td>(M, N, K) / cpu_time_used * 1e6;
#if DEBUG_PRINT
        std::cout << std::endl << "---gold---gold---gold---------------------" << std::endl;
        if(is_same<Td, rocblas_half>::value)
        {
            for(int i = 0; i < size_d; i++)
            {
                std::cout << half_to_float(hD_gold[i]) << "  ";
            }
        }
        else
        {
            for(int i = 0; i < size_d; i++)
            {
                std::cout << hD_gold[i] << "  ";
            }
        }
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
                        if(is_same<Td, rocblas_half>::value)
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

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order

        if(unit_check)
        {
            unit_check_general<Td>(M, N, batch_count, ldd, stride_d, hD_gold.data(), hD_1.data());
            unit_check_general<Td>(M, N, batch_count, ldd, stride_d, hD_gold.data(), hD_2.data());
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(Td) will not work here, because template deduction is matched
        // in compilation time
        if(norm_check)
        {
            rocblas_error = norm_check_general<Td>(
                'F', M, N, ldd, stride_d, batch_count, hD_gold.data(), hD_1.data());
            rocblas_error = norm_check_general<Td>(
                'F', M, N, ldd, stride_d, batch_count, hD_gold.data(), hD_2.data());
        }
    }

    std::cout << "before return" << std::endl;
    //  return rocblas_status_success;

    if(timing)
    {
        int number_cold_calls = 2;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_gemm_strided_batched_ex(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            &h_alpha_Tc,
                                            dA,
                                            a_type,
                                            lda,
                                            stride_a,
                                            dB,
                                            b_type,
                                            ldb,
                                            stride_b,
                                            &h_beta_Tc,
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
                                            workspace_size,
                                            workspace);
        }

        gpu_time_used = get_time_us(); // in microseconds
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
                                            a_type,
                                            lda,
                                            stride_a,
                                            dB,
                                            b_type,
                                            ldb,
                                            stride_b,
                                            &h_beta_Tc,
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
                                            workspace_size,
                                            workspace);
        }
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<Td>(M, N, K) * number_hot_calls / gpu_time_used * 1e6;

        cout << "transA,transB,M,N,K,alpha,lda,stride_a,ldb,stride_b,beta,ldc,stride_c,ldd,stride_"
                "d,batch_count,rocblas-Gflops,us";

        if(unit_check || norm_check)
            cout << ",CPU-Gflops(us),norm-error";

        cout << endl;

        cout << transA << "," << transB << "," << M << "," << N << "," << K << "," << h_alpha_Td
             << "," << lda << "," << stride_a << "," << ldb << "," << stride_b << "," << h_beta_Td
             << "," << ldc << "," << stride_c << "," << ldd << "," << stride_d << "," << batch_count
             << "," << rocblas_gflops << "," << gpu_time_used / number_hot_calls;

        if(unit_check || norm_check)
        {
            cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;
        }

        cout << endl;
    }
    return status;
}

rocblas_status testing_gemm_strided_batched_ex(Arguments argus)
{
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;
    rocblas_int ldd = argus.ldd;

    rocblas_int stride_a = argus.stride_a;
    rocblas_int stride_b = argus.stride_b;
    rocblas_int stride_c = argus.stride_c;
    rocblas_int stride_d = argus.stride_d;

    rocblas_int batch_count = argus.batch_count;

    rocblas_datatype a_type       = argus.a_type;
    rocblas_datatype b_type       = argus.b_type;
    rocblas_datatype c_type       = argus.c_type;
    rocblas_datatype d_type       = argus.d_type;
    rocblas_datatype compute_type = argus.compute_type;

    float alpha = argus.alpha;
    float beta  = argus.beta;

    rocblas_int norm_check = argus.norm_check;
    rocblas_int unit_check = argus.unit_check;
    rocblas_int timing     = argus.timing;
    int number_hot_calls   = argus.iters;

    if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
       c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r &&
       compute_type == rocblas_datatype_f16_r)
    {
        return testing_gemm_strided_batched_ex_template<rocblas_half, rocblas_half>(
            transA,
            transB,
            M,
            N,
            K,
            alpha,
            lda,
            ldb,
            beta,
            ldc,
            ldd,
            stride_a,
            stride_b,
            stride_c,
            stride_d,
            batch_count,
            norm_check,
            unit_check,
            timing,
            number_hot_calls,
            a_type,
            b_type,
            c_type,
            d_type,
            compute_type);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
            c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r &&
            compute_type == rocblas_datatype_f32_r)
    {
        return testing_gemm_strided_batched_ex_template<rocblas_half, float>(transA,
                                                                             transB,
                                                                             M,
                                                                             N,
                                                                             K,
                                                                             alpha,
                                                                             lda,
                                                                             ldb,
                                                                             beta,
                                                                             ldc,
                                                                             ldd,
                                                                             stride_a,
                                                                             stride_b,
                                                                             stride_c,
                                                                             stride_d,
                                                                             batch_count,
                                                                             norm_check,
                                                                             unit_check,
                                                                             timing,
                                                                             number_hot_calls,
                                                                             a_type,
                                                                             b_type,
                                                                             c_type,
                                                                             d_type,
                                                                             compute_type);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r &&
            c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r &&
            compute_type == rocblas_datatype_f32_r)
    {
        return testing_gemm_strided_batched_ex_template<float, float>(transA,
                                                                      transB,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      alpha,
                                                                      lda,
                                                                      ldb,
                                                                      beta,
                                                                      ldc,
                                                                      ldd,
                                                                      stride_a,
                                                                      stride_b,
                                                                      stride_c,
                                                                      stride_d,
                                                                      batch_count,
                                                                      norm_check,
                                                                      unit_check,
                                                                      timing,
                                                                      number_hot_calls,
                                                                      a_type,
                                                                      b_type,
                                                                      c_type,
                                                                      d_type,
                                                                      compute_type);
    }
    else if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r &&
            c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r &&
            compute_type == rocblas_datatype_f64_r)
    {
        return testing_gemm_strided_batched_ex_template<double, double>(transA,
                                                                        transB,
                                                                        M,
                                                                        N,
                                                                        K,
                                                                        alpha,
                                                                        lda,
                                                                        ldb,
                                                                        beta,
                                                                        ldc,
                                                                        ldd,
                                                                        stride_a,
                                                                        stride_b,
                                                                        stride_c,
                                                                        stride_d,
                                                                        batch_count,
                                                                        norm_check,
                                                                        unit_check,
                                                                        timing,
                                                                        number_hot_calls,
                                                                        a_type,
                                                                        b_type,
                                                                        c_type,
                                                                        d_type,
                                                                        compute_type);
    }
    else
    {
        return rocblas_status_not_implemented;
    }
}
