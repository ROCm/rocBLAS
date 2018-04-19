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
#include "norm.h"
#include "unit.h"
#include "flops.h"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template <typename T>
void testing_geam_bad_arg()
{
    const rocblas_int M = 100;
    const rocblas_int N = 100;

    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;
    const rocblas_int ldc = 100;

    const T h_alpha = 1.0;
    const T h_beta  = 1.0;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int size_A = N * lda;
    rocblas_int size_B = N * ldb;
    rocblas_int size_C = N * ldc;

    vector<T> hA(size_A);
    vector<T> hB(size_B);
    vector<T> hC(size_C);

    srand(1);
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hB, M, N, ldb);
    rocblas_init<T>(hC, M, N, ldc);

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                         rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),
                                         rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_C),
                                         rocblas_test::device_free};
    T* dA = (T*)dA_managed.get();
    T* dB = (T*)dB_managed.get();
    T* dC = (T*)dC_managed.get();
    if(!dA || !dB || !dC)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T) * size_C, hipMemcpyHostToDevice));

    {
        T* dA_null = nullptr;

        status = rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, dA_null, lda, &h_beta, dB, ldb, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: A is nullptr");
    }
    {
        T* dB_null = nullptr;

        status = rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB_null, ldb, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: B is nullptr");
    }
    {
        T* dC_null = nullptr;

        status = rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC_null, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: C is nullptr");
    }
    {
        T* h_alpha_null = nullptr;

        status = rocblas_geam<T>(
            handle, transA, transB, M, N, h_alpha_null, dA, lda, &h_beta, dB, ldb, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: h_alpha is nullptr");
    }
    {
        T* h_beta_null = nullptr;

        status = rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, dA, lda, h_beta_null, dB, ldb, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: h_beta is nullptr");
    }
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_geam<T>(
            handle_null, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc);

        verify_rocblas_status_invalid_handle(status);
    }
    return;
}

template <typename T>
rocblas_status testing_geam(Arguments argus)
{
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;

    T h_alpha = argus.alpha;
    T h_beta  = argus.beta;

    rocblas_int safe_size = 100; // arbitararily set to 100

    T* dC_in_place;

    rocblas_int size_A, size_B, size_C, A_row, A_col, B_row, B_col;
    rocblas_int inc1_A, inc2_A, inc1_B, inc2_B;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error_1 = std::numeric_limits<T>::max();
    T rocblas_error_2 = std::numeric_limits<T>::max();
    T rocblas_error   = std::numeric_limits<T>::max();

    rocblas_status status   = rocblas_status_success;
    rocblas_status status_h = rocblas_status_success;
    rocblas_status status_d = rocblas_status_success;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    if(transA == rocblas_operation_none)
    {
        A_row  = M;
        A_col  = N;
        inc1_A = 1;
        inc2_A = lda;
    }
    else
    {
        A_row  = N;
        A_col  = M;
        inc1_A = lda;
        inc2_A = 1;
    }
    if(transB == rocblas_operation_none)
    {
        B_row  = M;
        B_col  = N;
        inc1_B = 1;
        inc2_B = ldb;
    }
    else
    {
        B_row  = N;
        B_col  = M;
        inc1_B = ldb;
        inc2_B = 1;
    }

    size_A = lda * A_col;
    size_B = ldb * B_col;
    size_C = ldc * N;

    // check here to prevent undefined memory allocation error
    if(M <= 0 || N <= 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        T* dA = (T*)dA_managed.get();
        T* dB = (T*)dB_managed.get();
        T* dC = (T*)dC_managed.get();
        if(!dA || !dB || !dC)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        status = rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc);

        geam_arg_check(status, M, N, lda, ldb, ldc);

        return status;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A), hA_copy(size_A);
    vector<T> hB(size_B), hB_copy(size_B);
    vector<T> hC_1(size_C);
    vector<T> hC_2(size_C);
    vector<T> hC_gold(size_C);

    // Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, A_row, A_col, lda);
    rocblas_init<T>(hB, B_row, B_col, ldb);

    hA_copy = hA;
    hB_copy = hB;

    // allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                         rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),
                                         rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_C),
                                         rocblas_test::device_free};
    auto d_alpha_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    auto d_beta_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    T* dA      = (T*)dA_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* dC      = (T*)dC_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();
    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        // ROCBLAS
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_geam<T>(
            handle, transA, transB, M, N, d_alpha, dA, lda, d_beta, dB, ldb, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        // reference calculation for golden result
        cpu_time_used = get_time_us();

        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                hC_gold[i1 + i2 * ldc] = h_alpha * hA_copy[i1 * inc1_A + i2 * inc2_A] +
                                         h_beta * hB_copy[i1 * inc1_B + i2 * inc2_B];
            }
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = geam_gflop_count<T>(M, N) / cpu_time_used * 1e6;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_1.data());
            unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_2.data());
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched
        // in compilation time
        if(argus.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_1.data());
            rocblas_error_2 = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_2.data());
        }

        // inplace check for dC == dA
        {
            dC_in_place = dA;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            status_h = rocblas_geam<T>(handle,
                                       transA,
                                       transB,
                                       M,
                                       N,
                                       &h_alpha,
                                       dA,
                                       lda,
                                       &h_beta,
                                       dB,
                                       ldb,
                                       dC_in_place,
                                       ldc);

            if(lda != ldc || transA != rocblas_operation_none)
            {
                verify_rocblas_status_invalid_size(status_h, "rocblas_geam inplace C==A");
            }
            else
            {
                CHECK_HIP_ERROR(
                    hipMemcpy(hC_1.data(), dC_in_place, sizeof(T) * size_C, hipMemcpyDeviceToHost));
                // dA was clobbered by dC_in_place, so copy hA back to dA
                CHECK_HIP_ERROR(
                    hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

                // reference calculation
                for(int i1 = 0; i1 < M; i1++)
                {
                    for(int i2 = 0; i2 < N; i2++)
                    {
                        hC_gold[i1 + i2 * ldc] = h_alpha * hA_copy[i1 * inc1_A + i2 * inc2_A] +
                                                 h_beta * hB[i1 * inc1_B + i2 * inc2_B];
                    }
                }

                // enable unit check, notice unit check is not invasive, but norm check is,
                // unit check and norm check can not be interchanged their order
                if(argus.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_1.data());
                }

                // if enable norm check, norm check is invasive
                // any typeinfo(T) will not work here, because template deduction is matched
                // in compilation time
                if(argus.norm_check)
                {
                    rocblas_error =
                        norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_1.data());
                }
            }
        }

        // inplace check for dC == dB
        {
            dC_in_place = dB;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            status_h = rocblas_geam<T>(handle,
                                       transA,
                                       transB,
                                       M,
                                       N,
                                       &h_alpha,
                                       dA,
                                       lda,
                                       &h_beta,
                                       dB,
                                       ldb,
                                       dC_in_place,
                                       ldc);

            if(ldb != ldc || transB != rocblas_operation_none)
            {
                verify_rocblas_status_invalid_size(status_h, "rocblas_geam inplace C==A");
            }
            else
            {
                verify_rocblas_status_success(status_h, "status_h rocblas_geam inplace C==A");

                CHECK_HIP_ERROR(
                    hipMemcpy(hC_1.data(), dC_in_place, sizeof(T) * size_C, hipMemcpyDeviceToHost));

                // reference calculation
                for(int i1 = 0; i1 < M; i1++)
                {
                    for(int i2 = 0; i2 < N; i2++)
                    {
                        hC_gold[i1 + i2 * ldc] = h_alpha * hA_copy[i1 * inc1_A + i2 * inc2_A] +
                                                 h_beta * hB_copy[i1 * inc1_B + i2 * inc2_B];
                    }
                }

                // enable unit check, notice unit check is not invasive, but norm check is,
                // unit check and norm check can not be interchanged their order
                if(argus.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_1.data());
                }

                // if enable norm check, norm check is invasive
                // any typeinfo(T) will not work here, because template deduction is matched
                // in compilation time
                if(argus.norm_check)
                {
                    rocblas_error =
                        norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_1.data());
                }
            }
        }
    } // end of if unit/norm check

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 10;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            status = rocblas_geam<T>(
                handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc);
        }

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            status = rocblas_geam<T>(
                handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc);
        }
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = geam_gflop_count<T>(M, N) * number_hot_calls / gpu_time_used * 1e6;

        cout << "transA,transB,M,N,alpha,lda,beta,ldb,ldc,rocblas-Gflops,us";
        if(argus.unit_check || argus.norm_check)
        {
            cout << ",CPU-Gflops,us,norm_error_ptr_host,norm_error_ptr_dev";
        }
        cout << endl;

        cout << argus.transA_option << argus.transB_option << "," << M << "," << N << "," << h_alpha
             << "," << lda << "," << h_beta << "," << ldb << "," << ldc << "," << rocblas_gflops
             << "," << gpu_time_used/number_hot_calls << ",";

        if(argus.unit_check || argus.norm_check)
        {
            cout << cblas_gflops << "," << cpu_time_used << ",";
            cout << rocblas_error_1 << "," << rocblas_error_2;
        }
        cout << endl;
    }
    return status;
}
