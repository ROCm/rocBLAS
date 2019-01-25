/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "rocblas_datatype2string.hpp"
#include "utility.hpp"
#include "rocblas.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "flops.hpp"

/* ============================================================================================ */

template <typename T>
void testing_geam_bad_arg(const Arguments& arg)
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

    rocblas_local_handle handle;

    size_t size_A = N * static_cast<size_t>(lda);
    size_t size_B = N * static_cast<size_t>(ldb);
    size_t size_C = N * static_cast<size_t>(ldc);

    host_vector<T> hA(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hC(size_C);

    rocblas_seedrand();
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hB, M, N, ldb);
    rocblas_init<T>(hC, M, N, ldc);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> dC(size_C);
    if(!dA || !dB || !dC)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(T) * size_C, hipMemcpyHostToDevice));

    EXPECT_ROCBLAS_STATUS(
        rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, nullptr, lda, &h_beta, dB, ldb, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, nullptr, ldb, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, nullptr, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_geam<T>(handle, transA, transB, M, N, nullptr, dA, lda, &h_beta, dB, ldb, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_geam<T>(handle, transA, transB, M, N, &h_alpha, dA, lda, nullptr, dB, ldb, dC, ldc),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_geam<T>(
            nullptr, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_geam(const Arguments& arg)
{
    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T h_alpha = arg.alpha;
    T h_beta  = arg.beta;

    T* dC_in_place;

    rocblas_int A_row, A_col, B_row, B_col;
    rocblas_int inc1_A, inc2_A, inc1_B, inc2_B;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error_1 = std::numeric_limits<T>::max();
    T rocblas_error_2 = std::numeric_limits<T>::max();
    T rocblas_error   = std::numeric_limits<T>::max();

    rocblas_local_handle handle;

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

    size_t size_A = lda * static_cast<size_t>(A_col);
    size_t size_B = ldb * static_cast<size_t>(B_col);
    size_t size_C = ldc * static_cast<size_t>(N);

    // check here to prevent undefined memory allocation error
    if(M <= 0 || N <= 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        static const size_t safe_size = 100; // arbitararily set to 100
        device_vector<T> dA(safe_size);
        device_vector<T> dB(safe_size);
        device_vector<T> dC(safe_size);
        if(!dA || !dB || !dC)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        if(M == 0 && N == 0)
        {
            CHECK_ROCBLAS_ERROR(rocblas_geam<T>(
                handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc));
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(
                rocblas_geam<T>(
                    handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc),
                rocblas_status_invalid_size);
        }

        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A), hA_copy(size_A);
    host_vector<T> hB(size_B), hB_copy(size_B);
    host_vector<T> hC_1(size_C);
    host_vector<T> hC_2(size_C);
    host_vector<T> hC_gold(size_C);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA, A_row, A_col, lda);
    rocblas_init<T>(hB, B_row, B_col, ldb);

    hA_copy = hA;
    hB_copy = hB;

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> dC(size_C);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_geam<T>(
            handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_geam<T>(
            handle, transA, transB, M, N, d_alpha, dA, lda, d_beta, dB, ldb, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

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

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
            unit_check_general<T>(M, N, ldc, hC_gold, hC_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1);
            rocblas_error_2 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2);
        }

        // inplace check for dC == dA
        {
            dC_in_place = dA;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam<T>(handle,
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
                EXPECT_ROCBLAS_STATUS(status_h, rocblas_status_invalid_size);
            }
            else
            {
                CHECK_HIP_ERROR(
                    hipMemcpy(hC_1, dC_in_place, sizeof(T) * size_C, hipMemcpyDeviceToHost));
                // dA was clobbered by dC_in_place, so copy hA back to dA
                CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));

                // reference calculation
                for(int i1 = 0; i1 < M; i1++)
                {
                    for(int i2 = 0; i2 < N; i2++)
                    {
                        hC_gold[i1 + i2 * ldc] = h_alpha * hA_copy[i1 * inc1_A + i2 * inc2_A] +
                                                 h_beta * hB[i1 * inc1_B + i2 * inc2_B];
                    }
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
                }

                if(arg.norm_check)
                {
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1);
                }
            }
        }

        // inplace check for dC == dB
        {
            dC_in_place = dB;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam<T>(handle,
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
                EXPECT_ROCBLAS_STATUS(status_h, rocblas_status_invalid_size);
            }
            else
            {
                CHECK_ROCBLAS_ERROR(status_h);

                CHECK_HIP_ERROR(
                    hipMemcpy(hC_1, dC_in_place, sizeof(T) * size_C, hipMemcpyDeviceToHost));

                // reference calculation
                for(int i1 = 0; i1 < M; i1++)
                {
                    for(int i2 = 0; i2 < N; i2++)
                    {
                        hC_gold[i1 + i2 * ldc] = h_alpha * hA_copy[i1 * inc1_A + i2 * inc2_A] +
                                                 h_beta * hB_copy[i1 * inc1_B + i2 * inc2_B];
                    }
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
                }

                if(arg.norm_check)
                {
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1);
                }
            }
        }
    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 10;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_geam<T>(
                handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc);
        }

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_geam<T>(
                handle, transA, transB, M, N, &h_alpha, dA, lda, &h_beta, dB, ldb, dC, ldc);
        }
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = geam_gflop_count<T>(M, N) * number_hot_calls / gpu_time_used * 1e6;

        std::cout << "transA,transB,M,N,alpha,lda,beta,ldb,ldc,rocblas-Gflops,us";
        if(arg.unit_check || arg.norm_check)
        {
            std::cout << ",CPU-Gflops,us,norm_error_ptr_host,norm_error_ptr_dev";
        }
        std::cout << std::endl;

        std::cout << arg.transA << arg.transB << "," << M << "," << N << "," << h_alpha << ","
                  << lda << "," << h_beta << "," << ldb << "," << ldc << "," << rocblas_gflops
                  << "," << gpu_time_used / number_hot_calls << ",";

        if(arg.unit_check || arg.norm_check)
        {
            std::cout << cblas_gflops << "," << cpu_time_used << ",";
            std::cout << rocblas_error_1 << "," << rocblas_error_2;
        }
        std::cout << std::endl;
    }
}
