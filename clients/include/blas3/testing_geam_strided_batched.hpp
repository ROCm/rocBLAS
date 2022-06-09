/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

/* ============================================================================================ */

template <typename T>
void testing_geam_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_geam_strided_batched_fn = arg.fortran ? rocblas_geam_strided_batched<T, true>
                                                       : rocblas_geam_strided_batched<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_int M = 100;
        const rocblas_int N = 99;

        const rocblas_int lda = 100;
        const rocblas_int ldb = 100;
        const rocblas_int ldc = 100;

        const rocblas_int batch_count = 2;

        device_vector<T> alpha_d(1), beta_d(1), one_d(1), zero_d(1);

        const T alpha_h(1), beta_h(2), one_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* one   = &one_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        rocblas_int A_row = transA == rocblas_operation_none ? M : N;
        rocblas_int A_col = transA == rocblas_operation_none ? N : M;
        rocblas_int B_row = transB == rocblas_operation_none ? M : N;
        rocblas_int B_col = transB == rocblas_operation_none ? N : M;

        const rocblas_stride stride_a = size_t(lda) * A_col;
        const rocblas_stride stride_b = size_t(ldb) * B_col;
        const rocblas_stride stride_c = size_t(ldc) * N;

        // Allocate device memory
        device_strided_batch_matrix<T> dA(A_row, A_col, lda, stride_a, batch_count);
        device_strided_batch_matrix<T> dB(B_row, B_col, ldb, stride_b, batch_count);
        device_strided_batch_matrix<T> dC(M, N, ldc, stride_c, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dB.memcheck());
        CHECK_DEVICE_ALLOCATION(dC.memcheck());

        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(nullptr,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              stride_a,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              stride_b,
                                                              dC,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              rocblas_status_invalid_handle);

        // invalid values
        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              (rocblas_operation)rocblas_fill_full,
                                                              transB,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              stride_a,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              stride_b,
                                                              dC,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              transA,
                                                              (rocblas_operation)rocblas_fill_full,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              stride_a,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              stride_b,
                                                              dC,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              rocblas_status_invalid_value);

        // size in regular tests

        // alpha/beta
        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              nullptr,
                                                              dA,
                                                              lda,
                                                              stride_a,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              stride_b,
                                                              dC,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              dB,
                                                              ldb,
                                                              stride_b,
                                                              dC,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              rocblas_status_invalid_pointer);

        // invalid pointers
        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              stride_a,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              stride_b,
                                                              nullptr,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              rocblas_status_invalid_pointer);

        if(pointer_mode == rocblas_pointer_mode_host)
        {

            EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                                  transA,
                                                                  transB,
                                                                  M,
                                                                  N,
                                                                  alpha,
                                                                  nullptr,
                                                                  lda,
                                                                  stride_a,
                                                                  beta,
                                                                  dB,
                                                                  ldb,
                                                                  stride_b,
                                                                  dC,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count),
                                  rocblas_status_invalid_pointer);

            EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                                  transA,
                                                                  transB,
                                                                  M,
                                                                  N,
                                                                  alpha,
                                                                  dA,
                                                                  lda,
                                                                  stride_a,
                                                                  beta,
                                                                  nullptr,
                                                                  ldb,
                                                                  stride_b,
                                                                  dC,
                                                                  ldc,
                                                                  stride_c,
                                                                  batch_count),
                                  rocblas_status_invalid_pointer);
        }

        // batch_count==0 then all may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              nullptr,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              nullptr,
                                                              ldb,
                                                              stride_b,
                                                              nullptr,
                                                              ldc,
                                                              stride_c,
                                                              0),
                              rocblas_status_success);

        // M==0 then all may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              0,
                                                              N,
                                                              nullptr,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              nullptr,
                                                              ldb,
                                                              stride_b,
                                                              nullptr,
                                                              ldc,
                                                              stride_c,
                                                              0),
                              rocblas_status_success);

        // N==0 then all may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              0,
                                                              nullptr,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              nullptr,
                                                              ldb,
                                                              stride_b,
                                                              nullptr,
                                                              ldc,
                                                              stride_c,
                                                              0),
                              rocblas_status_success);

        // alpha==0 then A may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              zero,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              stride_b,
                                                              dC,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              rocblas_status_success);

        // beta==0 then B may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_strided_batched_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              stride_a,
                                                              zero,
                                                              nullptr,
                                                              ldb,
                                                              stride_b,
                                                              dC,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              rocblas_status_success);
    }
}

template <typename T>
void testing_geam_strided_batched(const Arguments& arg)
{
    auto rocblas_geam_strided_batched_fn = arg.fortran ? rocblas_geam_strided_batched<T, true>
                                                       : rocblas_geam_strided_batched<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;

    rocblas_int    lda         = arg.lda;
    rocblas_int    ldb         = arg.ldb;
    rocblas_int    ldc         = arg.ldc;
    rocblas_stride stride_a    = arg.stride_a;
    rocblas_stride stride_b    = arg.stride_b;
    rocblas_stride stride_c    = arg.stride_c;
    rocblas_int    batch_count = arg.batch_count;

    T alpha = arg.get_alpha<T>();
    T beta  = arg.get_beta<T>();

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;

    double rocblas_error_1 = std::numeric_limits<double>::max();
    double rocblas_error_2 = std::numeric_limits<double>::max();
    double rocblas_error   = std::numeric_limits<double>::max();

    rocblas_local_handle handle{arg};

    rocblas_int A_row = transA == rocblas_operation_none ? M : N;
    rocblas_int A_col = transA == rocblas_operation_none ? N : M;
    rocblas_int B_row = transB == rocblas_operation_none ? M : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : M;

    if(stride_a < size_t(lda) * A_col)
    {
        rocblas_cout << "WARNING: stride_a < lda * A_col, \n"
                        "setting stride_a = size_t(lda) * A_col"
                     << std::endl;
        stride_a = size_t(lda) * A_col;
    }
    if(stride_b < size_t(ldb) * B_col)
    {
        rocblas_cout << "WARNING: stride_b < ldb * B_col, \n"
                        "setting stride_b = size_t(ldb) * B_col"
                     << std::endl;
        stride_b = size_t(ldb) * B_col;
    }
    if(stride_c < size_t(ldc) * N)
    {
        rocblas_cout << "WARNING: stride_c < ldc * N), setting stride_c = size_t(ldc) * N"
                     << std::endl;
        stride_c = size_t(ldc) * N;
    }

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_geam_strided_batched_fn(
                handle,
                transA,
                transB,
                M,
                N,
                nullptr,
                nullptr,
                lda,
                stride_a,
                nullptr,
                nullptr,
                ldb,
                stride_b,
                (T*)0x1, // defeat C==A or B leading dim invalid checks, C nullptr in bad_arg
                ldc,
                stride_c,
                batch_count),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(A_row, A_col, lda, stride_a, batch_count),
        hA_copy(A_row, A_col, lda, stride_a, batch_count);
    host_strided_batch_matrix<T> hB(B_row, B_col, ldb, stride_b, batch_count),
        hB_copy(B_row, B_col, ldb, stride_b, batch_count);
    host_strided_batch_matrix<T> hC_1(M, N, ldc, stride_c, batch_count);
    host_strided_batch_matrix<T> hC_2(M, N, ldc, stride_c, batch_count);
    host_strided_batch_matrix<T> hC_gold(M, N, ldc, stride_c, batch_count);
    host_vector<T>               h_alpha(1);
    host_vector<T>               h_beta(1);

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hA_copy.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hB_copy.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_2.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(A_row, A_col, lda, stride_a, batch_count);
    device_strided_batch_matrix<T> dB(B_row, B_col, ldb, stride_b, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_c, batch_count);
    device_strided_batch_matrix<T> dC_in_place(M, N, ldc, stride_c, batch_count);
    device_vector<T>               d_alpha(1);
    device_vector<T>               d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC_in_place.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC_1, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);

    hA_copy.copy_from(hA);
    hB_copy.copy_from(hB);
    hC_2.copy_from(hC_1);

    // copy data from CPU to device
    CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
    CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_geam_strided_batched_fn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            &alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            &beta,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        CHECK_HIP_ERROR(dC.transfer_from(hC_2));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_ROCBLAS_ERROR(rocblas_geam_strided_batched_fn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            d_beta,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        for(size_t b = 0; b < batch_count; b++)
        {
            cblas_geam(transA,
                       transB,
                       M,
                       N,
                       (T*)h_alpha,
                       hA[b],
                       lda,
                       (T*)h_beta,
                       hB[b],
                       ldb,
                       hC_gold[b],
                       ldc);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // fetch GPU
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
            unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC_2, batch_count);
        }

        // inplace check for dC == dA
        {
            if((lda == ldc) && (transA == rocblas_operation_none))
                CHECK_HIP_ERROR(dC_in_place.transfer_from(hA));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam_strided_batched_fn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            &alpha,
                                                            dC_in_place,
                                                            lda,
                                                            stride_a,
                                                            &beta,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            dC_in_place,
                                                            ldc,
                                                            stride_c,
                                                            batch_count);

            if(lda != ldc || transA != rocblas_operation_none)
            {
                EXPECT_ROCBLAS_STATUS(status_h, rocblas_status_invalid_size);
            }
            else
            {
                CHECK_HIP_ERROR(hC_1.transfer_from(dC_in_place));
                // dA was clobbered by dC_in_place, so copy hA back to dA
                CHECK_HIP_ERROR(dA.transfer_from(hA));

                // reference calculation
                for(int b = 0; b < batch_count; b++)
                {
                    cblas_geam(transA,
                               transB,
                               M,
                               N,
                               (T*)h_alpha,
                               hA_copy[b],
                               lda,
                               (T*)h_beta,
                               hB_copy[b],
                               ldb,
                               hC_gold[b],
                               ldc);
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
                }

                if(arg.norm_check)
                {
                    rocblas_error = norm_check_general<T>(
                        'F', M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
                }
            }
        }

        // inplace check for dC == dB
        {
            if((ldb == ldc) && (transB == rocblas_operation_none))
                CHECK_HIP_ERROR(dC_in_place.transfer_from(hB));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam_strided_batched_fn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            &alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            &beta,
                                                            dC_in_place,
                                                            ldb,
                                                            stride_b,
                                                            dC_in_place,
                                                            ldc,
                                                            stride_c,
                                                            batch_count);

            if(ldb != ldc || transB != rocblas_operation_none)
            {
                EXPECT_ROCBLAS_STATUS(status_h, rocblas_status_invalid_size);
            }
            else
            {
                CHECK_ROCBLAS_ERROR(status_h);

                CHECK_HIP_ERROR(hC_1.transfer_from(dC_in_place));
                // reference calculation
                for(int b = 0; b < batch_count; b++)
                {
                    cblas_geam(transA,
                               transB,
                               M,
                               N,
                               (T*)h_alpha,
                               hA_copy[b],
                               lda,
                               (T*)h_beta,
                               hB_copy[b],
                               ldb,
                               hC_gold[b],
                               ldc);
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
                }

                if(arg.norm_check)
                {
                    rocblas_error = norm_check_general<T>(
                        'F', M, N, ldc, stride_c, hC_gold, hC_1, batch_count);
                }
            }
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_geam_strided_batched_fn(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            &alpha,
                                            dA,
                                            lda,
                                            stride_a,
                                            &beta,
                                            dB,
                                            ldb,
                                            stride_b,
                                            dC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_geam_strided_batched_fn(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            &alpha,
                                            dA,
                                            lda,
                                            stride_a,
                                            &beta,
                                            dB,
                                            ldb,
                                            stride_b,
                                            dC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_beta,
                      e_ldb,
                      e_stride_b,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         geam_gflop_count<T>(M, N),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
