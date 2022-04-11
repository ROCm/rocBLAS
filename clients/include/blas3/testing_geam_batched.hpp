/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
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
void testing_geam_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_geam_batched_fn
        = arg.fortran ? rocblas_geam_batched<T, true> : rocblas_geam_batched<T, false>;

    const rocblas_int M = 100;
    const rocblas_int N = 100;

    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;
    const rocblas_int ldc = 100;

    const rocblas_int batch_count = 5;

    const T h_alpha = 1.0;
    const T h_beta  = 1.0;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_local_handle handle{arg};

    rocblas_int A_row = transA == rocblas_operation_none ? M : N;
    rocblas_int A_col = transA == rocblas_operation_none ? N : M;
    rocblas_int B_row = transB == rocblas_operation_none ? M : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : M;

    // Allocate device memory
    device_batch_matrix<T> dA(A_row, A_col, lda, batch_count);
    device_batch_matrix<T> dB(B_row, B_col, ldb, batch_count);
    device_batch_matrix<T> dC(M, N, ldc, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  &h_alpha,
                                                  nullptr,
                                                  lda,
                                                  &h_beta,
                                                  dB,
                                                  ldb,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  &h_alpha,
                                                  dA,
                                                  lda,
                                                  &h_beta,
                                                  nullptr,
                                                  ldb,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
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
                                                  nullptr,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_geam_batched_fn(
            handle, transA, transB, M, N, nullptr, dA, lda, &h_beta, dB, ldb, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
                                                  transA,
                                                  transB,
                                                  M,
                                                  N,
                                                  &h_alpha,
                                                  dA,
                                                  lda,
                                                  nullptr,
                                                  dB,
                                                  ldb,
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(nullptr,
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
                                                  dC,
                                                  ldc,
                                                  batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_geam_batched(const Arguments& arg)
{
    auto rocblas_geam_batched_fn
        = arg.fortran ? rocblas_geam_batched<T, true> : rocblas_geam_batched<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;

    rocblas_int lda         = arg.lda;
    rocblas_int ldb         = arg.ldb;
    rocblas_int ldc         = arg.ldc;
    rocblas_int batch_count = arg.batch_count;

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

    size_t size_C = size_t(ldc) * size_t(N);

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_geam_batched_fn(handle,
                                                      transA,
                                                      transB,
                                                      M,
                                                      N,
                                                      nullptr,
                                                      nullptr,
                                                      lda,
                                                      nullptr,
                                                      nullptr,
                                                      ldb,
                                                      nullptr,
                                                      ldc,
                                                      batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(A_row, A_col, lda, batch_count),
        hA_copy(A_row, A_col, lda, batch_count);
    host_batch_matrix<T> hB(B_row, B_col, ldb, batch_count),
        hB_copy(B_row, B_col, ldb, batch_count);
    host_batch_matrix<T> hC_1(M, N, ldc, batch_count);
    host_batch_matrix<T> hC_2(M, N, ldc, batch_count);
    host_batch_matrix<T> hC_gold(M, N, ldc, batch_count);
    host_vector<T>       h_alpha(1);
    host_vector<T>       h_beta(1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hA_copy.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hB_copy.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_2.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(A_row, A_col, lda, batch_count);
    device_batch_matrix<T> dB(B_row, B_col, ldb, batch_count);
    device_batch_matrix<T> dC(M, N, ldc, batch_count);
    device_batch_matrix<T> dC_in_place(M, N, ldc, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(hB, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
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
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_geam_batched_fn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    &alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    &beta,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count));

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        CHECK_HIP_ERROR(dC.transfer_from(hC_2));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_ROCBLAS_ERROR(rocblas_geam_batched_fn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    d_alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    d_beta,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count));

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        for(int b = 0; b < batch_count; b++)
        {
            auto hA_copy_p = hA_copy[b];
            auto hB_copy_p = hB_copy[b];
            auto hC_gold_p = hC_gold[b];

            cblas_geam(transA,
                       transB,
                       M,
                       N,
                       (T*)h_alpha,
                       (T*)hA_copy_p,
                       lda,
                       (T*)h_beta,
                       (T*)hB_copy_p,
                       ldb,
                       (T*)hC_gold_p,
                       ldc);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // fetch GPU
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
            unit_check_general<T>(M, N, ldc, hC_gold, hC_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count);
            rocblas_error_2 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2, batch_count);
        }

        // inplace check for dC == dA
        {
            if((lda == ldc) && (transA == rocblas_operation_none))
                CHECK_HIP_ERROR(dC_in_place.transfer_from(hA));
            //
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam_batched_fn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    &alpha,
                                                    dC_in_place.ptr_on_device(),
                                                    lda,
                                                    &beta,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    dC_in_place.ptr_on_device(),
                                                    ldc,
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
                    auto hA_copy_p = hA_copy[b];
                    auto hB_copy_p = hB_copy[b];
                    auto hC_gold_p = hC_gold[b];

                    cblas_geam(transA,
                               transB,
                               M,
                               N,
                               (T*)h_alpha,
                               (T*)hA_copy_p,
                               lda,
                               (T*)h_beta,
                               (T*)hB_copy_p,
                               ldb,
                               (T*)hC_gold_p,
                               ldc);
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
                }

                if(arg.norm_check)
                {
                    rocblas_error
                        = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count);
                }
            }
        }

        // inplace check for dC == dB
        {
            if((ldb == ldc) && (transB == rocblas_operation_none))
                CHECK_HIP_ERROR(dC_in_place.transfer_from(hB));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam_batched_fn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    &alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    &beta,
                                                    dC_in_place.ptr_on_device(),
                                                    ldb,
                                                    dC_in_place.ptr_on_device(),
                                                    ldc,
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
                    auto hA_copy_p = hA_copy[b];
                    auto hB_copy_p = hB_copy[b];
                    auto hC_gold_p = hC_gold[b];

                    cblas_geam(transA,
                               transB,
                               M,
                               N,
                               (T*)h_alpha,
                               (T*)hA_copy_p,
                               lda,
                               (T*)h_beta,
                               (T*)hB_copy_p,
                               ldb,
                               (T*)hC_gold_p,
                               ldc);
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
                }

                if(arg.norm_check)
                {
                    rocblas_error
                        = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count);
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
            rocblas_geam_batched_fn(handle,
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    &alpha,
                                    dA.ptr_on_device(),
                                    lda,
                                    &beta,
                                    dB.ptr_on_device(),
                                    ldb,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_geam_batched_fn(handle,
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    &alpha,
                                    dA.ptr_on_device(),
                                    lda,
                                    &beta,
                                    dB.ptr_on_device(),
                                    ldb,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_beta,
                      e_ldb,
                      e_ldc,
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
