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
void testing_dgmm_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_dgmm_strided_batched_fn = arg.fortran ? rocblas_dgmm_strided_batched<T, true>
                                                       : rocblas_dgmm_strided_batched<T, false>;

    const rocblas_int M = 100;
    const rocblas_int N = 101;

    const rocblas_int lda  = 100;
    const rocblas_int incx = 1;
    const rocblas_int ldc  = 100;

    const rocblas_int  batch_count = 2;
    const rocblas_side side        = rocblas_side_left;

    // no device/host loop required as no difference
    rocblas_local_handle handle{arg};

    rocblas_int K = rocblas_side_right == side ? N : M;

    const rocblas_stride stride_a = N * size_t(lda);
    const rocblas_stride stride_x = K * size_t(incx);
    const rocblas_stride stride_c = N * size_t(ldc);

    // Allocate device memory
    device_strided_batch_matrix<T> dA(M, N, lda, stride_a, batch_count);
    device_strided_batch_vector<T> dx(K, incx ? incx : 1, stride_x, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_a, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(nullptr,
                                                          side,
                                                          M,
                                                          N,
                                                          dA,
                                                          lda,
                                                          stride_a,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          dC,
                                                          ldc,
                                                          stride_c,
                                                          batch_count),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(handle,
                                                          (rocblas_side)rocblas_fill_full,
                                                          M,
                                                          N,
                                                          dA,
                                                          lda,
                                                          stride_a,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          dC,
                                                          ldc,
                                                          stride_c,
                                                          batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(handle,
                                                          side,
                                                          M,
                                                          N,
                                                          nullptr,
                                                          lda,
                                                          stride_a,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          dC,
                                                          ldc,
                                                          stride_c,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(handle,
                                                          side,
                                                          M,
                                                          N,
                                                          dA,
                                                          lda,
                                                          stride_a,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          dC,
                                                          ldc,
                                                          stride_c,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(handle,
                                                          side,
                                                          M,
                                                          N,
                                                          dA,
                                                          lda,
                                                          stride_a,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          nullptr,
                                                          ldc,
                                                          stride_c,
                                                          batch_count),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_dgmm_strided_batched(const Arguments& arg)
{
    auto rocblas_dgmm_strided_batched_fn = arg.fortran ? rocblas_dgmm_strided_batched<T, true>
                                                       : rocblas_dgmm_strided_batched<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = rocblas_side_right == side ? size_t(N) : size_t(M);

    rocblas_int lda  = arg.lda;
    rocblas_int incx = arg.incx;
    rocblas_int ldc  = arg.ldc;

    rocblas_stride stride_a = arg.stride_a;
    rocblas_stride stride_x = arg.stride_x;
    if(!stride_x)
        stride_x = 1; // incx = 0 case
    rocblas_stride stride_c    = arg.stride_c;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_int abs_incx = incx > 0 ? incx : -incx;

    double gpu_time_used, cpu_time_used;

    double rocblas_error = std::numeric_limits<double>::max();

    if((stride_a > 0) && (stride_a < size_t(lda) * N))
    {
        rocblas_cout << "WARNING: stride_a < lda * N, setting stride_a = lda * N " << std::endl;
        stride_a = N * size_t(lda);
    }
    if((stride_c > 0) && (stride_c < size_t(ldc) * N))
    {
        rocblas_cout << "WARNING: stride_c < ldc * N, setting stride_c = lda * N" << std::endl;
        stride_c = N * size_t(ldc);
    }
    if((stride_x > 0) && (stride_x < size_t(abs_incx) * K))
    {
        rocblas_cout << "WARNING: stride_x < incx * (rocblas_side_right == side ? N : M)),\n"
                        "setting stride_x = incx * (rocblas_side_right == side ? N : M))"
                     << std::endl;
        stride_x = K * size_t(abs_incx);
    }

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0;
    if(invalid_size || M == 0 || N == 0 || batch_count == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_dgmm_strided_batched_fn(handle,
                                                              side,
                                                              M,
                                                              N,
                                                              nullptr,
                                                              lda,
                                                              stride_a,
                                                              nullptr,
                                                              incx,
                                                              stride_x,
                                                              nullptr,
                                                              ldc,
                                                              stride_c,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(M, N, lda, stride_a, batch_count);
    host_strided_batch_vector<T> hx(K, incx ? incx : 1, stride_x, batch_count);
    host_strided_batch_matrix<T> hC_1(M, N, ldc, stride_c, batch_count);
    host_strided_batch_matrix<T> hC_2(M, N, ldc, stride_c, batch_count);
    host_strided_batch_matrix<T> hC_gold(M, N, ldc, stride_c, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_2.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dA(M, N, lda, stride_a, batch_count);
    device_strided_batch_vector<T> dx(K, incx ? incx : 1, stride_x, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_c, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(hA, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);
    rocblas_init_matrix(hC_1, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dC.transfer_from(hC_1));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_dgmm_strided_batched_fn(handle,
                                                            side,
                                                            M,
                                                            N,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));
        handle.post_test(arg);

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        for(int b = 0; b < batch_count; b++)
            cblas_dgmm<T>(side, M, N, hA[b], lda, hx[b], incx, hC_gold[b], ldc);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // fetch GPU result
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC_2, batch_count);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_dgmm_strided_batched_fn(handle,
                                            side,
                                            M,
                                            N,
                                            dA,
                                            lda,
                                            stride_a,
                                            dx,
                                            incx,
                                            stride_x,
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
            rocblas_dgmm_strided_batched_fn(handle,
                                            side,
                                            M,
                                            N,
                                            dA,
                                            lda,
                                            stride_a,
                                            dx,
                                            incx,
                                            stride_x,
                                            dC,
                                            ldc,
                                            stride_c,
                                            batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side,
                      e_M,
                      e_N,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         dgmm_gflop_count<T>(M, N),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error);
    }
}
