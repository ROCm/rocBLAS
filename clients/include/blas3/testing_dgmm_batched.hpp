/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T>
void testing_dgmm_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_dgmm_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_dgmm_batched<T, true> : rocblas_dgmm_batched<T, false>;

    auto rocblas_dgmm_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_dgmm_batched_64<T, true>
                                                              : rocblas_dgmm_batched_64<T, false>;

    const int64_t M = 100;
    const int64_t N = 101;

    const int64_t lda  = 100;
    const int64_t incx = 1;
    const int64_t ldc  = 100;

    const int64_t batch_count = 2;

    const rocblas_side side = rocblas_side_left;

    // no device/host loop required as no difference
    rocblas_local_handle handle{arg};

    int64_t K = rocblas_side_right == side ? N : M;

    // Allocate device memory
    device_batch_matrix<T> dA(M, N, lda, batch_count);
    device_batch_vector<T> dx(K, incx, batch_count);
    device_batch_matrix<T> dC(M, N, ldc, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_dgmm_batched_fn,
                (nullptr, side, M, N, dA, lda, dx, incx, dC, ldc, batch_count));

    DAPI_EXPECT(
        rocblas_status_invalid_value,
        rocblas_dgmm_batched_fn,
        (handle, (rocblas_side)rocblas_fill_full, M, N, dA, lda, dx, incx, dC, ldc, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dgmm_batched_fn,
                (handle, side, M, N, nullptr, lda, dx, incx, dC, ldc, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dgmm_batched_fn,
                (handle, side, M, N, dA, lda, nullptr, incx, dC, ldc, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dgmm_batched_fn,
                (handle, side, M, N, dA, lda, dx, incx, nullptr, ldc, batch_count));
}

template <typename T>
void testing_dgmm_batched(const Arguments& arg)
{
    auto rocblas_dgmm_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_dgmm_batched<T, true> : rocblas_dgmm_batched<T, false>;

    auto rocblas_dgmm_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_dgmm_batched_64<T, true>
                                                              : rocblas_dgmm_batched_64<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    int64_t M = arg.M;
    int64_t N = arg.N;
    int64_t K = rocblas_side_right == side ? size_t(N) : size_t(M);

    int64_t lda  = arg.lda;
    int64_t incx = arg.incx;
    int64_t ldc  = arg.ldc;

    int64_t batch_count = arg.batch_count;

    double gpu_time_used, cpu_time_used;

    double rocblas_error = std::numeric_limits<double>::max();

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_dgmm_batched_fn,
                    (handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc, batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(M, N, lda, batch_count);
    host_batch_vector<T> hx(K, incx, batch_count);
    host_batch_matrix<T> hC(M, N, ldc, batch_count);
    host_batch_matrix<T> hC_gold(M, N, ldc, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(M, N, lda, batch_count);
    device_batch_vector<T> dx(K, incx, batch_count);
    device_batch_matrix<T> dC(M, N, ldc, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(hA, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        handle.pre_test(arg);
        DAPI_CHECK(rocblas_dgmm_batched_fn,
                   (handle,
                    side,
                    M,
                    N,
                    dA.ptr_on_device(),
                    lda,
                    dx.ptr_on_device(),
                    incx,
                    dC.ptr_on_device(),
                    ldc,
                    batch_count));
        handle.post_test(arg);

        // fetch GPU results
        CHECK_HIP_ERROR(hC.transfer_from(dC));

        if(arg.repeatability_check)
        {
            host_batch_matrix<T> hC_copy(M, N, ldc, batch_count);

            for(int i = 0; i < arg.iters; i++)
            {
                DAPI_CHECK(rocblas_dgmm_batched_fn,
                           (handle,
                            side,
                            M,
                            N,
                            dA.ptr_on_device(),
                            lda,
                            dx.ptr_on_device(),
                            incx,
                            dC.ptr_on_device(),
                            ldc,
                            batch_count));

                CHECK_HIP_ERROR(hC_copy.transfer_from(dC));
                unit_check_general<T>(M, N, ldc, hC, hC_copy, batch_count);
            }

            return;
        }

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        for(size_t b = 0; b < batch_count; b++)
            ref_dgmm<T>(side, M, N, hA[b], lda, hx[b], incx, hC_gold[b], ldc);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC, batch_count);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_dgmm_batched_fn,
                          (handle,
                           side,
                           M,
                           N,
                           dA.ptr_on_device(),
                           lda,
                           dx.ptr_on_device(),
                           incx,
                           dC.ptr_on_device(),
                           ldc,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_side, e_M, e_N, e_lda, e_incx, e_ldc, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            dgmm_gflop_count<T>(M, N),
            ArgumentLogging::NA_value,
            cpu_time_used,
            rocblas_error);
    }
}
