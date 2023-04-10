/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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
void testing_dgmm_bad_arg(const Arguments& arg)
{
    auto rocblas_dgmm_fn = arg.fortran ? rocblas_dgmm<T, true> : rocblas_dgmm<T, false>;

    const rocblas_int M = 100;
    const rocblas_int N = 101;

    const rocblas_int lda  = 100;
    const rocblas_int incx = 1;
    const rocblas_int ldc  = 100;

    rocblas_side side = rocblas_side_left;

    // no device/host loop required as no difference
    rocblas_local_handle handle{arg};

    rocblas_int K = rocblas_side_right == side ? N : M;

    // Allocate device memory
    device_matrix<T> dA(M, N, lda);
    device_vector<T> dx(K, incx);
    device_matrix<T> dC(M, N, ldc);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_fn(nullptr, side, M, N, dA, lda, dx, incx, dC, ldc),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(
        rocblas_dgmm_fn(handle, (rocblas_side)rocblas_fill_full, M, N, dA, lda, dx, incx, dC, ldc),
        rocblas_status_invalid_value);

    // sizes and quick returns done in normal test harness

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_fn(handle, side, M, N, nullptr, lda, dx, incx, dC, ldc),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_fn(handle, side, M, N, dA, lda, nullptr, incx, dC, ldc),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_fn(handle, side, M, N, dA, lda, dx, incx, nullptr, ldc),
                          rocblas_status_invalid_pointer);
}

template <typename T>
void testing_dgmm(const Arguments& arg)
{
    auto rocblas_dgmm_fn = arg.fortran ? rocblas_dgmm<T, true> : rocblas_dgmm<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = rocblas_side_right == side ? size_t(N) : size_t(M);

    rocblas_int lda  = arg.lda;
    rocblas_int incx = arg.incx;
    rocblas_int ldc  = arg.ldc;

    double gpu_time_used, cpu_time_used;

    double rocblas_error = std::numeric_limits<double>::max();

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || ldc < M;
    if(invalid_size || !M || !N)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_dgmm_fn(handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(M, N, lda);
    host_vector<T> hx(K, incx ? incx : 1);
    host_matrix<T> hC_1(M, N, ldc);
    host_matrix<T> hC_2(M, N, ldc);
    host_matrix<T> hC_gold(M, N, ldc);

    // Allocate device memory
    device_matrix<T> dA(M, N, lda);
    device_vector<T> dx(K, incx ? incx : 1);
    device_matrix<T> dC(M, N, ldc);

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
        CHECK_ROCBLAS_ERROR(rocblas_dgmm_fn(handle, side, M, N, dA, lda, dx, incx, dC, ldc));
        handle.post_test(arg);

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();
        cblas_dgmm<T>(side, M, N, hA, lda, hx, incx, hC_gold, ldc);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // fecth from GPU
        CHECK_HIP_ERROR(hC_2.transfer_from(dC));

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_2);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_2);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_dgmm_fn(handle, side, M, N, dA, lda, dx, incx, dC, ldc);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_dgmm_fn(handle, side, M, N, dA, lda, dx, incx, dC, ldc);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side, e_M, e_N, e_lda, e_incx, e_ldc>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            dgmm_gflop_count<T>(M, N),
            ArgumentLogging::NA_value,
            cpu_time_used,
            rocblas_error);
    }
}
