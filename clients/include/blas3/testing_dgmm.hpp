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
void testing_dgmm_bad_arg(const Arguments& arg)
{
    auto rocblas_dgmm_fn = arg.api & c_API_FORTRAN ? rocblas_dgmm<T, true> : rocblas_dgmm<T, false>;
    auto rocblas_dgmm_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_dgmm_64<T, true> : rocblas_dgmm_64<T, false>;

    const int64_t M = 100;
    const int64_t N = 101;

    const int64_t lda  = 100;
    const int64_t incx = 1;
    const int64_t ldc  = 100;

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

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_dgmm_fn,
                (nullptr, side, M, N, dA, lda, dx, incx, dC, ldc));

    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_dgmm_fn,
                (handle, (rocblas_side)rocblas_fill_full, M, N, dA, lda, dx, incx, dC, ldc));

    // sizes and quick returns done in normal test harness
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dgmm_fn,
                (handle, side, M, N, nullptr, lda, dx, incx, dC, ldc));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dgmm_fn,
                (handle, side, M, N, dA, lda, nullptr, incx, dC, ldc));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dgmm_fn,
                (handle, side, M, N, dA, lda, dx, incx, nullptr, ldc));
}

template <typename T>
void testing_dgmm(const Arguments& arg)
{
    auto rocblas_dgmm_fn = arg.api & c_API_FORTRAN ? rocblas_dgmm<T, true> : rocblas_dgmm<T, false>;
    auto rocblas_dgmm_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_dgmm_64<T, true> : rocblas_dgmm_64<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    int64_t M = arg.M;
    int64_t N = arg.N;
    int64_t K = rocblas_side_right == side ? size_t(N) : size_t(M);

    int64_t lda  = arg.lda;
    int64_t incx = arg.incx;
    int64_t ldc  = arg.ldc;

    double cpu_time_used;

    double rocblas_error = std::numeric_limits<double>::max();

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || ldc < M;
    if(invalid_size || !M || !N)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_dgmm_fn,
                    (handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(M, N, lda);
    host_vector<T> hx(K, incx);
    host_matrix<T> hC(M, N, ldc);
    host_matrix<T> hC_gold(M, N, ldc);

    // Allocate device memory
    device_matrix<T> dA(M, N, lda);
    device_vector<T> dx(K, incx);
    device_matrix<T> dC(M, N, ldc);

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
        DAPI_CHECK(rocblas_dgmm_fn, (handle, side, M, N, dA, lda, dx, incx, dC, ldc));
        handle.post_test(arg);

        // fecth from GPU
        CHECK_HIP_ERROR(hC.transfer_from(dC));

        if(arg.repeatability_check)
        {
            host_matrix<T> hC_copy(M, N, ldc);

            for(int i = 0; i < arg.iters; i++)
            {
                DAPI_CHECK(rocblas_dgmm_fn, (handle, side, M, N, dA, lda, dx, incx, dC, ldc));

                CHECK_HIP_ERROR(hC_copy.transfer_from(dC));
                unit_check_general<T>(M, N, ldc, hC, hC_copy);
            }
            return;
        }

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();
        ref_dgmm<T>(side, M, N, hA, lda, hx, incx, hC_gold, ldc);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC);
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

            DAPI_DISPATCH(rocblas_dgmm_fn, (handle, side, M, N, dA, lda, dx, incx, dC, ldc));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

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
