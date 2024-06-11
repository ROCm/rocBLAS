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

template <typename T>
void testing_sbmv_bad_arg(const Arguments& arg)
{
    auto rocblas_sbmv_fn = arg.api & c_API_FORTRAN ? rocblas_sbmv<T, true> : rocblas_sbmv<T, false>;
    auto rocblas_sbmv_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_sbmv_64<T, true> : rocblas_sbmv_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_fill uplo = rocblas_fill_upper;
        int64_t      N    = 100;
        int64_t      K    = 2;
        int64_t      incx = 1;
        int64_t      incy = 1;
        int64_t      lda  = 100;

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

        rocblas_int banded_matrix_row = K + 1;

        // Allocate device memory
        device_matrix<T> dAb(banded_matrix_row, N, lda);
        device_vector<T> dx(N, incx);
        device_vector<T> dy(N, incy);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dAb.memcheck());
        CHECK_DEVICE_ALLOCATION(dx.memcheck());
        CHECK_DEVICE_ALLOCATION(dy.memcheck());

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_sbmv_fn,
                    (nullptr, uplo, N, K, alpha, dAb, lda, dx, incx, beta, dy, incy));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_sbmv_fn,
                    (handle, rocblas_fill_full, N, K, alpha, dAb, lda, dx, incx, beta, dy, incy));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_sbmv_fn,
                    (handle, uplo, N, K, nullptr, dAb, lda, dx, incx, beta, dy, incy));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_sbmv_fn,
                    (handle, uplo, N, K, alpha, dAb, lda, dx, incx, nullptr, dy, incy));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_sbmv_fn,
                        (handle, uplo, N, K, alpha, nullptr, lda, dx, incx, beta, dy, incy));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_sbmv_fn,
                        (handle, uplo, N, K, alpha, dAb, lda, nullptr, incx, beta, dy, incy));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_sbmv_fn,
                        (handle, uplo, N, K, alpha, dAb, lda, dx, incx, beta, nullptr, incy));
        }
        // If N or k is 64 bit
        if(arg.api & c_API_64)
        {
            int64_t n_over_int32   = 2147483649;
            int64_t lda_over_int32 = 2147483649;
            int64_t k_over_int32   = 2147483648;
            DAPI_EXPECT(rocblas_status_invalid_size,
                        rocblas_sbmv_fn,
                        (handle, uplo, n_over_int32, K, alpha, dAb, lda, dx, incx, beta, dy, incy));
            DAPI_EXPECT(rocblas_status_invalid_size,
                        rocblas_sbmv_fn,
                        (handle,
                         uplo,
                         N,
                         k_over_int32,
                         alpha,
                         dAb,
                         lda_over_int32,
                         dx,
                         incx,
                         beta,
                         dy,
                         incy));
        }

        // N==0 all pointers may be null
        DAPI_CHECK(
            rocblas_sbmv_fn,
            (handle, uplo, 0, K, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy));

        // alpha==0 then A and x pointers may be null
        DAPI_CHECK(rocblas_sbmv_fn,
                   (handle, uplo, N, K, zero, nullptr, lda, nullptr, incx, beta, dy, incy));

        // alpha==0 and beta==1 all pointers may be null
        DAPI_CHECK(rocblas_sbmv_fn,
                   (handle, uplo, N, K, zero, nullptr, lda, nullptr, incx, one, nullptr, incy));
    }
}

template <typename T>
void testing_sbmv(const Arguments& arg)
{
    auto rocblas_sbmv_fn = arg.api & c_API_FORTRAN ? rocblas_sbmv<T, true> : rocblas_sbmv<T, false>;

    auto rocblas_sbmv_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_sbmv_64<T, true> : rocblas_sbmv_64<T, false>;

    int64_t N                 = arg.N;
    int64_t lda               = arg.lda;
    int64_t K                 = arg.K;
    int64_t incx              = arg.incx;
    int64_t incy              = arg.incy;
    int64_t banded_matrix_row = K + 1;

    host_vector<T> alpha(1);
    host_vector<T> beta(1);
    alpha[0] = arg.get_alpha<T>();
    beta[0]  = arg.get_beta<T>();

    rocblas_fill uplo = char2rocblas_fill(arg.uplo);

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N < 0 || lda < banded_matrix_row || K < 0 || !incx || !incy)
    {
        DAPI_EXPECT(
            rocblas_status_invalid_size,
            rocblas_sbmv_fn,
            (handle, uplo, N, K, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAb), `d` is in GPU (device) memory (eg dAb).
    // Allocate host memory
    host_matrix<T> hAb(banded_matrix_row, N, lda);
    host_vector<T> hx(N, incx);
    host_vector<T> hy(N, incy);
    host_vector<T> hy_gold(N, incy); // gold standard

    // Allocate device memory
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    // Allocate device memory
    device_matrix<T> dAb(banded_matrix_row, N, lda);
    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAb.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hAb, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

    // make copy in hy_gold which will later be used with CPU BLAS
    hy_gold = hy;

    // copy data from CPU to device
    dx.transfer_from(hx);
    dy.transfer_from(hy);
    dAb.transfer_from(hAb);

    double cpu_time_used;
    double error_host = 0.0, error_device = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_sbmv_fn,
                       (handle, uplo, N, K, alpha, dAb, lda, dx, incx, beta, dy, incy));
            handle.post_test(arg);

            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(d_alpha.transfer_from(alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(beta));

            dy.transfer_from(hy_gold);

            handle.pre_test(arg);

            DAPI_CHECK(rocblas_sbmv_fn,
                       (handle, uplo, N, K, d_alpha, dAb, lda, dx, incx, d_beta, dy, incy));

            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                host_vector<T> hy_copy(N, incy);
                CHECK_HIP_ERROR(hy_copy.memcheck());
                CHECK_HIP_ERROR(hy.transfer_from(dy));

                for(int i = 0; i < arg.iters; i++)
                {
                    dy.transfer_from(hy_gold);
                    DAPI_CHECK(rocblas_sbmv_fn,
                               (handle, uplo, N, K, d_alpha, dAb, lda, dx, incx, d_beta, dy, incy));
                    CHECK_HIP_ERROR(hy_copy.transfer_from(dy));
                    unit_check_general<T>(1, N, incy, hy, hy_copy);
                }
                return;
            }
        }

        // cpu ref
        cpu_time_used = get_time_us_no_sync();
        ref_sbmv<T>(uplo, N, K, alpha[0], hAb, lda, hx, incx, beta[0], hy_gold, incy);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incy, hy_gold, hy);
            }

            if(arg.norm_check)
            {
                error_host = norm_check_general<T>('F', 1, N, incy, hy_gold, hy);
            }
        }

        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incy, hy_gold, hy);
            }

            if(arg.norm_check)
            {
                error_device = norm_check_general<T>('F', 1, N, incy, hy_gold, hy);
            }
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream); // in microseconds

            DAPI_DISPATCH(rocblas_sbmv_fn,
                          (handle, uplo, N, K, alpha, dAb, lda, dx, incx, beta, dy, incy));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo, e_N, e_K, e_alpha, e_lda, e_incx, e_beta, e_incy>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            sbmv_gflop_count<T>(N, K),
            sbmv_gbyte_count<T>(N, K),
            cpu_time_used,
            error_host,
            error_device);
    }
}
