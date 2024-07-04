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
void testing_syr2_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_syr2_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_syr2_batched<T, true> : rocblas_syr2_batched<T, false>;

    auto rocblas_syr2_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_syr2_batched_64<T, true>
                                                              : rocblas_syr2_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_fill uplo        = rocblas_fill_upper;
        int64_t      N           = 100;
        int64_t      incx        = 1;
        int64_t      incy        = 1;
        int64_t      lda         = 100;
        int64_t      batch_count = 2;

        device_vector<T> alpha_d(1), zero_d(1);

        const T alpha_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        // Allocate device memory
        device_batch_matrix<T> dA_1(N, N, lda, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);
        device_batch_vector<T> dy(N, incy, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
        CHECK_DEVICE_ALLOCATION(dx.memcheck());
        CHECK_DEVICE_ALLOCATION(dy.memcheck());

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_syr2_batched_fn,
                    (nullptr, uplo, N, alpha, dx, incx, dy, incy, dA_1, lda, batch_count));

        DAPI_EXPECT(
            rocblas_status_invalid_value,
            rocblas_syr2_batched_fn,
            (handle, rocblas_fill_full, N, alpha, dx, incx, dy, incy, dA_1, lda, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_syr2_batched_fn,
                    (handle, uplo, N, nullptr, dx, incx, dy, incy, dA_1, lda, batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_syr2_batched_fn,
                        (handle, uplo, N, alpha, nullptr, incx, dy, incy, dA_1, lda, batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_syr2_batched_fn,
                        (handle, uplo, N, alpha, dx, incx, nullptr, incy, dA_1, lda, batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_syr2_batched_fn,
                        (handle, uplo, N, alpha, dx, incx, dy, incy, nullptr, lda, batch_count));
        }

        // N==0 all pointers may be null
        DAPI_CHECK(
            rocblas_syr2_batched_fn,
            (handle, uplo, 0, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, batch_count));

        // alpha==0 all pointers may be null
        DAPI_CHECK(
            rocblas_syr2_batched_fn,
            (handle, uplo, N, zero, nullptr, incx, nullptr, incy, nullptr, lda, batch_count));

        // batch_count==0 all pointers may be null
        DAPI_CHECK(rocblas_syr2_batched_fn,
                   (handle, uplo, N, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, 0));
    }
}

template <typename T>
void testing_syr2_batched(const Arguments& arg)
{
    auto rocblas_syr2_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_syr2_batched<T, true> : rocblas_syr2_batched<T, false>;

    auto rocblas_syr2_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_syr2_batched_64<T, true>
                                                              : rocblas_syr2_batched_64<T, false>;

    int64_t      N           = arg.N;
    int64_t      incx        = arg.incx;
    int64_t      incy        = arg.incy;
    int64_t      lda         = arg.lda;
    int64_t      batch_count = arg.batch_count;
    T            h_alpha     = arg.get_alpha<T>();
    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);

    rocblas_local_handle handle{arg};

    // argument check before allocating invalid memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
            rocblas_syr2_batched_fn,
            (handle, uplo, N, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(N, N, lda, batch_count);
    host_batch_matrix<T> hA_gold(N, N, lda, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_vector<T>       halpha(1);
    halpha[0] = h_alpha;

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hA_gold.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(N, N, lda, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<T>       d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_symmetric_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan);

    hA_gold.copy_from(hA);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double cpu_time_used;
    double rocblas_error_host;
    double rocblas_error_device;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_syr2_batched_fn,
                       (handle,
                        uplo,
                        N,
                        &h_alpha,
                        dx.ptr_on_device(),
                        incx,
                        dy.ptr_on_device(),
                        incy,
                        dA.ptr_on_device(),
                        lda,
                        batch_count));
            handle.post_test(arg);

            CHECK_HIP_ERROR(hA.transfer_from(dA));
        }

        if(arg.pointer_mode_device)
        {
            // copy data from CPU to device
            CHECK_HIP_ERROR(dA.transfer_from(hA_gold));
            CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);

            DAPI_CHECK(rocblas_syr2_batched_fn,
                       (handle,
                        uplo,
                        N,
                        d_alpha,
                        dx.ptr_on_device(),
                        incx,
                        dy.ptr_on_device(),
                        incy,
                        dA.ptr_on_device(),
                        lda,
                        batch_count));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                host_batch_matrix<T> hA_copy(N, N, lda, batch_count);
                CHECK_HIP_ERROR(hA_copy.memcheck());
                CHECK_HIP_ERROR(hA.transfer_from(dA));
                for(int i = 0; i < arg.iters; i++)
                {
                    CHECK_HIP_ERROR(dA.transfer_from(hA_gold));
                    DAPI_CHECK(rocblas_syr2_batched_fn,
                               (handle,
                                uplo,
                                N,
                                d_alpha,
                                dx.ptr_on_device(),
                                incx,
                                dy.ptr_on_device(),
                                incy,
                                dA.ptr_on_device(),
                                lda,
                                batch_count));
                    CHECK_HIP_ERROR(hA_copy.transfer_from(dA));
                    unit_check_general<T>(N, N, lda, hA, hA_copy, batch_count);
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; b++)
        {
            ref_syr2<T>(uplo, N, h_alpha, hx[b], incx, hy[b], incy, hA_gold[b], lda);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(N, N, lda, hA_gold, hA, batch_count);
            }
            if(arg.norm_check)
            {
                rocblas_error_host
                    = norm_check_general<T>('F', N, N, lda, hA_gold, hA, batch_count);
            }
        }
        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hA.transfer_from(dA));

            if(arg.unit_check)
            {
                unit_check_general<T>(N, N, lda, hA_gold, hA, batch_count);
            }
            if(arg.norm_check)
            {
                rocblas_error_device
                    = norm_check_general<T>('F', N, N, lda, hA_gold, hA, batch_count);
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
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_syr2_batched_fn,
                          (handle,
                           uplo,
                           N,
                           &h_alpha,
                           dx.ptr_on_device(),
                           incx,
                           dy.ptr_on_device(),
                           incy,
                           dA.ptr_on_device(),
                           lda,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo, e_N, e_alpha, e_lda, e_incx, e_incy, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            syr2_gflop_count<T>(N),
            syr2_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error_host,
            rocblas_error_device);
    }
}
