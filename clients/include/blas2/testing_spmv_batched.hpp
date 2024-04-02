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
void testing_spmv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_spmv_batched_fn
        = arg.api == FORTRAN ? rocblas_spmv_batched<T, true> : rocblas_spmv_batched<T, false>;

    auto rocblas_spmv_batched_fn_64 = arg.api == FORTRAN_64 ? rocblas_spmv_batched_64<T, true>
                                                            : rocblas_spmv_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_fill uplo        = rocblas_fill_upper;
        int64_t      N           = 100;
        int64_t      incx        = 1;
        int64_t      incy        = 1;
        int64_t      batch_count = 2;

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

        // Allocate device memory
        device_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);
        device_batch_vector<T> dy(N, incy, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dAp.memcheck());
        CHECK_DEVICE_ALLOCATION(dx.memcheck());
        CHECK_DEVICE_ALLOCATION(dy.memcheck());

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_spmv_batched_fn,
                    (nullptr,
                     uplo,
                     N,
                     alpha,
                     dAp.ptr_on_device(),
                     dx.ptr_on_device(),
                     incx,
                     beta,
                     dy.ptr_on_device(),
                     incy,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_spmv_batched_fn,
                    (handle,
                     rocblas_fill_full,
                     N,
                     alpha,
                     dAp.ptr_on_device(),
                     dx.ptr_on_device(),
                     incx,
                     beta,
                     dy.ptr_on_device(),
                     incy,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_spmv_batched_fn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     dAp.ptr_on_device(),
                     dx.ptr_on_device(),
                     incx,
                     beta,
                     dy.ptr_on_device(),
                     incy,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_spmv_batched_fn,
                    (handle,
                     uplo,
                     N,
                     alpha,
                     dAp.ptr_on_device(),
                     dx.ptr_on_device(),
                     incx,
                     nullptr,
                     dy.ptr_on_device(),
                     incy,
                     batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_spmv_batched_fn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         nullptr,
                         dx.ptr_on_device(),
                         incx,
                         beta,
                         dy.ptr_on_device(),
                         incy,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_spmv_batched_fn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dAp.ptr_on_device(),
                         nullptr,
                         incx,
                         beta,
                         dy.ptr_on_device(),
                         incy,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_spmv_batched_fn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dAp.ptr_on_device(),
                         dx.ptr_on_device(),
                         incx,
                         beta,
                         nullptr,
                         incy,
                         batch_count));
        }

        // N==0 all pointers may be null
        DAPI_CHECK(rocblas_spmv_batched_fn,
                   (handle,
                    uplo,
                    0,
                    nullptr,
                    nullptr,
                    nullptr,
                    incx,
                    nullptr,
                    nullptr,
                    incy,
                    batch_count));

        // alpha==0 and beta==1 all pointers may be null
        DAPI_CHECK(
            rocblas_spmv_batched_fn,
            (handle, uplo, N, zero, nullptr, nullptr, incx, one, nullptr, incy, batch_count));

        // batch_count==0 all pointers may be null
        DAPI_CHECK(rocblas_spmv_batched_fn,
                   (handle, uplo, N, nullptr, nullptr, nullptr, incx, nullptr, nullptr, incy, 0));
    }
}

template <typename T>
void testing_spmv_batched(const Arguments& arg)
{
    auto rocblas_spmv_batched_fn
        = arg.api == FORTRAN ? rocblas_spmv_batched<T, true> : rocblas_spmv_batched<T, false>;

    auto rocblas_spmv_batched_fn_64 = arg.api == FORTRAN_64 ? rocblas_spmv_batched_64<T, true>
                                                            : rocblas_spmv_batched_64<T, false>;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t incy        = arg.incy;
    int64_t batch_count = arg.batch_count;

    host_vector<T> h_alpha(1);
    host_vector<T> h_beta(1);
    h_alpha[0] = arg.get_alpha<T>();
    h_beta[0]  = arg.get_beta<T>();

    rocblas_fill uplo = char2rocblas_fill(arg.uplo);

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_spmv_batched_fn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     nullptr,
                     nullptr,
                     incx,
                     nullptr,
                     nullptr,
                     incy,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_batch_matrix<T> hA(N, N, N, batch_count);
    host_batch_matrix<T> hAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hy_gold(N, incy, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAp.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_symmetric_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

    // Helper function to convert regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, N);

    // Save a copy in hy_gold which will later get output of CPU BLAS
    hy_gold.copy_from(hy);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));

    double cpu_time_used;
    double rocblas_error_host = 0.0, rocblas_error_device = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_spmv_batched_fn,
                       (handle,
                        uplo,
                        N,
                        h_alpha,
                        dAp.ptr_on_device(),
                        dx.ptr_on_device(),
                        incx,
                        h_beta,
                        dy.ptr_on_device(),
                        incy,
                        batch_count));
            handle.post_test(arg);

            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));

            dy.transfer_from(hy_gold);

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_spmv_batched_fn,
                       (handle,
                        uplo,
                        N,
                        d_alpha,
                        dAp.ptr_on_device(),
                        dx.ptr_on_device(),
                        incx,
                        d_beta,
                        dy.ptr_on_device(),
                        incy,
                        batch_count));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                host_batch_vector<T> hy_copy(N, incy, batch_count);
                CHECK_HIP_ERROR(hy_copy.memcheck());
                CHECK_HIP_ERROR(hy.transfer_from(dy));
                for(int i = 0; i < arg.iters; i++)
                {
                    DAPI_CHECK(rocblas_spmv_batched_fn,
                               (handle,
                                uplo,
                                N,
                                d_alpha,
                                dAp.ptr_on_device(),
                                dx.ptr_on_device(),
                                incx,
                                d_beta,
                                dy.ptr_on_device(),
                                incy,
                                batch_count));
                    CHECK_HIP_ERROR(hy_copy.transfer_from(dy));
                    unit_check_general<T>(1, N, incy, hy, hy_copy, batch_count);
                }
                return;
            }
        }

        cpu_time_used = get_time_us_no_sync();
        // cpu reference
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_spmv<T>(uplo, N, h_alpha[0], hAp[b], hx[b], incx, h_beta[0], hy_gold[b], incy);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incy, hy_gold, hy, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_host
                    = norm_check_general<T>('F', 1, N, incy, hy_gold, hy, batch_count);
            }
        }

        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incy, hy_gold, hy, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_device
                    = norm_check_general<T>('F', 1, N, incy, hy_gold, hy, batch_count);
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
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_spmv_batched_fn,
                          (handle,
                           uplo,
                           N,
                           h_alpha,
                           dAp.ptr_on_device(),
                           dx.ptr_on_device(),
                           incx,
                           h_beta,
                           dy.ptr_on_device(),
                           incy,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_N, e_alpha, e_lda, e_incx, e_beta, e_incy, e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         spmv_gflop_count<T>(N),
                         spmv_gbyte_count<T>(N),
                         cpu_time_used,
                         rocblas_error_host,
                         rocblas_error_device);
    }
}
