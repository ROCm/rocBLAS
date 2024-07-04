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
 *
 * ************************************************************************ */

#pragma once

#include "testing_common.hpp"

template <typename T>
void testing_gbmv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_gbmv_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_gbmv_batched<T, true> : rocblas_gbmv_batched<T, false>;
    auto rocblas_gbmv_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_gbmv_batched_64<T, true>
                                                              : rocblas_gbmv_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_operation transA      = rocblas_operation_none;
        const int64_t           M           = 100;
        const int64_t           N           = 100;
        const int64_t           KL          = 5;
        const int64_t           KU          = 5;
        const int64_t           lda         = 100;
        const int64_t           incx        = 1;
        const int64_t           incy        = 1;
        const int64_t           batch_count = 2;

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

        const int64_t banded_matrix_row = KL + KU + 1;

        // Allocate device memory
        device_batch_matrix<T> dAb(banded_matrix_row, N, lda, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);
        device_batch_vector<T> dy(M, incy, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dAb.memcheck());
        CHECK_DEVICE_ALLOCATION(dx.memcheck());
        CHECK_DEVICE_ALLOCATION(dy.memcheck());

        auto dA_dev = dAb.ptr_on_device();
        auto dx_dev = dx.ptr_on_device();
        auto dy_dev = dy.ptr_on_device();

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_gbmv_batched_fn,
                    (nullptr,
                     transA,
                     M,
                     N,
                     KL,
                     KU,
                     alpha,
                     dA_dev,
                     lda,
                     dx_dev,
                     incx,
                     beta,
                     dy_dev,
                     incy,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_gbmv_batched_fn,
                    (handle,
                     (rocblas_operation)rocblas_fill_full,
                     M,
                     N,
                     KL,
                     KU,
                     alpha,
                     dA_dev,
                     lda,
                     dx_dev,
                     incx,
                     beta,
                     dy_dev,
                     incy,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_gbmv_batched_fn,
                    (handle,
                     transA,
                     M,
                     N,
                     KL,
                     KU,
                     nullptr,
                     dA_dev,
                     lda,
                     dx_dev,
                     incx,
                     beta,
                     dy_dev,
                     incy,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_gbmv_batched_fn,
                    (handle,
                     transA,
                     M,
                     N,
                     KL,
                     KU,
                     alpha,
                     dA_dev,
                     lda,
                     dx_dev,
                     incx,
                     nullptr,
                     dy_dev,
                     incy,
                     batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gbmv_batched_fn,
                        (handle,
                         transA,
                         M,
                         N,
                         KL,
                         KU,
                         alpha,
                         nullptr,
                         lda,
                         dx_dev,
                         incx,
                         beta,
                         dy_dev,
                         incy,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gbmv_batched_fn,
                        (handle,
                         transA,
                         M,
                         N,
                         KL,
                         KU,
                         alpha,
                         dA_dev,
                         lda,
                         nullptr,
                         incx,
                         beta,
                         dy_dev,
                         incy,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gbmv_batched_fn,
                        (handle,
                         transA,
                         M,
                         N,
                         KL,
                         KU,
                         alpha,
                         dA_dev,
                         lda,
                         dx_dev,
                         incx,
                         beta,
                         nullptr,
                         incy,
                         batch_count));
        }

        // When M==0, alpha, A, x, beta, and y may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_batched_fn,
                   (handle,
                    transA,
                    0,
                    N,
                    KL,
                    KU,
                    nullptr,
                    nullptr,
                    lda,
                    nullptr,
                    incx,
                    nullptr,
                    nullptr,
                    incy,
                    batch_count));

        // When N==0, alpha, A, x, beta, and Y may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_batched_fn,
                   (handle,
                    transA,
                    M,
                    0,
                    KL,
                    KU,
                    nullptr,
                    nullptr,
                    lda,
                    nullptr,
                    incx,
                    nullptr,
                    nullptr,
                    incy,
                    batch_count));

        // When alpha==0, A and x may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_batched_fn,
                   (handle,
                    transA,
                    M,
                    N,
                    KL,
                    KU,
                    zero,
                    nullptr,
                    lda,
                    nullptr,
                    incx,
                    beta,
                    dy_dev,
                    incy,
                    batch_count));

        // When alpha==0 && beta==1, A, x and y may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_batched_fn,
                   (handle,
                    transA,
                    M,
                    N,
                    KL,
                    KU,
                    zero,
                    nullptr,
                    lda,
                    nullptr,
                    incx,
                    one,
                    nullptr,
                    incy,
                    batch_count));

        // When batch_count==0, alpha, A, x, beta, and Y may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_batched_fn,
                   (handle,
                    transA,
                    M,
                    N,
                    KL,
                    KU,
                    nullptr,
                    nullptr,
                    lda,
                    nullptr,
                    incx,
                    nullptr,
                    nullptr,
                    incy,
                    0));
    }
}

template <typename T>
void testing_gbmv_batched(const Arguments& arg)
{
    auto rocblas_gbmv_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_gbmv_batched<T, true> : rocblas_gbmv_batched<T, false>;
    auto rocblas_gbmv_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_gbmv_batched_64<T, true>
                                                              : rocblas_gbmv_batched_64<T, false>;

    int64_t           M                 = arg.M;
    int64_t           N                 = arg.N;
    int64_t           KL                = arg.KL;
    int64_t           KU                = arg.KU;
    int64_t           lda               = arg.lda;
    int64_t           incx              = arg.incx;
    int64_t           incy              = arg.incy;
    T                 h_alpha           = arg.get_alpha<T>();
    T                 h_beta            = arg.get_beta<T>();
    rocblas_operation transA            = char2rocblas_operation(arg.transA);
    int64_t           batch_count       = arg.batch_count;
    int64_t           banded_matrix_row = KL + KU + 1;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < banded_matrix_row || !incx || !incy
                        || batch_count < 0 || KL < 0 || KU < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gbmv_batched_fn(handle,
                                                      transA,
                                                      M,
                                                      N,
                                                      KL,
                                                      KU,
                                                      nullptr,
                                                      nullptr,
                                                      lda,
                                                      nullptr,
                                                      incx,
                                                      nullptr,
                                                      nullptr,
                                                      incy,
                                                      batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    size_t dim_x;
    size_t dim_y;

    if(transA == rocblas_operation_none)
    {
        dim_x = N;
        dim_y = M;
    }
    else
    {
        dim_x = M;
        dim_y = N;
    }

    // Naming: `h` is in CPU (host) memory(eg hAb), `d` is in GPU (device) memory (eg dAb).
    // Allocate host memory
    host_batch_matrix<T> hAb(banded_matrix_row, N, lda, batch_count);
    host_batch_vector<T> hx(dim_x, incx, batch_count);
    host_batch_vector<T> hy(dim_y, incy, batch_count);
    host_batch_vector<T> hy_gold(dim_y, incy, batch_count);
    host_vector<T>       halpha(1);
    host_vector<T>       hbeta(1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hAb.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());

    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Allocate device memory
    device_batch_matrix<T> dAb(banded_matrix_row, N, lda, batch_count);
    device_batch_vector<T> dx(dim_x, incx, batch_count);
    device_batch_vector<T> dy(dim_y, incy, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAb.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hAb, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

    hy.copy_from(hy);
    hy_gold.copy_from(hy);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAb.transfer_from(hAb));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
    CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));

    double cpu_time_used;
    double error_host = 0.0, error_device = 0.0;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_gbmv_batched_fn,
                       (handle,
                        transA,
                        M,
                        N,
                        KL,
                        KU,
                        &h_alpha,
                        dAb.ptr_on_device(),
                        lda,
                        dx.ptr_on_device(),
                        incx,
                        &h_beta,
                        dy.ptr_on_device(),
                        incy,
                        batch_count));
            handle.post_test(arg);

            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }
        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(dy.transfer_from(hy_gold));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_gbmv_batched_fn,
                       (handle,
                        transA,
                        M,
                        N,
                        KL,
                        KU,
                        d_alpha,
                        dAb.ptr_on_device(),
                        lda,
                        dx.ptr_on_device(),
                        incx,
                        d_beta,
                        dy.ptr_on_device(),
                        incy,
                        batch_count));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                host_batch_vector<T> hy_copy(dim_y, incy, batch_count);
                CHECK_HIP_ERROR(hy_copy.memcheck());
                CHECK_HIP_ERROR(hy.transfer_from(dy));

                for(int i = 0; i < arg.iters; i++)
                {
                    CHECK_HIP_ERROR(dy.transfer_from(hy_gold));
                    DAPI_CHECK(rocblas_gbmv_batched_fn,
                               (handle,
                                transA,
                                M,
                                N,
                                KL,
                                KU,
                                d_alpha,
                                dAb.ptr_on_device(),
                                lda,
                                dx.ptr_on_device(),
                                incx,
                                d_beta,
                                dy.ptr_on_device(),
                                incy,
                                batch_count));
                    CHECK_HIP_ERROR(hy_copy.transfer_from(dy));
                    unit_check_general<T>(1, dim_y, incy, hy, hy_copy, batch_count);
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int64_t b = 0; b < batch_count; ++b)
        {
            ref_gbmv<T>(
                transA, M, N, KL, KU, h_alpha, hAb[b], lda, hx[b], incx, h_beta, hy_gold[b], incy);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(1, dim_y, incy, hy_gold, hy, batch_count);
            }
            if(arg.norm_check)
            {
                error_host = norm_check_general<T>('F', 1, dim_y, incy, hy_gold, hy, batch_count);
            }
        }
        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hy.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, dim_y, incy, hy_gold, hy, batch_count);
            }
            if(arg.norm_check)
            {
                error_device = norm_check_general<T>('F', 1, dim_y, incy, hy_gold, hy, batch_count);
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

            DAPI_DISPATCH(rocblas_gbmv_batched_fn,
                          (handle,
                           transA,
                           M,
                           N,
                           KL,
                           KU,
                           &h_alpha,
                           dAb.ptr_on_device(),
                           lda,
                           dx.ptr_on_device(),
                           incx,
                           &h_beta,
                           dy.ptr_on_device(),
                           incy,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_M,
                      e_N,
                      e_KL,
                      e_KU,
                      e_alpha,
                      e_lda,
                      e_incx,
                      e_beta,
                      e_incy,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gbmv_gflop_count<T>(transA, M, N, KL, KU),
                         gbmv_gbyte_count<T>(transA, M, N, KL, KU),
                         cpu_time_used,
                         error_host,
                         error_device);
    }
}
