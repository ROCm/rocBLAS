/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T, bool CONJ>
void testing_ger_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_ger_batched_fn
        = arg.fortran
              ? (CONJ ? rocblas_ger_batched<T, true, true> : rocblas_ger_batched<T, false, true>)
              : (CONJ ? rocblas_ger_batched<T, true, false> : rocblas_ger_batched<T, false, false>);

    rocblas_int       M           = 100;
    rocblas_int       N           = 100;
    rocblas_int       incx        = 1;
    rocblas_int       incy        = 1;
    rocblas_int       lda         = 100;
    T                 alpha       = 0.6;
    const rocblas_int batch_count = 5;

    size_t size_A = lda * size_t(N);

    rocblas_local_handle handle{arg};

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dx(M, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS((rocblas_ger_batched_fn(handle,
                                                  M,
                                                  N,
                                                  &alpha,
                                                  nullptr,
                                                  incx,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_batched_fn(handle,
                                                  M,
                                                  N,
                                                  &alpha,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  nullptr,
                                                  incy,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_batched_fn(handle,
                                                  M,
                                                  N,
                                                  &alpha,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  nullptr,
                                                  lda,
                                                  batch_count)),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_ger_batched_fn(nullptr,
                                                  M,
                                                  N,
                                                  &alpha,
                                                  dx.ptr_on_device(),
                                                  incx,
                                                  dy.ptr_on_device(),
                                                  incy,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  batch_count)),
                          rocblas_status_invalid_handle);
}

template <typename T, bool CONJ>
void testing_ger_batched(const Arguments& arg)
{
    auto rocblas_ger_batched_fn
        = arg.fortran
              ? (CONJ ? rocblas_ger_batched<T, true, true> : rocblas_ger_batched<T, false, true>)
              : (CONJ ? rocblas_ger_batched<T, true, false> : rocblas_ger_batched<T, false, false>);

    rocblas_int M           = arg.M;
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int lda         = arg.lda;
    T           h_alpha     = arg.get_alpha<T>();
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(
            (rocblas_ger_batched_fn(
                handle, M, N, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, batch_count)),

            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = size_t(lda) * N;
    size_t size_x   = M * abs_incx;
    size_t size_y   = N * abs_incy;

    device_batch_vector<T> dy(N, incy, batch_count);
    device_batch_vector<T> dx(M, incx, batch_count);
    device_batch_vector<T> dA_1(size_A, 1, batch_count);
    device_batch_vector<T> dA_2(size_A, 1, batch_count);
    device_vector<T>       d_alpha(1);
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dA_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    // Host-arrays of pointers to host memory
    host_batch_vector<T> hy(N, incy, batch_count);
    host_batch_vector<T> hx(M, incx, batch_count);
    host_batch_vector<T> hA_1(size_A, 1, batch_count);
    host_batch_vector<T> hA_2(size_A, 1, batch_count);
    host_batch_vector<T> hA_gold(size_A, 1, batch_count);
    host_vector<T>       halpha(1);
    halpha[0] = h_alpha;

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1;
    double rocblas_error_2;

    // Initial Data on CPU
    rocblas_init(hA_1, true);
    rocblas_init(hx, false);
    rocblas_init(hy, false);
    hA_2.copy_from(hA_1);
    hA_gold.copy_from(hA_1);

    CHECK_HIP_ERROR(dA_1.transfer_from(hA_1));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA_2.transfer_from(hA_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_ger_batched_fn(handle,
                                                    M,
                                                    N,
                                                    &h_alpha,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    dy.ptr_on_device(),
                                                    incy,
                                                    dA_1.ptr_on_device(),
                                                    lda,
                                                    batch_count)));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_ger_batched_fn(handle,
                                                    M,
                                                    N,
                                                    d_alpha,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    dy.ptr_on_device(),
                                                    incy,
                                                    dA_2.ptr_on_device(),
                                                    lda,
                                                    batch_count)));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_ger<T, CONJ>(M, N, h_alpha, hx[b], incx, hy[b], incy, hA_gold[b], lda);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA_1.transfer_from(dA_1));
        CHECK_HIP_ERROR(hA_2.transfer_from(dA_2));

        if(arg.unit_check)
        {
            if(std::is_same<T, float>{} || std::is_same<T, double>{})
            {
                unit_check_general<T>(M, N, lda, hA_gold, hA_1, batch_count);
                unit_check_general<T>(M, N, lda, hA_gold, hA_2, batch_count);
            }
            else
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(M, N, lda, hA_gold, hA_1, batch_count, tol);
                near_check_general<T>(M, N, lda, hA_gold, hA_2, batch_count, tol);
            }
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, lda, hA_gold, hA_1, batch_count);
            rocblas_error_2 = norm_check_general<T>('F', M, N, lda, hA_gold, hA_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_ger_batched_fn(handle,
                                   M,
                                   N,
                                   &h_alpha,
                                   dx.ptr_on_device(),
                                   incx,
                                   dy.ptr_on_device(),
                                   incy,
                                   dA_1.ptr_on_device(),
                                   lda,
                                   batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_ger_batched_fn(handle,
                                   M,
                                   N,
                                   &h_alpha,
                                   dx.ptr_on_device(),
                                   incx,
                                   dy.ptr_on_device(),
                                   incy,
                                   dA_1.ptr_on_device(),
                                   lda,
                                   batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_N, e_alpha, e_lda, e_incx, e_incy, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            ger_gflop_count<T, CONJ>(M, N),
            ger_gbyte_count<T>(M, N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}
