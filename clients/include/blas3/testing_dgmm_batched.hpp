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
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

/* ============================================================================================ */

template <typename T>
void testing_dgmm_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_dgmm_batched_fn
        = arg.fortran ? rocblas_dgmm_batched<T, true> : rocblas_dgmm_batched<T, false>;

    const rocblas_int M = 100;
    const rocblas_int N = 101;

    const rocblas_int lda  = 100;
    const rocblas_int incx = 1;
    const rocblas_int ldc  = 100;

    const rocblas_int batch_count = 5;

    const rocblas_side side = (rand() & 1) ? rocblas_side_right : rocblas_side_left;

    rocblas_local_handle handle{arg};

    size_t size_A = N * size_t(lda);
    size_t size_x = (rocblas_side_right == side ? N : M) * size_t(incx);
    size_t size_C = N * size_t(ldc);

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dx(size_x, 1, batch_count);
    device_batch_vector<T> dC(size_C, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(
        rocblas_dgmm_batched_fn(handle, side, M, N, nullptr, lda, dx, incx, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_dgmm_batched_fn(handle, side, M, N, dA, lda, nullptr, incx, dC, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_dgmm_batched_fn(handle, side, M, N, dA, lda, dx, incx, nullptr, ldc, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_dgmm_batched_fn(nullptr, side, M, N, dA, lda, dx, incx, dC, ldc, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_dgmm_batched(const Arguments& arg)
{
    auto rocblas_dgmm_batched_fn
        = arg.fortran ? rocblas_dgmm_batched<T, true> : rocblas_dgmm_batched<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = rocblas_side_right == side ? size_t(N) : size_t(M);

    rocblas_int lda      = arg.lda;
    rocblas_int incx     = arg.incx;
    rocblas_int ldc      = arg.ldc;
    rocblas_int abs_incx = incx > 0 ? incx : -incx;

    rocblas_int batch_count = arg.batch_count;

    double gpu_time_used, cpu_time_used;

    double rocblas_error = std::numeric_limits<double>::max();

    rocblas_local_handle handle{arg};

    size_t size_A = size_t(lda) * size_t(N);
    size_t size_C = size_t(ldc) * size_t(N);
    size_t size_x = size_t(abs_incx) * K;
    if(!size_x)
        size_x = 1;

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_dgmm_batched_fn(
                handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc, batch_count),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: dx is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> hx(size_x, 1, batch_count);
    host_batch_vector<T> hC(size_C, 1, batch_count);
    host_batch_vector<T> hC_1(size_C, 1, batch_count);
    host_batch_vector<T> hC_gold(size_C, 1, batch_count);
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hC_1.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // Initialize data on host memory
    rocblas_init_vector(hA, arg, rocblas_client_never_set_nan, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);
    rocblas_init_vector(hC, arg, rocblas_client_never_set_nan);

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dx(size_x, 1, batch_count);
    device_batch_vector<T> dC(size_C, 1, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        CHECK_ROCBLAS_ERROR(rocblas_dgmm_batched_fn(handle,
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

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        for(int b = 0; b < batch_count; b++)
            cblas_dgmm<T>(side, M, N, hA[b], lda, hx[b], incx, hC_gold[b], ldc);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // fetch GPU results
        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_dgmm_batched_fn(handle,
                                    side,
                                    M,
                                    N,
                                    dA.ptr_on_device(),
                                    lda,
                                    dx.ptr_on_device(),
                                    incx,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_dgmm_batched_fn(handle,
                                    side,
                                    M,
                                    N,
                                    dA.ptr_on_device(),
                                    lda,
                                    dx.ptr_on_device(),
                                    incx,
                                    dC.ptr_on_device(),
                                    ldc,
                                    batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

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
