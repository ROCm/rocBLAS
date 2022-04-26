/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_spr2_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_spr2_strided_batched_fn = arg.fortran ? rocblas_spr2_strided_batched<T, true>
                                                       : rocblas_spr2_strided_batched<T, false>;

    rocblas_fill   uplo        = rocblas_fill_upper;
    rocblas_int    N           = 10;
    rocblas_int    incx        = 1;
    rocblas_int    incy        = 1;
    T              alpha       = 0.6;
    rocblas_int    batch_count = 5;
    rocblas_stride stride_x    = 100;
    rocblas_stride stride_y    = 100;
    rocblas_stride stride_A    = 100;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_strided_batch_matrix<T> dAp_1(
        1, rocblas_packed_matrix_size(N), 1, stride_A, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_spr2_strided_batched_fn(handle,
                                                          rocblas_fill_full,
                                                          N,
                                                          &alpha,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          dAp_1,
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_spr2_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          nullptr,
                                                          incx,
                                                          stride_x,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          dAp_1,
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_spr2_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          nullptr,
                                                          incy,
                                                          stride_y,
                                                          dAp_1,
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_spr2_strided_batched_fn(handle,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          nullptr,
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_spr2_strided_batched_fn(nullptr,
                                                          uplo,
                                                          N,
                                                          &alpha,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          dy,
                                                          incy,
                                                          stride_y,
                                                          dAp_1,
                                                          stride_A,
                                                          batch_count),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_spr2_strided_batched(const Arguments& arg)
{
    auto rocblas_spr2_strided_batched_fn = arg.fortran ? rocblas_spr2_strided_batched<T, true>
                                                       : rocblas_spr2_strided_batched<T, false>;

    rocblas_int    N           = arg.N;
    rocblas_int    incx        = arg.incx;
    rocblas_int    incy        = arg.incy;
    T              h_alpha     = arg.get_alpha<T>();
    rocblas_fill   uplo        = char2rocblas_fill(arg.uplo);
    rocblas_stride stride_x    = arg.stride_x;
    rocblas_stride stride_y    = arg.stride_y;
    rocblas_stride stride_A    = arg.stride_a;
    rocblas_int    batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument check before allocating invalid memory
    bool invalid_size = N < 0 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_spr2_strided_batched_fn(handle,
                                                              uplo,
                                                              N,
                                                              nullptr,
                                                              nullptr,
                                                              incx,
                                                              stride_x,
                                                              nullptr,
                                                              incy,
                                                              stride_y,
                                                              nullptr,
                                                              stride_A,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    size_t size_A = rocblas_packed_matrix_size(N);

    // Naming: `h` is in CPU (host) memory(eg hAp_1), `d` is in GPU (device) memory (eg dAp_1).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(N, N, N, stride_A, batch_count);
    host_strided_batch_matrix<T> hAp_1(1, size_A, 1, stride_A, batch_count);
    host_strided_batch_matrix<T> hAp_2(1, size_A, 1, stride_A, batch_count);
    host_strided_batch_matrix<T> hA_gold(1, size_A, 1, stride_A, batch_count);
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy(N, incy, stride_y, batch_count);
    host_vector<T>               halpha(1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAp_1.memcheck());
    CHECK_HIP_ERROR(hAp_2.memcheck());
    CHECK_HIP_ERROR(hA_gold.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy.memcheck());

    halpha[0] = h_alpha;

    // Allocate device memory
    device_strided_batch_matrix<T> dAp_1(1, size_A, 1, stride_A, batch_count);
    device_strided_batch_matrix<T> dAp_2(1, size_A, 1, stride_A, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);
    device_vector<T>               d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dAp_2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_symmetric_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, false);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan);

    // Helper function to convert regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp_1, N);

    hAp_2.copy_from(hAp_1);
    hA_gold.copy_from(hAp_1);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAp_1.transfer_from(hAp_1));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1;
    double rocblas_error_2;

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dAp_2.transfer_from(hAp_1));
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_spr2_strided_batched_fn(handle,
                                                            uplo,
                                                            N,
                                                            &h_alpha,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            dy,
                                                            incy,
                                                            stride_y,
                                                            dAp_1,
                                                            stride_A,
                                                            batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_spr2_strided_batched_fn(handle,
                                                            uplo,
                                                            N,
                                                            d_alpha,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            dy,
                                                            incy,
                                                            stride_y,
                                                            dAp_2,
                                                            stride_A,
                                                            batch_count));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; b++)
        {
            cblas_spr2<T>(uplo, N, h_alpha, hx[b], incx, hy[b], incy, hA_gold[b]);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hAp_1.transfer_from(dAp_1));
        CHECK_HIP_ERROR(hAp_2.transfer_from(dAp_2));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, size_A, 1, stride_A, hA_gold, hAp_1, batch_count);
            unit_check_general<T>(1, size_A, 1, stride_A, hA_gold, hAp_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', 1, size_A, 1, stride_A, hA_gold, hAp_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('F', 1, size_A, 1, stride_A, hA_gold, hAp_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_spr2_strided_batched_fn(handle,
                                            uplo,
                                            N,
                                            &h_alpha,
                                            dx,
                                            incx,
                                            stride_x,
                                            dy,
                                            incy,
                                            stride_y,
                                            dAp_1,
                                            stride_A,
                                            batch_count);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_spr2_strided_batched_fn(handle,
                                            uplo,
                                            N,
                                            &h_alpha,
                                            dx,
                                            incx,
                                            stride_x,
                                            dy,
                                            incy,
                                            stride_y,
                                            dAp_1,
                                            stride_A,
                                            batch_count);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo,
                      e_N,
                      e_alpha,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_incy,
                      e_stride_y,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         spr2_gflop_count<T>(N),
                         spr2_gbyte_count<T>(N),
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
