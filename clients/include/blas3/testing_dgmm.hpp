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

    const rocblas_side side = (rand() & 1) ? rocblas_side_right : rocblas_side_left;

    rocblas_local_handle handle{arg};

    rocblas_int K = rocblas_side_right == side ? size_t(N) : size_t(M);

    // Allocate device memory
    device_matrix<T> dA(M, N, lda);
    device_vector<T> dx(K, incx);
    device_matrix<T> dC(M, N, ldc);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_fn(handle, side, M, N, nullptr, lda, dx, incx, dC, ldc),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_fn(handle, side, M, N, dA, lda, nullptr, incx, dC, ldc),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_fn(handle, side, M, N, dA, lda, dx, incx, nullptr, ldc),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_dgmm_fn(nullptr, side, M, N, dA, lda, dx, incx, dC, ldc),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_dgmm(const Arguments& arg)
{
    auto rocblas_dgmm_fn = arg.fortran ? rocblas_dgmm<T, true> : rocblas_dgmm<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = rocblas_side_right == side ? size_t(N) : size_t(M);

    rocblas_int lda      = arg.lda;
    rocblas_int incx     = arg.incx;
    rocblas_int ldc      = arg.ldc;
    rocblas_int abs_incx = incx > 0 ? incx : -incx;

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
        CHECK_ROCBLAS_ERROR(rocblas_dgmm_fn(handle, side, M, N, dA, lda, dx, incx, dC, ldc));

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
