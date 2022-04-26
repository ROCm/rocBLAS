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
void testing_hpr_bad_arg(const Arguments& arg)
{
    auto rocblas_hpr_fn = arg.fortran ? rocblas_hpr<T, true> : rocblas_hpr<T, false>;

    rocblas_fill         uplo  = rocblas_fill_upper;
    rocblas_int          N     = 100;
    rocblas_int          incx  = 1;
    real_t<T>            alpha = 0.6;
    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1);
    device_vector<T> dx(N, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    EXPECT_ROCBLAS_STATUS(rocblas_hpr_fn(handle, rocblas_fill_full, N, &alpha, dx, incx, dAp),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_hpr_fn(handle, uplo, N, &alpha, nullptr, incx, dAp),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpr_fn(handle, uplo, N, &alpha, dx, incx, nullptr),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_hpr_fn(nullptr, uplo, N, &alpha, dx, incx, dAp),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_hpr(const Arguments& arg)
{
    auto rocblas_hpr_fn = arg.fortran ? rocblas_hpr<T, true> : rocblas_hpr<T, false>;

    rocblas_int          N       = arg.N;
    rocblas_int          incx    = arg.incx;
    real_t<T>            h_alpha = arg.get_alpha<real_t<T>>();
    rocblas_fill         uplo    = char2rocblas_fill(arg.uplo);
    rocblas_local_handle handle{arg};

    // argument check before allocating invalid memory
    if(N < 0 || !incx)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_hpr_fn(handle, uplo, N, nullptr, nullptr, incx, nullptr),
                              rocblas_status_invalid_size);

        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t size_A   = rocblas_packed_matrix_size(N);

    // Naming: `h` is in CPU (host) memory(eg hAp_1), `d` is in GPU (device) memory (eg dAp_1).
    // Allocate host memory
    host_matrix<T>         hA(N, N, N);
    host_matrix<T>         hAp_1(1, size_A, 1);
    host_matrix<T>         hAp_2(1, size_A, 1);
    host_matrix<T>         hAp_gold(1, size_A, 1);
    host_vector<T>         hx(N, incx);
    host_vector<real_t<T>> halpha(1);

    halpha[0] = h_alpha;

    // Allocate device memory
    device_matrix<T>         dAp_1(1, size_A, 1);
    device_matrix<T>         dAp_2(1, size_A, 1);
    device_vector<T>         dx(N, incx);
    device_vector<real_t<T>> d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dAp_2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_hermitian_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);

    //regular to packed matrix conversion
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp_1, N);

    // copy matrix is easy in STL; hAp_gold = hAp_1: save a copy in hAp_gold which will be output of
    // CPU BLAS
    hAp_gold = hAp_1;
    hAp_2    = hAp_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAp_1.transfer_from(hAp_1));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1;
    double rocblas_error_2;

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dAp_2.transfer_from(hAp_1));
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_hpr_fn(handle, uplo, N, &h_alpha, dx, incx, dAp_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_hpr_fn(handle, uplo, N, d_alpha, dx, incx, dAp_2));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        cblas_hpr<T>(uplo, N, h_alpha, hx, incx, hAp_gold);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hAp_1.transfer_from(dAp_1));
        CHECK_HIP_ERROR(hAp_2.transfer_from(dAp_2));

        if(arg.unit_check)
        {
            const double tol = N * sum_error_tolerance<T>;
            near_check_general<T>(1, size_A, 1, hAp_gold, hAp_1, tol);
            near_check_general<T>(1, size_A, 1, hAp_gold, hAp_2, tol);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, size_A, 1, hAp_gold, hAp_1);
            rocblas_error_2 = norm_check_general<T>('F', 1, size_A, 1, hAp_gold, hAp_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_hpr_fn(handle, uplo, N, &h_alpha, dx, incx, dAp_1);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_hpr_fn(handle, uplo, N, &h_alpha, dx, incx, dAp_1);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_N, e_alpha, e_incx>{}.log_args<T>(rocblas_cout,
                                                                  arg,
                                                                  gpu_time_used,
                                                                  hpr_gflop_count<T>(N),
                                                                  hpr_gbyte_count<T>(N),
                                                                  cpu_time_used,
                                                                  rocblas_error_1,
                                                                  rocblas_error_2);
    }
}
