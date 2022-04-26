/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "near.hpp"
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

template <typename T>
void testing_gemv_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemv_fn = arg.fortran ? rocblas_gemv<T, true> : rocblas_gemv<T, false>;

        const rocblas_int M    = 100;
        const rocblas_int N    = 100;
        const rocblas_int lda  = 100;
        const rocblas_int incx = 1;
        const rocblas_int incy = 1;

        device_vector<T> alpha_d(1), beta_d(1), zero_d(1), one_d(1);
        const T          alpha_h(1), beta_h(1), zero_h(0), one_h(1);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* zero  = &zero_h;
        const T* one   = &one_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
        }

        const rocblas_operation transA = rocblas_operation_none;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
        // Allocate host memory
        host_matrix<T> hA(M, N, lda);
        host_vector<T> hx(N, incx);
        host_vector<T> hy(N, incy);

        // Allocate device memory
        device_matrix<T> dA(M, N, lda);
        device_vector<T> dx(N, incx);
        device_vector<T> dy(N, incy);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dx.memcheck());
        CHECK_DEVICE_ALLOCATION(dy.memcheck());

        // Initialize data on host memory
        rocblas_init_matrix(
            hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
        rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
        rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));

        EXPECT_ROCBLAS_STATUS(rocblas_gemv_fn(handle,
                                              (rocblas_operation)rocblas_fill_full,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dx,
                                              incx,
                                              beta,
                                              dy,
                                              incy),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(handle, transA, M, N, alpha, nullptr, lda, dx, incx, beta, dy, incy),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(handle, transA, M, N, alpha, dA, lda, nullptr, incx, beta, dy, incy),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(handle, transA, M, N, alpha, dA, lda, dx, incx, beta, nullptr, incy),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(handle, transA, M, N, nullptr, dA, lda, dx, incx, beta, dy, incy),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(handle, transA, M, N, alpha, dA, lda, dx, incx, nullptr, dy, incy),
            rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(nullptr, transA, M, N, alpha, dA, lda, dx, incx, beta, dy, incy),
            rocblas_status_invalid_handle);

        // If M==0, then all pointers may be nullptr without error
        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(
                handle, transA, 0, N, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy),
            rocblas_status_success);

        // If N==0, then all pointers may be nullptr without error
        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(
                handle, transA, M, 0, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy),
            rocblas_status_success);

        // We can only test alpha==0 if pointer_mode == rocblas_pointer_mode_host
        if(pointer_mode == rocblas_pointer_mode_host)
        {
            // If alpha==0, then A and X may be nullptr without error
            EXPECT_ROCBLAS_STATUS(
                rocblas_gemv_fn(
                    handle, transA, M, N, zero, nullptr, lda, nullptr, incx, beta, dy, incy),
                rocblas_status_success);

            // If alpha==0 && beta==1, then A, X and Y may be nullptr without error
            EXPECT_ROCBLAS_STATUS(
                rocblas_gemv_fn(
                    handle, transA, M, N, zero, nullptr, lda, nullptr, incx, one, nullptr, incy),
                rocblas_status_success);
        }
    }
}

template <typename T>
void testing_gemv(const Arguments& arg)
{
    auto rocblas_gemv_fn = arg.fortran ? rocblas_gemv<T, true> : rocblas_gemv<T, false>;

    rocblas_int       M       = arg.M;
    rocblas_int       N       = arg.N;
    rocblas_int       lda     = arg.lda;
    rocblas_int       incx    = arg.incx;
    rocblas_int       incy    = arg.incy;
    T                 h_alpha = arg.get_alpha<T>();
    T                 h_beta  = arg.get_beta<T>();
    rocblas_operation transA  = char2rocblas_operation(arg.transA);
    bool              HMM     = arg.HMM;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy;
    if(invalid_size || !M || !N)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_gemv_fn(
                handle, transA, M, N, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    size_t dim_x, abs_incx;
    size_t dim_y, abs_incy;

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

    abs_incx = incx >= 0 ? incx : -incx;
    abs_incy = incy >= 0 ? incy : -incy;

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(M, N, lda);
    host_vector<T> hx(dim_x, incx);
    host_vector<T> hy_1(dim_y, incy);
    host_vector<T> hy_2(dim_y, incy);
    host_vector<T> hy_gold(dim_y, incy);
    host_vector<T> halpha(1);
    host_vector<T> hbeta(1);
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Allocate device memory
    device_matrix<T> dA(M, N, lda, HMM);
    device_vector<T> dx(dim_x, incx, HMM);
    device_vector<T> dy_1(dim_y, incy, HMM);
    device_vector<T> dy_2(dim_y, incy, HMM);
    device_vector<T> d_alpha(1, 1, HMM);
    device_vector<T> d_beta(1, 1, HMM);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy_2.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy_1, arg, rocblas_client_beta_sets_nan);

    // copy vector is easy in STL; hy_gold = hy_1: save a copy in hy_gold which will be output of
    // CPU BLAS
    hy_gold = hy_1;
    hy_2    = hy_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy_1.transfer_from(hy_1));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(dy_2.transfer_from(hy_2));
        CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gemv_fn(
            handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            rocblas_gemv_fn(handle, transA, M, N, d_alpha, dA, lda, dx, incx, d_beta, dy_2, incy));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        cblas_gemv<T>(transA, M, N, h_alpha, hA, lda, hx, incx, h_beta, hy_gold, incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_1.transfer_from(dy_1));
        CHECK_HIP_ERROR(hy_2.transfer_from(dy_2));

        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold, hy_1);
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold, hy_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold, hy_1);
            rocblas_error_2 = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_gemv_fn(handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_gemv_fn(handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA, e_M, e_N, e_alpha, e_lda, e_incx, e_beta, e_incy>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            gemv_gflop_count<T>(transA, M, N),
            gemv_gbyte_count<T>(transA, M, N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}
