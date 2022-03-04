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
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_tpmv_bad_arg(const Arguments& arg)
{
    auto rocblas_tpmv_fn = arg.fortran ? rocblas_tpmv<T, true> : rocblas_tpmv<T, false>;

    const rocblas_int       M      = 100;
    const rocblas_int       incx   = 1;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_fill      uplo   = rocblas_fill_lower;
    const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    size_t size_A = (M * (M + 1)) / 2;
    size_t size_x = M * size_t(incx);

    host_vector<T> hA((rocblas_int)size_A, (rocblas_int)1);
    CHECK_HIP_ERROR(hA.memcheck());
    host_vector<T> hx((rocblas_int)size_x, (rocblas_int)1);
    CHECK_HIP_ERROR(hx.memcheck());
    device_vector<T> dA(size_A);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    device_vector<T> dx(size_x);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    //
    // Checks.
    //
    EXPECT_ROCBLAS_STATUS(rocblas_tpmv_fn(handle, uplo, transA, diag, M, nullptr, dx, incx),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_tpmv_fn(handle, uplo, transA, diag, M, dA, nullptr, incx),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_tpmv_fn(nullptr, uplo, transA, diag, M, dA, dx, incx),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_tpmv(const Arguments& arg)
{
    auto rocblas_tpmv_fn = arg.fortran ? rocblas_tpmv<T, true> : rocblas_tpmv<T, false>;

    rocblas_int M = arg.M, incx = arg.incx;

    char char_uplo = arg.uplo, char_transA = arg.transA, char_diag = arg.diag;

    rocblas_fill         uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation    transA = char2rocblas_operation(char_transA);
    rocblas_diagonal     diag   = char2rocblas_diagonal(char_diag);
    rocblas_local_handle handle{arg};

    bool invalid_size = M < 0 || !incx;
    if(invalid_size || !M)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_tpmv_fn(handle, uplo, transA, diag, M, nullptr, nullptr, incx),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);

        return;
    }

    rocblas_int size_A = (M * (M + 1)) / 2;
    rocblas_int size_x, dim_x, abs_incx;
    dim_x    = M;
    abs_incx = incx >= 0 ? incx : -incx;
    size_x   = dim_x * abs_incx;

    host_vector<T> hA(size_A, 1);
    CHECK_HIP_ERROR(hA.memcheck());
    host_vector<T> hx(M, incx);
    CHECK_HIP_ERROR(hx.memcheck());
    device_vector<T> dA(size_A);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    device_vector<T> dx(size_x);
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    host_vector<T> hres(M, incx);
    CHECK_HIP_ERROR(hres.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(hA, arg, size_A, 1, 1, 0, 1, rocblas_client_never_set_nan, true);
    rocblas_init_vector(hx, arg, M, abs_incx, 0, 1, rocblas_client_never_set_nan, false, true);

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    /* =====================================================================
     ROCBLAS
     =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {

        //
        // ROCBLAS
        //
        CHECK_ROCBLAS_ERROR(rocblas_tpmv_fn(handle, uplo, transA, diag, M, dA, dx, incx));

        //
        // CPU BLAS
        //
        {
            cpu_time_used = get_time_us_no_sync();
            cblas_tpmv<T>(uplo, transA, diag, M, hA, hx, incx);
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // fetch GPU
        CHECK_HIP_ERROR(hres.transfer_from(dx));

        //
        // Unit check.
        //
        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_x, abs_incx, hx, hres);
        }

        //
        // Norm check.
        //
        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, dim_x, abs_incx, hx, hres);
        }
    }

    if(arg.timing)
    {
        //
        // Warmup
        //
        {
            int number_cold_calls = arg.cold_iters;
            for(int iter = 0; iter < number_cold_calls; iter++)
            {
                rocblas_tpmv_fn(handle, uplo, transA, diag, M, dA, dx, incx);
            }
        }

        //
        // Go !
        //
        {
            hipStream_t stream;
            CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
            gpu_time_used        = get_time_us_sync(stream); // in microseconds
            int number_hot_calls = arg.iters;
            for(int iter = 0; iter < number_hot_calls; iter++)
            {
                rocblas_tpmv_fn(handle, uplo, transA, diag, M, dA, dx, incx);
            }
            gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        }

        //
        // Log performance.
        //
        ArgumentModel<e_uplo, e_transA, e_diag, e_M, e_incx>{}.log_args<T>(rocblas_cout,
                                                                           arg,
                                                                           gpu_time_used,
                                                                           tpmv_gflop_count<T>(M),
                                                                           tpmv_gbyte_count<T>(M),
                                                                           cpu_time_used,
                                                                           rocblas_error);
    }
}
