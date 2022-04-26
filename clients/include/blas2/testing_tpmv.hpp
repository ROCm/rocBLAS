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
void testing_tpmv_bad_arg(const Arguments& arg)
{
    auto rocblas_tpmv_fn = arg.fortran ? rocblas_tpmv<T, true> : rocblas_tpmv<T, false>;

    const rocblas_int       M      = 100;
    const rocblas_int       incx   = 1;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_fill      uplo   = rocblas_fill_lower;
    const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_matrix<T> dAp(1, rocblas_packed_matrix_size(M), 1);
    device_vector<T> dx(M, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    //
    // Checks.
    //
    EXPECT_ROCBLAS_STATUS(
        rocblas_tpmv_fn(handle, rocblas_fill_full, transA, diag, M, dAp, dx, incx),
        rocblas_status_invalid_value);
    // arg_checks code shared so transA, diag tested only in non-batched

    EXPECT_ROCBLAS_STATUS(
        rocblas_tpmv_fn(handle, uplo, (rocblas_operation)rocblas_fill_full, diag, M, dAp, dx, incx),
        rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(
        rocblas_tpmv_fn(
            handle, uplo, transA, (rocblas_diagonal)rocblas_fill_full, M, dAp, dx, incx),
        rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(rocblas_tpmv_fn(handle, uplo, transA, diag, M, nullptr, dx, incx),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_tpmv_fn(handle, uplo, transA, diag, M, dAp, nullptr, incx),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_tpmv_fn(nullptr, uplo, transA, diag, M, dAp, dx, incx),
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

    size_t abs_incx = incx >= 0 ? incx : -incx;

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_matrix<T> hA(M, M, M);
    host_matrix<T> hAp(1, rocblas_packed_matrix_size(M), 1);
    host_vector<T> hx(M, incx);
    host_vector<T> hres(M, incx);

    // Allocate device memory
    device_matrix<T> dAp(1, rocblas_packed_matrix_size(M), 1);
    device_vector<T> dx(M, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, M);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    /* =====================================================================
     ROCBLAS
     =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {

        // ROCBLAS
        CHECK_ROCBLAS_ERROR(rocblas_tpmv_fn(handle, uplo, transA, diag, M, dAp, dx, incx));

        // CPU BLAS
        {
            cpu_time_used = get_time_us_no_sync();
            cblas_tpmv<T>(uplo, transA, diag, M, hAp, hx, incx);
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // fetch GPU
        CHECK_HIP_ERROR(hres.transfer_from(dx));

        // Unit check.
        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, abs_incx, hx, hres);
        }

        // Norm check.
        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, M, abs_incx, hx, hres);
        }
    }

    if(arg.timing)
    {
        // Warmup
        {
            int number_cold_calls = arg.cold_iters;
            for(int iter = 0; iter < number_cold_calls; iter++)
            {
                rocblas_tpmv_fn(handle, uplo, transA, diag, M, dAp, dx, incx);
            }
        }

        // Go !
        {
            hipStream_t stream;
            CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
            gpu_time_used        = get_time_us_sync(stream); // in microseconds
            int number_hot_calls = arg.iters;
            for(int iter = 0; iter < number_hot_calls; iter++)
            {
                rocblas_tpmv_fn(handle, uplo, transA, diag, M, dAp, dx, incx);
            }
            gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        }

        // Log performance.
        ArgumentModel<e_uplo, e_transA, e_diag, e_M, e_incx>{}.log_args<T>(rocblas_cout,
                                                                           arg,
                                                                           gpu_time_used,
                                                                           tpmv_gflop_count<T>(M),
                                                                           tpmv_gbyte_count<T>(M),
                                                                           cpu_time_used,
                                                                           rocblas_error);
    }
}
