/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
void testing_tpsv_bad_arg(const Arguments& arg)
{
    auto rocblas_tpsv_fn = arg.api & c_API_FORTRAN ? rocblas_tpsv<T, true> : rocblas_tpsv<T, false>;

    auto rocblas_tpsv_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_tpsv_64<T, true> : rocblas_tpsv_64<T, false>;

    const int64_t           N      = 100;
    const int64_t           incx   = 1;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_fill      uplo   = rocblas_fill_lower;
    const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1);
    device_vector<T> dx(N, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    // Checks
    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_tpsv_fn,
                (nullptr, uplo, transA, diag, N, dAp, dx, incx));

    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_tpsv_fn,
                (handle, rocblas_fill_full, transA, diag, N, dAp, dx, incx));

    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_tpsv_fn,
                (handle, uplo, (rocblas_operation)rocblas_fill_full, diag, N, dAp, dx, incx));

    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_tpsv_fn,
                (handle, uplo, transA, (rocblas_diagonal)rocblas_fill_full, N, dAp, dx, incx));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_tpsv_fn,
                (handle, uplo, transA, diag, N, nullptr, dx, incx));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_tpsv_fn,
                (handle, uplo, transA, diag, N, dAp, nullptr, incx));
}

template <typename T>
void testing_tpsv(const Arguments& arg)
{
    auto rocblas_tpsv_fn = arg.api & c_API_FORTRAN ? rocblas_tpsv<T, true> : rocblas_tpsv<T, false>;

    auto rocblas_tpsv_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_tpsv_64<T, true> : rocblas_tpsv_64<T, false>;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    char    char_uplo   = arg.uplo;
    char    char_transA = arg.transA;
    char    char_diag   = arg.diag;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = N < 0 || !incx;
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_tpsv_fn,
                    (handle, uplo, transA, diag, N, nullptr, nullptr, incx));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_matrix<T> hA(N, N, N);
    host_matrix<T> hAp(1, rocblas_packed_matrix_size(N), 1);
    host_vector<T> hb(N, incx);
    host_vector<T> hx(N, incx);
    host_vector<T> hx_or_b(N, incx);
    host_vector<T> cpu_x_or_b(N, incx);

    // Allocate device memory
    device_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1);
    device_vector<T> dx_or_b(N, incx);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_or_b.memcheck());

    // Initialize hA on host memory
    rocblas_init_matrix(hA,
                        arg,
                        rocblas_client_never_set_nan,
                        rocblas_client_diagonally_dominant_triangular_matrix,
                        true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    if(diag == rocblas_diagonal_unit)
    {
        make_unit_diagonal(uplo, (T*)hA, N, N);
    }

    hb = hx;

    // Calculate hb = hA*hx;
    ref_trmv<T>(uplo, transA, diag, N, hA, N, hb, incx);
    cpu_x_or_b = hb; // cpuXorB <- B
    hx_or_b    = hb;

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, (T*)hA, (T*)hAp, N);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b));

    double max_err                 = 0.0;
    double max_res                 = 0.0;
    double error_eps_multiplier    = 40.0;
    double residual_eps_multiplier = 20.0;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();
    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b
        handle.pre_test(arg);
        DAPI_CHECK(rocblas_tpsv_fn, (handle, uplo, transA, diag, N, dAp, dx_or_b, incx));
        handle.post_test(arg);
        CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));

        if(arg.repeatability_check)
        {
            host_vector<T> hx_or_b_copy(N, incx);
            CHECK_HIP_ERROR(hx_or_b_copy.memcheck());
            for(int i = 0; i < arg.iters; i++)
            {
                CHECK_HIP_ERROR(dx_or_b.transfer_from(hb));
                DAPI_CHECK(rocblas_tpsv_fn, (handle, uplo, transA, diag, N, dAp, dx_or_b, incx));
                CHECK_HIP_ERROR(hx_or_b_copy.transfer_from(dx_or_b));
                unit_check_general<T>(1, N, incx, hx_or_b, hx_or_b_copy);
            }
            return;
        }

        max_err = rocblas_abs(vector_norm_1<T>(N, incx, hx, hx_or_b));

        if(arg.unit_check)
            trsm_err_res_check(max_err, N, error_eps_multiplier, eps);

        ref_trmv<T>(uplo, transA, diag, N, hA, N, hx_or_b, incx);
        // hx_or_b contains A * (calculated X), so residual = A * (calculated x) - b
        //                                                  = hx_or_b - hb
        // res is the one norm of the scaled residual for each column

        max_res = rocblas_abs(vector_norm_1<T>(N, incx, hx_or_b, hb));

        if(arg.unit_check)
            trsm_err_res_check(max_res, N, residual_eps_multiplier, eps);
    }

    if(arg.timing)
    {
        double gpu_time_used, cpu_time_used = 0.0;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_tpsv_fn, (handle, uplo, transA, diag, N, dAp, dx_or_b, incx));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        if(arg.unit_check || arg.norm_check)
        {
            // CPU cblas
            cpu_time_used = get_time_us_no_sync();

            ref_tpsv<T>(uplo, transA, diag, N, hAp, cpu_x_or_b, incx);

            cpu_time_used = get_time_us_no_sync() - cpu_time_used; // in microseconds
        }

        ArgumentModel<e_uplo, e_transA, e_diag, e_N, e_incx>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            tpsv_gflop_count<T>(N),
            ArgumentLogging::NA_value,
            cpu_time_used,
            max_err,
            max_res);
    }
}
