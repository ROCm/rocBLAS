/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_solve.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_tpsv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_tpsv_strided_batched_fn = arg.fortran ? rocblas_tpsv_strided_batched<T, true>
                                                       : rocblas_tpsv_strided_batched<T, false>;

    const rocblas_int       N           = 100;
    const rocblas_int       incx        = 1;
    const rocblas_stride    stride_x    = 200;
    const rocblas_stride    stride_a    = size_t(N) * N;
    const rocblas_int       batch_count = 3;
    const rocblas_operation transA      = rocblas_operation_none;
    const rocblas_fill      uplo        = rocblas_fill_lower;
    const rocblas_diagonal  diag        = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_strided_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1, stride_a, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());

    //
    // Checks.
    //
    EXPECT_ROCBLAS_STATUS(rocblas_tpsv_strided_batched_fn(handle,
                                                          rocblas_fill_full,
                                                          transA,
                                                          diag,
                                                          N,
                                                          dAp,
                                                          stride_a,
                                                          dx,
                                                          incx,
                                                          stride_x,
                                                          batch_count),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocblas_tpsv_strided_batched_fn(
            handle, uplo, transA, diag, N, nullptr, stride_a, dx, incx, stride_x, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_tpsv_strided_batched_fn(
            handle, uplo, transA, diag, N, dAp, stride_a, nullptr, incx, stride_x, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_tpsv_strided_batched_fn(
            nullptr, uplo, transA, diag, N, dAp, stride_a, dx, incx, stride_x, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_tpsv_strided_batched(const Arguments& arg)
{
    auto rocblas_tpsv_strided_batched_fn = arg.fortran ? rocblas_tpsv_strided_batched<T, true>
                                                       : rocblas_tpsv_strided_batched<T, false>;

    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    char        char_uplo   = arg.uplo;
    char        char_transA = arg.transA;
    char        char_diag   = arg.diag;
    rocblas_int stride_ap   = arg.stride_a;
    rocblas_int stride_x    = arg.stride_x;
    rocblas_int stride_a    = N * N;
    rocblas_int batch_count = arg.batch_count;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(rocblas_tpsv_strided_batched_fn(handle,
                                                              uplo,
                                                              transA,
                                                              diag,
                                                              N,
                                                              nullptr,
                                                              stride_ap,
                                                              nullptr,
                                                              incx,
                                                              stride_x,
                                                              batch_count),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    size_t abs_incx = size_t(incx >= 0 ? incx : -incx);

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_strided_batch_matrix<T> hA(N, N, N, stride_a, batch_count);
    host_strided_batch_matrix<T> hAp(1, rocblas_packed_matrix_size(N), 1, stride_ap, batch_count);
    host_strided_batch_matrix<T> AAT(N, N, N, stride_a, batch_count);
    host_strided_batch_vector<T> hb(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx_or_b_1(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx_or_b_2(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> cpu_x_or_b(N, incx, stride_x, batch_count);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hAp.memcheck());
    CHECK_HIP_ERROR(AAT.memcheck());
    CHECK_HIP_ERROR(hb.memcheck());
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hx_or_b_1.memcheck());
    CHECK_HIP_ERROR(hx_or_b_2.memcheck());
    CHECK_HIP_ERROR(cpu_x_or_b.memcheck());

    // Allocate device memory
    device_strided_batch_matrix<T> dAp(1, rocblas_packed_matrix_size(N), 1, stride_ap, batch_count);
    device_strided_batch_vector<T> dx_or_b(N, incx, stride_x, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dAp.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_or_b.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    //  calculate AAT = hA * hA ^ T or AAT = hA * hA ^ H if complex
    for(int b = 0; b < batch_count; b++)
    {
        prepare_triangular_solve((T*)hA[b], N, (T*)AAT[b], N, char_uplo);
        if(diag == rocblas_diagonal_unit)
        {
            make_unit_diagonal(uplo, (T*)hA[b], N, N);
        }
    }

    hb.copy_from(hx);

    // Calculate hb = hA*hx;
    for(int b = 0; b < batch_count; b++)
    {
        cblas_trmv<T>(uplo, transA, diag, N, hA[b], N, hb[b], incx);
    }

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, N);

    cpu_x_or_b.copy_from(hb);
    hx_or_b_1.copy_from(hb);
    hx_or_b_2.copy_from(hb);

    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;
    double gpu_time_used, cpu_time_used;
    double rocblas_error;
    double error_eps_multiplier    = 40.0;
    double residual_eps_multiplier = 20.0;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_tpsv_strided_batched_fn(
            handle, uplo, transA, diag, N, dAp, stride_ap, dx_or_b, incx, stride_x, batch_count));

        CHECK_HIP_ERROR(hx_or_b_1.transfer_from(dx_or_b));

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_2));

        CHECK_ROCBLAS_ERROR(rocblas_tpsv_strided_batched_fn(
            handle, uplo, transA, diag, N, dAp, stride_ap, dx_or_b, incx, stride_x, batch_count));

        CHECK_HIP_ERROR(hx_or_b_2.transfer_from(dx_or_b));

        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate norm 1 of vector E
        for(int b = 0; b < batch_count; b++)
        {
            max_err_1 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx[b], hx_or_b_1[b]));
            max_err_2 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx[b], hx_or_b_2[b]));

            //unit test
            trsm_err_res_check<T>(max_err_1, N, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, N, error_eps_multiplier, eps);
        }

        // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmv<T>(uplo, transA, diag, N, hA[b], N, hx_or_b_1[b], incx);
            cblas_trmv<T>(uplo, transA, diag, N, hA[b], N, hx_or_b_2[b], incx);
        }

        //calculate norm 1 of res
        for(int b = 0; b < batch_count; b++)
        {
            max_err_1 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx_or_b_1[b], hb[b]));
            max_err_2 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx_or_b_1[b], hb[b]));

            //unit test
            trsm_err_res_check<T>(max_err_1, N, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, N, error_eps_multiplier, eps);
        }
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        hx_or_b_1.copy_from(cpu_x_or_b);
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_tpsv_strided_batched_fn(handle,
                                            uplo,
                                            transA,
                                            diag,
                                            N,
                                            dAp,
                                            stride_ap,
                                            dx_or_b,
                                            incx,
                                            stride_x,
                                            batch_count);

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_tpsv_strided_batched_fn(handle,
                                            uplo,
                                            transA,
                                            diag,
                                            N,
                                            dAp,
                                            stride_ap,
                                            dx_or_b,
                                            incx,
                                            stride_x,
                                            batch_count);

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // CPU cblas
        cpu_time_used = get_time_us_no_sync();

        if(arg.norm_check)
            for(int b = 0; b < batch_count; b++)
                cblas_tpsv<T>(uplo, transA, diag, N, hAp[b], cpu_x_or_b[b], incx);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        ArgumentModel<e_uplo,
                      e_transA,
                      e_diag,
                      e_N,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         tpsv_gflop_count<T>(N),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         max_err_1,
                         max_err_2);
    }
}
