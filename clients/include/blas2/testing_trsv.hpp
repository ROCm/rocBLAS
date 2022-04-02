/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
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

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 40

template <typename T>
void testing_trsv(const Arguments& arg)
{
    auto rocblas_trsv_fn = arg.fortran ? rocblas_trsv<T, true> : rocblas_trsv<T, false>;

    rocblas_int M           = arg.M;
    rocblas_int lda         = arg.lda;
    rocblas_int incx        = arg.incx;
    char        char_uplo   = arg.uplo;
    char        char_transA = arg.transA;
    char        char_diag   = arg.diag;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || lda < M || lda < 1 || !incx;
    if(invalid_size || !M)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_trsv_fn(handle, uplo, transA, diag, M, nullptr, lda, nullptr, incx),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    size_t abs_incx = size_t(incx >= 0 ? incx : -incx);
    size_t size_x   = M * abs_incx;

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(M, M, lda);
    host_matrix<T> hAAT(M, M, lda);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);

    // Allocate device memory
    device_matrix<T> dA(M, M, lda);
    device_vector<T> dx_or_b(size_x);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_or_b.memcheck());

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_vector(hx, arg, M, abs_incx, 0, 1, rocblas_client_never_set_nan, false, true);

    //  calculate hAAT = hA * hA ^ T or hAAT = hA * hA ^ H if complex
    cblas_gemm<T>(rocblas_operation_none,
                  rocblas_operation_conjugate_transpose,
                  M,
                  M,
                  M,
                  T(1.0),
                  hA,
                  lda,
                  hA,
                  lda,
                  T(0.0),
                  hAAT,
                  lda);

    //  copy hAAT into hA, make hA strictly diagonal dominant, and therefore SPD
    copy_hAAT_to_hA<T>((T*)hAAT, (T*)hA, M, size_t(lda));

    //  calculate Cholesky factorization of SPD (or Hermitian if complex) matrix hA
    cblas_potrf<T>(char_uplo, M, hA, lda);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(diag == rocblas_diagonal_unit)
    {
        make_unit_diagonal(uplo, (T*)hA, lda, M);
    }

    hb = hx;

    // Calculate hb = hA*hx;
    cblas_trmv<T>(uplo, transA, diag, M, hA, lda, hb, incx);
    cpu_x_or_b = hb; // cpuXorB <- B
    hx_or_b_1  = hb;
    hx_or_b_2  = hb;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;
    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    if(!ROCBLAS_REALLOC_ON_DEMAND)
    {
        // Compute size
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocblas_trsv_fn(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx));
        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        // Allocate memory
        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_1, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_trsv_fn(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx));
        CHECK_HIP_ERROR(hipMemcpy(hx_or_b_1, dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_2, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_trsv_fn(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx));
        CHECK_HIP_ERROR(hipMemcpy(hx_or_b_2, dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate norm 1 of vector E
        max_err_1 = rocblas_abs(vector_norm_1<T>(M, abs_incx, hx, hx_or_b_1));
        max_err_2 = rocblas_abs(vector_norm_1<T>(M, abs_incx, hx, hx_or_b_2));

        //unit test
        trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

        // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
        cblas_trmv<T>(uplo, transA, diag, M, hA, lda, hx_or_b_1, incx);
        cblas_trmv<T>(uplo, transA, diag, M, hA, lda, hx_or_b_2, incx);

        // Calculate norm 1 of vector res
        max_err_1 = rocblas_abs(vector_norm_1<T>(M, abs_incx, hx_or_b_1, hb));
        max_err_2 = rocblas_abs(vector_norm_1<T>(M, abs_incx, hx_or_b_1, hb));

        //unit test
        trsm_err_res_check<T>(max_err_1, M, residual_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, M, residual_eps_multiplier, eps);
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_1, sizeof(T) * size_x, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_trsv_fn(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_trsv_fn(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // CPU cblas
        cpu_time_used = get_time_us_no_sync();

        if(arg.norm_check)
            cblas_trsv<T>(uplo, transA, diag, M, hA, lda, cpu_x_or_b, incx);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        ArgumentModel<e_uplo, e_transA, e_diag, e_M, e_lda, e_incx>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            trsv_gflop_count<T>(M),
            ArgumentLogging::NA_value,
            cpu_time_used,
            max_err_1,
            max_err_2);
    }
}
