/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 40

template <typename T>
void testing_trsv_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_trsv_batched_fn
        = FORTRAN ? rocblas_trsv_batched<T, true> : rocblas_trsv_batched<T, false>;

    rocblas_int M           = arg.M;
    rocblas_int lda         = arg.lda;
    rocblas_int incx        = arg.incx;
    char        char_uplo   = arg.uplo;
    char        char_transA = arg.transA;
    char        char_diag   = arg.diag;
    rocblas_int batch_count = arg.batch_count;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle;

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || lda < M || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !M || !batch_count)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_trsv_batched_fn(
                handle, uplo, transA, diag, M, nullptr, lda, nullptr, incx, batch_count),
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    size_t size_A   = lda * size_t(M);
    size_t abs_incx = size_t(incx >= 0 ? incx : -incx);
    size_t size_x   = M * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> AAT(size_A, 1, batch_count);
    host_batch_vector<T> hb(size_x, 1, batch_count);
    host_batch_vector<T> hx(size_x, 1, batch_count);
    host_batch_vector<T> hx_or_b_1(size_x, 1, batch_count);
    host_batch_vector<T> hx_or_b_2(size_x, 1, batch_count);
    host_batch_vector<T> cpu_x_or_b(size_x, 1, batch_count);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;
    double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dx_or_b(M, incx, batch_count);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx_or_b.memcheck());

    for(int b = 0; b < batch_count; b++)
    {
        rocblas_init<T>(hA[b], M, M, lda);

        //  calculate AAT = hA * hA ^ T or AAT = hA * hA ^ H if complex
        cblas_gemm<T>(rocblas_operation_none,
                      rocblas_operation_conjugate_transpose,
                      M,
                      M,
                      M,
                      T(1.0),
                      hA[b],
                      lda,
                      hA[b],
                      lda,
                      T(0.0),
                      AAT[b],
                      lda);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < M; i++)
        {
            T t = 0.0;
            for(int j = 0; j < M; j++)
            {
                int idx    = i + j * lda;
                hA[b][idx] = AAT[b][idx];
                t += rocblas_abs(AAT[b][idx]);
            }
            hA[b][i + i * lda] = t;
        }

        //  calculate Cholesky factorization of SPD (or hermitian if complex) matrix hA
        cblas_potrf<T>(char_uplo, M, hA[b], lda);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(char_diag == 'U' || char_diag == 'u')
        {
            if('L' == char_uplo || 'l' == char_uplo)
            {
                for(int i = 0; i < M; i++)
                {
                    T diag = hA[b][i + i * lda];
                    for(int j = 0; j <= i; j++)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / diag;
                }
            }
            else
            {
                for(int j = 0; j < M; j++)
                {
                    T diag = hA[b][j + j * lda];
                    for(int i = 0; i <= j; i++)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / diag;
                }
            }
        }
    }

    rocblas_init(hx, false);
    hb.copy_from(hx);

    for(int b = 0; b < batch_count; b++)
    {
        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hb[b], incx);
    }

    cpu_x_or_b.copy_from(hb);
    hx_or_b_1.copy_from(hb);
    hx_or_b_2.copy_from(hb);

    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_trsv_batched_fn(handle,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    M,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dx_or_b.ptr_on_device(),
                                                    incx,
                                                    batch_count));

        CHECK_HIP_ERROR(hx_or_b_1.transfer_from(dx_or_b));

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_2));

        CHECK_ROCBLAS_ERROR(rocblas_trsv_batched_fn(handle,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    M,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dx_or_b.ptr_on_device(),
                                                    incx,
                                                    batch_count));

        CHECK_HIP_ERROR(hx_or_b_2.transfer_from(dx_or_b));

        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate norm 1 of vector E
        for(int b = 0; b < batch_count; b++)
        {
            max_err_1 = rocblas_abs(vector_norm_1<T>(M, abs_incx, hx[b], hx_or_b_1[b]));
            max_err_2 = rocblas_abs(vector_norm_1<T>(M, abs_incx, hx[b], hx_or_b_2[b]));
            //unit test
            trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);
        }

        // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hx_or_b_1[b], incx);
            cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hx_or_b_2[b], incx);
        }

        //calculate norm 1 of res
        for(int b = 0; b < batch_count; b++)
        {
            max_err_1 = rocblas_abs(vector_norm_1<T>(M, abs_incx, hx_or_b_1[b], hb[b]));
            max_err_2 = rocblas_abs(vector_norm_1<T>(M, abs_incx, hx_or_b_1[b], hb[b]));
            //unit test
            trsm_err_res_check<T>(max_err_1, M, residual_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, residual_eps_multiplier, eps);
        }
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_trsv_batched_fn(handle,
                                    uplo,
                                    transA,
                                    diag,
                                    M,
                                    dA.ptr_on_device(),
                                    lda,
                                    dx_or_b.ptr_on_device(),
                                    incx,
                                    batch_count);

        gpu_time_used = get_time_us(); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_trsv_batched_fn(handle,
                                    uplo,
                                    transA,
                                    diag,
                                    M,
                                    dA.ptr_on_device(),
                                    lda,
                                    dx_or_b.ptr_on_device(),
                                    incx,
                                    batch_count);

        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops
            = batch_count * trsv_gflop_count<T>(M) * number_hot_calls / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        if(arg.norm_check)
            for(int b = 0; b < batch_count; b++)
                cblas_trsv<T>(uplo, transA, diag, M, hA[b], lda, cpu_x_or_b[b], incx);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * trsv_gflop_count<T>(M) / cpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "M,lda,incx,uplo,transA,diag,batch_count,rocblas-Gflops,us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        rocblas_cout << std::endl;

        rocblas_cout << M << ',' << lda << ',' << incx << ',' << char_uplo << ',' << char_transA
                     << ',' << char_diag << ',' << batch_count << ',' << rocblas_gflops << ","
                     << gpu_time_used / number_hot_calls;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                         << max_err_2;

        rocblas_cout << std::endl;
    }
}
