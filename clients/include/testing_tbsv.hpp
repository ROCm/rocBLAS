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

#define ERROR_EPS_MULTIPLIER 60
#define RESIDUAL_EPS_MULTIPLIER 40

template <typename T>
void testing_tbsv(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int K           = arg.K;
    rocblas_int lda         = arg.lda;
    rocblas_int incx        = arg.incx;
    char        char_uplo   = arg.uplo;
    char        char_transA = arg.transA;
    char        char_diag   = arg.diag;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle;

    // check here to prevent undefined memory allocation error
    if(N < 0 || K < 0 || lda < K + 1 || !incx)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_tbsv<T>(handle, uplo, transA, diag, N, K, nullptr, lda, nullptr, incx),
            rocblas_status_invalid_size);
        return;
    }

    size_t size_A   = size_t(lda) * size_t(N);
    size_t abs_incx = size_t(incx >= 0 ? incx : -incx);
    size_t size_x   = N * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hAB(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;
    double error_eps_multiplier    = 40.0;
    double residual_eps_multiplier = 40.0;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx_or_b(size_x);

    rocblas_init<T>(hA, true);

    // initialize "exact" answer hx
    rocblas_init<T>(hx, false);
    hb = hx;

    // Made hA a banded matrix with k sub/super-diagonals
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if(uplo == rocblas_fill_upper)
                if(j > K + i || i > j)
                    hA[j * lda + i] = T(0);
                else if(i > K + j || j > i)
                    hA[j * lda + i] = T(0);
        }
    }

    //  calculate AAT = hA * hA ^ T or AAT = hA * hA ^ H if complex
    cblas_gemm<T>(rocblas_operation_none,
                  rocblas_operation_conjugate_transpose,
                  N,
                  N,
                  N,
                  T(1.0),
                  hA,
                  lda,
                  hA,
                  lda,
                  T(0.0),
                  AAT,
                  lda);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < N; i++)
    {
        T t = 0.0;
        for(int j = 0; j < N; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += rocblas_abs(AAT[i + j * lda]);
        }
        hA[i + i * lda] = t;
    }

    //  calculate Cholesky factorization of SPD (or hermitian if complex) matrix hA
    cblas_potrf<T>(char_uplo, N, hA, lda);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        if('L' == char_uplo || 'l' == char_uplo)
            for(int i = 0; i < N; i++)
            {
                T diag = hA[i + i * lda];
                for(int j = 0; j <= i; j++)
                    hA[i + j * lda] = hA[i + j * lda] / diag;
            }
        else
            for(int j = 0; j < N; j++)
            {
                T diag = hA[j + j * lda];
                for(int i = 0; i <= j; i++)
                    hA[i + j * lda] = hA[i + j * lda] / diag;
            }

        for(int i = 0; i < N; i++)
            rocblas_init<T>(hA + i * lda + i, 1, 1, 1);
    }

    // convert regular hA matrix to banded hAB matrix
    if(uplo == rocblas_fill_upper)
    {
        for(int j = 0; j < N; j++)
        {
            // Move bands of hA into new banded hAB format.
            rocblas_int m = K - j;
            for(int i = std::max(0, j - K); i <= j; i++)
            {
                hAB[j * lda + (m + i)] = hA[j * lda + i];
            }

            // fill in bottom with random data
            // to ensure we aren't using it.
            for(int i = K + 1; i < N; i++)
            {
                rocblas_init<T>(hAB + j * lda + i, 1, 1, 1);
            }

            // fill top left triangle with random data
            // to ensure we aren't using it.
            for(int i = 0; i < m; i++)
            {
                rocblas_init<T>(hAB + j * lda + i, 1, 1, 1);
            }
        }
    }
    else
    {
        for(int j = 0; j < N; j++)
        {
            // Move bands of hA into new banded hAB format.
            for(int i = j; i <= std::min(N - 1, j + K); i++)
            {
                hAB[j * lda + (i - j)] = hA[j * lda + i];
            }

            // fill in bottom rows and bottom right triangle
            // with random data to ensure we aren't using it.
            rocblas_int m = std::min(K + 1, N - j);
            for(int i = m; i < N; i++)
            {
                rocblas_init<T>(hAB + j * lda + i, 1, 1, 1);
            }
        }
    }

    // std::cout << "N: " << N << ", K: " << K << "\n";
    // std::cout << "hAB: \n";
    // for(int i = 0; i < N; i++)
    // {
    //     for(int j = 0; j < N; j++)
    //     {
    //         std::cout << hAB[j * lda + i] << " ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n\n";

    cblas_tbmv<T>(uplo, transA, diag, N, K, hAB, lda, hb, incx);
    cpu_x_or_b = hb;
    hx_or_b_1  = hb;
    hx_or_b_2  = hb;

    // std::cout << "hx:\n";
    // for(int i = 0; i < N; i++)
    // {
    //     std::cout << hb[i * incx] << " ";
    // }
    // std::cout << "\n\n";
    CHECK_HIP_ERROR(dA.transfer_from(hAB));

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

        CHECK_ROCBLAS_ERROR(
            rocblas_tbsv<T>(handle, uplo, transA, diag, N, K, dA, lda, dx_or_b, incx));

        CHECK_HIP_ERROR(hx_or_b_1.transfer_from(dx_or_b));

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_2));

        CHECK_ROCBLAS_ERROR(
            rocblas_tbsv<T>(handle, uplo, transA, diag, N, K, dA, lda, dx_or_b, incx));
        CHECK_HIP_ERROR(hx_or_b_2.transfer_from(dx_or_b));
        // if(incx > 0) {
        //         std::cout << "gpu:\n";
        //         for(int i = 0; i < N; i++)
        //             std::cout << hx_or_b_1[i * incx] << " ";
        //         std::cout << "\ncpu:\n";
        //         for(int i = 0; i < N; i++)
        //             std::cout << hx[i * incx] << " ";
        //         std::cout << "\n";
        // }
        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate norm 1 of vector E
        max_err_1 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx, hx_or_b_1));
        max_err_2 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx, hx_or_b_2));

        //unit test
        trsm_err_res_check<T>(max_err_1, N, error_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, N, error_eps_multiplier, eps);

        // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
        cblas_tbmv<T>(uplo, transA, diag, N, K, hAB, lda, hx_or_b_1, incx);
        cblas_tbmv<T>(uplo, transA, diag, N, K, hAB, lda, hx_or_b_2, incx);

        // Calculate norm 1 of vector res
        max_err_1 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx_or_b_1, hb));
        max_err_2 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx_or_b_1, hb));

        //unit test
        trsm_err_res_check<T>(max_err_1, N, residual_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, N, residual_eps_multiplier, eps);
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_tbsv<T>(handle, uplo, transA, diag, N, K, dA, lda, dx_or_b, incx);

        gpu_time_used = get_time_us(); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_tbsv<T>(handle, uplo, transA, diag, N, K, dA, lda, dx_or_b, incx);

        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = tbsv_gflop_count<T>(N) * number_hot_calls / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        if(arg.norm_check)
            cblas_tbsv<T>(uplo, transA, diag, N, K, hAB, lda, cpu_x_or_b, incx);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = tbsv_gflop_count<T>(N) / cpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "N,K,lda,incx,uplo,transA,diag,rocblas-Gflops,us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << N << ',' << K << ',' << lda << ',' << incx << ',' << char_uplo << ','
                  << char_transA << ',' << char_diag << ',' << rocblas_gflops << ","
                  << gpu_time_used / number_hot_calls;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                      << max_err_2;

        std::cout << std::endl;
    }
}
