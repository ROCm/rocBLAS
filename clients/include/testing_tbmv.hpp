/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

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
void full_matrix_to_band_matrix(
    int n, int k, int lda, rocblas_fill uplo, host_vector<T> full, host_vector<T>& banded)
{
    if(uplo == rocblas_fill_upper)
    {
        for(int j = 1; j <= n; j++)
        {
            int m = k + 1 - j;
            for(int i = std::max(1, j - k); i <= j; i++)
            {
                banded[(j - 1) * lda + m + (i - 1)] = full[(j - 1) * lda + (i - 1)];
            }
        }
    }
    else if(uplo == rocblas_fill_lower)
    {
        for(int j = 1; j <= n; j++)
        {
            int m = 1 - j;
            for(int i = j; i <= std::min(n, j + k); i++)
            {
                banded[(j - 1) * lda + m + (i - 1)] = full[(j - 1) * lda + (i - 1)];
            }
        }
    }
}

template <typename T>
void testing_tbmv_bad_arg(const Arguments& arg)
{
    // const rocblas_int M    = 100;
    // const rocblas_int N    = 100;
    // const rocblas_int lda  = 100;
    // const rocblas_int incx = 1;
    // const rocblas_int incy = 1;
    // T                 alpha;
    // T                 beta;
    // alpha = beta = 1.0;

    // const rocblas_operation transA = rocblas_operation_none;

    // rocblas_local_handle handle;

    // size_t size_A = lda * size_t(N);
    // size_t size_x = N * size_t(incx);
    // size_t size_y = M * size_t(incy);

    // // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    // host_vector<T> hA(size_A);
    // host_vector<T> hx(size_x);
    // host_vector<T> hy(size_y);

    // // Initial Data on CPU
    // rocblas_seedrand();
    // rocblas_init<T>(hA, M, N, lda);
    // rocblas_init<T>(hx, 1, N, incx);
    // rocblas_init<T>(hy, 1, M, incy);

    // // allocate memory on device
    // device_vector<T> dA(size_A);
    // device_vector<T> dx(size_x);
    // device_vector<T> dy(size_y);
    // if(!dA || !dx || !dy)
    // {
    //     CHECK_HIP_ERROR(hipErrorOutOfMemory);
    //     return;
    // }

    // // copy data from CPU to device
    // CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    // CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    // CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

    // EXPECT_ROCBLAS_STATUS(
    //     rocblas_gemv<T>(handle, transA, M, N, &alpha, nullptr, lda, dx, incx, &beta, dy, incy),
    //     rocblas_status_invalid_pointer);

    // EXPECT_ROCBLAS_STATUS(
    //     rocblas_gemv<T>(handle, transA, M, N, &alpha, dA, lda, nullptr, incx, &beta, dy, incy),
    //     rocblas_status_invalid_pointer);

    // EXPECT_ROCBLAS_STATUS(
    //     rocblas_gemv<T>(handle, transA, M, N, &alpha, dA, lda, dx, incx, &beta, nullptr, incy),
    //     rocblas_status_invalid_pointer);

    // EXPECT_ROCBLAS_STATUS(
    //     rocblas_gemv<T>(handle, transA, M, N, nullptr, dA, lda, dx, incx, &beta, dy, incy),
    //     rocblas_status_invalid_pointer);

    // EXPECT_ROCBLAS_STATUS(
    //     rocblas_gemv<T>(handle, transA, M, N, &alpha, dA, lda, dx, incx, nullptr, dy, incy),
    //     rocblas_status_invalid_pointer);

    // EXPECT_ROCBLAS_STATUS(
    //     rocblas_gemv<T>(nullptr, transA, M, N, &alpha, dA, lda, dx, incx, &beta, dy, incy),
    //     rocblas_status_invalid_handle);
}

template <typename T>
void testing_tbmv(const Arguments& arg)
{
    rocblas_int       M         = arg.M;
    rocblas_int       K         = arg.K;
    rocblas_int       lda       = arg.lda;
    rocblas_int       incx      = arg.incx;
    char              char_uplo = arg.uplo;
    rocblas_fill      uplo      = char2rocblas_fill(char_uplo);
    rocblas_operation transA    = char2rocblas_operation(arg.transA);
    rocblas_diagonal  diag      = char2rocblas_diagonal(arg.diag);

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(M < 0 || K < 0 || lda < M || lda < 1 || !incx)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dA1(safe_size);
        device_vector<T>    dx1(safe_size);
        if(!dA1 || !dx1)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(
            rocblas_tbmv<T>(handle, uplo, transA, diag, M, K, dA1, lda, dx1, incx),
            rocblas_status_invalid_size);

        return;
    }

    size_t size_A = lda * size_t(M);
    size_t size_x, abs_incx;

    abs_incx = incx >= 0 ? incx : -incx;
    size_x   = M * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    // host_vector<T> hA(size_A);
    host_vector<T> hA_reg(size_A);
    host_vector<T> hx(size_x);
    host_vector<T> hx_1(size_x);
    host_vector<T> hx_gold(size_x);

    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    if((!dA && size_A) || (!dx && size_x))
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA_reg, M, M, lda);
    rocblas_init<T>(hx, 1, M, abs_incx);
    hx_gold = hx;

    // make hA_reg a banded matrix with k sub/super diagonals
    // for(int i = 0; i < M; i++)
    //     for(int j = 0; j < M; j++)
    //         if(j > i + K || j < i)
    //         {
    //             int idx     = ('U' == char_uplo || 'u' == char_uplo) ? i + j * lda : j + i * lda;
    //             hA_reg[idx] = 0;
    //         }

    //  TODO: make hA unit diagonal if diag == rocblas_diagonal_unit
    // if(char_diag == 'U' || char_diag == 'u')
    // {
    //     if('L' == char_uplo || 'l' == char_uplo)
    //         for(int i = 0; i < M; i++)
    //         {
    //             T diag = hA_reg[i + i * lda];
    //             for(int j = 0; j <= i; j++)
    //                 hA_reg[i + j * lda] = hA_reg[i + j * lda] / diag;
    //         }
    //     else
    //         for(int j = 0; j < M; j++)
    //         {
    //             T diag = hA_reg[j + j * lda];
    //             for(int i = 0; i <= j; i++)
    //                 hA_reg[i + j * lda] = hA_reg[i + j * lda] / diag;
    //         }
    // }
    // full_matrix_to_band_matrix(M, K, lda, uplo, hA_reg, hA);
    // std::cout << "ha\n";
    // for(int i = 0; i < M; i++)
    // {
    //     for(int j = 0; j < M; j++)
    //     {
    //         std::cout << hA_reg[j * lda + i] << " ";
    //         // if(hA_reg[j * lda + i] != 0)
    //         // std::cout << "(" << j << ", " << i << ") = " << hA[j * lda + i] << "\n";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "-----\nhA\n";
    std::cout << "\nx:\n";
    for(int i = 0; i < M; i++)
    {
        std::cout << hx[i * incx] << " ";
    }
    std::cout << "\n\n";

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA_reg, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_tbmv<T>(handle, uplo, transA, diag, M, K, dA, lda, dx, incx));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx_1, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        // std::cout << "uplo: " << uplo << ", transA: " << transA << ", diag: " << diag << ", M: " << M << ", K: " << K << ", lda: " << lda << ", incx: " << incx << "\n";
        cblas_tbmv<T>(uplo, transA, diag, M, K, hA_reg, lda, hx_gold, incx);
        // std::cout << "done cblas\n";

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = tbmv_gflop_count<T>(M, K) / cpu_time_used * 1e6;

        std::cout << "gpu\n";
        for(int i = 0; i < M; i++)
        {
            std::cout << hx_1[i * incx] << " ";
        }
        std::cout << "\n-----\ncpu\n";
        for(int i = 0; i < M; i++)
        {
            std::cout << hx_gold[i * incx] << " ";
        }
        std::cout << "\n----\n";
        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, abs_incx, hx_gold, hx_1);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, M, abs_incx, hx_gold, hx_1);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_tbmv<T>(handle, uplo, transA, diag, M, K, dA, lda, dx, incx);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_tbmv<T>(handle, uplo, transA, diag, M, K, dA, lda, dx, incx);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = tbmv_gflop_count<T>(M, K) / gpu_time_used * 1e6;
        rocblas_bandwidth = (1.0 * M * M) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "M,K,lda,incx,rocblas-Gflops,rocblas-GB/s,";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,norm_error_device_ptr";
        }
        std::cout << std::endl;

        std::cout << M << "," << K << "," << lda << "," << incx << "," << rocblas_gflops << ","
                  << rocblas_bandwidth << ",";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ',';
            std::cout << rocblas_error_1;
        }

        std::cout << std::endl;
    }
}
