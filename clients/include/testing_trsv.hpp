/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
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
#define RESIDUAL_EPS_MULTIPLIER 20

template <typename T>
void testing_trsv(const Arguments& arg)
{
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
    rocblas_local_handle handle;

    // check here to prevent undefined memory allocation error
    if(M < 0 || lda < M || !incx)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dx_or_b(safe_size);
        device_vector<T>    dA(safe_size);

        if(!dA || !dx_or_b)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_trsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx),
            rocblas_status_invalid_size);
        return;
    }

    size_t size_A   = size_t(lda) * size_t(M);
    size_t abs_incx = size_t(incx >= 0 ? incx : -incx);
    size_t size_x   = M * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);
    host_vector<T> my_cpu_x_or_b(size_x);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;
    T      error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    T      residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    T      eps                     = std::numeric_limits<T>::epsilon();

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx_or_b(size_x);

    rocblas_init<T>(hA, M, M, lda);

    //  calculate AAT = hA * hA ^ T
    cblas_gemm<T, T>(rocblas_operation_none,
                     rocblas_operation_transpose,
                     M,
                     M,
                     M,
                     1.0,
                     hA,
                     lda,
                     hA,
                     lda,
                     0.0,
                     AAT,
                     lda);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < M; i++)
    {
        T t = 0.0;
        for(int j = 0; j < M; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += AAT[i + j * lda] > 0 ? AAT[i + j * lda] : -AAT[i + j * lda];
        }
        hA[i + i * lda] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf<T>(char_uplo, M, hA, lda);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        if('L' == char_uplo || 'l' == char_uplo)
            for(int i = 0; i < M; i++)
            {
                T diag = hA[i + i * lda];
                for(int j = 0; j <= i; j++)
                    hA[i + j * lda] = hA[i + j * lda] / diag;
            }
        else
            for(int j = 0; j < M; j++)
            {
                T diag = hA[j + j * lda];
                for(int i = 0; i <= j; i++)
                    hA[i + j * lda] = hA[i + j * lda] / diag;
            }
    }

    rocblas_init<T>(hx, 1, M, abs_incx);
    hb = hx;

    // Calculate hb = hA*hx;
    cblas_trmv<T>(uplo, transA, diag, M, hA, lda, hb, incx);
    cpu_x_or_b    = hb; // cpuXorB <- B
    hx_or_b_1     = hb;
    hx_or_b_2     = hb;
    my_cpu_x_or_b = hb;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_1, sizeof(T) * size_x, hipMemcpyHostToDevice));
    T max_err_1 = 0.0;
    T max_err_2 = 0.0;
    T max_res_1 = 0.0;
    T max_res_2 = 0.0;
    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_trsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx));
        CHECK_HIP_ERROR(hipMemcpy(hx_or_b_1, dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_2, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_trsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx));
        CHECK_HIP_ERROR(hipMemcpy(hx_or_b_2, dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        T err_1 = 0.0;
        T err_2 = 0.0;
        for(int i = 0; i < M; i++)
            if(hx[i * abs_incx] != 0)
            {
                err_1 += std::abs((hx[i * abs_incx] - hx_or_b_1[i * abs_incx]) / hx[i * abs_incx]);
                err_2 += std::abs((hx[i * abs_incx] - hx_or_b_2[i * abs_incx]) / hx[i * abs_incx]);
            }
            else
            {
                err_1 += std::abs(hx_or_b_1[i * abs_incx]);
                err_2 += std::abs(hx_or_b_2[i * abs_incx]);
            }
        max_err_1 = max_err_1 > err_1 ? max_err_1 : err_1;
        max_err_2 = max_err_2 > err_2 ? max_err_2 : err_2;
        trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

        cblas_trmv<T>(uplo, transA, diag, M, hA, lda, hx_or_b_1, incx);
        cblas_trmv<T>(uplo, transA, diag, M, hA, lda, hx_or_b_2, incx);
        // hx_or_b contains A * (calculated X), so residual = A * (calculated x) - b
        //                                                  = hx_or_b - hb
        // res is the one norm of the scaled residual for each column

        T res_1 = 0.0;
        T res_2 = 0.0;
        for(int i = 0; i < M; i++)
            if(hb[i * abs_incx] != 0)
            {
                res_1 += std::abs((hx_or_b_1[i * abs_incx] - hb[i * abs_incx]) / hb[i * abs_incx]);
                res_2 += std::abs((hx_or_b_2[i * abs_incx] - hb[i * abs_incx]) / hb[i * abs_incx]);
            }
            else
            {
                res_1 += std::abs(hx_or_b_1[i * abs_incx]);
                res_2 += std::abs(hx_or_b_2[i * abs_incx]);
            }
        max_res_1 = max_res_1 > res_1 ? max_res_1 : res_1;
        max_res_2 = max_res_2 > res_2 ? max_res_2 : res_2;

        trsm_err_res_check<T>(max_res_1, M, residual_eps_multiplier, eps);
        trsm_err_res_check<T>(max_res_2, M, residual_eps_multiplier, eps);
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(hipMemcpy(dx_or_b, hx_or_b_1, sizeof(T) * size_x, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_trsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);

        gpu_time_used = get_time_us(); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_trsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);

        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = trsv_gflop_count<T>(M) * number_hot_calls / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        if(arg.norm_check)
            cblas_trsv<T>(uplo, transA, diag, M, hA, lda, cpu_x_or_b, incx);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = trsv_gflop_count<T>(M) / cpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "M,lda,incx,uplo,transA,diag,rocblas-Gflops,us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << M << ',' << lda << ',' << incx << ',' << char_uplo << ',' << char_transA << ','
                  << char_diag << ',' << rocblas_gflops << "," << gpu_time_used / number_hot_calls;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                      << max_err_2;

        std::cout << std::endl;
    }
}
