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
void testing_trsv_batched(const Arguments& arg)
{
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
    if(M < 0 || lda < M || !incx || batch_count <= 0)
    {
        device_vector<T*, 0, T> dA(1);
        device_vector<T*, 0, T> dx_or_b(1);

        if(!dA || !dx_or_b)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        if(batch_count == 0)
            CHECK_ROCBLAS_ERROR(rocblas_trsv_batched<T>(
                handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx, batch_count));
        else
            EXPECT_ROCBLAS_STATUS(
                rocblas_trsv_batched<T>(
                    handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx, batch_count),
                rocblas_status_invalid_size);
        return;
    }

    size_t size_A   = lda * size_t(M);
    size_t abs_incx = size_t(incx >= 0 ? incx : -incx);
    size_t size_x   = M * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> AAT[batch_count];
    host_vector<T> hb[batch_count];
    host_vector<T> hx[batch_count];
    host_vector<T> hx_or_b_1[batch_count];
    host_vector<T> hx_or_b_2[batch_count];
    host_vector<T> cpu_x_or_b[batch_count];

    for(int b = 0; b < batch_count; b++)
    {
        hA[b]         = host_vector<T>(size_A);
        AAT[b]        = host_vector<T>(size_A);
        hb[b]         = host_vector<T>(size_x);
        hx[b]         = host_vector<T>(size_x);
        hx_or_b_1[b]  = host_vector<T>(size_x);
        hx_or_b_2[b]  = host_vector<T>(size_x);
        cpu_x_or_b[b] = host_vector<T>(size_x);
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;
    T      error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    T      residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    T      eps                     = std::numeric_limits<T>::epsilon();

    // allocate memory on device
    device_vector<T*, 0, T> dA(batch_count); //(size_A);
    device_vector<T*, 0, T> dx_or_b(batch_count); //(size_B);

    device_batch_vector<T> Av(batch_count, size_A);
    device_batch_vector<T> XorBv(batch_count, size_x);

    int last = batch_count - 1;
    if(!dA || !dx_or_b || (!Av[last] && size_A) || (!XorBv[last] && size_x))
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    for(int b = 0; b < batch_count; b++)
    {
        rocblas_init<T>(hA[b], M, M, lda);

        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T, T>(rocblas_operation_none,
                         rocblas_operation_transpose,
                         M,
                         M,
                         M,
                         1.0,
                         hA[b],
                         lda,
                         hA[b],
                         lda,
                         0.0,
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
                t += AAT[b][idx] > 0 ? AAT[b][idx] : -AAT[b][idx];
            }
            hA[b][i + i * lda] = t;
        }

        //  calculate Cholesky factorization of SPD matrix hA
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

        rocblas_init<T>(hx[b], 1, M, abs_incx);
        hb[b] = hx[b];

        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hb[b], incx);

        cpu_x_or_b[b] = hb[b]; // cpuXorB <- B
        hx_or_b_1[b]  = hb[b];
        hx_or_b_2[b]  = hb[b];
    }

    // 1. User intermediate arrays to access device memory from host
    for(int b = 0; b < batch_count; b++)
    {
        CHECK_HIP_ERROR(hipMemcpy(Av[b], hA[b], sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(XorBv[b], hx_or_b_1[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
    }
    // 2. Copy intermediate arrays into device arrays
    CHECK_HIP_ERROR(hipMemcpy(dA, Av, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_or_b, XorBv, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    T max_err_1 = 0.0;
    T max_err_2 = 0.0;
    T max_res_1 = 0.0;
    T max_res_2 = 0.0;
    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_trsv_batched<T>(
            handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx, batch_count));

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(hx_or_b_1[b], XorBv[b], sizeof(T) * size_x, hipMemcpyDeviceToHost));
        }

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(XorBv[b], hx_or_b_2[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(dx_or_b, XorBv, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_trsv_batched<T>(
            handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx, batch_count));

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(hx_or_b_2[b], XorBv[b], sizeof(T) * size_x, hipMemcpyDeviceToHost));
        }

        for(int b = 0; b < batch_count; b++)
        {
            max_err_1 = max_err_2 = 0;
            T err_1               = 0.0;
            T err_2               = 0.0;

            for(int i = 0; i < M; i++)
            {
                int idx = i * abs_incx;
                if(hx[b][idx] != 0)
                {
                    err_1 += std::abs((hx[b][idx] - hx_or_b_1[b][idx]) / hx[b][idx]);
                    err_2 += std::abs((hx[b][idx] - hx_or_b_2[b][idx]) / hx[b][idx]);
                }
                else
                {
                    err_1 += std::abs(hx_or_b_1[b][idx]);
                    err_2 += std::abs(hx_or_b_2[b][idx]);
                }
            }
            max_err_1 = max_err_1 > err_1 ? max_err_1 : err_1;
            max_err_2 = max_err_2 > err_2 ? max_err_2 : err_2;
            trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);
        }

        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hx_or_b_1[b], incx);
            cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hx_or_b_2[b], incx);
        }

        // hx_or_b contains A * (calculated X), so residual = A * (calculated x) - b
        //                                                  = hx_or_b - hb
        // res is the one norm of the scaled residual for each column
        for(int b = 0; b < batch_count; b++)
        {
            max_res_1 = max_res_2 = 0;
            T res_1               = 0.0;
            T res_2               = 0.0;

            for(int i = 0; i < M; i++)
            {
                int idx = i * abs_incx;
                if(hb[b][idx] != 0)
                {
                    res_1 += std::abs((hx_or_b_1[b][idx] - hb[b][idx]) / hb[b][idx]);
                    res_2 += std::abs((hx_or_b_2[b][idx] - hb[b][idx]) / hb[b][idx]);
                }
                else
                {
                    res_1 += std::abs(hx_or_b_1[b][idx]);
                    res_2 += std::abs(hx_or_b_2[b][idx]);
                }
            }

            max_res_1 = max_res_1 > res_1 ? max_res_1 : res_1;
            max_res_2 = max_res_2 > res_2 ? max_res_2 : res_2;

            trsm_err_res_check<T>(max_res_1, M, residual_eps_multiplier, eps);
            trsm_err_res_check<T>(max_res_2, M, residual_eps_multiplier, eps);
        }
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(XorBv[b], hx_or_b_1[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(dx_or_b, XorBv, sizeof(T) * batch_count, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_trsv_batched<T>(
                handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx, batch_count);

        gpu_time_used = get_time_us(); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_trsv_batched<T>(
                handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx, batch_count);

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
        std::cout << "M,lda,incx,uplo,transA,diag,batch_count,rocblas-Gflops,us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << M << ',' << lda << ',' << incx << ',' << char_uplo << ',' << char_transA << ','
                  << char_diag << ',' << batch_count << ',' << rocblas_gflops << ","
                  << gpu_time_used / number_hot_calls;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                      << max_err_2;

        std::cout << std::endl;
    }
}
