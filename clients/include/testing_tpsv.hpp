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
#define RESIDUAL_EPS_MULTIPLIER 20

template <typename T>
host_vector<T> regular_to_packed(bool upper, host_vector<T> A, rocblas_int n)
{
    size_t         size_AP = size_t(n) * (n + 1) / 2;
    host_vector<T> AP(size_AP);

    int index = 0;
    if(upper)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }
    else
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = i; j < n; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }

    return AP;
}

template <typename T>
void testing_tpsv(const Arguments& arg)
{
    rocblas_int N           = arg.N;
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
    if(N < 0 || !incx)
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
        EXPECT_ROCBLAS_STATUS(rocblas_tpsv<T>(handle, uplo, transA, diag, N, dA, dx_or_b, incx),
                              rocblas_status_invalid_size);
        return;
    }

    size_t size_A   = size_t(N) * N;
    size_t size_AP  = size_t(N) * (N + 1) / 2.0;
    size_t abs_incx = size_t(incx >= 0 ? incx : -incx);
    size_t size_x   = N * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hAP(size_AP);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);
    host_vector<T> my_cpu_x_or_b(size_x);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;
    double error_eps_multiplier    = 1.1; //ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = 1.1; //RESIDUAL_EPS_MULTIPLIER;
    double eps                     = 5.0; //std::numeric_limits<rocblas_real_t<T>>::epsilon();

    // allocate memory on device
    device_vector<T> dAP(size_AP);
    device_vector<T> dx_or_b(size_x);
    CHECK_HIP_ERROR(dAP.memcheck());
    CHECK_HIP_ERROR(dx_or_b.memcheck());

    rocblas_init<T>(hA, true);

    //  calculate AAT = hA * hA ^ T
    cblas_gemm<T, T>(rocblas_operation_none,
                     rocblas_operation_conjugate_transpose,
                     N,
                     N,
                     N,
                     T(1.0),
                     hA,
                     N,
                     hA,
                     N,
                     T(0.0),
                     AAT,
                     N);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < N; i++)
    {
        T t = 0.0;
        for(int j = 0; j < N; j++)
        {
            hA[i + j * N] = AAT[i + j * N];
            t += rocblas_abs(AAT[i + j * N]);
        }
        hA[i + i * N] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf<T>(char_uplo, N, hA, N);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        // if('L' == char_uplo || 'l' == char_uplo)
        //     for(int i = 0; i < N; i++)
        //     {
        //         T diag = hA[i + i * N];
        //         for(int j = 0; j <= i; j++)
        //             hA[i + j * N] = hA[i + j * N] / diag;
        //     }
        // else
        //     for(int j = 0; j < N; j++)
        //     {
        //         T diag = hA[j + j * N];
        //         for(int i = 0; i <= j; i++)
        //             hA[i + j * N] = hA[i + j * N] / diag;
        //     }

        // for(int i = 0; i < N; i++)
        // {
        //     hA[i * N + i] = 5.0;
        // }
    }

    rocblas_init<T>(hx, 1, N, abs_incx);
    hb = hx;

    // Calculate hb = hA*hx;
    cblas_trmv<T>(uplo, transA, diag, N, hA, N, hb, incx);
    cpu_x_or_b    = hb; // cpuXorB <- B
    hx_or_b_1     = hb;
    hx_or_b_2     = hb;
    my_cpu_x_or_b = hb;

    hAP = regular_to_packed(uplo == rocblas_fill_upper, hA, N);

    std::cout << "ha reg:\n";
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            std::cout << hA[j * N + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\nha packed:\n";
    for(int i = 0; i < size_AP; i++)
    {
        std::cout << hAP[i] << " ";
    }
    std::cout << "\n";
    std::cout << "inputx: \n";
    for(int i = 0; i < N; i++)
    {
        std::cout << hx_or_b_1[i * incx] << " ";
    }
    std::cout << "\n";

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAP.transfer_from(hAP));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;
    double max_res_1 = 0.0;
    double max_res_2 = 0.0;
    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_tpsv<T>(handle, uplo, transA, diag, N, dAP, dx_or_b, incx));
        CHECK_HIP_ERROR(hx_or_b_1.transfer_from(dx_or_b));

        // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_2));

        CHECK_ROCBLAS_ERROR(rocblas_tpsv<T>(handle, uplo, transA, diag, N, dAP, dx_or_b, incx));
        CHECK_HIP_ERROR(hx_or_b_2.transfer_from(dx_or_b));

        std::cout << "cpu result:\n";
        for(int i = 0; i < N; i++)
            std::cout << hx[i * abs_incx] << " ";
        std::cout << "\ngpu result:\n";
        for(int i = 0; i < N; i++)
            std::cout << hx_or_b_1[i * abs_incx] << " ";
        std::cout << "\n";

        max_err_1 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx, hx_or_b_1));
        max_err_2 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx, hx_or_b_2));

        // T err_1 = 0.0;
        // T err_2 = 0.0;
        // for(int i = 0; i < N; i++)
        //     if(hx[i * abs_incx] != 0)
        //     {
        //         err_1 += std::abs((hx[i * abs_incx] - hx_or_b_1[i * abs_incx]) / hx[i * abs_incx]);
        //         err_2 += std::abs((hx[i * abs_incx] - hx_or_b_2[i * abs_incx]) / hx[i * abs_incx]);
        //     }
        //     else
        //     {
        //         err_1 += std::abs(hx_or_b_1[i * abs_incx]);
        //         err_2 += std::abs(hx_or_b_2[i * abs_incx]);
        //     }
        // max_err_1 = max_err_1 > err_1 ? max_err_1 : err_1;
        // max_err_2 = max_err_2 > err_2 ? max_err_2 : err_2;
        trsm_err_res_check<T>(max_err_1, N, error_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, N, error_eps_multiplier, eps);

        cblas_trmv<T>(uplo, transA, diag, N, hA, N, hx_or_b_1, incx);
        cblas_trmv<T>(uplo, transA, diag, N, hA, N, hx_or_b_2, incx);
        // hx_or_b contains A * (calculated X), so residual = A * (calculated x) - b
        //                                                  = hx_or_b - hb
        // res is the one norm of the scaled residual for each column

        max_err_1 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx_or_b_1, hb));
        max_err_1 = rocblas_abs(vector_norm_1<T>(N, abs_incx, hx_or_b_2, hb));

        // T res_1 = 0.0;
        // T res_2 = 0.0;
        // for(int i = 0; i < N; i++)
        //     if(hb[i * abs_incx] != 0)
        //     {
        //         res_1 += std::abs((hx_or_b_1[i * abs_incx] - hb[i * abs_incx]) / hb[i * abs_incx]);
        //         res_2 += std::abs((hx_or_b_2[i * abs_incx] - hb[i * abs_incx]) / hb[i * abs_incx]);
        //     }
        //     else
        //     {
        //         res_1 += std::abs(hx_or_b_1[i * abs_incx]);
        //         res_2 += std::abs(hx_or_b_2[i * abs_incx]);
        //     }
        // max_res_1 = max_res_1 > res_1 ? max_res_1 : res_1;
        // max_res_2 = max_res_2 > res_2 ? max_res_2 : res_2;

        trsm_err_res_check<T>(max_res_1, N, residual_eps_multiplier, eps);
        trsm_err_res_check<T>(max_res_2, N, residual_eps_multiplier, eps);
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
            rocblas_tpsv<T>(handle, uplo, transA, diag, N, dAP, dx_or_b, incx);

        gpu_time_used = get_time_us(); // in microseconds

        for(int i = 0; i < number_hot_calls; i++)
            rocblas_tpsv<T>(handle, uplo, transA, diag, N, dAP, dx_or_b, incx);

        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = trsv_gflop_count<T>(N) * number_hot_calls / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        if(arg.norm_check)
            cblas_tpsv<T>(uplo, transA, diag, N, hAP, cpu_x_or_b, incx);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = trsv_gflop_count<T>(N) / cpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "N,incx,uplo,transA,diag,rocblas-Gflops,us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << N << ',' << incx << ',' << char_uplo << ',' << char_transA << ',' << char_diag
                  << ',' << rocblas_gflops << "," << gpu_time_used / number_hot_calls;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                      << max_err_2;

        std::cout << std::endl;
    }
}
