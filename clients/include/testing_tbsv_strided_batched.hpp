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
void testing_tbsv_strided_batched(const Arguments& arg)
{
    // rocblas_int N           = arg.N;
    // rocblas_int K           = arg.K;
    // rocblas_int lda         = arg.lda;
    // rocblas_int incx        = arg.incx;
    // char        char_uplo   = arg.uplo;
    // char        char_transA = arg.transA;
    // char        char_diag   = arg.diag;
    // rocblas_int stride_a    = arg.stride_a;
    // rocblas_int stride_x    = arg.stride_x;
    // rocblas_int batch_count = arg.batch_count;

    // rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    // rocblas_operation transA = char2rocblas_operation(char_transA);
    // rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    // rocblas_status       status;
    // rocblas_local_handle handle;

    // // check here to prevent undefined memory allocation error
    // if(N < 0 || K < 0 || lda < K + 1 || !incx || batch_count <= 0)
    // {
    //     CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
    //     if(batch_count == 0)
    //         CHECK_ROCBLAS_ERROR(rocblas_tbsv_strided_batched<T>(handle,
    //                                                             uplo,
    //                                                             transA,
    //                                                             diag,
    //                                                             N,
    //                                                             K,
    //                                                             nullptr,
    //                                                             lda,
    //                                                             stride_a,
    //                                                             nullptr,
    //                                                             incx,
    //                                                             stride_x,
    //                                                             batch_count));
    //     else
    //         EXPECT_ROCBLAS_STATUS(rocblas_tbsv_strided_batched<T>(handle,
    //                                                               uplo,
    //                                                               transA,
    //                                                               diag,
    //                                                               N,
    //                                                               K,
    //                                                               nullptr,
    //                                                               lda,
    //                                                               stride_a,
    //                                                               nullptr,
    //                                                               incx,
    //                                                               stride_x,
    //                                                               batch_count),
    //                               rocblas_status_invalid_size);
    //     return;
    // }

    // size_t size_A   = lda * size_t(N) + stride_a * (batch_count - 1);
    // size_t abs_incx = size_t(incx >= 0 ? incx : -incx);
    // size_t size_x   = N * abs_incx + stride_x * (batch_count - 1);

    // // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    // host_vector<T> hA(size_A);
    // host_vector<T> AAT(size_A);
    // host_vector<T> hb(size_x);
    // host_vector<T> hx(size_x);
    // host_vector<T> hx_or_b_1(size_x);
    // host_vector<T> hx_or_b_2(size_x);
    // host_vector<T> cpu_x_or_b(size_x);

    // double gpu_time_used, cpu_time_used;
    // double rocblas_gflops, cblas_gflops;
    // double rocblas_error;
    // double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    // double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    // double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    // // allocate memory on device
    // device_vector<T> dA(size_A);
    // device_vector<T> dx_or_b(size_x);
    // CHECK_DEVICE_ALLOCATION(dA.memcheck());
    // CHECK_DEVICE_ALLOCATION(dx_or_b.memcheck());

    // rocblas_init<T>(hA, true);

    // // initialize "exact" answer hx
    // rocblas_init<T>(hx, false);
    // hb = hx;

    // //  calculate AAT = hA * hA ^ T or AAT = hA * hA ^ H if complex
    // for(int b = 0; b < batch_count; b++)
    // {
    //     // //  make hA unit diagonal if diag == rocblas_diagonal_unit
    //     // if(char_diag == 'U' || char_diag == 'u')
    //     // {
    //     //     if('L' == char_uplo || 'l' == char_uplo)
    //     //     {
    //     //         for(int i = 0; i < N; i++)
    //     //         {
    //     //             T diag = hA[i + i * lda + stride_a * b];
    //     //             for(int j = 0; j <= i; j++)
    //     //                 hA[i + j * lda + stride_a * b] = hA[i + j * lda + stride_a * b] / diag;
    //     //         }
    //     //     }
    //     //     else
    //     //     {
    //     //         for(int j = 0; j < N; j++)
    //     //         {
    //     //             T diag = hA[j + j * lda + stride_a * b];
    //     //             for(int i = 0; i <= j; i++)
    //     //                 hA[i + j * lda + stride_a * b] = hA[i + j * lda + stride_a * b] / diag;
    //     //         }
    //     //     }
    //     // }
    // }

    // // Calculate hb = hA*hx;
    // for(int b = 0; b < batch_count; b++)
    // {
    //     cblas_tbmv<T>(uplo, transA, diag, N, K, hA + stride_a * b, lda, hb + stride_x * b, incx);
    // }
    // cpu_x_or_b = hb; // cpuXorB <- B
    // hx_or_b_1  = hb;
    // hx_or_b_2  = hb;

    // // copy data from CPU to device
    // CHECK_HIP_ERROR(dA.transfer_from(hA));
    // CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

    // double max_err_1 = 0.0;
    // double max_err_2 = 0.0;

    // if(arg.unit_check || arg.norm_check)
    // {
    //     // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
    //     CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    //     CHECK_ROCBLAS_ERROR(rocblas_tbsv_strided_batched<T>(handle,
    //                                                         uplo,
    //                                                         transA,
    //                                                         diag,
    //                                                         N,
    //                                                         K,
    //                                                         dA,
    //                                                         lda,
    //                                                         stride_a,
    //                                                         dx_or_b,
    //                                                         incx,
    //                                                         stride_x,
    //                                                         batch_count));

    //     CHECK_HIP_ERROR(hx_or_b_1.transfer_from(dx_or_b));

    //     // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
    //     CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
    //     CHECK_HIP_ERROR(dx_or_b.transfer_From(hx_or_b_2));

    //     CHECK_ROCBLAS_ERROR(rocblas_tbsv_strided_batched<T>(handle,
    //                                                         uplo,
    //                                                         transA,
    //                                                         diag,
    //                                                         N,
    //                                                         K,
    //                                                         dA,
    //                                                         lda,
    //                                                         stride_a,
    //                                                         dx_or_b,
    //                                                         incx,
    //                                                         stride_x,
    //                                                         batch_count));

    //     CHECK_HIP_ERROR(hx_or_b_2.transfer_from(dx_or_b));

    //     //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
    //     // calculate norm 1 of vector E
    //     for(int b = 0; b < batch_count; b++)
    //     {
    //         max_err_1 = rocblas_abs(
    //             vector_norm_1<T>(N, abs_incx, &hx[b * stride_x], &hx_or_b_1[b * stride_x]));
    //         max_err_2 = rocblas_abs(
    //             vector_norm_1<T>(N, abs_incx, &hx[b * stride_x], &hx_or_b_2[b * stride_x]));

    //         //unit test
    //         trsm_err_res_check<T>(max_err_1, N, error_eps_multiplier, eps);
    //         trsm_err_res_check<T>(max_err_2, N, error_eps_multiplier, eps);
    //     }

    //     // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
    //     for(int b = 0; b < batch_count; b++)
    //     {
    //         cblas_tbmv<T>(
    //             uplo, transA, diag, N, K, hA + b * stride_a, lda, hx_or_b_1 + b * stride_x, incx);
    //         cblas_tbmv<T>(
    //             uplo, transA, diag, N, K, hA + b * stride_a, lda, hx_or_b_2 + b * stride_x, incx);
    //     }

    //     //calculate norm 1 of res
    //     for(int b = 0; b < batch_count; b++)
    //     {
    //         max_err_1 = rocblas_abs(
    //             vector_norm_1<T>(N, abs_incx, &hx_or_b_1[b * stride_x], &hb[b * stride_x]));
    //         max_err_2 = rocblas_abs(
    //             vector_norm_1<T>(N, abs_incx, &hx_or_b_1[b * stride_x], &hb[b * stride_x]));

    //         //unit test
    //         trsm_err_res_check<T>(max_err_1, N, error_eps_multiplier, eps);
    //         trsm_err_res_check<T>(max_err_2, N, error_eps_multiplier, eps);
    //     }
    // }

    // if(arg.timing)
    // {
    //     // GPU rocBLAS
    //     CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b_1));

    //     CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    //     int number_cold_calls = arg.cold_iters;
    //     int number_hot_calls  = arg.iters;

    //     for(int i = 0; i < number_cold_calls; i++)
    //         rocblas_tbsv_strided_batched<T>(handle,
    //                                         uplo,
    //                                         transA,
    //                                         diag,
    //                                         N,
    //                                         K,
    //                                         dA,
    //                                         lda,
    //                                         stride_a,
    //                                         dx_or_b,
    //                                         incx,
    //                                         stride_x,
    //                                         batch_count);

    //     gpu_time_used = get_time_us(); // in microseconds

    //     for(int i = 0; i < number_hot_calls; i++)
    //         rocblas_tbsv_strided_batched<T>(handle,
    //                                         uplo,
    //                                         transA,
    //                                         diag,
    //                                         N,
    //                                         K,
    //                                         dA,
    //                                         lda,
    //                                         stride_a,
    //                                         dx_or_b,
    //                                         incx,
    //                                         stride_x,
    //                                         batch_count);

    //     gpu_time_used = get_time_us() - gpu_time_used;
    //     rocblas_gflops
    //         = batch_count * tbsv_gflop_count<T>(N) * number_hot_calls / gpu_time_used * 1e6;

    //     // CPU cblas
    //     cpu_time_used = get_time_us();

    //     if(arg.norm_check)
    //         for(int b = 0; b < batch_count; b++)
    //             cblas_tbsv<T>(
    //                 uplo, transA, diag, N, K, hA + b * stride_a, lda, cpu_x_or_b + b * stride_x, incx);

    //     cpu_time_used = get_time_us() - cpu_time_used;
    //     cblas_gflops  = batch_count * tbsv_gflop_count<T>(N) / cpu_time_used * 1e6;

    //     // only norm_check return an norm error, unit check won't return anything
    //     std::cout << "N,lda,incx,uplo,transA,diag,batch_count,rocblas-Gflops,us";

    //     if(arg.norm_check)
    //         std::cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

    //     std::cout << std::endl;

    //     std::cout << N << ',' << K << ',' << lda << ',' << incx << ',' << char_uplo << ',' << char_transA << ','
    //               << char_diag << ',' << batch_count << ',' << rocblas_gflops << ","
    //               << gpu_time_used / number_hot_calls;

    //     if(arg.norm_check)
    //         std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
    //                   << max_err_2;

    //     std::cout << std::endl;
    // }
}
