/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
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
void testing_trsm_strided_batched(const Arguments& arg)
{
    rocblas_int M           = arg.M;
    rocblas_int N           = arg.N;
    rocblas_int lda         = arg.lda;
    rocblas_int ldb         = arg.ldb;
    rocblas_int stride_a    = arg.stride_a;
    rocblas_int stride_b    = arg.stride_b;
    rocblas_int batch_count = arg.batch_count;

    char char_side   = arg.side;
    char char_uplo   = arg.uplo;
    char char_transA = arg.transA;
    char char_diag   = arg.diag;
    T    alpha_h     = arg.alpha;

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_int K      = side == rocblas_side_left ? M : N;
    size_t      size_A = lda * size_t(K) + stride_a * (batch_count - 1);
    size_t      size_B = ldb * size_t(N) + stride_b * (batch_count - 1);

    rocblas_local_handle handle;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dA(safe_size);
        device_vector<T>    dXorB(safe_size);
        if(!dA || !dXorB)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        rocblas_status status = rocblas_trsm_strided_batched<T>(handle,
                                                                side,
                                                                uplo,
                                                                transA,
                                                                diag,
                                                                M,
                                                                N,
                                                                &alpha_h,
                                                                dA,
                                                                lda,
                                                                stride_a,
                                                                dXorB,
                                                                ldb,
                                                                stride_b,
                                                                batch_count);
        if(batch_count == 0)
            CHECK_ROCBLAS_ERROR(status);
        else
            EXPECT_ROCBLAS_STATUS(status, rocblas_status_invalid_size);

        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hX(size_B);
    host_vector<T> hXorB_1(size_B);
    host_vector<T> hXorB_2(size_B);
    host_vector<T> cpuXorB(size_B);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    T      error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    T      residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    T      eps                     = std::numeric_limits<T>::epsilon();

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dXorB(size_B);
    device_vector<T> alpha_d(1);
    if(!dA || !dXorB || !alpha_d)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //  Random lower triangular matrices have condition number
    //  that grows exponentially with matrix size. Random full
    //  matrices have condition that grows linearly with
    //  matrix size.
    //
    //  We want a triangular matrix with condition number that grows
    //  lineary with matrix size. We start with full random matrix A.
    //  Calculate symmetric AAT <- A A^T. Make AAT strictly diagonal
    //  dominant. A strictly diagonal dominant matrix is SPD so we
    //  can use Cholesky to calculate L L^T = AAT. These L factors
    //  should have condition number approximately equal to
    //  the condition number of the original matrix A.

    //  initialize full random matrix hA with all entries in [1, 10]
    rocblas_init<T>(hA, K, K, lda, stride_a, batch_count);

    //  pad untouched area into zero
    for(int b = 0; b < batch_count; b++)
    {
        for(int i = K; i < lda; i++)
            for(int j = 0; j < K; j++)
                hA[i + j * lda + b * stride_a] = 0.0;

        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T, T>(rocblas_operation_none,
                         rocblas_operation_transpose,
                         K,
                         K,
                         K,
                         1.0,
                         hA + stride_a * b,
                         lda,
                         hA + stride_a * b,
                         lda,
                         0.0,
                         AAT + stride_a * b,
                         lda);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < K; i++)
        {
            T t = 0.0;
            for(int j = 0; j < K; j++)
            {
                int idx = i + j * lda + b * stride_a;
                hA[idx] = AAT[idx];
                t += AAT[idx] > 0 ? AAT[idx] : -AAT[idx];
            }
            hA[i + i * lda + b * stride_a] = t;
        }

        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(char_uplo, K, hA + stride_a * b, lda);
    }

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        if('L' == char_uplo || 'l' == char_uplo)
        {
            for(int b = 0; b < batch_count; b++)
            {
                for(int i = 0; i < K; i++)
                {
                    T diag = hA[i + i * lda + b * stride_a];
                    for(int j = 0; j <= i; j++)
                        hA[i + j * lda + b * stride_a] = hA[i + j * lda + b * stride_a] / diag;
                }
            }
        }
        else
        {
            for(int b = 0; b < batch_count; b++)
            {
                for(int j = 0; j < K; j++)
                {
                    T diag = hA[j + j * lda + b * stride_a];
                    for(int i = 0; i <= j; i++)
                        hA[i + j * lda + b * stride_a] = hA[i + j * lda + b * stride_a] / diag;
                }
            }
        }
    }

    // Initial hX
    rocblas_init<T>(hX, M, N, ldb, stride_b, batch_count);
    // pad untouched area into zero
    for(int b = 0; b < batch_count; b++)
        for(int i = M; i < ldb; i++)
            for(int j = 0; j < N; j++)
                hX[i + j * ldb + b * stride_b] = 0.0;
    hB = hX;

    // Calculate hB = hA*hX;
    for(int b = 0; b < batch_count; b++)
        cblas_trmm<T>(side,
                      uplo,
                      transA,
                      diag,
                      M,
                      N,
                      1.0 / alpha_h,
                      hA + stride_a * b,
                      lda,
                      hB + stride_b * b,
                      ldb);

    hXorB_1 = hB; // hXorB <- B
    hXorB_2 = hB; // hXorB <- B
    cpuXorB = hB; // cpuXorB <- B

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB_1, sizeof(T) * size_B, hipMemcpyHostToDevice));

    T max_err_1 = 0.0;
    T max_err_2 = 0.0;
    T max_res_1 = 0.0;
    T max_res_2 = 0.0;
    if(arg.unit_check || arg.norm_check)
    {
        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB_1, sizeof(T) * size_B, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_trsm_strided_batched<T>(handle,
                                                            side,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            N,
                                                            &alpha_h,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dXorB,
                                                            ldb,
                                                            stride_b,
                                                            batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hXorB_1, dXorB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB_2, sizeof(T) * size_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(alpha_d, &alpha_h, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_trsm_strided_batched<T>(handle,
                                                            side,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            N,
                                                            alpha_d,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dXorB,
                                                            ldb,
                                                            stride_b,
                                                            batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hXorB_2, dXorB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        // Error Check
        // hXorB contains calculated X, so error is hX - hXorB

        // err is the one norm of the scaled error for a single column
        // max_err is the maximum of err for all columns
        for(int b = 0; b < batch_count; b++)
        {
            max_err_1 = max_err_2 = 0;
            for(int i = 0; i < N; i++)
            {
                T err_1 = 0.0;
                T err_2 = 0.0;
                for(int j = 0; j < M; j++)
                {
                    int idx = j + i * ldb + b * stride_b;
                    if(hX[idx] != 0)
                    {
                        err_1 += std::abs((hX[idx] - hXorB_1[idx]) / hX[idx]);
                        err_2 += std::abs((hX[idx] - hXorB_2[idx]) / hX[idx]);
                    }
                    else
                    {
                        err_1 += std::abs(hXorB_1[idx]);
                        err_2 += std::abs(hXorB_2[idx]);
                    }
                }
                max_err_1 = max_err_1 > err_1 ? max_err_1 : err_1;
                max_err_2 = max_err_2 > err_2 ? max_err_2 : err_2;
            }
            trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

            // Residual Check
            // hXorB <- hA * (A^(-1) B) ;
            cblas_trmm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          1.0 / alpha_h,
                          hA + stride_a * b,
                          lda,
                          hXorB_1 + stride_b * b,
                          ldb);
            cblas_trmm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          1.0 / alpha_h,
                          hA + stride_a * b,
                          lda,
                          hXorB_2 + stride_b * b,
                          ldb);

            // hXorB contains A * (calculated X), so residual = A * (calculated X) - B
            //                                                = hXorB - hB
            // res is the one norm of the scaled residual for each column
            max_res_1 = max_res_2 = 0;
            for(int i = 0; i < N; i++)
            {
                T res_1 = 0.0;
                T res_2 = 0.0;
                for(int j = 0; j < M; j++)
                {
                    int idx = j + i * ldb + b * stride_b;
                    if(hB[j + i * ldb] != 0)
                    {
                        res_1 += std::abs((hXorB_1[idx] - hB[idx]) / hB[idx]);
                        res_2 += std::abs((hXorB_2[idx] - hB[idx]) / hB[idx]);
                    }
                    else
                    {
                        res_1 += std::abs(hXorB_1[idx]);
                        res_2 += std::abs(hXorB_2[idx]);
                    }
                }
                max_res_1 = max_res_1 > res_1 ? max_res_1 : res_1;
                max_res_2 = max_res_2 > res_2 ? max_res_2 : res_2;
            }
            trsm_err_res_check<T>(max_res_1, M, residual_eps_multiplier, eps);
            trsm_err_res_check<T>(max_res_2, M, residual_eps_multiplier, eps);
        }
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(hipMemcpy(dXorB, hXorB_1, sizeof(T) * size_B, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        gpu_time_used = get_time_us(); // in microseconds

        CHECK_ROCBLAS_ERROR(rocblas_trsm_strided_batched<T>(handle,
                                                            side,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            N,
                                                            &alpha_h,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dXorB,
                                                            ldb,
                                                            stride_b,
                                                            batch_count));

        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = batch_count * trsm_gflop_count<T>(M, N, K) / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        for(int b = 0; b < batch_count; b++)
            cblas_trsm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          alpha_h,
                          hA + stride_a * b,
                          lda,
                          cpuXorB + stride_b * b,
                          ldb);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * trsm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "M,N,lda,ldb,side,uplo,transA,diag,batch_count,rocblas-Gflops,us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << M << ',' << N << ',' << lda << ',' << ldb << ',' << char_side << ','
                  << char_uplo << ',' << char_transA << ',' << char_diag << ',' << batch_count
                  << ',' << rocblas_gflops << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                      << max_err_2;

        std::cout << std::endl;
    }
}
