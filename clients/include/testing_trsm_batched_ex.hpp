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
#define TRSM_BLOCK 128

template <typename T>
void testing_trsm_batched_ex(const Arguments& arg)
{
    rocblas_int M   = arg.M;
    rocblas_int N   = arg.N;
    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;

    char        char_side   = arg.side;
    char        char_uplo   = arg.uplo;
    char        char_transA = arg.transA;
    char        char_diag   = arg.diag;
    T           alpha_h     = arg.alpha;
    rocblas_int batch_count = arg.batch_count;

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_int K      = side == rocblas_side_left ? M : N;
    size_t      size_A = lda * size_t(K);
    size_t      size_B = ldb * size_t(N);

    rocblas_local_handle handle;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count <= 0)
    {
        static const size_t     safe_size = 100; // arbitrarily set to 100
        rocblas_int             num_batch = batch_count < 0 ? 1 : batch_count;
        device_vector<T*, 0, T> dA(1);
        device_vector<T*, 0, T> dXorB(1);
        device_vector<T*, 0, T> dinvA(1);

        if(!dA || !dXorB)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        rocblas_status status = rocblas_trsm_batched_ex(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        &alpha_h,
                                                        dA,
                                                        lda,
                                                        dXorB,
                                                        ldb,
                                                        batch_count,
                                                        dinvA,
                                                        TRSM_BLOCK * K,
                                                        arg.compute_type);

        if(batch_count == 0)
            CHECK_ROCBLAS_ERROR(status);
        else
            EXPECT_ROCBLAS_STATUS(status, rocblas_status_invalid_size);

        return;
    }

    // Device-arrays of pointers to device memory
    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dXorB(batch_count);
    device_vector<T*, 0, T> dinvA(batch_count);
    device_vector<T*, 0, T> dX_tmp(batch_count);
    device_vector<T>        alpha_d(1);

    // Host-arrays of pointers to host memory
    host_vector<T> hA[batch_count];
    host_vector<T> AAT[batch_count];
    host_vector<T> hB[batch_count];
    host_vector<T> hX[batch_count];
    host_vector<T> hXorB_1[batch_count];
    host_vector<T> hXorB_2[batch_count];
    host_vector<T> cpuXorB[batch_count];

    for(int b = 0; b < batch_count; b++)
    {
        hA[b]      = host_vector<T>(size_A);
        AAT[b]     = host_vector<T>(size_A);
        hB[b]      = host_vector<T>(size_B);
        hX[b]      = host_vector<T>(size_B);
        hXorB_1[b] = host_vector<T>(size_B);
        hXorB_2[b] = host_vector<T>(size_B);
        cpuXorB[b] = host_vector<T>(size_B);
    }

    // Host-arrays of pointers to device memory
    device_batch_vector<T> bA(batch_count, size_A);
    device_batch_vector<T> bXorB(batch_count, size_B);
    device_batch_vector<T> binvA(batch_count, TRSM_BLOCK * K);

    int last = batch_count - 1;
    if((!bA[last] && size_A) || (!bXorB[last] && size_B) || (!binvA[last] && TRSM_BLOCK * K))
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    T      error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    T      residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    T      eps                     = std::numeric_limits<T>::epsilon();

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
    for(int b = 0; b < batch_count; b++)
    {
        rocblas_init<T>(hA[b], K, K, lda);

        //  pad untouched area into zero
        for(int i = K; i < lda; i++)
            for(int j = 0; j < K; j++)
                hA[b][i + j * lda] = 0.0;

        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T, T>(rocblas_operation_none,
                         rocblas_operation_transpose,
                         K,
                         K,
                         K,
                         1.0,
                         hA[b],
                         lda,
                         hA[b],
                         lda,
                         0.0,
                         AAT[b],
                         lda);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < K; i++)
        {
            T t = 0.0;
            for(int j = 0; j < K; j++)
            {
                hA[b][i + j * lda] = AAT[b][i + j * lda];
                t += AAT[b][i + j * lda] > 0 ? AAT[b][i + j * lda] : -AAT[b][i + j * lda];
            }
            hA[b][i + i * lda] = t;
        }

        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(char_uplo, K, hA[b], lda);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(char_diag == 'U' || char_diag == 'u')
        {
            if('L' == char_uplo || 'l' == char_uplo)
                for(int i = 0; i < K; i++)
                {
                    T diag = hA[b][i + i * lda];
                    for(int j = 0; j <= i; j++)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / diag;
                }
            else
                for(int j = 0; j < K; j++)
                {
                    T diag = hA[b][j + j * lda];
                    for(int i = 0; i <= j; i++)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / diag;
                }
        }

        // Initial hX
        rocblas_init<T>(hX[b], M, N, ldb);
        // pad untouched area into zero
        for(int i = M; i < ldb; i++)
            for(int j = 0; j < N; j++)
                hX[b][i + j * ldb] = 0.0;
        hB[b] = hX[b];

        // Calculate hB = hA*hX;
        cblas_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA[b], lda, hB[b], ldb);

        hXorB_1[b] = hB[b]; // hXorB <- B
        hXorB_2[b] = hB[b]; // hXorB <- B
        cpuXorB[b] = hB[b]; // cpuXorB <- B

        // copy data from CPU to device
        // 1. Use intermediate arrays to access device memory from host
        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b], sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bXorB[b], hXorB_1[b], sizeof(T) * size_B, hipMemcpyHostToDevice));
    }

    // 2. Copy intermediate arrays into device arrays
    CHECK_HIP_ERROR(hipMemcpy(dA, bA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dXorB, bXorB, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dinvA, binvA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    rocblas_int stride_A    = TRSM_BLOCK * lda + TRSM_BLOCK;
    rocblas_int stride_invA = TRSM_BLOCK * TRSM_BLOCK;

    int blocks = K / TRSM_BLOCK;

    T max_err_1 = 0.0;
    T max_err_2 = 0.0;
    T max_res_1 = 0.0;
    T max_res_2 = 0.0;
    if(arg.unit_check || arg.norm_check)
    {
        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(bXorB[b], hXorB_1[b], sizeof(T) * size_B, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(dXorB, bXorB, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

        hipStream_t rocblas_stream;
        rocblas_get_stream(handle, &rocblas_stream);

        for(int b = 0; b < batch_count; b++)
        {
            if(blocks > 0)
            {
                CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched<T>(handle,
                                                                     uplo,
                                                                     diag,
                                                                     TRSM_BLOCK,
                                                                     bA[b],
                                                                     lda,
                                                                     stride_A,
                                                                     binvA[b],
                                                                     TRSM_BLOCK,
                                                                     stride_invA,
                                                                     blocks));
            }

            if(K % TRSM_BLOCK != 0 || blocks == 0)
            {
                CHECK_ROCBLAS_ERROR(
                    rocblas_trtri_strided_batched<T>(handle,
                                                     uplo,
                                                     diag,
                                                     K - TRSM_BLOCK * blocks,
                                                     bA[b] + stride_A * blocks,
                                                     lda,
                                                     stride_A,
                                                     binvA[b] + stride_invA * blocks,
                                                     TRSM_BLOCK,
                                                     stride_invA,
                                                     1));
            }
        }

        size_t x_temp_size = M * N;
        CHECK_ROCBLAS_ERROR(rocblas_trsm_batched_ex(handle,
                                                    side,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    M,
                                                    N,
                                                    &alpha_h,
                                                    dA,
                                                    lda,
                                                    dXorB,
                                                    ldb,
                                                    batch_count,
                                                    dinvA,
                                                    TRSM_BLOCK * K,
                                                    arg.compute_type));

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(hXorB_1[b], bXorB[b], sizeof(T) * size_B, hipMemcpyDeviceToHost));
        }

        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(bXorB[b], hXorB_2[b], sizeof(T) * size_B, hipMemcpyHostToDevice));
        }

        CHECK_HIP_ERROR(hipMemcpy(dXorB, bXorB, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(alpha_d, &alpha_h, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_trsm_batched_ex(handle,
                                                    side,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    M,
                                                    N,
                                                    alpha_d,
                                                    dA,
                                                    lda,
                                                    dXorB,
                                                    ldb,
                                                    batch_count,
                                                    dinvA,
                                                    TRSM_BLOCK * K,
                                                    arg.compute_type));

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(hXorB_2[b], bXorB[b], sizeof(T) * size_B, hipMemcpyDeviceToHost));
        }

        // Error Check
        // hXorB contains calculated X, so error is hX - hXorB

        // err is the one norm of the scaled error for a single column
        // max_err is the maximum of err for all columns
        for(int b = 0; b < batch_count; b++)
        {
            for(int i = 0; i < N; i++)
            {
                T err_1 = 0.0;
                T err_2 = 0.0;
                for(int j = 0; j < M; j++)
                {
                    if(hX[b][j + i * ldb] != 0)
                    {
                        err_1 += std::abs((hX[b][j + i * ldb] - hXorB_1[b][j + i * ldb])
                                          / hX[b][j + i * ldb]);
                        err_2 += std::abs((hX[b][j + i * ldb] - hXorB_2[b][j + i * ldb])
                                          / hX[b][j + i * ldb]);
                    }
                    else
                    {
                        err_1 += std::abs(hXorB_1[b][j + i * ldb]);
                        err_2 += std::abs(hXorB_2[b][j + i * ldb]);
                    }
                }
                max_err_1 = max_err_1 > err_1 ? max_err_1 : err_1;
                max_err_2 = max_err_2 > err_2 ? max_err_2 : err_2;
            }

            trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

            // Residual Check
            // hXorB <- hA * (A^(-1) B) ;
            cblas_trmm<T>(
                side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA[b], lda, hXorB_1[b], ldb);
            cblas_trmm<T>(
                side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA[b], lda, hXorB_2[b], ldb);

            // hXorB contains A * (calculated X), so residual = A * (calculated X) - B
            //                                                = hXorB - hB
            // res is the one norm of the scaled residual for each column
            for(int i = 0; i < N; i++)
            {
                T res_1 = 0.0;
                T res_2 = 0.0;
                for(int j = 0; j < M; j++)
                {
                    if(hB[b][j + i * ldb] != 0)
                    {
                        res_1 += std::abs((hXorB_1[b][j + i * ldb] - hB[b][j + i * ldb])
                                          / hB[b][j + i * ldb]);
                        res_2 += std::abs((hXorB_2[b][j + i * ldb] - hB[b][j + i * ldb])
                                          / hB[b][j + i * ldb]);
                    }
                    else
                    {
                        res_1 += std::abs(hXorB_1[b][j + i * ldb]);
                        res_2 += std::abs(hXorB_2[b][j + i * ldb]);
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
        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(bXorB[b], hXorB_1[b], sizeof(T) * size_B, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(dXorB, bXorB, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        gpu_time_used = get_time_us(); // in microseconds

        CHECK_ROCBLAS_ERROR(rocblas_trsm_batched_ex(handle,
                                                    side,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    M,
                                                    N,
                                                    &alpha_h,
                                                    dA,
                                                    lda,
                                                    dXorB,
                                                    ldb,
                                                    batch_count,
                                                    dinvA,
                                                    TRSM_BLOCK * K,
                                                    arg.compute_type));

        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = batch_count * trsm_gflop_count<T>(M, N, K) / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        for(int b = 0; b < batch_count; b++)
        {
            cblas_trsm<T>(side, uplo, transA, diag, M, N, alpha_h, hA[b], lda, cpuXorB[b], ldb);
        }

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
