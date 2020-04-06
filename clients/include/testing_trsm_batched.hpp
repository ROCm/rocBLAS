/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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
void testing_trsm_batched(const Arguments& arg)
{
    rocblas_int M           = arg.M;
    rocblas_int N           = arg.N;
    rocblas_int lda         = arg.lda;
    rocblas_int ldb         = arg.ldb;
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
    size_t      size_A = lda * size_t(K);
    size_t      size_B = ldb * size_t(N);

    rocblas_local_handle handle;
    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100

        device_batch_vector<T> dA(safe_size, 1, 1);
        device_batch_vector<T> dXorB(safe_size, 1, 1);
        CHECK_DEVICE_ALLOCATION(dA.memcheck());
        CHECK_DEVICE_ALLOCATION(dXorB.memcheck());

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        rocblas_status status = rocblas_trsm_batched<T>(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        &alpha_h,
                                                        dA.ptr_on_device(),
                                                        lda,
                                                        dXorB.ptr_on_device(),
                                                        ldb,
                                                        batch_count);

        if(batch_count == 0) // || M == 0 || N == 0 || K == 0)
            CHECK_ROCBLAS_ERROR(status);
        else
            EXPECT_ROCBLAS_STATUS(status, rocblas_status_invalid_size);

        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> AAT(size_A, 1, batch_count);
    host_batch_vector<T> hB(size_B, 1, batch_count);
    host_batch_vector<T> hX(size_B, 1, batch_count);
    host_batch_vector<T> hXorB_1(size_B, 1, batch_count);
    host_batch_vector<T> hXorB_2(size_B, 1, batch_count);
    host_batch_vector<T> cpuXorB(size_B, 1, batch_count);
    host_vector<T>       halpha(1);
    halpha[0] = alpha_h;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    // allocate memory on device
    device_batch_vector<T> dA(size_A, 1, batch_count);
    device_batch_vector<T> dXorB(size_B, 1, batch_count);
    device_vector<T>       alpha_d(1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dXorB.memcheck());
    CHECK_DEVICE_ALLOCATION(alpha_d.memcheck());

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
        for(int i = K; i < lda; i++)
            for(int j = 0; j < K; j++)
                hA[b][i + j * lda] = 0.0;

        //  calculate AAT = hA * hA ^ T or AAT = hA * hA ^ H if complex
        cblas_gemm<T>(rocblas_operation_none,
                      rocblas_operation_conjugate_transpose,
                      K,
                      K,
                      K,
                      T(1.0),
                      hA[b],
                      lda,
                      hA[b],
                      lda,
                      T(0.0),
                      AAT[b],
                      lda);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < K; i++)
        {
            T t = 0.0;
            for(int j = 0; j < K; j++)
            {
                int idx    = i + j * lda;
                hA[b][idx] = AAT[b][idx];
                t += rocblas_abs(AAT[b][idx]);
            }
            hA[b][i + i * lda] = t;
        }

        //  calculate Cholesky factorization of SPD (or hermitian if complex) matrix hA
        cblas_potrf<T>(char_uplo, K, hA[b], lda);
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
                    T diag = hA[b][i + i * lda];
                    for(int j = 0; j <= i; j++)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / diag;
                }
            }
        }
        else
        {
            for(int b = 0; b < batch_count; b++)
            {
                for(int j = 0; j < K; j++)
                {
                    T diag = hA[b][j + j * lda];
                    for(int i = 0; i <= j; i++)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / diag;
                }
            }
        }
    }

    // Initialize "exact" answer hx
    for(int b = 0; b < batch_count; b++)
    {
        rocblas_init<T>(hX[b], M, N, ldb);
        for(int i = M; i < ldb; i++)
            for(int j = 0; j < N; j++)
                hX[b][i + j * ldb] = 0.0;
    }

    hB.copy_from(hX);

    for(int b = 0; b < batch_count; b++)
    {
        // Calculate hB = hA*hX
        cblas_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA[b], lda, hB[b], ldb);
    }

    hXorB_1.copy_from(hB);
    hXorB_2.copy_from(hB);
    cpuXorB.copy_from(hB);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

        CHECK_ROCBLAS_ERROR(rocblas_trsm_batched<T>(handle,
                                                    side,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    M,
                                                    N,
                                                    &alpha_h,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dXorB.ptr_on_device(),
                                                    ldb,
                                                    batch_count));

        CHECK_HIP_ERROR(hXorB_1.transfer_from(dXorB));

        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(alpha_d.transfer_from(halpha));
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_2));

        CHECK_ROCBLAS_ERROR(rocblas_trsm_batched<T>(handle,
                                                    side,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    M,
                                                    N,
                                                    alpha_d,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dXorB.ptr_on_device(),
                                                    ldb,
                                                    batch_count));

        CHECK_HIP_ERROR(hXorB_2.transfer_from(dXorB));

        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate vector-induced-norm 1 of matrix E
        for(int b = 0; b < batch_count; b++)
        {
            max_err_1 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hX[b], hXorB_1[b]));
            max_err_2 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hX[b], hXorB_2[b]));

            //unit test
            trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

            // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
            cblas_trmm<T>(
                side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA[b], lda, hXorB_1[b], ldb);
            cblas_trmm<T>(
                side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA[b], lda, hXorB_2[b], ldb);

            // calculate vector-induced-norm 1 of matrix res
            max_err_1 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hXorB_1[b], hB[b]));
            max_err_2 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hXorB_2[b], hB[b]));

            //unit test
            trsm_err_res_check<T>(max_err_1, M, residual_eps_multiplier, eps);
            trsm_err_res_check<T>(max_err_2, M, residual_eps_multiplier, eps);
        }
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        gpu_time_used = get_time_us(); // in microseconds

        CHECK_ROCBLAS_ERROR(rocblas_trsm_batched<T>(handle,
                                                    side,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    M,
                                                    N,
                                                    &alpha_h,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dXorB.ptr_on_device(),
                                                    ldb,
                                                    batch_count));

        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = batch_count * trsm_gflop_count<T>(M, N, K) / gpu_time_used * 1e6;

        // CPU cblas
        cpu_time_used = get_time_us();

        for(int b = 0; b < batch_count; b++)
            cblas_trsm<T>(side, uplo, transA, diag, M, N, alpha_h, hA[b], lda, cpuXorB[b], ldb);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * trsm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "M,N,lda,ldb,side,uplo,transA,diag,batch_count,rocblas-Gflops,us";

        if(arg.norm_check)
            rocblas_cout << ",CPU-Gflops,us,norm_error_host_ptr,norm_error_dev_ptr";

        rocblas_cout << std::endl;

        rocblas_cout << M << ',' << N << ',' << lda << ',' << ldb << ',' << char_side << ','
                     << char_uplo << ',' << char_transA << ',' << char_diag << ',' << batch_count
                     << ',' << rocblas_gflops << "," << gpu_time_used;

        if(arg.norm_check)
            rocblas_cout << "," << cblas_gflops << "," << cpu_time_used << "," << max_err_1 << ","
                         << max_err_2;

        rocblas_cout << std::endl;
    }
}
