/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
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
void testing_trmm(const Arguments& arg)
{
    bool nantest = rocblas_isnan(arg.alpha) || rocblas_isnan(arg.alphai);
    if(!std::is_same<T, float>{} && !std::is_same<T, double>{} && !std::is_same<T, rocblas_half>{}
       && !is_complex<T> && nantest)
        return; // Exclude integers or other types which don't support NaN

    rocblas_int M   = arg.M;
    rocblas_int N   = arg.N;
    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;

    char char_side   = arg.side;
    char char_uplo   = arg.uplo;
    char char_transA = arg.transA;
    char char_diag   = arg.diag;
    T    h_alpha_T   = arg.get_alpha<T>();

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_int K      = side == rocblas_side_left ? M : N;
    size_t      size_A = lda * size_t(K);
    size_t      size_B = ldb * size_t(N);

    rocblas_local_handle handle;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dA(safe_size);
        device_vector<T>    dB(safe_size);
        if(!dA || !dB)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm<T>(handle, side, uplo, transA, diag, M, N, &h_alpha_T, dA, lda, dB, ldb),
            rocblas_status_invalid_size);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hB_1(size_B);
    host_vector<T> hB_2(size_B);
    host_vector<T> cpuB(size_B);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error = 0.0;

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> alpha_d(1);
    if(!dA || !dB || !alpha_d)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //  initialize full random matrix hA with all entries in [1, 10]
    rocblas_init<T>(hA, K, K, lda);

    //  pad untouched area into zero
    for(int i = K; i < lda; i++)
        for(int j = 0; j < K; j++)
            hA[i + j * lda] = 0.0;

    // Initial hB
    if(nantest)
        rocblas_init_nan<T>(hB, M, N, ldb);
    else
        rocblas_init<T>(hB, M, N, ldb);
    // pad untouched area into zero
    for(int i = M; i < ldb; i++)
        for(int j = 0; j < N; j++)
            hB[i + j * ldb] = 0.0;

    hB_1 = hB; // hXorB <- B
    hB_2 = hB; // hXorB <- B
    cpuB = hB; // cpuB <- B

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB_1, sizeof(T) * size_B, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(
            rocblas_trmm<T>(handle, side, uplo, transA, diag, M, N, &h_alpha_T, dA, lda, dB, ldb));

        CHECK_HIP_ERROR(hipMemcpy(hB_1, dB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        // calculate dB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB_2, sizeof(T) * size_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(alpha_d, &h_alpha_T, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(
            rocblas_trmm<T>(handle, side, uplo, transA, diag, M, N, alpha_d, dA, lda, dB, ldb));

        CHECK_HIP_ERROR(hipMemcpy(hB_2, dB, sizeof(T) * size_B, hipMemcpyDeviceToHost));

        // CPU BLAS
        if(arg.timing)
        {
            cpu_time_used = get_time_us();
        }

        cblas_trmm<T>(side, uplo, transA, diag, M, N, h_alpha_T, hA, lda, cpuB, ldb);

        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = trmm_gflop_count<T>(M, N, side) / cpu_time_used * 1e6;
        }

        if(arg.unit_check)
        {
            if(std::is_same<T, rocblas_half>{} && K > 10000)
            {
                // For large K, rocblas_half tends to diverge proportional to K
                // Tolerance is slightly greater than 1 / 1024.0
                const double tol = K * sum_error_tolerance<T>;
                near_check_general<T>(M, N, ldb, cpuB, hB_1, tol);
                near_check_general<T>(M, N, ldb, cpuB, hB_2, tol);
            }
            else
            {
                unit_check_general<T>(M, N, ldb, cpuB, hB_1);
                unit_check_general<T>(M, N, ldb, cpuB, hB_2);
            }
        }

        if(arg.norm_check)
        {
            auto err1     = std::abs(norm_check_general<T>('F', M, N, ldb, cpuB, hB_1));
            auto err2     = std::abs(norm_check_general<T>('F', M, N, ldb, cpuB, hB_2));
            rocblas_error = err1 > err2 ? err1 : err2;
            rocblas_error = err1;
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_trmm<T>(
                handle, side, uplo, transA, diag, M, N, &h_alpha_T, dA, lda, dB, ldb));
        }

        gpu_time_used = get_time_us(); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_trmm<T>(handle, side, uplo, transA, diag, M, N, &h_alpha_T, dA, lda, dB, ldb);
        }
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = trmm_gflop_count<T>(M, N, side) * number_hot_calls / gpu_time_used * 1e6;

        std::cout << "M,N,alpha,lda,ldb,side,uplo,transA,diag,rocblas-Gflops,us";

        if(arg.unit_check || arg.norm_check)
            std::cout << ",CPU-Gflops,us,norm-error";

        std::cout << std::endl;

        std::cout << M << ',' << N << ',' << arg.get_alpha<T>() << ',' << lda << ',' << ldb << ','
                  << char_side << ',' << char_uplo << ',' << char_transA << ',' << char_diag << ','
                  << rocblas_gflops << "," << gpu_time_used / number_hot_calls;

        if(arg.unit_check || arg.norm_check)
            std::cout << ", " << cblas_gflops << ", " << cpu_time_used << ", " << rocblas_error;

        std::cout << std::endl;
    }
}
