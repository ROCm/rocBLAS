/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_test.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_vector.hpp"
#include "rocblas_init.hpp"
#include "rocblas_datatype2string.hpp"
#include "utility.hpp"
#include "rocblas.hpp"
#include "cblas_interface.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "flops.hpp"

template <typename T>
void testing_trmm(const Arguments& arg)
{
    rocblas_int M   = arg.M;
    rocblas_int N   = arg.N;
    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;

    char char_side   = arg.side;
    char char_uplo   = arg.uplo;
    char char_transA = arg.transA;
    char char_diag   = arg.diag;
    T alpha          = arg.alpha;

    rocblas_side side        = char2rocblas_side(char_side);
    rocblas_fill uplo        = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal diag    = char2rocblas_diagonal(char_diag);

    rocblas_int K = (side == rocblas_side_left ? M : N);
    size_t size_A = lda * static_cast<size_t>(K);
    size_t size_B = ldb * static_cast<size_t>(N);

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < 0 || ldb < 0)
    {
        static const size_t safe_size = 100;
        device_vector<T> dA(safe_size);
        device_vector<T> dB(safe_size);
        device_vector<T> dC(safe_size);
        if(!dA || !dB || !dC)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(
            rocblas_trmm<T>(
                handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb, dC, ldc),
            rocblas_status_invalid_size);

        return;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hB(size_B);
    host_vector<T> hC(size_B);
    host_vector<T> hB_copy(size_B);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;

    rocblas_local_handle handle;

    // allocate memory on device
    // dB and dC are exact the same size
    device_vector<T> dA(size_A);
    device_vector<T> dB(size_B);
    device_vector<T> dC(size_B);
    if(!dA || !dB || !dC)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init_symmetric<T>(hA, K, lda);
    rocblas_init<T>(hB, M, N, ldb);
    hB_copy = hB;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * size_B, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    CHECK_ROCBLAS_ERROR(
        rocblas_trmm<T>(handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb, dC, ldc));

    if(arg.timing)
    {
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = trmm_gflop_count<T>(M, N, K) / gpu_time_used * 1e6;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC, dC, sizeof(T) * size_B, hipMemcpyDeviceToHost));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(arg.timing)
        {
            cpu_time_used = get_time_us();
        }

        cblas_trmm<T>(side, uplo, transA, diag, M, N, alpha, hA, lda, hB_copy, ldb);

        if(arg.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = trmm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;
        }

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldb, hB_copy, hC);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', M, N, ldb, hB_copy, hC);
        }
    }

    if(arg.timing)
    {
        // only norm_check return an norm error, unit check won't return anything
        cout << "M, N, lda, rocblas-Gflops (us) ";
        if(arg.norm_check)
        {
            cout << "CPU-Gflops(us), norm-error";
        }
        cout << endl;

        cout << M << ',' << N << ',' << lda << ',' << rocblas_gflops << "(" << gpu_time_used
             << "),";

        if(arg.norm_check)
        {
            cout << cblas_gflops << "(" << cpu_time_used << "),";
            cout << rocblas_error;
        }

        cout << endl;
    }
}
