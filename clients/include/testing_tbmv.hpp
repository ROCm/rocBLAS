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
void testing_tbmv_bad_arg(const Arguments& arg)
{
    const rocblas_int M    = 100;
    const rocblas_int K    = 5;
    const rocblas_int lda  = 100;
    const rocblas_int incx = 1;

    const rocblas_fill      uplo   = rocblas_fill_upper;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    rocblas_local_handle handle;

    size_t size_A = lda * size_t(M);
    size_t size_x = M * size_t(incx);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hx(size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA, M, M, lda);
    rocblas_init<T>(hx, 1, M, incx);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    if(!dA || !dx)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));

    EXPECT_ROCBLAS_STATUS(rocblas_tbmv<T>(handle, uplo, transA, diag, M, K, nullptr, lda, dx, incx),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_tbmv<T>(handle, uplo, transA, diag, M, K, dA, lda, nullptr, incx),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_tbmv<T>(nullptr, uplo, transA, diag, M, K, dA, lda, dx, incx),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_tbmv(const Arguments& arg)
{
    rocblas_int       M         = arg.M;
    rocblas_int       K         = arg.K;
    rocblas_int       lda       = arg.lda;
    rocblas_int       incx      = arg.incx;
    char              char_uplo = arg.uplo;
    char              char_diag = arg.diag;
    rocblas_fill      uplo      = char2rocblas_fill(char_uplo);
    rocblas_operation transA    = char2rocblas_operation(arg.transA);
    rocblas_diagonal  diag      = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(M < 0 || K < 0 || lda < M || lda < 1 || !incx || K >= lda)
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
    host_vector<T> hA(size_A);
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
    rocblas_init<T>(hA, M, M, lda);
    rocblas_init<T>(hx, 1, M, abs_incx);
    hx_gold = hx;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
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
        cblas_tbmv<T>(uplo, transA, diag, M, K, hA, lda, hx_gold, incx);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = tbmv_gflop_count<T>(M, K) / cpu_time_used * 1e6;

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
        int number_hot_calls  = arg.iters;
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
        std::cout << "M,K,lda,incx,rocblas-Gflops,rocblas-GB/s,us,";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,us,norm_error_device_ptr";
        }
        std::cout << std::endl;

        std::cout << M << "," << K << "," << lda << "," << incx << "," << rocblas_gflops << ","
                  << rocblas_bandwidth << "," << gpu_time_used / number_hot_calls << ",";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ',' << cpu_time_used << ',';
            std::cout << rocblas_error_1;
        }

        std::cout << std::endl;
    }
}
