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
void testing_tbmv_batched_bad_arg(const Arguments& arg)
{
    const rocblas_int M           = 100;
    const rocblas_int K           = 5;
    const rocblas_int lda         = 100;
    const rocblas_int incx        = 1;
    const rocblas_int batch_count = 5;

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
    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dx(batch_count);
    if(!dA || !dx)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(
        rocblas_tbmv_batched<T>(
            handle, uplo, transA, diag, M, K, nullptr, lda, dx, incx, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_tbmv_batched<T>(
            handle, uplo, transA, diag, M, K, dA, lda, nullptr, incx, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_tbmv_batched<T>(nullptr, uplo, transA, diag, M, K, dA, lda, dx, incx, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_tbmv_batched(const Arguments& arg)
{
    rocblas_int       M           = arg.M;
    rocblas_int       K           = arg.K;
    rocblas_int       lda         = arg.lda;
    rocblas_int       incx        = arg.incx;
    char              char_uplo   = arg.uplo;
    char              char_diag   = arg.diag;
    rocblas_fill      uplo        = char2rocblas_fill(char_uplo);
    rocblas_operation transA      = char2rocblas_operation(arg.transA);
    rocblas_diagonal  diag        = char2rocblas_diagonal(char_diag);
    rocblas_int       batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(M < 0 || K < 0 || lda < M || lda < 1 || !incx || K >= lda || batch_count <= 0)
    {
        static const size_t     safe_size = 100; // arbitrarily set to 100
        device_vector<T*, 0, T> dA1(safe_size);
        device_vector<T*, 0, T> dx1(safe_size);
        if(!dA1 || !dx1)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(
            rocblas_tbmv_batched<T>(
                handle, uplo, transA, diag, M, K, dA1, lda, dx1, incx, batch_count),
            (M < 0 || K < 0 || lda < M || lda < 1 || !incx || K >= lda || batch_count < 0)
                ? rocblas_status_invalid_size
                : rocblas_status_success);

        return;
    }

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dx(batch_count);

    if(!dA || !dx)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    size_t size_A = lda * size_t(M);
    size_t size_x, abs_incx;

    abs_incx = incx >= 0 ? incx : -incx;
    size_x   = M * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hx[batch_count];
    host_vector<T> hx_1[batch_count];
    host_vector<T> hx_gold[batch_count];

    device_batch_vector<T> bA(batch_count, size_A);
    device_batch_vector<T> bx(batch_count, size_x);

    int last = batch_count - 1;
    if((!bA[last] && size_A) || (!bx[last] && size_x))
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]      = host_vector<T>(size_A);
        hx[b]      = host_vector<T>(size_x);
        hx_1[b]    = host_vector<T>(size_x);
        hx_gold[b] = host_vector<T>(size_x);

        rocblas_init<T>(hA[b], M, M, lda);
        rocblas_init<T>(hx[b], 1, M, abs_incx);
        hx_gold[b] = hx[b];

        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b], sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bx[b], hx[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dA, bA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */

    if(arg.unit_check || arg.norm_check)
    {
        // nothing ever on host for tbmv
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_tbmv_batched<T>(
            handle, uplo, transA, diag, M, K, dA, lda, dx, incx, batch_count));

        // copy output from device to CPU
        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(hipMemcpy(hx_1[b], bx[b], sizeof(T) * size_x, hipMemcpyDeviceToHost));
        }

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; b++)
            cblas_tbmv<T>(uplo, transA, diag, M, K, hA[b], lda, hx_gold[b], incx);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * tbmv_gflop_count<T>(M, K) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, batch_count, abs_incx, hx_gold, hx_1);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', 1, M, abs_incx, batch_count, hx_gold, hx_1);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_tbmv_batched<T>(
                handle, uplo, transA, diag, M, K, dA, lda, dx, incx, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_tbmv_batched<T>(
                handle, uplo, transA, diag, M, K, dA, lda, dx, incx, batch_count);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * tbmv_gflop_count<T>(M, K) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * (1.0 * M * M) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "M,K,lda,incx,batch_count,rocblas-Gflops,rocblas-GB/s,us,";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,us,norm_error_device_ptr";
        }
        std::cout << std::endl;

        std::cout << M << "," << K << "," << lda << "," << incx << "," << batch_count << ","
                  << rocblas_gflops << "," << rocblas_bandwidth << ","
                  << gpu_time_used / number_hot_calls << ",";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ',' << cpu_time_used << ',';
            std::cout << rocblas_error_1;
        }

        std::cout << std::endl;
    }
}
