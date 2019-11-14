/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
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

template <typename T>
void testing_gemv_batched_bad_arg(const Arguments& arg)
{
    const rocblas_int M           = 100;
    const rocblas_int N           = 100;
    const rocblas_int lda         = 100;
    const rocblas_int incx        = 1;
    const rocblas_int incy        = 1;
    const T           alpha       = 1.0;
    const T           beta        = 1.0;
    const rocblas_int batch_count = 5;

    const rocblas_operation transA = rocblas_operation_none;

    rocblas_local_handle handle;

    // allocate memory on device
    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dy(batch_count);

    if(!dA || !dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_batched<T>(
            handle, transA, M, N, &alpha, nullptr, lda, dx, incx, &beta, dy, incy, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_batched<T>(
            handle, transA, M, N, &alpha, dA, lda, nullptr, incx, &beta, dy, incy, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_batched<T>(
            handle, transA, M, N, &alpha, dA, lda, dx, incx, &beta, nullptr, incy, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_batched<T>(
            handle, transA, M, N, nullptr, dA, lda, dx, incx, &beta, dy, incy, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_batched<T>(
            handle, transA, M, N, &alpha, dA, lda, dx, incx, nullptr, dy, incy, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_gemv_batched<T>(
            nullptr, transA, M, N, &alpha, dA, lda, dx, incx, &beta, dy, incy, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_gemv_batched(const Arguments& arg)
{
    rocblas_int       M           = arg.M;
    rocblas_int       N           = arg.N;
    rocblas_int       lda         = arg.lda;
    rocblas_int       incx        = arg.incx;
    rocblas_int       incy        = arg.incy;
    T                 h_alpha     = arg.get_alpha<T>();
    T                 h_beta      = arg.get_beta<T>();
    rocblas_operation transA      = char2rocblas_operation(arg.transA);
    rocblas_int       batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || lda < M || lda < 1 || !incx || !incy || batch_count <= 0)
    {
        static constexpr size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T*, 0, T> dAA1(safe_size);
        device_vector<T*, 0, T> dxA1(safe_size);
        device_vector<T*, 0, T> dy_1A1(safe_size);

        if(!dAA1 || !dxA1 || !dy_1A1)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_gemv_batched<T>(handle,
                                                      transA,
                                                      M,
                                                      N,
                                                      &h_alpha,
                                                      dAA1,
                                                      lda,
                                                      dxA1,
                                                      incx,
                                                      &h_beta,
                                                      dy_1A1,
                                                      incy,
                                                      batch_count),
                              M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy
                                      || batch_count < 0
                                  ? rocblas_status_invalid_size
                                  : rocblas_status_success);
        return;
    }

    //Device-arrays of pointers to device memory
    device_vector<T*, 0, T> dAA(batch_count);
    device_vector<T*, 0, T> dxA(batch_count);
    device_vector<T*, 0, T> dy_1A(batch_count);
    device_vector<T*, 0, T> dy_2A(batch_count);

    if(!dAA || !dxA || !dy_1A || !dy_2A)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    size_t size_A = lda * static_cast<size_t>(N);
    size_t size_x, dim_x, abs_incx;
    size_t size_y, dim_y, abs_incy;

    if(transA == rocblas_operation_none)
    {
        dim_x = N;
        dim_y = M;
    }
    else
    {
        dim_x = M;
        dim_y = N;
    }

    abs_incx = incx >= 0 ? incx : -incx;
    abs_incy = incy >= 0 ? incy : -incy;

    size_x = dim_x * abs_incx;
    size_y = dim_y * abs_incy;

    // Host-arrays of pointers to host memory
    host_vector<T> hAA[batch_count];
    host_vector<T> hxA[batch_count];
    host_vector<T> hy_1A[batch_count];
    host_vector<T> hy_2A[batch_count];
    host_vector<T> hy_goldA[batch_count];
    for(int b = 0; b < batch_count; ++b)
    {
        hAA[b]      = host_vector<T>(size_A);
        hxA[b]      = host_vector<T>(size_x);
        hy_1A[b]    = host_vector<T>(size_y);
        hy_2A[b]    = host_vector<T>(size_y);
        hy_goldA[b] = host_vector<T>(size_y);
    }

    // Host-arrays of pointers to device memory
    // (intermediate arrays used for the transfers)
    device_batch_vector<T> AA(batch_count, size_A);
    device_batch_vector<T> xA(batch_count, size_x);
    device_batch_vector<T> y_1A(batch_count, size_y);
    device_batch_vector<T> y_2A(batch_count, size_y);

    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    int last = batch_count - 1;
    if((!AA[last] && size_A) || (!xA[last] && size_x) || ((!y_1A[last] || !y_2A[last]) && size_y)
       || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    for(int b = 0; b < batch_count; ++b)
    {
        rocblas_init<T>(hAA[b], M, N, lda);
        rocblas_init<T>(hxA[b], 1, dim_x, abs_incx);
        if(rocblas_isnan(arg.beta))
            rocblas_init_nan<T>(hy_1A[b], 1, dim_y, abs_incy);
        else
            rocblas_init<T>(hy_1A[b], 1, dim_y, abs_incy);
        hy_goldA[b] = hy_1A[b];
        hy_2A[b]    = hy_1A[b];
    }

    // copy data from CPU to device
    // 1. Use intermediate arrays to access device memory from host
    for(int b = 0; b < batch_count; ++b)
    {
        CHECK_HIP_ERROR(hipMemcpy(AA[b], hAA[b], sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(xA[b], hxA[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(y_1A[b], hy_1A[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
    }
    // 2. Copy intermediate arrays into device arrays
    CHECK_HIP_ERROR(hipMemcpy(dAA, AA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dxA, xA, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1A, y_1A, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        for(int b = 0; b < batch_count; ++b)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(y_2A[b], hy_2A[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(dy_2A, y_2A, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gemv_batched<T>(handle,
                                                    transA,
                                                    M,
                                                    N,
                                                    &h_alpha,
                                                    dAA,
                                                    lda,
                                                    dxA,
                                                    incx,
                                                    &h_beta,
                                                    dy_1A,
                                                    incy,
                                                    batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_gemv_batched<T>(
            handle, transA, M, N, d_alpha, dAA, lda, dxA, incx, d_beta, dy_2A, incy, batch_count));

        // copy output from device to CPU
        // Use intermediate arrays to access device memory from host
        for(int b = 0; b < batch_count; ++b)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(hy_1A[b], y_1A[b], sizeof(T) * size_y, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(
                hipMemcpy(hy_2A[b], y_2A[b], sizeof(T) * size_y, hipMemcpyDeviceToHost));
        }

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_gemv<T>(
                transA, M, N, h_alpha, hAA[b], lda, hxA[b], incx, h_beta, hy_goldA[b], incy);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * gemv_gflop_count<T>(transA, M, N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_y, batch_count, abs_incy, hy_goldA, hy_1A);
            unit_check_general<T>(1, dim_y, batch_count, abs_incy, hy_goldA, hy_2A);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', 1, dim_y, abs_incy, batch_count, hy_goldA, hy_1A);
            rocblas_error_2
                = norm_check_general<T>('F', 1, dim_y, abs_incy, batch_count, hy_goldA, hy_2A);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_gemv_batched<T>(handle,
                                    transA,
                                    M,
                                    N,
                                    &h_alpha,
                                    dAA,
                                    lda,
                                    dxA,
                                    incx,
                                    &h_beta,
                                    dy_1A,
                                    incy,
                                    batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_gemv_batched<T>(handle,
                                    transA,
                                    M,
                                    N,
                                    &h_alpha,
                                    dAA,
                                    lda,
                                    dxA,
                                    incx,
                                    &h_beta,
                                    dy_1A,
                                    incy,
                                    batch_count);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * gemv_gflop_count<T>(transA, M, N) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * (1.0 * M * N) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "M,N,alpha,lda,incx,beta,incy,batch_count,rocblas-Gflops,rocblas-GB/s,";
        if(arg.norm_check)
        {
            std::cout << "CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";
        }
        std::cout << std::endl;

        std::cout << M << "," << N << "," << h_alpha << "," << lda << "," << incx << "," << h_beta
                  << "," << incy << "," << batch_count << "," << rocblas_gflops << ","
                  << rocblas_bandwidth << ",";

        if(arg.norm_check)
        {
            std::cout << cblas_gflops << ',';
            std::cout << rocblas_error_1 << ',' << rocblas_error_2;
        }

        std::cout << std::endl;
    }
}
