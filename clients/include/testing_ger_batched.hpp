/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T>
void testing_ger_batched_bad_arg(const Arguments& arg)
{
    rocblas_int       M           = 100;
    rocblas_int       N           = 100;
    rocblas_int       incx        = 1;
    rocblas_int       incy        = 1;
    rocblas_int       lda         = 100;
    T                 alpha       = 0.6;
    const rocblas_int batch_count = 5;

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
        rocblas_ger_batched<T>(handle, M, N, &alpha, nullptr, incx, dy, incy, dA, lda, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_ger_batched<T>(handle, M, N, &alpha, dx, incx, nullptr, incy, dA, lda, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_ger_batched<T>(handle, M, N, &alpha, dx, incx, dy, incy, nullptr, lda, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocblas_ger_batched<T>(nullptr, M, N, &alpha, dx, incx, dy, incy, dA, lda, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_ger_batched(const Arguments& arg)
{
    rocblas_int M           = arg.M;
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int incy        = arg.incy;
    rocblas_int lda         = arg.lda;
    T           h_alpha     = arg.get_alpha<T>();
    rocblas_int batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument check before allocating invalid memory
    if(M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy || batch_count < 0)
    {
        device_vector<T*, 0, T> dA(1);
        device_vector<T*, 0, T> dx(1);
        device_vector<T*, 0, T> dy(1);
        if(!dA || !dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_ger_batched<T>(
                                  handle, M, N, &h_alpha, dx, incx, dy, incy, dA, lda, batch_count),
                              rocblas_status_invalid_size);

        return;
    }

    //quick return
    if(!M || !N || !batch_count)
    {
        device_vector<T*, 0, T> dA(1);
        device_vector<T*, 0, T> dx(1);
        device_vector<T*, 0, T> dy(1);
        if(!dA || !dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_ger_batched<T>(
                                  handle, M, N, &h_alpha, dx, incx, dy, incy, dA, lda, batch_count),
                              rocblas_status_success);

        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = lda * N;
    size_t size_x   = M * abs_incx;
    size_t size_y   = N * abs_incy;

    //Device-arrays of pointers to device memory
    device_vector<T*, 0, T> dy(batch_count);
    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dA_1(batch_count);
    device_vector<T*, 0, T> dA_2(batch_count);
    device_vector<T>        d_alpha(1);
    if(!dA_1 || !dA_2 || !dx || !dy || !d_alpha)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    // Host-arrays of pointers to host memory
    host_vector<T> hy[batch_count];
    host_vector<T> hx[batch_count];
    host_vector<T> hA_1[batch_count];
    host_vector<T> hA_2[batch_count];
    host_vector<T> hA_gold[batch_count];

    for(int b = 0; b < batch_count; ++b)
    {
        hy[b]      = host_vector<T>(size_y);
        hx[b]      = host_vector<T>(size_x);
        hA_1[b]    = host_vector<T>(size_A);
        hA_2[b]    = host_vector<T>(size_A);
        hA_gold[b] = host_vector<T>(size_A);
    }

    // Host-arrays of pointers to device memory
    // (intermediate arrays used for the transfers)
    device_batch_vector<T> A_1(batch_count, size_A);
    device_batch_vector<T> A_2(batch_count, size_A);
    device_batch_vector<T> y(batch_count, size_y);
    device_batch_vector<T> x(batch_count, size_x);

    int last = batch_count - 1;
    if((!y[last] && size_y) || (!x[last] && size_x) || ((!A_1[last] || !A_2[last]) && size_A)
       || !d_alpha)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    // Initial Data on CPU
    rocblas_seedrand();

    for(int b = 0; b < batch_count; ++b)
    {
        if(lda >= M)
        {
            rocblas_init<T>(hA_1[b], M, N, lda);
        }
        rocblas_init<T>(hx[b], 1, M, abs_incx);
        rocblas_init<T>(hy[b], 1, N, abs_incy);

        // copy matrix is easy in STL; hA_gold = hA_1: save a copy in hA_gold which will be output of
        // CPU BLAS
        hA_gold[b] = hA_1[b];
        hA_2[b]    = hA_1[b];
    }

    // copy data from CPU to device
    // 1. Use intermediate arrays to access device memory from host
    for(int b = 0; b < batch_count; ++b)
    {
        CHECK_HIP_ERROR(hipMemcpy(A_1[b], hA_1[b], sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(x[b], hx[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(y[b], hy[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
    }

    // 2. Copy intermediate arrays into device arrays
    CHECK_HIP_ERROR(hipMemcpy(dA_1, A_1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, x, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, y, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        for(int b = 0; b < batch_count; ++b)
        {
            CHECK_HIP_ERROR(hipMemcpy(A_2[b], hA_2[b], sizeof(T) * size_A, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(dA_2, A_2, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_ger_batched<T>(
            handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda, batch_count));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_ger_batched<T>(
            handle, M, N, d_alpha, dx, incx, dy, incy, dA_2, lda, batch_count));
        // copy output from device to CPU
        for(int b = 0; b < batch_count; ++b)
        {
            hipMemcpy(hA_1[b], A_1[b], sizeof(T) * size_A, hipMemcpyDeviceToHost);
            hipMemcpy(hA_2[b], A_2[b], sizeof(T) * size_A, hipMemcpyDeviceToHost);
        }

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int b = 0; b < batch_count; ++b)
        {
            cblas_ger<T>(M, N, h_alpha, hx[b], incx, hy[b], incy, hA_gold[b], lda);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * ger_gflop_count<T>(M, N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, lda, hA_gold, hA_1);
            unit_check_general<T>(M, N, batch_count, lda, hA_gold, hA_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, lda, batch_count, hA_gold, hA_1);
            rocblas_error_2 = norm_check_general<T>('F', M, N, lda, batch_count, hA_gold, hA_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_ger_batched<T>(
                handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_ger_batched<T>(
                handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda, batch_count);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * ger_gflop_count<T>(M, N) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * (2.0 * M * N) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "M,N,alpha,incx,incy,lda,batch_count,rocblas-Gflops,rocblas-GB/s";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << M << "," << N << "," << h_alpha << "," << incx << "," << incy << "," << lda
                  << "," << batch_count << "," << rocblas_gflops << "," << rocblas_bandwidth;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}
