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
void testing_syr_batched_bad_arg()
{
    rocblas_fill         uplo        = rocblas_fill_upper;
    rocblas_int          N           = 100;
    rocblas_int          incx        = 1;
    rocblas_int          lda         = 100;
    T                    alpha       = 0.6;
    rocblas_int          batch_count = 2;
    rocblas_local_handle handle;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t size_A   = lda * N * batch_count;
    size_t size_x   = N * abs_incx * batch_count;

    // allocate memory on device
    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dA_1(batch_count);
    if(!dx || !dA_1)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(
        rocblas_syr_batched<T>(handle, uplo, N, &alpha, nullptr, incx, dA_1, lda, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_syr_batched<T>(handle, uplo, N, &alpha, dx, incx, nullptr, lda, batch_count),
        rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(
        rocblas_syr_batched<T>(nullptr, uplo, N, &alpha, dx, incx, dA_1, lda, batch_count),
        rocblas_status_invalid_handle);
}

template <typename T>
void testing_syr_batched(const Arguments& arg)
{
    rocblas_int  N           = arg.N;
    rocblas_int  incx        = arg.incx;
    rocblas_int  lda         = arg.lda;
    T            h_alpha     = arg.get_alpha<T>();
    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);
    rocblas_int  batch_count = arg.batch_count;

    rocblas_local_handle handle;

    // argument check before allocating invalid memory
    if(N < 0 || lda < N || lda < 1 || !incx || batch_count < 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100

        device_vector<T*, 0, T> dx(std::max(1, batch_count));
        device_vector<T*, 0, T> dA_1(std::max(1, batch_count));
        if(!dx || !dA_1)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(
            rocblas_syr_batched<T>(handle, uplo, N, &h_alpha, dx, incx, dA_1, lda, batch_count),
            rocblas_status_invalid_size);
        return;
    }

    if(N <= 0 || batch_count == 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100

        device_vector<T*, 0, T> dx(std::max(1, batch_count));
        device_vector<T*, 0, T> dA_1(std::max(1, batch_count));
        if(!dx || !dA_1)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(
            rocblas_syr_batched<T>(handle, uplo, N, &h_alpha, dx, incx, dA_1, lda, batch_count));
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    if(abs_incx == 0)
        abs_incx = 1;

    size_t size_A = size_t(lda) * N;
    size_t size_x = size_t(N) * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA_1[batch_count];
    host_vector<T> hA_2[batch_count];
    host_vector<T> hA_gold[batch_count];
    host_vector<T> hx[batch_count];

    for(int i = 0; i < batch_count; i++)
    {
        hA_1[i]    = host_vector<T>(size_A);
        hA_2[i]    = host_vector<T>(size_A);
        hA_gold[i] = host_vector<T>(size_A);
        hx[i]      = host_vector<T>(size_x);
    }

    // allocate memory on device
    device_batch_vector<T> dA_1(batch_count, size_A);
    device_batch_vector<T> dA_2(batch_count, size_A);
    device_batch_vector<T> dx(batch_count, size_x);
    device_vector<T>       d_alpha(1);
    if(!dA_1 || !dA_2 || !dx || !d_alpha)
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
    for(int i = 0; i < batch_count; i++)
    {
        if(lda >= N)
        {
            rocblas_init_symmetric<T>(hA_1[i], N, lda);
        }
        rocblas_init<T>(hx[i], 1, N, abs_incx);
    }

    // copy matrix is easy in STL; hA_gold = hA_1: save a copy in hA_gold which will be output of
    // CPU BLAS
    for(int i = 0; i < batch_count; i++)
    {
        hA_gold[i] = hA_1[i];
        hA_2[i]    = hA_1[i];
    }

    // copy data from CPU to device
    for(int i = 0; i < batch_count; i++)
    {
        CHECK_HIP_ERROR(hipMemcpy(dA_1[i], hA_1[i], sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dA_2[i], hA_2[i], sizeof(T) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dx[i], hx[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
    }

    // vector pointers on gpu
    device_vector<T*, 0, T> dx_pvec(batch_count);
    device_vector<T*, 0, T> dA1_pvec(batch_count);
    device_vector<T*, 0, T> dA2_pvec(batch_count);
    if(!dx_pvec || !dA1_pvec || !dA2_pvec)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy gpu vector pointers from host to device pointer array
    CHECK_HIP_ERROR(hipMemcpy(dx_pvec, dx, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA1_pvec, dA_1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA2_pvec, dA_2, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        //CHECK_HIP_ERROR(hipMemcpy(dA_2, hA_2, sizeof(T) * lda * N, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_syr_batched<T>(
            handle, uplo, N, &h_alpha, dx_pvec, incx, dA1_pvec, lda, batch_count));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_syr_batched<T>(
            handle, uplo, N, d_alpha, dx_pvec, incx, dA2_pvec, lda, batch_count));

        // copy output from device to CPU
        for(int i = 0; i < batch_count; i++)
        {
            hipMemcpy(hA_1[i], dA_1[i], sizeof(T) * size_A, hipMemcpyDeviceToHost);
            hipMemcpy(hA_2[i], dA_2[i], sizeof(T) * size_A, hipMemcpyDeviceToHost);
        }

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_syr<T>(uplo, N, h_alpha, hx[i], incx, hA_gold[i], lda);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * syr_gflop_count<T>(N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            for(int i = 0; i < batch_count; i++)
            {
                unit_check_general<T>(N, N, lda, hA_gold[i], hA_1[i]);
                unit_check_general<T>(N, N, lda, hA_gold[i], hA_2[i]);
            }
        }

        if(arg.norm_check)
        {
            for(int i = 0; i < batch_count; i++)
            {
                rocblas_error_1 = norm_check_general<T>('F', N, N, lda, hA_gold[i], hA_1[i]);
                rocblas_error_2 = norm_check_general<T>('F', N, N, lda, hA_gold[i], hA_2[i]);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_syr_batched<T>(handle, uplo, N, &h_alpha, dx, incx, dA_1, lda, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_syr_batched<T>(handle, uplo, N, &h_alpha, dx, incx, dA_1, lda, batch_count);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * syr_gflop_count<T>(N) / gpu_time_used * 1e6;
        rocblas_bandwidth = batch_count * (2.0 * N * (N + 1)) / 2 * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "N,alpha,incx,lda,batch_count,rocblas-Gflops,rocblas-GB/s";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << N << "," << h_alpha << "," << incx << "," << lda << "," << batch_count << ","
                  << rocblas_gflops << "," << rocblas_bandwidth;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}
