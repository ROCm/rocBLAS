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
void testing_ger_bad_arg(const Arguments& arg)
{
    rocblas_int M     = 100;
    rocblas_int N     = 100;
    rocblas_int incx  = 1;
    rocblas_int incy  = 1;
    rocblas_int lda   = 100;
    T           alpha = 0.6;

    rocblas_local_handle handle;

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    size_t      size_A   = lda * size_t(N);
    size_t      size_x   = M * size_t(abs_incx);
    size_t      size_y   = N * size_t(abs_incy);

    // allocate memory on device
    device_vector<T> dA_1(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    if(!dA_1 || !dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(rocblas_ger<T>(handle, M, N, &alpha, nullptr, incx, dy, incy, dA_1, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_ger<T>(handle, M, N, &alpha, dx, incx, nullptr, incy, dA_1, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_ger<T>(handle, M, N, &alpha, dx, incx, dy, incy, nullptr, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocblas_ger<T>(nullptr, M, N, &alpha, dx, incx, dy, incy, dA_1, lda),
                          rocblas_status_invalid_handle);
}

template <typename T>
void testing_ger(const Arguments& arg)
{
    rocblas_int M       = arg.M;
    rocblas_int N       = arg.N;
    rocblas_int incx    = arg.incx;
    rocblas_int incy    = arg.incy;
    rocblas_int lda     = arg.lda;
    T           h_alpha = arg.get_alpha<T>();

    rocblas_local_handle handle;

    // argument check before allocating invalid memory
    if(M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dA_1(safe_size);
        device_vector<T>    dx(safe_size);
        device_vector<T>    dy(safe_size);
        if(!dA_1 || !dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCBLAS_STATUS(rocblas_ger<T>(handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda),
                              rocblas_status_invalid_size);

        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_A   = lda * N;
    size_t size_x   = M * abs_incx;
    size_t size_y   = N * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA_1(size_A);
    host_vector<T> hA_2(size_A);
    host_vector<T> hA_gold(size_A);
    host_vector<T> hx(M * abs_incx);
    host_vector<T> hy(N * abs_incy);

    // allocate memory on device
    device_vector<T> dA_1(size_A);
    device_vector<T> dA_2(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> d_alpha(1);
    if(!dA_1 || !dA_2 || !dx || !dy || !d_alpha)
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
    if(lda >= M)
    {
        rocblas_init<T>(hA_1, M, N, lda);
    }
    rocblas_init<T>(hx, 1, M, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    // copy matrix is easy in STL; hA_gold = hA_1: save a copy in hA_gold which will be output of
    // CPU BLAS
    hA_gold = hA_1;
    hA_2    = hA_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA_1, hA_1, sizeof(T) * lda * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * M * abs_incx, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * N * abs_incy, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dA_2, hA_2, sizeof(T) * lda * N, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_ger<T>(handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_ger<T>(handle, M, N, d_alpha, dx, incx, dy, incy, dA_2, lda));

        // copy output from device to CPU
        hipMemcpy(hA_1, dA_1, sizeof(T) * N * lda, hipMemcpyDeviceToHost);
        hipMemcpy(hA_2, dA_2, sizeof(T) * N * lda, hipMemcpyDeviceToHost);

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_ger<T>(M, N, h_alpha, hx, incx, hy, incy, hA_gold, lda);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = ger_gflop_count<T>(M, N) / cpu_time_used * 1e6;

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, lda, hA_gold, hA_1);
            unit_check_general<T>(M, N, lda, hA_gold, hA_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', M, N, lda, hA_gold, hA_1);
            rocblas_error_2 = norm_check_general<T>('F', M, N, lda, hA_gold, hA_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_ger<T>(handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_ger<T>(handle, M, N, &h_alpha, dx, incx, dy, incy, dA_1, lda);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = ger_gflop_count<T>(M, N) / gpu_time_used * 1e6;
        rocblas_bandwidth = (2.0 * M * N) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        std::cout << "M,N,alpha,incx,incy,lda,rocblas-Gflops,rocblas-GB/s";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        std::cout << std::endl;

        std::cout << M << "," << N << "," << h_alpha << "," << incx << "," << incy << "," << lda
                  << "," << rocblas_gflops << "," << rocblas_bandwidth;

        if(arg.norm_check)
            std::cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
    }
}
