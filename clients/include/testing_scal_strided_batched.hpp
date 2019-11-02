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

template <typename T, typename U = T>
void testing_scal_strided_batched(const Arguments& arg)
{
    rocblas_int N           = arg.N;
    rocblas_int incx        = arg.incx;
    rocblas_int stridex     = arg.stride_x;
    rocblas_int batch_count = arg.batch_count;
    U           h_alpha     = arg.get_alpha<U>();

    rocblas_local_handle handle;

    // argument sanity check before allocating invalid memory
    // --- do no checking for stride_x ---
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        device_vector<T>    dx(safe_size);
        if(!dx)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        if(batch_count < 0)
            EXPECT_ROCBLAS_STATUS((rocblas_scal_strided_batched<T, U>)(handle,
                                                                       N,
                                                                       &h_alpha,
                                                                       dx,
                                                                       incx,
                                                                       stridex,
                                                                       batch_count),
                                  rocblas_status_invalid_size);
        else
            CHECK_ROCBLAS_ERROR((rocblas_scal_strided_batched<T, U>)(handle,
                                                                     N,
                                                                     &h_alpha,
                                                                     dx,
                                                                     incx,
                                                                     stridex,
                                                                     batch_count));

        return;
    }

    size_t size_x = N * size_t(incx) + size_t(stridex) * size_t(batch_count - 1);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx_1(size_x);
    host_vector<T> hx_2(size_x);
    host_vector<T> hx_gold(size_x);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx_1, 1, N, incx, stridex, batch_count);

    // copy vector is easy in STL; hx_gold = hx: save a copy in hx_gold which will be output of CPU
    // BLAS
    hx_2    = hx_1;
    hx_gold = hx_1;

    // allocate memory on device
    device_vector<T> dx_1(size_x);
    device_vector<T> dx_2(size_x);
    device_vector<U> d_alpha(1);
    if(!dx_1 || !dx_2 || !d_alpha)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_1, hx_1, sizeof(T) * size_x, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    CHECK_HIP_ERROR(hipMemcpy(dx_1, hx_1, sizeof(T) * size_x, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dx_2, hx_2, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));

        // GPU BLAS, rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR((rocblas_scal_strided_batched<T, U>(
            handle, N, &h_alpha, dx_1, incx, stridex, batch_count)));

        // GPU BLAS, rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR((rocblas_scal_strided_batched<T, U>(
            handle, N, d_alpha, dx_2, incx, stridex, batch_count)));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx_1, dx_1, sizeof(T) * size_x, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hx_2, dx_2, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(int i = 0; i < batch_count; i++)
        {
            cblas_scal<T, U>(N, h_alpha, hx_gold + i * stridex, incx);
        }

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = batch_count * scal_gflop_count<T, U>(N) / cpu_time_used * 1e6 * 1;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, incx, stridex, hx_gold, hx_1);
            unit_check_general<T>(1, N, batch_count, incx, stridex, hx_gold, hx_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('F', 1, N, incx, stridex, batch_count, hx_gold, hx_1);
            rocblas_error_2
                = norm_check_general<T>('F', 1, N, incx, stridex, batch_count, hx_gold, hx_2);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_scal_strided_batched<T, U>(
                handle, N, &h_alpha, dx_1, incx, stridex, batch_count);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_scal_strided_batched<T, U>(
                handle, N, &h_alpha, dx_1, incx, stridex, batch_count);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = batch_count * scal_gflop_count<T, U>(N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = batch_count * (2.0 * N) * sizeof(T) / gpu_time_used / 1e3;

        std::cout << "N,alpha,incx,rocblas-Gflops,rocblas-GB/s,rocblas-us";

        if(arg.norm_check)
            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";

        std::cout << std::endl;

        std::cout << N << "," << h_alpha << "," << incx << "," << rocblas_gflops << ","
                  << rocblas_bandwidth << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << cblas_gflops << ',' << rocblas_error_1 << ',' << rocblas_error_2;

        std::cout << std::endl;
    }
}

template <typename T, typename U = T>
void testing_scal_strided_batched_bad_arg(const Arguments& arg)
{
    rocblas_int N           = 100;
    rocblas_int incx        = 1;
    U           h_alpha     = U(1.0);
    rocblas_int batch_count = 5;
    rocblas_int stridex     = 50;

    rocblas_local_handle handle;

    size_t size_x = N * size_t(incx);

    // allocate memory on device
    device_vector<T> dx(size_x);
    if(!dx)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    EXPECT_ROCBLAS_STATUS(
        (rocblas_scal_strided_batched<T, U>)(handle, N, nullptr, dx, incx, stridex, batch_count),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS((rocblas_scal_strided_batched<T, U>)(handle,
                                                               N,
                                                               &h_alpha,
                                                               nullptr,
                                                               incx,
                                                               stridex,
                                                               batch_count),
                          rocblas_status_invalid_pointer);
}
