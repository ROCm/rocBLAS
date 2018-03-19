/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "rocblas.hpp"
#include "arg_check.h"
#include "rocblas_test_unique_ptr.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

using namespace std;

template <typename T>
void testing_syr_bad_arg()
{
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int N     = 100;
    rocblas_int incx  = 1;
    rocblas_int lda   = 100;
    T alpha           = 0.6;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int size_A   = lda * N;
    rocblas_int size_x   = N * abs_incx;

    // allocate memory on device
    auto dA_1_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                           rocblas_test::device_free};
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),
                                         rocblas_test::device_free};
    T* dA_1 = (T*)dA_1_managed.get();
    T* dx   = (T*)dx_managed.get();
    if(!dA_1 || !dx)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // test if (nullptr == dx)
    {
        T* dx_null = nullptr;
        status     = rocblas_syr<T>(handle, uplo, N, (T*)&alpha, dx_null, incx, dA_1, lda);

        verify_rocblas_status_invalid_pointer(status, "ERROR: A or x is null pointer");
    }
    // test if (nullptr == dA_1)
    {
        T* dA_1_null = nullptr;
        status       = rocblas_syr<T>(handle, uplo, N, (T*)&alpha, dx, incx, dA_1_null, lda);

        verify_rocblas_status_invalid_pointer(status, "ERROR: A or x is null pointer");
    }
    // test if (handle == nullptr)
    {
        rocblas_handle handle_null = nullptr;
        status = rocblas_syr<T>(handle_null, uplo, N, (T*)&alpha, dx, incx, dA_1, lda);

        verify_rocblas_status_invalid_handle(status);
    }
    return;
}

template <typename T>
rocblas_status testing_syr(Arguments argus)
{
    rocblas_int N    = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int lda  = argus.lda;
    T h_alpha        = (T)argus.alpha;

    rocblas_int safe_size = 100; // arbitrarily set to 100

    rocblas_fill uplo = char2rocblas_fill(argus.uplo_option);

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    // argument check before allocating invalid memory
    if(N <= 0 || lda < N || lda < 1 || 0 == incx)
    {
        auto dA_1_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                               rocblas_test::device_free};
        auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),
                                             rocblas_test::device_free};
        T* dA_1 = (T*)dA_1_managed.get();
        T* dx   = (T*)dx_managed.get();
        if(!dA_1 || !dx)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_syr<T>(handle, uplo, N, (T*)&h_alpha, dx, incx, dA_1, lda);

        syr_arg_check(status, N, lda, incx);

        return status;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int size_A   = lda * N;
    rocblas_int size_x   = N * abs_incx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA_1(size_A);
    vector<T> hA_2(size_A);
    vector<T> hA_gold(size_A);
    vector<T> hx(N * abs_incx);

    // allocate memory on device
    auto dA_1_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                           rocblas_test::device_free};
    auto dA_2_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                           rocblas_test::device_free};
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),
                                         rocblas_test::device_free};
    auto d_alpha_managed =
        rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)), rocblas_test::device_free};
    T* dA_1    = (T*)dA_1_managed.get();
    T* dA_2    = (T*)dA_2_managed.get();
    T* dx      = (T*)dx_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    if(!dA_1 || !dA_2 || !dx || !d_alpha)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    // Initial Data on CPU
    srand(1);
    if(lda >= N)
    {
        rocblas_init_symmetric<T>(hA_1, N, lda);
    }
    rocblas_init<T>(hx, 1, N, abs_incx);

    // copy matrix is easy in STL; hA_gold = hA_1: save a copy in hA_gold which will be output of
    // CPU BLAS
    hA_gold = hA_1;
    hA_2    = hA_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA_1, hA_1.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * N * abs_incx, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dA_2, hA_2.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(
            rocblas_syr<T>(handle, uplo, N, (T*)&h_alpha, dx, incx, dA_1, lda));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_syr<T>(handle, uplo, N, d_alpha, dx, incx, dA_2, lda));

        // copy output from device to CPU
        hipMemcpy(hA_1.data(), dA_1, sizeof(T) * N * lda, hipMemcpyDeviceToHost);
        hipMemcpy(hA_2.data(), dA_2, sizeof(T) * N * lda, hipMemcpyDeviceToHost);

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_syr<T>(uplo, N, h_alpha, hx.data(), incx, hA_gold.data(), lda);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = syr_gflop_count<T>(N) / cpu_time_used * 1e6;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, lda, hA_gold.data(), hA_1.data());
            unit_check_general<T>(N, N, lda, hA_gold.data(), hA_2.data());
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', N, N, lda, hA_gold.data(), hA_1.data());
            rocblas_error_2 = norm_check_general<T>('F', N, N, lda, hA_gold.data(), hA_2.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_syr<T>(handle, uplo, N, (T*)&h_alpha, dx, incx, dA_1, lda);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_syr<T>(handle, uplo, N, (T*)&h_alpha, dx, incx, dA_1, lda);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = syr_gflop_count<T>(N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (2.0 * N * N) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        cout << "N,alpha,incx,lda,rocblas-Gflops,rocblas-GB/s";

        if(argus.norm_check)
            cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";

        cout << endl;

        cout <<  N << "," << h_alpha << "," << incx << "," << lda << "," << rocblas_gflops
             << "," << rocblas_bandwidth;

        if(argus.norm_check)
            cout << "," << cblas_gflops << "," << rocblas_error_1 << "," << rocblas_error_2;

        cout << endl;
    }

    return rocblas_status_success;
}
