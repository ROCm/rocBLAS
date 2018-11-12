/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "rocblas.hpp"
#include "arg_check.h"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

using namespace std;

template <typename T>
void testing_gemv_bad_arg()
{
    const rocblas_int M            = 100;
    const rocblas_int N            = 100;
    const rocblas_int lda          = 100;
    const rocblas_int incx         = 1;
    const rocblas_int incy         = 1;
    const T alpha                  = 1.0;
    const T beta                   = 1.0;
    const rocblas_operation transA = rocblas_operation_none;

    rocblas_local_handle handle;

    rocblas_status status;

    rocblas_int size_A = lda * N;
    rocblas_int size_x = N * incx;
    rocblas_int size_y = M * incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hx, 1, N, incx);
    rocblas_init<T>(hy, 1, M, incy);

    // allocate memory on device
    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    if(!dA || !dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));

    {
        T* dA_null = nullptr;

        status =
            rocblas_gemv<T>(handle, transA, M, N, &alpha, dA_null, lda, dx, incx, &beta, dy, incy);

        verify_rocblas_status_invalid_pointer(status, "rocBLAS TEST ERROR: A is null pointer");
    }
    {
        T* dx_null = nullptr;
        status =
            rocblas_gemv<T>(handle, transA, M, N, &alpha, dA, lda, dx_null, incx, &beta, dy, incy);

        verify_rocblas_status_invalid_pointer(status, "rocBLAS TEST ERROR: x is null pointer");
    }
    {
        T* dy_null = nullptr;
        status =
            rocblas_gemv<T>(handle, transA, M, N, &alpha, dA, lda, dx, incx, &beta, dy_null, incy);

        verify_rocblas_status_invalid_pointer(status, "rocBLAS TEST ERROR: y is null pointer");
    }
    {
        T* alpha_null = nullptr;
        status =
            rocblas_gemv<T>(handle, transA, M, N, alpha_null, dA, lda, dx, incx, &beta, dy, incy);

        verify_rocblas_status_invalid_pointer(status, "rocBLAS TEST ERROR: alpha is null pointer");
    }
    {
        T* beta_null = nullptr;
        status =
            rocblas_gemv<T>(handle, transA, M, N, &alpha, dA, lda, dx, incx, beta_null, dy, incy);

        verify_rocblas_status_invalid_pointer(status, "rocBLAS TEST ERROR: beta is null pointer");
    }
    {
        rocblas_handle handle_null = nullptr;

        status =
            rocblas_gemv<T>(handle_null, transA, M, N, &alpha, dA, lda, dx, incx, &beta, dy, incy);

        verify_rocblas_status_invalid_handle(status);
    }
    return;
}

template <typename T>
rocblas_status testing_gemv(Arguments argus)
{
    rocblas_int M            = argus.M;
    rocblas_int N            = argus.N;
    rocblas_int lda          = argus.lda;
    rocblas_int incx         = argus.incx;
    rocblas_int incy         = argus.incy;
    T h_alpha                = (T)argus.alpha;
    T h_beta                 = (T)argus.beta;
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_int safe_size    = 100; // arbitrarily set to 100

    rocblas_local_handle handle;

    rocblas_status status;

    // argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || lda < M || lda < 1 || 0 == incx || 0 == incy)
    {
        device_vector<T> dA1(safe_size);
        device_vector<T> dx1(safe_size);
        device_vector<T> dy1(safe_size);
        if(!dA1 || !dx1 || !dy1)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_gemv<T>(
            handle, transA, M, N, &h_alpha, dA1, lda, dx1, incx, &h_beta, dy1, incy);

        gemv_ger_arg_check(status, M, N, lda, incx, incy);

        return status;
    }

    rocblas_int size_A = lda * N;
    rocblas_int size_x, dim_x, abs_incx;
    rocblas_int size_y, dim_y, abs_incy;

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

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hx(size_x);
    host_vector<T> hy_1(size_y);
    host_vector<T> hy_2(size_y);
    host_vector<T> hy_gold(size_y);

    device_vector<T> dA(size_A);
    device_vector<T> dx(size_x);
    device_vector<T> dy_1(size_y);
    device_vector<T> dy_2(size_y);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    if((!dA && (size_A != 0)) || (!dx && (size_x != 0)) || ((!dy_1 || !dy_2) && (size_y != 0)) ||
       !d_alpha || !d_beta)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hx, 1, dim_x, abs_incx);
    rocblas_init<T>(hy_1, 1, dim_y, abs_incy);

    // copy vector is easy in STL; hy_gold = hy_1: save a copy in hy_gold which will be output of
    // CPU BLAS
    hy_gold = hy_1;
    hy_2    = hy_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(
            handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(
            rocblas_gemv<T>(handle, transA, M, N, d_alpha, dA, lda, dx, incx, d_beta, dy_2, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * size_y, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_gemv<T>(transA, M, N, h_alpha, hA, lda, hx, incx, h_beta, hy_gold, incy);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = gemv_gflop_count<T>(M, N) / cpu_time_used * 1e6;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold, hy_1);
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold, hy_2);
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold, hy_1);
            rocblas_error_2 = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold, hy_2);
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_gemv<T>(handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_gemv<T>(handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy_1, incy);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        rocblas_gflops    = gemv_gflop_count<T>(M, N) / gpu_time_used * 1e6;
        rocblas_bandwidth = (1.0 * M * N) * sizeof(T) / gpu_time_used / 1e3;

        // only norm_check return an norm error, unit check won't return anything
        cout << "M,N,alpha,lda,incx,incy,rocblas-Gflops,rocblas-GB/s,";
        if(argus.norm_check)
        {
            cout << "CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";
        }
        cout << endl;

        cout << M << "," << N << "," << h_alpha << "," << lda << "," << incx << "," << incy << ","
             << rocblas_gflops << "," << rocblas_bandwidth << ",";

        if(argus.norm_check)
        {
            cout << cblas_gflops << ',';
            cout << rocblas_error_1 << ',' << rocblas_error_2;
        }

        cout << endl;
    }
    return rocblas_status_success;
}
