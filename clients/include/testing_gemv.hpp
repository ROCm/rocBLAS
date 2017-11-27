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
#include "rocblas_test_unique_ptr.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

using namespace std;

template<typename T>
void testing_gemv_bad_arg()
{
    const rocblas_int M = 100;
    const rocblas_int N = 100;
    const rocblas_int lda = 100;
    const rocblas_int incx = 1;
    const rocblas_int incy = 1;
    const T alpha = 1.0;
    const T beta = 1.0;
    const rocblas_operation transA = rocblas_operation_none;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_status status;

    rocblas_int size_A = lda * N;
    rocblas_int size_x = N * incx;
    rocblas_int size_y = M * incy;;

    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hx(size_x);
    vector<T> hy(size_y);

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hx, 1, N, incx);
    rocblas_init<T>(hy, 1, M, incy);

    //allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),rocblas_test::device_free};
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),rocblas_test::device_free};
    auto dy_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_y),rocblas_test::device_free};
    T* dA = (T*) dA_managed.get();
    T* dx = (T*) dx_managed.get();
    T* dy = (T*) dy_managed.get();
    if (!dA || !dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*size_A,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*size_x * incx, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T)*size_y * incy, hipMemcpyHostToDevice));

    {
        T *dA_null = nullptr;
        
        status = rocblas_gemv<T>(handle, transA, M, N, (T*)&alpha, dA_null, lda, dx, incx, (T*)&beta, dy, incy);

        verify_rocblas_status_invalid_pointer(status,"rocBLAS TEST ERROR: A is null pointer");
    }
    {
        T *dx_null = nullptr;
        status = rocblas_gemv<T>(handle, transA, M, N, (T*)&alpha, dA, lda, dx_null, incx, (T*)&beta, dy, incy);

        verify_rocblas_status_invalid_pointer(status,"rocBLAS TEST ERROR: x is null pointer");
    }
    {
        T *dy_null = nullptr;
        status = rocblas_gemv<T>(handle, transA, M, N, (T*)&alpha, dA, lda, dx, incx, (T*)&beta, dy_null, incy);

        verify_rocblas_status_invalid_pointer(status,"rocBLAS TEST ERROR: y is null pointer");
    }
    {
        T *beta_null = nullptr;
        status = rocblas_gemv<T>(handle, transA, M, N, (T*)&alpha, dA, lda, dx, incx, beta_null, dy, incy);

        verify_rocblas_status_invalid_pointer(status,"rocBLAS TEST ERROR: beta is null pointer");
    }
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_gemv<T>(handle_null, transA, M, N, (T*)&alpha, dA, lda, dx, incx, (T*)&beta, dy, incy);

        verify_rocblas_status_invalid_handle(status);
    }
    return;
}

template<typename T>
rocblas_status testing_gemv(Arguments argus)
{
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;
    T h_alpha = (T)argus.alpha;
    T h_beta = (T)argus.beta;
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_int safe_size = 100;  // arbitrarily set to 100

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_status status;

    //argument sanity check before allocating invalid memory
    if (M <= 0 || N <= 0 || lda < M || lda < 1 || 0 == incx || 0 == incy)
    {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        auto dy_1_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        T* dA1 = (T*) dA_managed.get();
        T* dx1 = (T*) dx_managed.get();
        T* dy1 = (T*) dy_1_managed.get();

        if (!dA1 || !dx1 || !dy1)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_gemv<T>(handle, transA, M, N, (T*)&h_alpha, dA1, lda, dx1, incx, (T*)&h_beta, dy1, incy);

        gemv_ger_arg_check(status, M, N, lda, incx, incy);

        return status;
    }

    rocblas_int size_A = lda * N;
    rocblas_int size_x, dim_x, abs_incx;
    rocblas_int size_y, dim_y, abs_incy;

    if(transA == rocblas_operation_none)
    {
        dim_x = N ;
        dim_y = M ;
    }
    else
    {
        dim_x = M ;
        dim_y = N ;
    }

    abs_incx = incx >= 0 ? incx : -incx;
    abs_incy = incy >= 0 ? incy : -incy;

    size_x = dim_x * abs_incx;
    size_y = dim_y * abs_incy;
  
    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hx(size_x);
    vector<T> hy_1(size_y);
    vector<T> hy_2(size_y);
    vector<T> hy_gold(size_y);

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),rocblas_test::device_free};
    auto dx_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_x),rocblas_test::device_free};
    auto dy_1_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_y),rocblas_test::device_free};
    auto dy_2_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_y),rocblas_test::device_free};
    auto d_alpha_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)),rocblas_test::device_free};
    auto d_beta_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)),rocblas_test::device_free};
    T* dA = (T*) dA_managed.get();
    T* dx = (T*) dx_managed.get();
    T* dy_1 = (T*) dy_1_managed.get();
    T* dy_2 = (T*) dy_2_managed.get();
    T* d_alpha = (T*) d_alpha_managed.get();
    T* d_beta= (T*) d_beta_managed.get();

    if ((!dA && (size_A != 0)) || (!dx && (size_x != 0)) || ((!dy_1 || !dy_2) && (size_y != 0)) || !d_alpha || !d_beta)
    {
       PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
       return rocblas_status_memory_error;
    }

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hx, 1, dim_x, abs_incx);
    rocblas_init<T>(hy_1, 1, dim_y, abs_incy);

    //copy vector is easy in STL; hy_gold = hy_1: save a copy in hy_gold which will be output of CPU BLAS
    hy_gold = hy_1;
    hy_2 = hy_1;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T)*size_y, hipMemcpyHostToDevice));

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        int number_timing_iterations = 1;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        gpu_time_used = get_time_us();// in microseconds

        for (int iter = 0; iter < number_timing_iterations; iter++)
        {
            rocblas_gemv<T>(handle, transA, M, N, (T*)&h_alpha, dA, lda, dx, incx, (T*)&h_beta, dy_1, incy);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_timing_iterations;
        rocblas_gflops = gemv_gflop_count<T> (M, N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (1.0 * M * N) * sizeof(T)/ gpu_time_used / 1e3;
    }

    if(argus.unit_check || argus.norm_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T)*size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T)*size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice))
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice))

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle, transA, M, N, (T*)&h_alpha, dA, lda, dx, incx, (T*)&h_beta, dy_1, incy));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle, transA, M, N, d_alpha, dA, lda, dx, incx, d_beta, dy_2, incy));

        //copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T)*size_y, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T)*size_y, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_gemv<T>(transA, M, N, h_alpha, hA.data(), lda, hx.data(), incx, h_beta, hy_gold.data(), incy);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops = gemv_gflop_count<T>(M, N) / cpu_time_used * 1e6;

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold.data(), hy_1.data());
            unit_check_general<T>(1, dim_y, abs_incy, hy_gold.data(), hy_2.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check)
        {
            rocblas_error_1 = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold.data(), hy_1.data());
            rocblas_error_2 = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_gold.data(), hy_2.data());
        }
    }

    if(argus.timing)
    {
        //only norm_check return an norm error, unit check won't return anything
        cout << "M,N,lda,rocblas-Gflops,rocblas-GB/s,";
        if(argus.norm_check)
        {
            cout << "CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr" ;
        }
        cout << endl;

        cout << "GGG,"<< M << ',' << N <<',' << lda <<','<< rocblas_gflops << ',' << rocblas_bandwidth << ','  ;

        if(argus.norm_check)
        {
            cout << cblas_gflops << ',';
            cout << rocblas_error_1 << ',' << rocblas_error_2;
        }

        cout << endl;
    }
    return rocblas_status_success;
}
