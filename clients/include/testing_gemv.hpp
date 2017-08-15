/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "rocblas.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "arg_check.h"
#include "flops.h"

using namespace std;


/* ============================================================================================ */

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

    rocblas_handle handle;
    T *dA, *dx, *dy;

    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) {
        rocblas_destroy_handle(handle);
        return;
    }

    rocblas_int A_size = lda * N;
    rocblas_int X_size = N;
    rocblas_int Y_size = M;

    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hx(X_size * incx);
    vector<T> hy(Y_size * incy);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx, X_size * incx * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, Y_size * incy * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hx, 1, X_size, incx);
    rocblas_init<T>(hy, 1, Y_size, incy);

    //copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T)*lda*N,  hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T)*X_size * incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T)*Y_size * incy, hipMemcpyHostToDevice);

    {
        T *dA_null = nullptr;

        status = rocblas_gemv<T>(handle,
                     transA, M, N,
                     (T*)&alpha,
                     dA_null, lda,
                     dx, incx,
                     (T*)&beta,
                     dy, incy);

        verify_rocblas_status_invalid_pointer(status,"ERROR: A is null pointer");
    }
    {
        T *dx_null = nullptr;
        status = rocblas_gemv<T>(handle,
                     transA, M, N,
                     (T*)&alpha,
                     dA, lda,
                     dx_null, incx,
                     (T*)&beta,
                     dy, incy);

        verify_rocblas_status_invalid_pointer(status,"ERROR: x is null pointer");
    }
    {
        T *dy_null = nullptr;
        status = rocblas_gemv<T>(handle,
                     transA, M, N,
                     (T*)&alpha,
                     dA, lda,
                     dx, incx,
                     (T*)&beta,
                     dy_null, incy);

        verify_rocblas_status_invalid_pointer(status,"ERROR: y is null pointer");
    }
    {
        T *beta_null = nullptr;
        status = rocblas_gemv<T>(handle,
                     transA, M, N,
                     (T*)&alpha,
                     dA, lda,
                     dx, incx,
                     beta_null,
                     dy, incy);

        verify_rocblas_status_invalid_pointer(status,"ERROR: beta is null pointer");
    }
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_gemv<T>(handle_null,
                     transA, M, N,
                     (T*)&alpha,
                     dA, lda,
                     dx, incx,
                     (T*)&beta,
                     dy, incy);

        verify_rocblas_status_invalid_handle(status);
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    rocblas_destroy_handle(handle);
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
    T alpha = (T)argus.alpha;
    T beta = (T)argus.beta;
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);

    T *dA, *dx, *dy;

    rocblas_handle handle;
    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) {
        rocblas_destroy_handle(handle);
        return status;
    }

    //argument sanity check before allocating invalid memory
    if (M < 0 || N < 0 || lda < M || lda < 1 || 0 == incx || 0 == incy)
    {
        CHECK_HIP_ERROR(hipMalloc(&dA, 100 * sizeof(T)));  //  100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));
        CHECK_HIP_ERROR(hipMalloc(&dy, 100 * sizeof(T)));

        status = rocblas_gemv<T>(handle,
                     transA, M, N,
                     (T*)&alpha,
                     dA, lda,
                     dx, incx,
                     (T*)&beta,
                     dy, incy);

        gemv_ger_arg_check(status, M, N, lda, incx, incy);

        return status;
    }

    rocblas_int A_size = lda * N;
    rocblas_int X_size;
    rocblas_int Y_size;

    //transA = rocblas_operation_transpose;
    if(transA == rocblas_operation_none){
        X_size = N ;
        Y_size = M ;
    }
    else{
        X_size = M ;
        Y_size = N ;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;

    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hx(X_size * abs_incx);
    vector<T> hy(Y_size * abs_incy);
    vector<T> hz(Y_size * abs_incy);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx, X_size * abs_incx * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, Y_size * abs_incy * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hx, 1, X_size, abs_incx);
    rocblas_init<T>(hy, 1, Y_size, abs_incy);

    //copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hz = hy;

    //copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T)*lda*N,  hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T)*X_size * abs_incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T)*Y_size * abs_incy, hipMemcpyHostToDevice);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

    for(int iter=0;iter<1;iter++){

        status = rocblas_gemv<T>(handle,
                     transA, M, N,
                     (T*)&alpha,
                     dA, lda,
                     dx, incx,
                     (T*)&beta,
                     dy, incy);

        if (status != rocblas_status_success) {
            CHECK_HIP_ERROR(hipFree(dA));
            CHECK_HIP_ERROR(hipFree(dx));
            CHECK_HIP_ERROR(hipFree(dy));
            rocblas_destroy_handle(handle);
            return status;
        }
    }
    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = gemv_gflop_count<T> (M, N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (1.0 * M * N) * sizeof(T)/ gpu_time_used / 1e3;
    }

    //copy output from device to CPU
    hipMemcpy(hy.data(), dy, sizeof(T)*Y_size*abs_incy, hipMemcpyDeviceToHost);

    if(argus.unit_check || argus.norm_check){
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        cblas_gemv<T>(transA, M, N,
               alpha,
               hA.data(), lda,
               hx.data(), incx,
               beta,
               hz.data(), incy);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = gemv_gflop_count<T>(M, N) / cpu_time_used * 1e6;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(1, Y_size, abs_incy, hz.data(), hy.data());
        }


        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            rocblas_error = norm_check_general<T>('F', 1, Y_size, abs_incy, hz.data(), hy.data());
        }
    }

    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything
        cout << "M, N, lda, rocblas-Gflops, rocblas-GB/s, ";
        if(argus.norm_check){
            cout << "CPU-Gflops, norm-error" ;
        }
        cout << endl;

        cout << "GEMV,"<< M << ',' << N <<',' << lda <<','<< rocblas_gflops << ',' << rocblas_bandwidth << ','  ;

        if(argus.norm_check){
            cout << cblas_gflops << ',';
            cout << rocblas_error;
        }

        cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
