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
void testing_ger_bad_arg()
{
    rocblas_int M = 100;
    rocblas_int N = 100;
    rocblas_int incx = 1;
    rocblas_int incy = 1;
    rocblas_int lda = 100;
    T alpha = 0.6;

    rocblas_handle handle;
    T *dA, *dx, *dy;

    rocblas_status status;
    status = rocblas_create_handle(&handle);

    if(status != rocblas_status_success) {
        printf("ERROR: rocblas_create_handle status = %d\n",status);
        rocblas_destroy_handle(handle);
        return;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    rocblas_int sizeA = lda * N;

    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(sizeA);
    vector<T> hB(sizeA);
    vector<T> hx(M * abs_incx);
    vector<T> hy(N * abs_incy);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, sizeA * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx, M * abs_incx * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, N * abs_incy * sizeof(T)));

    // test if (nullptr == dx)
    {
        T *dx_null = nullptr;
        status = rocblas_ger<T>(handle,
                     M, N,
                     (T*)&alpha,
                     dx_null, incx,
                     dy, incy,
                     dA, lda);

        pointer_check(status,"ERROR: A or x or y is null pointer");
    }
    // test if (nullptr == dy)
    {
        T *dy_null = nullptr;
        status = rocblas_ger<T>(handle,
                     M, N,
                     (T*)&alpha,
                     dx, incx,
                     dy_null, incy,
                     dA, lda);

        pointer_check(status,"ERROR: A or x or y is null pointer");
    }
    // test if (nullptr == dA)
    {
        T *dA_null = nullptr;
        status = rocblas_ger<T>(handle,
                     M, N,
                     (T*)&alpha,
                     dx, incx,
                     dy, incy,
                     dA_null, lda);

        pointer_check(status,"ERROR: A or x or y is null pointer");
    }
    // test if (handle == nullptr)
    {
        rocblas_handle handle_null = nullptr;
        status = rocblas_ger<T>(handle_null,
                     M, N,
                     (T*)&alpha,
                     dx, incx,
                     dy, incy,
                     dA, lda);

        handle_check(status);
    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    CHECK_HIP_ERROR(hipFree(dA));
    rocblas_destroy_handle(handle);
    return;
}

template<typename T>
rocblas_status testing_ger(Arguments argus)
{
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;
    rocblas_int lda = argus.lda;
    T alpha = (T)argus.alpha;

    rocblas_handle handle;
    T *dA, *dx, *dy;

    rocblas_status status;
    status = rocblas_create_handle(&handle);

    if(status != rocblas_status_success) {
        printf("ERROR: rocblas_create_handle status = %d\n",status);
        return status;
    }

    //argument check before allocating invalid memory
    if ( M <= 0 || N <= 0 || lda < M || lda < 1 || 0 == incx || 0 == incy )
    {
        CHECK_HIP_ERROR(hipMalloc(&dA, 100 * sizeof(T)));  //  100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&dx, 100 * sizeof(T)));
        CHECK_HIP_ERROR(hipMalloc(&dy, 100 * sizeof(T)));

        status = rocblas_ger<T>(handle,
                     M, N,
                     (T*)&alpha,
                     dx, incx,
                     dy, incy,
                     dA, lda);

        gemv_ger_arg_check(status, M, N, lda, incx, incy);

        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        CHECK_HIP_ERROR(hipFree(dA));
        rocblas_destroy_handle(handle);
        return status;
    }

    rocblas_int abs_incx = incx >= 0 ? incx : -incx;
    rocblas_int abs_incy = incy >= 0 ? incy : -incy;
    rocblas_int sizeA = lda * N;

    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(sizeA);
    vector<T> hB(sizeA);
    vector<T> hx(M * abs_incx);
    vector<T> hy(N * abs_incy);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, sizeA * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx, M * abs_incx * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, N * abs_incy * sizeof(T)));

    if (nullptr == dx || nullptr == dy || nullptr == dA)
    {
        status = rocblas_ger<T>(handle,
                     M, N,
                     (T*)&alpha,
                     dx, incx,
                     dy, incy,
                     dA, lda);

        pointer_check(status,"ERROR: A or x or y is null pointer");

        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        CHECK_HIP_ERROR(hipFree(dA));
        rocblas_destroy_handle(handle);
        return status;
    }
    else if (handle == nullptr)
    {
        status = rocblas_ger<T>(handle,
                     M, N,
                     (T*)&alpha,
                     dx, incx,
                     dy, incy,
                     dA, lda);

        handle_check(status);

        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        CHECK_HIP_ERROR(hipFree(dA));
        rocblas_destroy_handle(handle);
        return status;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
    double rocblas_error;

    //Initial Data on CPU
    srand(1);
    if( lda >= M ) 
    {
        rocblas_init<T>(hA, M, N, lda);
    }
    rocblas_init<T>(hx, 1, M, abs_incx);
    rocblas_init<T>(hy, 1, N, abs_incy);

    //copy matrix is easy in STL; hB = hA: save a copy in hB which will be output of CPU BLAS
    hB = hA;

    //copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T)*lda*N,  hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T)*M * abs_incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T)*N * abs_incy, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

    for(int iter=0;iter<1;iter++){

        status = rocblas_ger<T>(handle,
                     M, N,
                     (T*)&alpha,
                     dx, incx,
                     dy, incy,
                     dA, lda);

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
        rocblas_gflops = ger_gflop_count<T> (M, N) / gpu_time_used * 1e6 * 1;
        rocblas_bandwidth = (2.0 * M * N) * sizeof(T)/ gpu_time_used / 1e3;
    }

    //copy output from device to CPU
    hipMemcpy(hA.data(), dA, sizeof(T)*N*lda, hipMemcpyDeviceToHost);

    if(argus.unit_check || argus.norm_check){
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        cblas_ger<T>(M, N,
               alpha,
               hx.data(), incx,
               hy.data(), incy,
               hB.data(), lda);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = ger_gflop_count<T>(M, N) / cpu_time_used * 1e6;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(M, N, lda, hB.data(), hA.data());
        }


        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            rocblas_error = norm_check_general<T>('F', M, N, lda, hB.data(), hA.data());
        }
    }

    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything
        cout << "M, N, lda, rocblas-Gflops, rocblas-GB/s, ";
        if(argus.norm_check){
            cout << "CPU-Gflops, norm-error" ;
        }
        cout << endl;

        cout << M << ',' << N << ',' << lda << ',' << rocblas_gflops << ',' << rocblas_bandwidth << ','  ;

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
