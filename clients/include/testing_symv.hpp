/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "rocblas.h"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include "flops.h"

using namespace std;


/* ============================================================================================ */

template<typename T>
rocblas_status testing_symv(Arguments argus)
{

    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int incx = argus.incx;
    rocblas_int incy = argus.incy;

    rocblas_int A_size = lda * N;
    rocblas_int X_size = N * incx;
    rocblas_int Y_size = N * incy;

    rocblas_status status = rocblas_status_success;

    //argument sanity check, quick return if input parameters are invalid before allocating invalid memory
    if ( N < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( lda < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }
    else if ( incx < 0 ){
        status = rocblas_status_invalid_size;
        return status;
    }


    //Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hx(X_size);
    vector<T> hy(Y_size);
    vector<T> hz(Y_size);

    T *dA, *dx, *dy;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;

    T alpha = (T)argus.alpha;
    T beta = (T)argus.beta;

    rocblas_handle handle;

    char char_fill = argus.uplo_option;
    rocblas_fill uplo = char2rocblas_fill(char_fill);

    rocblas_create_handle(&handle);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx, X_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, Y_size * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init_symmetric<T>(hA, N, lda);
    rocblas_init<T>(hx, 1, N, incx);
    rocblas_init<T>(hy, 1, N, incy);

    //copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hz = hy;

    //copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T)*lda*N,  hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T)*N*incy, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

    for(int iter=0;iter<1;iter++){

        status = rocblas_symv<T>(handle,
                     uplo, N,
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
        rocblas_gflops = symv_gflop_count<T> (N) / gpu_time_used * 1e6 * 1;
    }

    //copy output from device to CPU
    hipMemcpy(hy.data(), dy, sizeof(T)*N*incy, hipMemcpyDeviceToHost);

    if(argus.unit_check || argus.norm_check){
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        cblas_symv<T>(uplo, N,
               alpha,
               hA.data(),lda,
               hx.data(), incx,
               beta,
               hz.data(), incy);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = symv_gflop_count<T>(N) / cpu_time_used * 1e6;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(1, N, incy, hz.data(), hy.data());
        }

    for(int i=0; i<32; i++){
        printf("CPU[%d]=%f, GPU[%d]=%f\n", i, hz[i], i, hy[i]);
    }
        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            rocblas_error = norm_check_general<T>('F', 1, N, incy, hz.data(), hy.data());
        }
    }

    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything
            cout << "N, lda, rocblas-Gflops (us) ";
            if(argus.norm_check){
                cout << "CPU-Gflops(us), norm-error" ;
            }
            cout << endl;

            cout << N <<',' << lda <<','<< rocblas_gflops << "(" << gpu_time_used << "),";

            if(argus.norm_check){
                cout << cblas_gflops << "(" << cpu_time_used << "),";
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
