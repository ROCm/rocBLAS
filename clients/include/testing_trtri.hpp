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
rocblas_status testing_trtri(Arguments argus)
{
    rocblas_int N = argus.N;
    rocblas_int lda = argus.lda;

    rocblas_int size_A = lda * N;

    // check here to prevent undefined memory allocation error
    if(N < 0 || lda < 0)
    {
        return rocblas_status_invalid_size;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hB(size_A);

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;
    double rocblas_error;

    char char_uplo = argus.uplo_option;
    char char_diag = argus.diag_option;

    rocblas_fill uplo     = char2rocblas_fill(char_uplo);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),
                                         rocblas_test::device_free};
    T* dA    = (T*)dA_managed.get();
    if(!dA)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    // Initial Data on CPU
    srand(1);
    rocblas_init_symmetric<T>(hA, N, lda);

    // proprocess the matrix to avoid ill-conditioned matrix
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hA[i + j * lda] *= 0.01;

            if(j % 2)
                hA[i + j * lda] *= -1;
            if(i == j)
            {
                if(diag == rocblas_diagonal_unit)
                    hA[i + j * lda] = 1.0;
                else
                    hA[i + j * lda] *= 100.0;
            }
        }
    }
    hB = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    status = rocblas_trtri<T>(handle, uplo, diag, N, dA, lda);

    if(argus.timing)
    {
        gpu_time_used  = get_time_us() - gpu_time_used;
        rocblas_gflops = trtri_gflop_count<T>(N) / gpu_time_used * 1e6;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA.data(), dA, sizeof(T) * size_A, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(argus.timing)
        {
            cpu_time_used = get_time_us();
        }

        rocblas_int info = cblas_trtri<T>(char_uplo, char_diag, N, hB.data(), lda);

        if(info != 0)
            printf("error in cblas_trtri\n");

        if(argus.timing)
        {
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = trtri_gflop_count<T>(N) / cpu_time_used * 1e6;
        }

#ifndef NDEBUG
        print_matrix(hB, hA, N, N, lda);
#endif

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, lda, hB.data(), hA.data());
        }

        // if enable norm check, norm check is invasive
        // any typeinfo(T) will not work here, because template deduction is matched in compilation
        // time
        if(argus.norm_check)
        {
            rocblas_error = norm_check_symmetric<T>('F', char_uplo, N, lda, hB.data(), hA.data());
        }
    }

    if(argus.timing)
    {
        // only norm_check return an norm error, unit check won't return anything
        cout << "N, lda, uplo, diag, rocblas-Gflops (us) ";
        if(argus.norm_check)
        {
            cout << "CPU-Gflops(us), norm-error";
        }
        cout << endl;

        cout << N << ',' << lda << ',' << char_uplo << ',' << char_diag << ',' << rocblas_gflops
             << "(" << gpu_time_used << "),";

        if(argus.norm_check)
        {
            cout << cblas_gflops << "(" << cpu_time_used << "),";
            cout << rocblas_error;
        }

        cout << endl;
    }

    return rocblas_status_success;
}
