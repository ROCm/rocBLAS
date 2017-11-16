/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <sys/time.h>
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
#include <typeinfo>

using namespace std;

template<typename T>
rocblas_status testing_gemm_strided_batched(Arguments argus)
{
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    T alpha = argus.alpha;
    T beta = argus.beta;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;
    rocblas_int batch_count = argus.batch_count;
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int safe_size = 100;  // arbitrarily set to 100

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

//  make bsa, bsb, bsc two times minimum size so matrices are non-contiguous
    rocblas_int bsa = lda * A_col * 2; 
    rocblas_int bsb = ldb * B_col * 2; 
    rocblas_int bsc = ldc * N * 2;

    //check here to prevent undefined memory allocation error
    if( M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 || batch_count < 0)
    {
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
        T* dA = (T*) dA_managed.get();
        T* dB = (T*) dB_managed.get();
        T* dC = (T*) dC_managed.get();
        if (!dA || !dB || !dC)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }

        status = rocblas_gemm_strided_batched<T>(handle, transA, transB,
                    M, N, K,
                    &alpha, dA, lda, bsa,
                    dB, ldb, bsb,
                    &beta, dC, ldc, bsc, batch_count);

        gemm_strided_batched_arg_check(status, M, N, K, lda, ldb, ldc, batch_count);

        return status;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error = 0.0;

    rocblas_int size_A = bsa * batch_count;
    rocblas_int size_B = bsb * batch_count;
    rocblas_int size_C = bsc * batch_count;

    //allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_C),rocblas_test::device_free};
    T* dA = (T*) dA_managed.get();
    T* dB = (T*) dB_managed.get();
    T* dC = (T*) dC_managed.get();
    if ((!dA && (size_A != 0)) || (!dB && (size_B != 0)) || (!dC && (size_C != 0)))
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }
    
    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hA(size_A);
    vector<T> hB(size_B);
    vector<T> hC(size_C);
    vector<T> hC_gold(size_C);

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, A_row, A_col * batch_count, lda);
    rocblas_init<T>(hB, B_row, B_col * batch_count, ldb);
    rocblas_init<T>(hC, M, N * batch_count, ldc);
    hC_gold = hC;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*size_A,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T)*size_B,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T)*size_C,  hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        // ROCBLAS
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched<T>(handle, transA, transB,
                    M, N, K,
                    &alpha, dA, lda, bsa,
                    dB, ldb, bsb,
                    &beta, dC, ldc, bsc, batch_count));

        //copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(T)*size_C, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();
        for(rocblas_int i=0;i<batch_count; i++)
        {
            cblas_gemm<T>(
                     transA, transB, M, N, K,
                     alpha, hA.data() + bsa * i, lda,
                     hB.data() + bsb * i, ldb,
                     beta, hC_gold.data() + bsc * i, ldc);
        }
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops = gemm_gflop_count<T>(M, N, K) * batch_count / cpu_time_used * 1e6;

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N*batch_count, lda, hC_gold.data(), hC.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', M, N*batch_count, lda, hC_gold.data(), hC.data());
        }
    }

    if(argus.timing)
    {
        gpu_time_used = get_time_us();  // in microseconds

        rocblas_gemm_strided_batched<T>(handle, transA, transB,
                    M, N, K,
                    &alpha, dA, lda, bsa,
                    dB, ldb, bsb,
                    &beta, dC, ldc, bsc, batch_count);

        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<T> (M, N, K) * batch_count / gpu_time_used * 1e6;

        cout << "Shape,Batch_Count,M,N,K,lda,ldb,ldc,rocblas-Gflops,us";

        if(argus.norm_check) cout << ",CPU-Gflops,us,norm-error" ;

        cout << endl;

        cout << argus.transA_option << argus.transB_option << ',' << batch_count;
        cout << ',' << M <<','<< N <<',' << K <<',' << lda <<','<< ldb <<',' << ldc;
        cout << ',' << rocblas_gflops << "," << gpu_time_used;

        if(argus.norm_check) cout << "," << cblas_gflops << "," << cpu_time_used << "," << rocblas_error;

        cout << endl;
    }

    return rocblas_status_success;
}
