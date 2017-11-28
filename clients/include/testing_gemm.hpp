/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

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

/* ============================================================================================ */
template<typename T>
void testing_gemm_NaN(Arguments argus)
{
    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;

    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int  size_A, size_B, size_C, A_row, A_col, B_row, B_col;
    T alpha = argus.alpha;
    T beta = argus.beta;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    A_row = transA == rocblas_operation_none ? M : K;
    A_col = transA == rocblas_operation_none ? K : M;
    B_row = transB == rocblas_operation_none ? K : N;
    B_col = transB == rocblas_operation_none ? N : K;

    size_A = lda * A_col; size_B = ldb * B_col; size_C = ldc * N;

    //check here to prevent undefined memory allocation error
    if( M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M )
    {
        // bad arguments are tested in other tests
        return;
    }

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory.
    vector<T> hA(size_A);
    vector<T> hB(size_B);
    vector<T> hC(size_C);

    //allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_C),rocblas_test::device_free};
    T* dA = (T*) dA_managed.get();
    T* dB = (T*) dB_managed.get();
    T* dC = (T*) dC_managed.get();
    if (!dA || !dB || !dC)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //Initial Data on CPU
    for (int i = 0; i < size_A; i++) hA[i] = 1.0;
    for (int i = 0; i < size_B; i++) hB[i] = 1.0;
    for (int i = 0; i < size_C; i++) hC[i] = 1.0;

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < M; j++)
        {
            hC[j + i*ldc] = std::numeric_limits<T>::quiet_NaN();
        }
    }

    //copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T) * size_C, hipMemcpyHostToDevice));

    // ROCBLAS
    status = rocblas_gemm<T>(handle, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);

    verify_rocblas_status_success(status, "rocblas_gemm return error in testing_gemm_NaN");

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < M; j++)
        {
            verify_not_nan(hC[j + i*ldc]);
            if(hC[j + i*ldc] != hC[j + i*ldc]) goto finish_nested_loops;
        }
    }
    finish_nested_loops:

    return;
}

template<typename T>
void testing_gemm_bad_arg()
{
    const rocblas_int M = 100;
    const rocblas_int N = 100;
    const rocblas_int K = 100;

    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;
    const rocblas_int ldc = 100;

    const T alpha = 1.0;
    const T beta = 1.0;

    const rocblas_int safe_size = 100;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    //allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * safe_size),rocblas_test::device_free};
    T* dA = (T*) dA_managed.get();
    T* dB = (T*) dB_managed.get();
    T* dC = (T*) dC_managed.get();
    if (!dA || !dB || !dC)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    {
        T *dA_null = nullptr;

        status = rocblas_gemm<T>(handle, transA, transB, M, N, K, &alpha, dA_null, lda, dB, ldb, &beta, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: A is nullptr");
    }
    {
        T *dB_null = nullptr;

        status = rocblas_gemm<T>(handle, transA, transB, M, N, K, &alpha, dA, lda, dB_null, ldb, &beta, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: B is nullptr");
    }
    {
        T *dC_null = nullptr;

        status = rocblas_gemm<T>(handle, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC_null, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: C is nullptr");
    }
    {
        T *alpha_null = nullptr;

        status = rocblas_gemm<T>(handle, transA, transB, M, N, K, alpha_null, dA, lda, dB, ldb, &beta, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: C is nullptr");
    }
    {
        T *beta_null= nullptr;

        status = rocblas_gemm<T>(handle, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, beta_null, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: C is nullptr");
    }
    {
        rocblas_handle handle_null = nullptr;

        status = rocblas_gemm<T>(handle_null, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);

        verify_rocblas_status_invalid_handle(status);
    }

    return;
}

template<typename T>
rocblas_status testing_gemm(Arguments argus)
{
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;

    T h_alpha = argus.alpha;
    T h_beta = argus.beta;

    rocblas_int safe_size = 100;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error = 0.0;

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    //check here to prevent undefined memory allocation error
    if( M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M )
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

        status = rocblas_gemm<T>(handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);

        gemm_arg_check(status, M, N, K, lda, ldb, ldc);

        return status;
    }

    rocblas_int size_A = lda * A_col;
    rocblas_int size_B = ldb * B_col;
    rocblas_int size_C = ldc * N;

    //allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_C),rocblas_test::device_free};
    auto d_alpha_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)),rocblas_test::device_free};
    auto d_beta_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T)),rocblas_test::device_free};
    T* dA = (T*) dA_managed.get();
    T* dB = (T*) dB_managed.get();
    T* dC = (T*) dC_managed.get();
    T* d_alpha = (T*) d_alpha_managed.get();
    T* d_beta  = (T*) d_beta_managed.get();
    if (!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> hB(size_B);
    vector<T> hC_1(size_C);
    vector<T> hC_2(size_C);
    vector<T> hC_gold(size_C);

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, A_row, A_col, lda);
    rocblas_init<T>(hB, B_row, B_col, ldb);
    rocblas_init<T>(hC_1, M, N, ldc);

    hC_2 = hC_1;
    hC_gold = hC_1;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * size_B, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_1.data(), sizeof(T) * size_C, hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_2.data(), sizeof(T) * size_C, hipMemcpyHostToDevice));

        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC, sizeof(T) * size_C, hipMemcpyDeviceToHost));

        // CPU BLAS
        cpu_time_used = get_time_us();

        cblas_gemm<T>(transA, transB, M, N, K, h_alpha, hA.data(), lda, hB.data(), ldb, h_beta, hC_gold.data(), ldc);

        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops = gemm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;

        #ifndef NDEBUG
        print_matrix(hC_gold, hC, min(M,3), min(N,3), ldc);
        #endif

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_1.data());
            unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_2.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched 
        //in compilation time
        if(argus.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_1.data());
            rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_2.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls = 10;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for (int i = 0; i < number_cold_calls; i++)
        {
            rocblas_gemm<T>(handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
        }

        gpu_time_used = get_time_us();   // in microseconds
        for (int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm<T>(handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
        }
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<T> (M, N, K) * number_hot_calls / gpu_time_used * 1e6;

        cout << "Shape,M,N,K,lda,ldb,ldc,rocblas-Gflops,us";

        if(argus.unit_check || argus.norm_check) cout << ",CPU-Gflops(us),norm-error";

        cout << endl;

        cout << argus.transA_option << argus.transB_option << ',' 
            << M << ',' << N << ',' << K << ',' << lda << ',' << ldb << ',' << ldc <<',' 
            << rocblas_gflops << "," << gpu_time_used;

        if(argus.unit_check || argus.norm_check)
        {
            cout << cblas_gflops << "," << cpu_time_used << ',' << rocblas_error;
        }

        cout << endl;
    }
    return status;
}

/* ============================================================================================ */
/*! \brief   Bencharking GEMM, allocate a large matrix once.
subsequent steps get a sub-matrix. The memory allocation/deallocation overhead is amortized
Intended for long time running
*/

template<typename T>
rocblas_status range_testing_gemm(Arguments argus)
{
    rocblas_int start = argus.start;
    rocblas_int step = argus.step;
    rocblas_int end = argus.end;

    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int  size_A, size_B, size_C;
    T alpha = argus.alpha;
    T beta = argus.beta;

    double rocblas_gflops, cblas_gflops;
    double gpu_time_used, cpu_time_used;

    T rocblas_error = 0.0;

    //argument sanity check, quick return if input parameters are invalid before allocating invalid memory
    if( start < 0 || end < 0 || step < 0 || end < start )
    {
        cout << "Invalid matrix dimension input, will return" << endl;
        return rocblas_status_invalid_size;
    }

    size_A = size_B = size_C = end * end;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hA(size_A);
    vector<T> hB(size_B);
    vector<T> hC(size_C);
    vector<T> hC_gold(size_C);

    rocblas_status status;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    //allocate memory on device
    auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_A),rocblas_test::device_free};
    auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_B),rocblas_test::device_free};
    auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T) * size_C),rocblas_test::device_free};
    T* dA = (T*) dA_managed.get();
    T* dB = (T*) dB_managed.get();
    T* dC = (T*) dC_managed.get();
    if (!dA || !dB || !dC)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, end, end, end);
    rocblas_init<T>(hB, end, end, end);
    rocblas_init<T>(hC, end, end, end);

    //copy vector is easy in STL; hz = hx: save a copy in hC_gold which will be output of CPU BLAS
    hC_gold = hC;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*end*end,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T)*end*end,  hipMemcpyHostToDevice));

    char precision = type2char<T>(); // like turn float-> 's'

    string filename = string("benchmark_")  + precision + string("gemm_") + argus.transA_option + argus.transB_option +
            string("_") + to_string(start) + "to" + to_string(end) + "_step" + to_string(step) + ".csv";

    ofstream myfile;
    myfile.open(filename);
    if (myfile.is_open())
    {
        myfile << "M, N, K, lda, ldb, ldc, rocblas-Gflops (us) ";
        if(argus.norm_check)
        {
            myfile << "CPU-Gflops(us), norm-error" ;
        }
        myfile << endl;
    }

    for(rocblas_int size = start; size <= end; size += step)
    {
        cout << "Benchmarking M:" << (int)size << ", N:"<< (int) size <<", K:" << (int)size << endl ;

        //make sure CPU and GPU routines see the same input
        hC = hC_gold;
        CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T)*size*size,      hipMemcpyHostToDevice));

        /* =====================================================================
             ROCBLAS
        =================================================================== */

        gpu_time_used = get_time_us();// in microseconds
        rocblas_gflops = gemm_gflop_count<T> (size, size, size) / gpu_time_used * 1e6 ;

        //library interface
        status = rocblas_gemm<T>(handle, transA, transB,
                        size, size, size,
                        &alpha, dA, size,
                        dB, size,
                        &beta, dC, size);

        gpu_time_used = get_time_us() - gpu_time_used;

        //copy output from device to CPU
        hipMemcpy(hC.data(), dC, sizeof(T)*size*size,      hipMemcpyDeviceToHost);

        if(argus.norm_check)
        {
             /* =====================================================================
                         CPU BLAS
             =================================================================== */

            cpu_time_used = get_time_us();

            cblas_gemm<T>(
                         transA, transB, size, size, size,
                         alpha, hA.data(), size,
                         hB.data(), size,
                         beta, hC_gold.data(), size);


            cpu_time_used = get_time_us() - cpu_time_used;

            cblas_gflops = gemm_gflop_count<T> (size, size, size) / cpu_time_used * 1e6 ;

            //if enable norm check, norm check is invasive
            //any typeinfo(T) will not work here, because template deduction is matched in compilation time
            if(argus.norm_check) 
            {
                rocblas_error = norm_check_general<T>('F', size, size, size, hC_gold.data(), hC.data());
            }

        }// end of if unit/norm check

        if (myfile.is_open())
        {
        //only norm_check return an norm error, unit check won't return anything, only return the real part, imag part does not make sense
            myfile << size <<','<< size <<',' << size <<',' << size <<','<< size <<',' << size <<',' << rocblas_gflops << "(" << gpu_time_used << "),";

            if(argus.norm_check)
            {
                myfile << cblas_gflops << "(" << cpu_time_used << "),";
                //cout << rocblas_error;
            }

            myfile << endl;
        }
    }// end of loop

    if (myfile.is_open()) myfile.close();

    return status;
}

template<typename T>
rocblas_status benchmark_gemm(Arguments argus)
{
    //if negative, fall back to specific matrix size testing
    //otherwise, range testing. only norm check is enabled in range_test
    if(argus.start < 0 || argus.end < 0 || argus.step < 0)
    {
        cout << "Specific matrix size testing: output will be displayed on terminal" << endl;
        cout << endl;
        return testing_gemm<T>(argus);
    }
    else
    {
        argus.timing = 1; //timing is enabled
        cout << "Range matrix size testing: output will be benchmark_xgemm_(transpose)_(begin)to(end)_(step).csv ..." << endl;
        cout << endl;
        //cout << "==================================================================" << endl;
        return range_testing_gemm<T>(argus);
    }
}
