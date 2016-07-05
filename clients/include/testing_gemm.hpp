/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <sys/time.h>
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
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template<typename T>
rocblas_status testing_gemm(Arguments argus)
{

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;
    rocblas_int K = argus.K;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;

    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int  A_size, B_size, C_size, A_row, A_col, B_row, B_col;
    T alpha = argus.alpha;
    T beta = argus.beta;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error = 0.0;
    rocblas_handle handle;
    rocblas_status status = rocblas_status_success;
    rocblas_create_handle(&handle);

    if(transA == rocblas_operation_none){
        A_row =  M; A_col = K;
    }
    else{
        A_row = K; A_col = M;
    }

    if(transB == rocblas_operation_none){
        B_row =  K; B_col = N;
    }
    else{
        B_row = N; B_col = K;
    }

    A_size = A_row * A_col; B_size = B_row * B_col; C_size = M * N;

    //check here to prevent undefined memory allocation error
    if( M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 ){
        return rocblas_status_invalid_size;
    }
    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hC(C_size);
    vector<T> hC_copy(C_size);

    T *dA, *dB, *dC;

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dC, C_size * sizeof(T)));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, A_row, A_col, lda);
    rocblas_init<T>(hB, B_row, B_col, ldb);
    rocblas_init<T>(hC, M, N, ldc);

    //copy vector is easy in STL; hz = hx: save a copy in hC_copy which will be output of CPU BLAS
    hC_copy = hC;

    //copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*lda*A_col,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T)*ldb*B_col,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T)*ldc*N,      hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

#if 1
    //library interface
    status = rocblas_gemm<T>(handle, transA, transB,
                    M, N, K,
                    &alpha, dA, lda,
                    dB, ldb,
                    &beta, dC, ldc);

    if (status != rocblas_status_success) {
        hipFree(dA);
        hipFree(dB);
        hipFree(dC);
        return status;
    }
#endif
 //    sleep(1);
    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = gemm_gflop_count<T> (M, N, K) / gpu_time_used * 1e6;
    }

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(T)*ldc*N,      hipMemcpyDeviceToHost));


    if(argus.unit_check || argus.norm_check){

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }


        cblas_gemm<T>(
                     transA, transB, M, N, K,
                     alpha, hA.data(), lda,
                     hB.data(), ldb,
                     beta, hC_copy.data(), ldc);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = gemm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;
        }

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check){
            unit_check_general<T>(M, N, lda, hC_copy.data(), hC.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            rocblas_error = norm_check_general<T>('F', M, N, lda, hC_copy.data(), hC.data());
        }

    }// end of if unit/norm check

    if(argus.timing){
        //only norm_check return an norm error, unit check won't return anything,
            cout << "M, N, K, lda, ldb, ldc, rocblas-Gflops (us) ";
            if(argus.norm_check){
                cout << "CPU-Gflops(us), norm-error" ;
            }
            cout << endl;

            cout << "GG," << M <<','<< N <<',' << K <<',' << lda <<','<< ldb <<',' << ldc <<',' << rocblas_gflops << "(" << gpu_time_used << "),";

            if(argus.norm_check){
                cout << cblas_gflops << "(" << cpu_time_used << "),";
                cout << rocblas_error;
            }

            cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));

    rocblas_destroy_handle(handle);
    return rocblas_status_success;
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

    rocblas_int  A_size, B_size, C_size;
    T alpha = argus.alpha;
    T beta = argus.beta;

    double rocblas_gflops, cblas_gflops;
    double gpu_time_used, cpu_time_used;

    T rocblas_error = 0.0;
    rocblas_handle handle;
    rocblas_status status = rocblas_status_success;

    //argument sanity check, quick return if input parameters are invalid before allocating invalid memory
    if( start < 0 || end < 0 || step < 0 || end < start ){
        cout << "Invalid matrix dimension input, will return" << endl;
        return rocblas_status_invalid_size;
    }


    A_size = B_size = C_size = end * end;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hC(C_size);
    vector<T> hC_copy(C_size);

    T *dA, *dB, *dC;

    rocblas_create_handle(&handle);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dC, C_size * sizeof(T)));

    //rocblas_malloc_device(&dx, sizeX * sizeof(T));

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, end, end, end);
    rocblas_init<T>(hB, end, end, end);
    rocblas_init<T>(hC, end, end, end);

    //copy vector is easy in STL; hz = hx: save a copy in hC_copy which will be output of CPU BLAS
    hC_copy = hC;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*end*end,  hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T)*end*end,  hipMemcpyHostToDevice));

    char precision = type2char<T>(); // like turn float-> 's'

    string filename = string("benchmark_")  + precision + string("gemm_") + argus.transA_option + argus.transB_option +
            string("_") + to_string(start) + "to" + to_string(end) + "_step" + to_string(step) + ".csv";

    ofstream myfile;
    myfile.open(filename);
    if (myfile.is_open()){
        myfile << "M, N, K, lda, ldb, ldc, rocblas-Gflops (us) ";
        if(argus.norm_check){
            myfile << "CPU-Gflops(us), norm-error" ;
        }
        myfile << endl;
    }

    for(rocblas_int size = start; size <= end; size += step){

        cout << "Benchmarking M:" << (int)size << ", N:"<< (int) size <<", K:" << (int)size << endl ;

        //make sure CPU and GPU routines see the same input
        hC = hC_copy;
        CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T)*size*size,      hipMemcpyHostToDevice));

        /* =====================================================================
             ROCBLAS
        =================================================================== */

        gpu_time_used = get_time_us();// in microseconds
        rocblas_gflops = gemm_gflop_count<T> (size, size, size) / gpu_time_used * 1e6 ;

    #if 0
        //library interface
        status = rocblas_gemm<T>(handle, transA, transB,
                        size, size, size,
                        &alpha, dA, size,
                        dB, size,
                        &beta, dC, size);

        if (status != rocblas_status_success) {
            hipFree(dA);
            hipFree(dB);
            hipFree(dC);
            return status;
        }
    #endif

        gpu_time_used = get_time_us() - gpu_time_used;

        //copy output from device to CPU
        hipMemcpy(hC.data(), dC, sizeof(T)*size*size,      hipMemcpyDeviceToHost);

        if(argus.norm_check){

             /* =====================================================================
                         CPU BLAS
             =================================================================== */

            cpu_time_used = get_time_us();

            cblas_gemm<T>(
                         transA, transB, size, size, size,
                         alpha, hA.data(), size,
                         hB.data(), size,
                         beta, hC_copy.data(), size);


            cpu_time_used = get_time_us() - cpu_time_used;

            cblas_gflops = gemm_gflop_count<T> (size, size, size) / cpu_time_used * 1e6 ;

            //if enable norm check, norm check is invasive
            //any typeinfo(T) will not work here, because template deduction is matched in compilation time
            if(argus.norm_check) {
                rocblas_error = norm_check_general<T>('F', size, size, size, hC_copy.data(), hC.data());
            }

        }// end of if unit/norm check

        if (myfile.is_open()){
        //only norm_check return an norm error, unit check won't return anything, only return the real part, imag part does not make sense
            myfile << size <<','<< size <<',' << size <<',' << size <<','<< size <<',' << size <<',' << rocblas_gflops << "(" << gpu_time_used << "),";

            if(argus.norm_check){
                myfile << cblas_gflops << "(" << cpu_time_used << "),";
                //cout << rocblas_error;
            }

            myfile << endl;
        }
    }// end of loop

    if (myfile.is_open()) myfile.close();

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));

    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}


template<typename T>
rocblas_status benchmark_gemm(Arguments argus)
{
    //if negative, fall back to specific matrix size testing
    //otherwise, range testing. only norm check is enabled in range_test
    if(argus.start < 0 || argus.end < 0 || argus.step < 0){
        cout << "Specific matrix size testing: output will be displayed on terminal" << endl;
        cout << endl;
        return testing_gemm<T>(argus);
    }
    else{
        argus.timing = 1; //timing is enabled
        cout << "Range matrix size testing: output will be benchmark_xgemm_(transpose)_(begin)to(end)_(step).csv ..." << endl;
        cout << endl;
        //cout << "==================================================================" << endl;
        return range_testing_gemm<T>(argus);
    }
}
