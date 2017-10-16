/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

#include "rocblas.hpp"
#include "utility.h"
#include "norm.h"
#include "unit.h"
#include "arg_check.h"
#include "flops.h"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template<typename T>
void testing_geam_bad_arg()
{
    const rocblas_int M = 100;
    const rocblas_int N = 100;

    const rocblas_int lda = 100;
    const rocblas_int ldb = 100;
    const rocblas_int ldc = 100;

    const T h_alpha = 1.0;
    const T h_beta = 1.0;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_handle handle;
    T *dA, *dB, *dC;

    rocblas_status status;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) 
    {
        rocblas_destroy_handle(handle);
        return;
    }

    rocblas_int A_size = N * lda;
    rocblas_int B_size = N * ldb;
    rocblas_int C_size = N * ldc;

    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hC(C_size);

    srand(1);
    rocblas_init<T>(hA, M, N, lda);
    rocblas_init<T>(hB, M, N, ldb);
    rocblas_init<T>(hC, M, N, ldc);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dC, C_size * sizeof(T)));

    //copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T)*A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T)*B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T)*C_size, hipMemcpyHostToDevice));

    {
        T *dA_null = nullptr;
        status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA_null, lda,
                    dB, ldb,
                    &h_beta, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: A is nullptr");
    }
    {
        T *dB_null = nullptr;
        status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB_null, ldb,
                    &h_beta, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: B is nullptr");
    }
    {
        T *dC_null = nullptr;
        status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB, ldb,
                    &h_beta, dC_null, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: C is nullptr");
    }
    {
        T *h_alpha_null = nullptr;
        status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    h_alpha_null, dA, lda,
                    dB, ldb,
                    &h_beta, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: h_alpha is nullptr");
    }
    {
        T *h_beta_null= nullptr;
        status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB, ldb,
                    h_beta_null, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: h_beta is nullptr");
    }
    {
        rocblas_handle handle_null = nullptr;
        status = rocblas_geam<T>(handle_null, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB, ldb,
                    &h_beta, dC, ldc);

        verify_rocblas_status_invalid_handle(status);
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));

    rocblas_destroy_handle(handle);
    return;
}

template<typename T>
rocblas_status testing_geam(Arguments argus)
{
    rocblas_operation transA = char2rocblas_operation(argus.transA_option);
    rocblas_operation transB = char2rocblas_operation(argus.transB_option);

    rocblas_int M = argus.M;
    rocblas_int N = argus.N;

    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int ldc = argus.ldc;

    T h_alpha = argus.alpha;
    T h_beta = argus.beta;
    T* d_alpha, *d_beta;

    T *dA, *dB, *dC, *dC_in_place;

    rocblas_int  A_size, B_size, C_size, A_row, A_col, B_row, B_col;
    rocblas_int  inc1_A, inc2_A, inc1_B, inc2_B;

    double gpu_time_used, cpu_time_used;
    double rocblas_gflops, cblas_gflops;

    T rocblas_error = 0.0;

    rocblas_handle handle;
    rocblas_status status, status_h, status_d;
    status = rocblas_create_handle(&handle);
    verify_rocblas_status_success(status,"ERROR: rocblas_create_handle");

    if(status != rocblas_status_success) 
    {
        rocblas_destroy_handle(handle);
        return status;
    }
    if(transA == rocblas_operation_none)
    {
        A_row = M; A_col = N;
        inc1_A = 1; inc2_A = lda;
    }
    else
    {
        A_row = N; A_col = M;
        inc1_A = lda; inc2_A = 1;
    }
    if(transB == rocblas_operation_none)
    {
        B_row = M; B_col = N;
        inc1_B = 1; inc2_B = ldb;
    }
    else
    {
        B_row = N; B_col = M;
        inc1_B = ldb; inc2_B = 1;
    }

    A_size = lda * A_col; B_size = ldb * B_col; C_size = ldc * N;

    //check here to prevent undefined memory allocation error
    if( M <= 0 || N <= 0 || lda < A_row || ldb < B_row || ldc < M )
    {
        CHECK_HIP_ERROR(hipMalloc(&dA, 100 * sizeof(T))); // 100 is arbitary
        CHECK_HIP_ERROR(hipMalloc(&dB, 100 * sizeof(T)));
        CHECK_HIP_ERROR(hipMalloc(&dC, 100 * sizeof(T)));

        status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB, ldb,
                    &h_beta, dC, ldc);

        geam_arg_check(status, M, N, lda, ldb, ldc);

        CHECK_HIP_ERROR(hipFree(dA));
        CHECK_HIP_ERROR(hipFree(dB));
        CHECK_HIP_ERROR(hipFree(dC));

        rocblas_destroy_handle(handle);

        return status;
    }

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dC, C_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_alpha, sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_beta, sizeof(T)));

    if (nullptr == dA || nullptr == dB || nullptr == dC )
    {
        status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB, ldb,
                    &h_beta, dC, ldc);

        verify_rocblas_status_invalid_pointer(status, "ERROR: A or B or C is nullptr");

        CHECK_HIP_ERROR(hipFree(dA));
        CHECK_HIP_ERROR(hipFree(dB));
        CHECK_HIP_ERROR(hipFree(dC));

        rocblas_destroy_handle(handle);

        return status;
    }
    else if (nullptr == handle )
    {
        status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB, ldb,
                    &h_beta, dC, ldc);

        verify_rocblas_status_invalid_handle(status);

        CHECK_HIP_ERROR(hipFree(dA));
        CHECK_HIP_ERROR(hipFree(dB));
        CHECK_HIP_ERROR(hipFree(dC));

        rocblas_destroy_handle(handle);

        return status;
    }

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size), hA_copy(A_size);
    vector<T> hB(B_size), hB_copy(B_size);
    vector<T> hC_h(C_size);
    vector<T> hC_d(C_size);
    vector<T> hC_gold(C_size);

    //Initial Data on CPU
    srand(1);
    rocblas_init<T>(hA, A_row, A_col, lda);
    rocblas_init<T>(hB, B_row, B_col, ldb);

    hA_copy = hA;
    hB_copy = hB;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
             ROCBLAS
        =================================================================== */
        // &h_alpha and &h_beta are host pointers
        status_h = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB, ldb,
                    &h_beta, dC, ldc);

        CHECK_HIP_ERROR(hipMemcpy(hC_h.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));


        // d_alpha and d_beta are device pointers
        status_d = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    d_alpha, dA, lda,
                    dB, ldb,
                    d_beta, dC, ldc);

        CHECK_HIP_ERROR(hipMemcpy(hC_d.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));


         /* =====================================================================
                 CPU BLAS
         =================================================================== */
        if(status_d == rocblas_status_success && status_h == rocblas_status_success) //only valid size compare with cblas
        {
            cpu_time_used = get_time_us();

            // reference calculation
            for (int i1 = 0; i1 < M; i1++)
            {
                for (int i2 = 0; i2 < N; i2++)
                {
                    hC_gold[i1 + i2*ldc] = h_alpha * hA_copy[i1*inc1_A + i2*inc2_A] + h_beta * hB_copy[i1*inc1_B + i2*inc2_B];
                }
            }

            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops = geam_gflop_count<T>(M, N) / cpu_time_used * 1e6;
        }
        else
        {
            verify_rocblas_status_success(status_h, "status_h  rocblas_geam  error");
            verify_rocblas_status_success(status_d, "status_d  rocblas_geam  error");
        }

        #ifndef NDEBUG
        print_matrix(hC_gold, hC_h, min(M,3), min(N,3), ldc);
        #endif

        //enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_h.data());
            unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_d.data());
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched 
        //in compilation time
        if(argus.norm_check)
        {
            cout << ", norm-error";
            rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_h.data());
            rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_d.data());
        }


        // inplace check for dC == dA

        {
            dC_in_place = dA;

            status_h = rocblas_geam<T>(handle, transA, transB,
                        M, N,
                        &h_alpha, dA, lda,
                        dB, ldb,
                        &h_beta, dC_in_place, ldc);

            if (lda != ldc || transA != rocblas_operation_none)
            {
                verify_rocblas_status_invalid_size(status_h, "rocblas_geam inplace C==A");
            }
            else
            {
                verify_rocblas_status_success(status_h, "status_h rocblas_geam inplace C==A");
              //verify_rocblas_status_success(status_d, "status_d rocblas_geam inplace C==A");

                CHECK_HIP_ERROR(hipMemcpy(hC_h.data(), dC_in_place, sizeof(T) * C_size, hipMemcpyDeviceToHost));
                // dA was clobbered by dC_in_place, so copy hA back to dA
                CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));

                //CHECK_HIP_ERROR(hipMemcpy(hC_d.data(), dC_in_place, sizeof(T) * C_size, hipMemcpyDeviceToHost));

                // reference calculation
                for (int i1 = 0; i1 < M; i1++)
                {
                    for (int i2 = 0; i2 < N; i2++)
                    {
                        hC_gold[i1 + i2*ldc] = h_alpha * hA_copy[i1*inc1_A + i2*inc2_A] + h_beta * hB[i1*inc1_B + i2*inc2_B];
                    }
                }

                //enable unit check, notice unit check is not invasive, but norm check is,
                // unit check and norm check can not be interchanged their order
                if(argus.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_h.data());
                    unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_d.data());
                }

                //if enable norm check, norm check is invasive
                //any typeinfo(T) will not work here, because template deduction is matched 
                //in compilation time
                if(argus.norm_check)
                {
                    cout << ", norm-error";
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_h.data());
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_d.data());
                }
            }
        }

        // inplace check for dC == dB

        {
            dC_in_place = dB;

            status_h = rocblas_geam<T>(handle, transA, transB,
                        M, N,
                        &h_alpha, dA, lda,
                        dB, ldb,
                        &h_beta, dC_in_place, ldc);

            if (ldb != ldc || transB != rocblas_operation_none)
            {
                verify_rocblas_status_invalid_size(status_h, "rocblas_geam inplace C==A");
            }
            else
            {
                verify_rocblas_status_success(status_h, "status_h rocblas_geam inplace C==A");
              //verify_rocblas_status_success(status_d, "status_d rocblas_geam inplace C==A");

                CHECK_HIP_ERROR(hipMemcpy(hC_h.data(), dC_in_place, sizeof(T) * C_size, hipMemcpyDeviceToHost));
                // dA was clobbered by dC_in_place, so copy hA back to dA
        //      CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));

                //CHECK_HIP_ERROR(hipMemcpy(hC_d.data(), dC_in_place, sizeof(T) * C_size, hipMemcpyDeviceToHost));

                // reference calculation
                for (int i1 = 0; i1 < M; i1++)
                {
                    for (int i2 = 0; i2 < N; i2++)
                    {
                        hC_gold[i1 + i2*ldc] = h_alpha * hA_copy[i1*inc1_A + i2*inc2_A] + h_beta * hB_copy[i1*inc1_B + i2*inc2_B];
                    }
                }

                //enable unit check, notice unit check is not invasive, but norm check is,
                // unit check and norm check can not be interchanged their order
                if(argus.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_h.data());
                    unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_d.data());
                }

                //if enable norm check, norm check is invasive
                //any typeinfo(T) will not work here, because template deduction is matched 
                //in compilation time
                if(argus.norm_check)
                {
                    cout << ", norm-error";
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_h.data());
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold.data(), hC_d.data());
                }
            }
        }



    }// end of if unit/norm check

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls = 10;

        for (int i = 0; i < number_cold_calls; i++)
        {
            status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB, ldb,
                    &h_beta, dC, ldc);
        }

        gpu_time_used = get_time_us();   // in microseconds
        for (int i = 0; i < number_hot_calls; i++)
        {
            status = rocblas_geam<T>(handle, transA, transB,
                    M, N,
                    &h_alpha, dA, lda,
                    dB, ldb,
                    &h_beta, dC, ldc);
        }
        gpu_time_used = get_time_us() - gpu_time_used;
        rocblas_gflops = geam_gflop_count<T> (M, N) * number_hot_calls / gpu_time_used * 1e6;

        cout << "Shape, M, N, lda, ldb, ldc, rocblas-Gflops (us) ";
        if(argus.unit_check || argus.norm_check)
        {
            cout << "CPU-Gflops(us), norm-error" ;
        }
        cout << endl;

        cout << argus.transA_option << argus.transB_option << ',' 
            << M << ',' << N << ',' << lda << ',' << ldb << ',' << ldc <<',' 
            << rocblas_gflops << " (" << gpu_time_used << "), ";

        if(argus.unit_check || argus.norm_check)
        {
            cout << cblas_gflops << " (" << cpu_time_used << "), ";
            cout << rocblas_error;
        }

        cout << endl;
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));

    rocblas_destroy_handle(handle);
    return status;
}
