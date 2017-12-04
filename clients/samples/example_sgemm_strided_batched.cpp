/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <limits>
#include "rocblas.h"

using namespace std;

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(error)                              \
    if(error != rocblas_status_success)                         \
    {                                                           \
        fprintf(stderr, "rocBLAS error: ");                     \
        if(error == rocblas_status_invalid_handle)              \
            fprintf(stderr, "rocblas_status_invalid_handle");   \
        if(error == rocblas_status_not_implemented)             \
            fprintf(stderr, " rocblas_status_not_implemented"); \
        if(error == rocblas_status_invalid_pointer)             \
            fprintf(stderr, "rocblas_status_invalid_pointer");  \
        if(error == rocblas_status_invalid_size)                \
            fprintf(stderr, "rocblas_status_invalid_size");     \
        if(error == rocblas_status_memory_error)                \
            fprintf(stderr, "rocblas_status_memory_error");     \
        if(error == rocblas_status_internal_error)              \
            fprintf(stderr, "rocblas_status_internal_error");   \
        fprintf(stderr, "\n");                                  \
        exit(EXIT_FAILURE);                                     \
    }
#endif

#define DIM1 127
#define DIM2 128
#define DIM3 129
#define BATCH_COUNT 10

template <typename T>
void mat_mat_mult(T alpha,
                  T beta,
                  int M,
                  int N,
                  int K,
                  T* A,
                  int As1,
                  int As2,
                  T* B,
                  int Bs1,
                  int Bs2,
                  T* C,
                  int Cs1,
                  int Cs2)
{
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            T t = 0.0;
            for(int i3 = 0; i3 < K; i3++)
            {
                t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
            }
            C[i1 * Cs1 + i2 * Cs2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
        }
    }
}

int main()
{
    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
    float alpha = 1.1, beta = 0.9;

    rocblas_int m = DIM1, n = DIM2, k = DIM3, batch_count = BATCH_COUNT;
    rocblas_int lda, ldb, ldc, bsa, bsb, bsc;
    int a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    cout << "sgemm_strided_batched example" << endl;
    if(transa == rocblas_operation_none)
    {
        lda        = m;
        bsa        = k * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
        cout << "N";
    }
    else
    {
        lda        = k;
        bsa        = m * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
        cout << "T";
    }
    if(transb == rocblas_operation_none)
    {
        ldb        = k;
        bsb        = n * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
        cout << "N: ";
    }
    else
    {
        ldb        = n;
        bsb        = k * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
        cout << "T: ";
    }
    ldc = m;
    bsc = n * ldc;

    cout << "M, N, K, lda, bsa, ldb, bsb, ldc, bsc = " << m << ", " << n << ", " << k << ", " << lda
         << ", " << bsa << ", " << ldb << ", " << bsb << ", " << ldc << ", " << bsc << endl;

    int size_a = bsa * batch_count;
    int size_b = bsb * batch_count;
    int size_c = bsc * batch_count;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    vector<float> ha(size_a);
    vector<float> hb(size_b);
    vector<float> hc(size_c);
    vector<float> hc_gold(size_c);

    // initial data on host
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = rand() % 17;
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = rand() % 17;
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = rand() % 17;
    }
    hc_gold = hc;

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    CHECK_ROCBLAS_ERROR(rocblas_sgemm_strided_batched(handle,
                                                      transa,
                                                      transb,
                                                      m,
                                                      n,
                                                      k,
                                                      &alpha,
                                                      da,
                                                      lda,
                                                      bsa,
                                                      db,
                                                      ldb,
                                                      bsb,
                                                      &beta,
                                                      dc,
                                                      ldc,
                                                      bsc,
                                                      batch_count));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

    // calculate golden or correct result
    for(int i = 0; i < batch_count; i++)
    {
        float* a_ptr = &ha[i * bsa];
        float* b_ptr = &hb[i * bsb];
        float* c_ptr = &hc_gold[i * bsc];
        mat_mat_mult<float>(alpha,
                            beta,
                            m,
                            n,
                            k,
                            a_ptr,
                            a_stride_1,
                            a_stride_2,
                            b_ptr,
                            b_stride_1,
                            b_stride_2,
                            c_ptr,
                            1,
                            ldc);
    }

    float max_relative_error = numeric_limits<float>::min();
    for(int i = 0; i < size_c; i++)
    {
        float relative_error = (hc_gold[i] - hc[i]) / hc_gold[i];
        relative_error       = relative_error > 0 ? relative_error : -relative_error;
        max_relative_error =
            relative_error < max_relative_error ? max_relative_error : relative_error;
    }
    float eps       = numeric_limits<float>::epsilon();
    float tolerance = 10;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        cout << "FAIL: max_relative_error = " << max_relative_error << endl;
    }
    else
    {
        cout << "PASS: max_relative_error = " << max_relative_error << endl;
    }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
