/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "utility.hpp"
#include <hip/hip_runtime.h>
#include <math.h> // isnan
#include <rocblas/rocblas.h>

#define DIM1 63
#define DIM2 64

// reference code for trmm (triangle matrix matrix multiplication)
template <typename T>
void trmm_reference(rocblas_side      side,
                    rocblas_fill      uplo,
                    rocblas_operation trans,
                    rocblas_diagonal  diag,
                    int               M,
                    int               N,
                    T                 alpha,
                    const T*          A,
                    int               lda,
                    const T*          B,
                    int               ldb,
                    T*                C,
                    int               ldc)
{
    int As1 = rocblas_operation_none == trans ? 1 : lda;
    int As2 = rocblas_operation_none == trans ? lda : 1;

    // this is 3 loop gemm algorithm with non-relevant triangle part masked
    if(rocblas_side_left == side)
    {
        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                T t = 0.0;
                for(int i3 = 0; i3 < M; i3++)
                {
                    if((i1 == i3) && (rocblas_diagonal_unit == diag))
                    {
                        t += B[i3 + i2 * ldb];
                    }
                    else if(((i3 > i1) && (rocblas_fill_upper == uplo))
                            || ((i1 > i3) && (rocblas_fill_lower == uplo))
                            || ((i1 == i3) && (rocblas_diagonal_non_unit == diag)))
                    {
                        t += A[i1 * As1 + i3 * As2] * B[i3 + i2 * ldb];
                    }
                }
                C[i1 + i2 * ldc] = alpha * t;
            }
        }
    }
    else if(rocblas_side_right == side)
    {
        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                T t = 0.0;
                for(int i3 = 0; i3 < N; i3++)
                {
                    if((i3 == i2) && (rocblas_diagonal_unit == diag))
                    {
                        t += B[i1 + i3 * ldb];
                    }
                    else if(((i2 > i3) && (rocblas_fill_upper == uplo))
                            || ((i3 > i2) && (rocblas_fill_lower == uplo))
                            || ((i3 == i2) && (rocblas_diagonal_non_unit == diag)))
                    {
                        t += B[i1 + i3 * ldb] * A[i3 * As1 + i2 * As2];
                    }
                }
                C[i1 + i2 * ldc] = alpha * t;
            }
        }
    }
}

int main()
{
    //  rocblas_side      side   = rocblas_side_right;
    rocblas_side      side   = rocblas_side_left;
    rocblas_fill      uplo   = rocblas_fill_upper;
    rocblas_operation transa = rocblas_operation_none;
    rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    float alpha = 1.1;

    rocblas_int m = DIM1, n = DIM2;
    rocblas_int lda, ldb, ldc, size_a, size_b, size_c;

    rocblas_cout << "trmm V3 example" << std::endl;
    if(rocblas_side_left == side)
    {
        lda    = m;
        size_a = m * lda;
        rocblas_cout << "left";
    }
    else if(rocblas_side_right == side)
    {
        lda    = n;
        size_a = n * lda;
        rocblas_cout << "right";
    }
    rocblas_fill_upper == uplo ? rocblas_cout << ",upper" : rocblas_cout << ",lower";
    rocblas_operation_none == transa ? rocblas_cout << ",N" : rocblas_cout << ",T";
    rocblas_diagonal_non_unit == diag ? rocblas_cout << ",non_unit_diag:"
                                      : rocblas_cout << ",unit_diag:";

    ldb    = m;
    size_b = n * ldb;

    ldc    = m;
    size_c = n * ldc;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<float> ha(size_a);
    std::vector<float> hb(size_b);
    std::vector<float> hc(size_c);
    std::vector<float> hc_gold(size_c);

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

#ifdef ROCBLAS_V3
    //  API for ROCBLAS_V3
    CHECK_ROCBLAS_ERROR(
        rocblas_strmm(handle, side, uplo, transa, diag, m, n, &alpha, da, lda, db, ldb, dc, ldc));
#else
    //  API from before ROCBLAS_V3
    CHECK_ROCBLAS_ERROR(rocblas_strmm_outofplace(
        handle, side, uplo, transa, diag, m, n, &alpha, da, lda, db, ldb, dc, ldc));
#endif

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

    rocblas_cout << "m, n, lda, ldb, ldc = " << m << ", " << n << ", " << lda << ", " << ldb << ", "
                 << ldc << std::endl;

    trmm_reference<float>(
        side, uplo, transa, diag, m, n, alpha, ha.data(), lda, hb.data(), ldb, hc_gold.data(), ldc);

    float relative_error, max_relative_error = 0;
    for(int i = 0; i < size_c; i++)
    {
        relative_error = hc_gold[i] != 0 ? (hc_gold[i] - hc[i]) / hc_gold[i] : 0;
        relative_error = relative_error > 0 ? relative_error : -relative_error;
        max_relative_error
            = relative_error > max_relative_error ? relative_error : max_relative_error;
    }
    float eps       = std::numeric_limits<float>::epsilon();
    float tolerance = 10;
    //  fail if max_relative_error is NaN or if error greater than allowed
    if(isnan(max_relative_error) || max_relative_error > eps * tolerance)
    {
        rocblas_cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
    }
    else
    {
        rocblas_cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
