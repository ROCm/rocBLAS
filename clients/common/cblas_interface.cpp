/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************/
#include "cblas_interface.hpp"
#include "rocblas_vector.hpp"
#include "utility.hpp"
#include <bitset>
#include <omp.h>

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

template <>
void cblas_nrm2<rocblas_half>(rocblas_int         n,
                              const rocblas_half* x,
                              rocblas_int         incx,
                              rocblas_half*       result)
{
    if(n <= 0 || incx <= 0)
        return;

    host_vector<float> x_float(n * incx);

    for(size_t i = 0; i < n; i++)
        x_float[i * incx] = x[i * incx];

    *result = rocblas_half(cblas_snrm2(n, x_float, incx));
}

template <>
void cblas_nrm2<rocblas_bfloat16>(rocblas_int             n,
                                  const rocblas_bfloat16* x,
                                  rocblas_int             incx,
                                  rocblas_bfloat16*       result)
{
    if(n <= 0 || incx <= 0)
        return;

    host_vector<float> x_float(n * incx);

    for(size_t i = 0; i < n; i++)
        x_float[i * incx] = float(x[i * incx]);

    *result = rocblas_bfloat16(cblas_snrm2(n, x_float, incx));
}

template <>
void cblas_axpy<rocblas_half>(rocblas_int   n,
                              rocblas_half  alpha,
                              rocblas_half* x,
                              rocblas_int   incx,
                              rocblas_half* y,
                              rocblas_int   incy)
{
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = n * abs_incx;
    size_t size_y   = n * abs_incy;
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;
    host_vector<float> x_float(size_x), y_float(size_y);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = x[i * abs_incx];
        y_float[i * abs_incy] = y[i * abs_incy];
    }

    cblas_saxpy(n, alpha, x_float, incx, y_float, incy);

    for(size_t i = 0; i < n; i++)
    {
        x[i * abs_incx] = rocblas_half(x_float[i * abs_incx]);
        y[i * abs_incy] = rocblas_half(y_float[i * abs_incy]);
    }
}

/**
  *
  * cblas_scal(rocblas_int n, T alpha, U x, rocblas_int incx)
  *
  * Info about cblas_scal function:
  *
  *    The reason why we call cblas_scal(our CPU implementation) instead of BLIS SCAL is because of the different resultant output vector produced
  *    when initialized with input parameters alpha == 0 and vector `x` to NaN. For this input (alpha == 0 and vector `x` to NaN) BLIS SCAL produces
  *    resultant vector filled with zeros whereas rocBLAS, cuBLAS, MAGMA produces resultant vector filled with NaN's.
  *
  * Parameters   : n     : Number of elements in `x`.
  *                alpha : scalar alpha value.
  *                x     : Host pointer storing vector `x`.
  *                incx  : Specifies the increment for the elements of `x`.
  *
  * Return Value : Void
  *
**/

template <typename T, typename U>
void cblas_scal(int64_t n, T alpha, U x, int64_t incx)
{
    if(n <= 0 || incx <= 0)
        return;

    if(incx == 1)
    {
#pragma omp parallel for
        for(int64_t i = 0; i < n; i++)
            x[i] = alpha * x[i];
    }
    else
    {
#pragma omp parallel for
        for(int64_t i = 0; i < n; i++)
            x[i * incx] = alpha * x[i * incx];
    }
}

//Scal Instantiation
template void cblas_scal<float, float*>(int64_t n, float alpha, float* x, int64_t incx);
template void cblas_scal<double, double*>(int64_t n, double alpha, double* x, int64_t incx);
template void cblas_scal<rocblas_half, rocblas_half*>(int64_t       n,
                                                      rocblas_half  alpha,
                                                      rocblas_half* x,
                                                      int64_t       incx);
template void
    cblas_scal<float, rocblas_half*>(int64_t n, float alpha, rocblas_half* x, int64_t incx);

template void cblas_scal<rocblas_bfloat16, rocblas_bfloat16*>(int64_t           n,
                                                              rocblas_bfloat16  alpha,
                                                              rocblas_bfloat16* x,
                                                              int64_t           incx);

template void
    cblas_scal<float, rocblas_bfloat16*>(int64_t n, float alpha, rocblas_bfloat16* x, int64_t incx);

template void cblas_scal<rocblas_complex_num<float>, rocblas_complex_num<float>*>(
    int64_t n, rocblas_complex_num<float> alpha, rocblas_complex_num<float>* x, int64_t incx);
template void cblas_scal<rocblas_complex_num<double>, rocblas_complex_num<double>*>(
    int64_t n, rocblas_complex_num<double> alpha, rocblas_complex_num<double>* x, int64_t incx);
template void cblas_scal<float, rocblas_complex_num<float>*>(int64_t                     n,
                                                             float                       alpha,
                                                             rocblas_complex_num<float>* x,
                                                             int64_t                     incx);
template void cblas_scal<double, rocblas_complex_num<double>*>(int64_t                      n,
                                                               double                       alpha,
                                                               rocblas_complex_num<double>* x,
                                                               int64_t                      incx);

template <>
void cblas_dot<rocblas_half>(rocblas_int         n,
                             const rocblas_half* x,
                             rocblas_int         incx,
                             const rocblas_half* y,
                             rocblas_int         incy,
                             rocblas_half*       result)
{
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = n * abs_incx;
    size_t size_y   = n * abs_incy;
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;
    host_vector<float> x_float(size_x);
    host_vector<float> y_float(size_y);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = x[i * abs_incx];
        y_float[i * abs_incy] = y[i * abs_incy];
    }

    *result = rocblas_half(cblas_sdot(n, x_float, incx, y_float, incy));
}

template <>
void cblas_dot<rocblas_bfloat16>(rocblas_int             n,
                                 const rocblas_bfloat16* x,
                                 rocblas_int             incx,
                                 const rocblas_bfloat16* y,
                                 rocblas_int             incy,
                                 rocblas_bfloat16*       result)
{
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = n * abs_incx;
    size_t size_y   = n * abs_incy;
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;
    host_vector<float> x_float(size_x);
    host_vector<float> y_float(size_y);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = float(x[i * abs_incx]);
        y_float[i * abs_incy] = float(y[i * abs_incy]);
    }

    *result = rocblas_bfloat16(cblas_sdot(n, x_float, incx, y_float, incy));
}

template <>
void cblas_dotc<float>(rocblas_int  n,
                       const float* x,
                       rocblas_int  incx,
                       const float* y,
                       rocblas_int  incy,
                       float*       result)
{
    cblas_dot(n, x, incx, y, incy, result);
}

template <>
void cblas_dotc<double>(rocblas_int   n,
                        const double* x,
                        rocblas_int   incx,
                        const double* y,
                        rocblas_int   incy,
                        double*       result)
{
    cblas_dot(n, x, incx, y, incy, result);
}

template <>
void cblas_dotc<rocblas_half>(rocblas_int         n,
                              const rocblas_half* x,
                              rocblas_int         incx,
                              const rocblas_half* y,
                              rocblas_int         incy,
                              rocblas_half*       result)
{
    cblas_dot(n, x, incx, y, incy, result);
}

template <>
void cblas_dotc<rocblas_bfloat16>(rocblas_int             n,
                                  const rocblas_bfloat16* x,
                                  rocblas_int             incx,
                                  const rocblas_bfloat16* y,
                                  rocblas_int             incy,
                                  rocblas_bfloat16*       result)
{
    cblas_dot(n, x, incx, y, incy, result);
}

// rot
template <>
void cblas_rot<rocblas_half>(rocblas_int         n,
                             rocblas_half*       x,
                             rocblas_int         incx,
                             rocblas_half*       y,
                             rocblas_int         incy,
                             const rocblas_half* c,
                             const rocblas_half* s)
{
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = n * abs_incx;
    size_t size_y   = n * abs_incy;
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;
    host_vector<float> x_float(size_x);
    host_vector<float> y_float(size_y);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = x[i * abs_incx];
        y_float[i * abs_incy] = y[i * abs_incy];
    }

    const float c_float = float(*c);
    const float s_float = float(*s);

    cblas_srot(n, x_float, incx, y_float, incy, c_float, s_float);

    for(size_t i = 0; i < n; i++)
    {
        x[i * abs_incx] = x_float[i * abs_incx];
        y[i * abs_incy] = y_float[i * abs_incy];
    }
}

template <>
void cblas_rot<rocblas_bfloat16>(rocblas_int             n,
                                 rocblas_bfloat16*       x,
                                 rocblas_int             incx,
                                 rocblas_bfloat16*       y,
                                 rocblas_int             incy,
                                 const rocblas_bfloat16* c,
                                 const rocblas_bfloat16* s)
{
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = n * abs_incx;
    size_t size_y   = n * abs_incy;
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;
    host_vector<float> x_float(size_x);
    host_vector<float> y_float(size_y);

    for(size_t i = 0; i < n; i++)
    {
        x_float[i * abs_incx] = x[i * abs_incx];
        y_float[i * abs_incy] = y[i * abs_incy];
    }

    const float c_float = rocblas_bfloat16(*c);
    const float s_float = rocblas_bfloat16(*s);

    cblas_srot(n, x_float, incx, y_float, incy, c_float, s_float);

    for(size_t i = 0; i < n; i++)
    {
        x[i * abs_incx] = rocblas_bfloat16(x_float[i * abs_incx]);
        y[i * abs_incy] = rocblas_bfloat16(y_float[i * abs_incy]);
    }
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */
/**
  *
  * cblas_dgmm(rocblas_side side, rocblas_int m, rocblas_int n, T* A, rocblas_int lda, T *x, rocblas_int incx, T *C, rocblas_int ldc)
  *
  * Parameters   : side  : specifies the side of diag(x)
  *                m     : Number of rows in matrices `A` and `C`.
  *                n     : Number of cols in matrices `A` and `C`.
  *                A     : Host pointer storing matrix `A`.
  *                lda   : Leading dimension of matrix `A`.
  *                x     : Host pointer storing vector `x`.
  *                incx  : Specifies the increment of the elements in `x`.
  *                C     : Host pointer storing matrix `C`.
  *                ldc   : Leading dimension of matrix `C`.
  *
  * Return Value : Void
  *
**/

template <typename T>
void cblas_dgmm(rocblas_side side,
                rocblas_int  m,
                rocblas_int  n,
                T*           A,
                rocblas_int  lda,
                T*           x,
                rocblas_int  incx,
                T*           C,
                rocblas_int  ldc)
{
    if(!m || !n)
        return;

    rocblas_int K = rocblas_side_right == side ? size_t(n) : size_t(m);

    ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (K - 1) : 0;

    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            if(rocblas_side_right == side)
            {
                C[i + j * ldc] = A[i + j * lda] * x[shift_x + j * incx];
            }
            else
            {
                C[i + j * ldc] = A[i + j * lda] * x[shift_x + i * incx];
            }
        }
    }
}

//dgmm Instantiation
template void cblas_dgmm<float>(rocblas_side side,
                                rocblas_int  m,
                                rocblas_int  n,
                                float*       A,
                                rocblas_int  lda,
                                float*       x,
                                rocblas_int  incx,
                                float*       C,
                                rocblas_int  ldc);
template void cblas_dgmm<double>(rocblas_side side,
                                 rocblas_int  m,
                                 rocblas_int  n,
                                 double*      A,
                                 rocblas_int  lda,
                                 double*      x,
                                 rocblas_int  incx,
                                 double*      C,
                                 rocblas_int  ldc);
template void cblas_dgmm<rocblas_complex_num<float>>(rocblas_side                side,
                                                     rocblas_int                 m,
                                                     rocblas_int                 n,
                                                     rocblas_complex_num<float>* A,
                                                     rocblas_int                 lda,
                                                     rocblas_complex_num<float>* x,
                                                     rocblas_int                 incx,
                                                     rocblas_complex_num<float>* C,
                                                     rocblas_int                 ldc);
template void cblas_dgmm<rocblas_complex_num<double>>(rocblas_side                 side,
                                                      rocblas_int                  m,
                                                      rocblas_int                  n,
                                                      rocblas_complex_num<double>* A,
                                                      rocblas_int                  lda,
                                                      rocblas_complex_num<double>* x,
                                                      rocblas_int                  incx,
                                                      rocblas_complex_num<double>* C,
                                                      rocblas_int                  ldc);

// geam
template <typename T>
inline T geam_conj_helper(const T& x)
{
    return x;
}

template <>
inline rocblas_float_complex geam_conj_helper(const rocblas_float_complex& x)
{
    return std::conj(x);
}

template <>
inline rocblas_double_complex geam_conj_helper(const rocblas_double_complex& x)
{
    return std::conj(x);
}

template <typename T>
void cblas_geam_helper(rocblas_operation transA,
                       rocblas_operation transB,
                       rocblas_int       M,
                       rocblas_int       N,
                       T                 alpha,
                       T*                A,
                       rocblas_int       lda,
                       T                 beta,
                       T*                B,
                       rocblas_int       ldb,
                       T*                C,
                       rocblas_int       ldc)
{
    rocblas_int inc1_A = transA == rocblas_operation_none ? 1 : lda;
    rocblas_int inc2_A = transA == rocblas_operation_none ? lda : 1;
    rocblas_int inc1_B = transB == rocblas_operation_none ? 1 : ldb;
    rocblas_int inc2_B = transB == rocblas_operation_none ? ldb : 1;

#pragma omp parallel for
    for(rocblas_int i = 0; i < M; i++)
    {
        for(rocblas_int j = 0; j < N; j++)
        {
            T a_val = alpha ? A[i * inc1_A + j * inc2_A] : 0;
            T b_val = beta ? B[i * inc1_B + j * inc2_B] : 0;
            if(transA == rocblas_operation_conjugate_transpose)
                a_val = geam_conj_helper(a_val);
            if(transB == rocblas_operation_conjugate_transpose)
                b_val = geam_conj_helper(b_val);
            C[i + j * ldc] = alpha * a_val + beta * b_val;
        }
    }
}

template <>
void cblas_geam(rocblas_operation transa,
                rocblas_operation transb,
                rocblas_int       m,
                rocblas_int       n,
                float*            alpha,
                float*            A,
                rocblas_int       lda,
                float*            beta,
                float*            B,
                rocblas_int       ldb,
                float*            C,
                rocblas_int       ldc)
{
    return cblas_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

template <>
void cblas_geam(rocblas_operation transa,
                rocblas_operation transb,
                rocblas_int       m,
                rocblas_int       n,
                double*           alpha,
                double*           A,
                rocblas_int       lda,
                double*           beta,
                double*           B,
                rocblas_int       ldb,
                double*           C,
                rocblas_int       ldc)
{
    return cblas_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

template <>
void cblas_geam(rocblas_operation      transa,
                rocblas_operation      transb,
                rocblas_int            m,
                rocblas_int            n,
                rocblas_float_complex* alpha,
                rocblas_float_complex* A,
                rocblas_int            lda,
                rocblas_float_complex* beta,
                rocblas_float_complex* B,
                rocblas_int            ldb,
                rocblas_float_complex* C,
                rocblas_int            ldc)
{
    return cblas_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

template <>
void cblas_geam(rocblas_operation       transa,
                rocblas_operation       transb,
                rocblas_int             m,
                rocblas_int             n,
                rocblas_double_complex* alpha,
                rocblas_double_complex* A,
                rocblas_int             lda,
                rocblas_double_complex* beta,
                rocblas_double_complex* B,
                rocblas_int             ldb,
                rocblas_double_complex* C,
                rocblas_int             ldc)
{
    return cblas_geam_helper(transa, transb, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

// gemm
template <>
void cblas_gemm<rocblas_bfloat16, float, float>(rocblas_operation       transA,
                                                rocblas_operation       transB,
                                                rocblas_int             m,
                                                rocblas_int             n,
                                                rocblas_int             k,
                                                float                   alpha,
                                                const rocblas_bfloat16* A,
                                                rocblas_int             lda,
                                                const rocblas_bfloat16* B,
                                                rocblas_int             ldb,
                                                float                   beta,
                                                float*                  C,
                                                rocblas_int             ldc,
                                                bool                    alt)
{
    // cblas does not support rocblas_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);

    host_vector<float> A_float(sizeA), B_float(sizeB);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<rocblas_bfloat16, rocblas_bfloat16, float>(rocblas_operation       transA,
                                                           rocblas_operation       transB,
                                                           rocblas_int             m,
                                                           rocblas_int             n,
                                                           rocblas_int             k,
                                                           float                   alpha,
                                                           const rocblas_bfloat16* A,
                                                           rocblas_int             lda,
                                                           const rocblas_bfloat16* B,
                                                           rocblas_int             ldb,
                                                           float                   beta,
                                                           rocblas_bfloat16*       C,
                                                           rocblas_int             ldc,
                                                           bool                    alt)
{
    // cblas does not support rocblas_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<rocblas_bfloat16>(C_float[i]);
}

template <>
void cblas_gemm<rocblas_half, float, float>(rocblas_operation   transA,
                                            rocblas_operation   transB,
                                            rocblas_int         m,
                                            rocblas_int         n,
                                            rocblas_int         k,
                                            float               alpha,
                                            const rocblas_half* A,
                                            rocblas_int         lda,
                                            const rocblas_half* B,
                                            rocblas_int         ldb,
                                            float               beta,
                                            float*              C,
                                            rocblas_int         ldc,
                                            bool                alt)
{
    // cblas does not support rocblas_half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);

    host_vector<float> A_float(sizeA), B_float(sizeB);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = A[i];
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = B[i];

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<rocblas_half, rocblas_half, float>(rocblas_operation   transA,
                                                   rocblas_operation   transB,
                                                   rocblas_int         m,
                                                   rocblas_int         n,
                                                   rocblas_int         k,
                                                   float               alpha,
                                                   const rocblas_half* A,
                                                   rocblas_int         lda,
                                                   const rocblas_half* B,
                                                   rocblas_int         ldb,
                                                   float               beta,
                                                   rocblas_half*       C,
                                                   rocblas_int         ldc,
                                                   bool                alt)
{
    // cblas does not support rocblas_half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = rocblas_bfloat16(float(A[i]),
                                          rocblas_bfloat16::rocblas_truncate_t::rocblas_truncate);
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = rocblas_bfloat16(float(B[i]),
                                          rocblas_bfloat16::rocblas_truncate_t::rocblas_truncate);
        for(size_t i = 0; i < sizeC; i++)
            C_float[i] = rocblas_bfloat16(float(C[i]),
                                          rocblas_bfloat16::rocblas_truncate_t::rocblas_truncate);
    }
    else
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = A[i];
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = B[i];
        for(size_t i = 0; i < sizeC; i++)
            C_float[i] = C[i];
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = rocblas_half(C_float[i]);
}

template <>
void cblas_gemm<rocblas_half, rocblas_half, rocblas_half>(rocblas_operation   transA,
                                                          rocblas_operation   transB,
                                                          rocblas_int         m,
                                                          rocblas_int         n,
                                                          rocblas_int         k,
                                                          rocblas_half        alpha,
                                                          const rocblas_half* A,
                                                          rocblas_int         lda,
                                                          const rocblas_half* B,
                                                          rocblas_int         ldb,
                                                          rocblas_half        beta,
                                                          rocblas_half*       C,
                                                          rocblas_int         ldc,
                                                          bool                alt)
{
    // cblas does not support rocblas_half, so convert to higher precision float
    // This will give more precise result which is acceptable for testing
    float alpha_float = alpha;
    float beta_float  = beta;

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = A[i];
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = B[i];
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = C[i];

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha_float,
                A_float,
                lda,
                B_float,
                ldb,
                beta_float,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = rocblas_half(C_float[i]);
}

template <>
void cblas_gemm<int8_t, int32_t, int32_t>(rocblas_operation transA,
                                          rocblas_operation transB,
                                          rocblas_int       m,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          int32_t           alpha,
                                          const int8_t*     A,
                                          rocblas_int       lda,
                                          const int8_t*     B,
                                          rocblas_int       ldb,
                                          int32_t           beta,
                                          int32_t*          C,
                                          rocblas_int       ldc,
                                          bool              alt)
{
    // cblas does not support int8_t input / int32_t output, however non-overflowing
    // 32-bit integer operations can be represented accurately with double-precision
    // floats, so convert to doubles and downcast result down to int32_t.
    // NOTE: This will not properly account for 32-bit integer overflow, however
    //       the result should be acceptable for testing.

    size_t const sizeA = ((transA == rocblas_operation_none) ? k : m) * size_t(lda);
    size_t const sizeB = ((transB == rocblas_operation_none) ? n : k) * size_t(ldb);
    size_t const sizeC = n * size_t(ldc);

    host_vector<double> A_double(sizeA);
    host_vector<double> B_double(sizeB);
    host_vector<double> C_double(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_double[i] = static_cast<double>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_double[i] = static_cast<double>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_double[i] = static_cast<double>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    cblas_dgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_double,
                lda,
                B_double,
                ldb,
                beta,
                C_double,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<int32_t>(C_double[i]);
}

template <typename T>
void cblas_geam_min_plus(rocblas_operation transA,
                         rocblas_operation transB,
                         rocblas_int       m,
                         rocblas_int       n,
                         rocblas_int       k,
                         const T           alpha,
                         const T*          A,
                         rocblas_int       lda,
                         const T*          B,
                         rocblas_int       ldb,
                         const T           beta,
                         const T*          C,
                         rocblas_int       ldc,
                         T*                D,
                         rocblas_int       ldd)
{
    bool TRANSA = transA != rocblas_operation_none;
    bool TRANSB = transB != rocblas_operation_none;

#pragma omp parallel for
    for(int n1 = 0; n1 < n; n1++)
    {
        for(int m1 = 0; m1 < m; m1++)
        {
            size_t idxC = size_t(ldc) * n1 + m1;
            size_t idxD = size_t(ldd) * n1 + m1;
            D[idxD]     = beta * C[idxC];
            for(int k1 = 0; k1 < k; k1++)
            {
                size_t idxA = TRANSA ? size_t(lda) * m1 + k1 : size_t(lda) * k1 + m1;
                size_t idxB = TRANSB ? size_t(ldb) * k1 + n1 : size_t(ldb) * n1 + k1;
                D[idxD]     = std::min(alpha * (A[idxA] + B[idxB]), D[idxD]);
            }
        }
    }
}

template <typename T>
void cblas_geam_plus_min(rocblas_operation transA,
                         rocblas_operation transB,
                         rocblas_int       m,
                         rocblas_int       n,
                         rocblas_int       k,
                         const T           alpha,
                         const T*          A,
                         rocblas_int       lda,
                         const T*          B,
                         rocblas_int       ldb,
                         const T           beta,
                         const T*          C,
                         rocblas_int       ldc,
                         T*                D,
                         rocblas_int       ldd)
{
    bool TRANSA = transA != rocblas_operation_none;
    bool TRANSB = transB != rocblas_operation_none;

#pragma omp parallel for
    for(int n1 = 0; n1 < n; n1++)
    {
        for(int m1 = 0; m1 < m; m1++)
        {
            size_t idxC = size_t(ldc) * n1 + m1;
            size_t idxD = size_t(ldd) * n1 + m1;
            D[idxD]     = beta * C[idxC];
            for(int k1 = 0; k1 < k; k1++)
            {
                size_t idxA = TRANSA ? size_t(lda) * m1 + k1 : size_t(lda) * k1 + m1;
                size_t idxB = TRANSB ? size_t(ldb) * k1 + n1 : size_t(ldb) * n1 + k1;
                D[idxD] += std::min(alpha * A[idxA], alpha * B[idxB]);
            }
        }
    }
}

template <typename T, typename U>
void cblas_herkx(rocblas_fill      uplo,
                 rocblas_operation transA,
                 rocblas_int       n,
                 rocblas_int       k,
                 const T*          alpha,
                 const T*          A,
                 rocblas_int       lda,
                 const T*          B,
                 rocblas_int       ldb,
                 const U*          beta,
                 T*                C,
                 rocblas_int       ldc)
{
    if(n <= 0 || (*beta == 1 && (k == 0 || *alpha == 0)))
        return;

    if(transA == rocblas_operation_none)
    {
        if(uplo == rocblas_fill_upper)
        {
#pragma omp parallel for
            for(int j = 0; j < n; ++j)
            {
                for(int i = 0; i <= j; i++)
                {
                    C[i + j * ldc] = *beta ? *beta * C[i + j * ldc] : 0;
                }

                if(*alpha)
                    for(int l = 0; l < k; l++)
                    {
                        T temp = *alpha * std::conj(B[j + l * ldb]);
                        for(int i = 0; i <= j; ++i)
                        {
                            C[i + j * ldc] += temp * A[i + l * lda];
                        }
                    }
                C[j + j * ldc].imag(0);
            }
        }
        else // lower
        {
#pragma omp parallel for
            for(int j = 0; j < n; ++j)
            {
                for(int i = j; i < n; i++)
                {
                    C[i + j * ldc] = *beta ? *beta * C[i + j * ldc] : 0;
                }

                if(*alpha)
                    for(int l = 0; l < k; l++)
                    {
                        T temp = *alpha * std::conj(B[j + l * ldb]);
                        for(int i = j; i < n; ++i)
                        {
                            C[i + j * ldc] += temp * A[i + l * lda];
                        }
                    }
                C[j + j * ldc].imag(0);
            }
        }
    }
    else // conjugate transpose
    {
        if(uplo == rocblas_fill_upper)
        {
#pragma omp parallel for
            for(int j = 0; j < n; ++j)
            {
                for(int i = 0; i <= j; i++)
                {
                    C[i + j * ldc] = *beta ? *beta * C[i + j * ldc] : 0;

                    if(*alpha)
                    {
                        T temp(0);
                        for(int l = 0; l < k; l++)
                        {
                            temp += std::conj(A[l + i * lda]) * B[l + j * ldb];
                        }
                        C[i + j * ldc] += *alpha * temp;
                    }

                    if(i == j)
                        C[j + j * ldc].imag(0);
                }
            }
        }
        else // lower
        {
#pragma omp parallel for
            for(int j = 0; j < n; ++j)
            {
                for(int i = j; i < n; i++)
                {
                    C[i + j * ldc] = *beta ? *beta * C[i + j * ldc] : 0;

                    if(*alpha)
                    {
                        T temp(0);
                        for(int l = 0; l < k; l++)
                        {
                            temp += std::conj(A[l + i * lda]) * B[l + j * ldb];
                        }
                        C[i + j * ldc] += *alpha * temp;
                    }

                    if(i == j)
                        C[j + j * ldc].imag(0);
                }
            }
        }
    }
}

// instantiations
template void cblas_geam_min_plus<float>(rocblas_operation transA,
                                         rocblas_operation transB,
                                         rocblas_int       m,
                                         rocblas_int       n,
                                         rocblas_int       k,
                                         const float       alpha,
                                         const float*      A,
                                         rocblas_int       lda,
                                         const float*      B,
                                         rocblas_int       ldb,
                                         const float       beta,
                                         const float*      C,
                                         rocblas_int       ldc,
                                         float*            D,
                                         rocblas_int       ldd);

template void cblas_geam_min_plus<double>(rocblas_operation transA,
                                          rocblas_operation transB,
                                          rocblas_int       m,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          const double      alpha,
                                          const double*     A,
                                          rocblas_int       lda,
                                          const double*     B,
                                          rocblas_int       ldb,
                                          const double      beta,
                                          const double*     C,
                                          rocblas_int       ldc,
                                          double*           D,
                                          rocblas_int       ldd);

template void cblas_geam_min_plus<rocblas_half>(rocblas_operation   transA,
                                                rocblas_operation   transB,
                                                rocblas_int         m,
                                                rocblas_int         n,
                                                rocblas_int         k,
                                                const rocblas_half  alpha,
                                                const rocblas_half* A,
                                                rocblas_int         lda,
                                                const rocblas_half* B,
                                                rocblas_int         ldb,
                                                const rocblas_half  beta,
                                                const rocblas_half* C,
                                                rocblas_int         ldc,
                                                rocblas_half*       D,
                                                rocblas_int         ldd);

template void cblas_geam_plus_min<float>(rocblas_operation transA,
                                         rocblas_operation transB,
                                         rocblas_int       m,
                                         rocblas_int       n,
                                         rocblas_int       k,
                                         const float       alpha,
                                         const float*      A,
                                         rocblas_int       lda,
                                         const float*      B,
                                         rocblas_int       ldb,
                                         const float       beta,
                                         const float*      C,
                                         rocblas_int       ldc,
                                         float*            D,
                                         rocblas_int       ldd);

template void cblas_geam_plus_min<double>(rocblas_operation transA,
                                          rocblas_operation transB,
                                          rocblas_int       m,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          const double      alpha,
                                          const double*     A,
                                          rocblas_int       lda,
                                          const double*     B,
                                          rocblas_int       ldb,
                                          const double      beta,
                                          const double*     C,
                                          rocblas_int       ldc,
                                          double*           D,
                                          rocblas_int       ldd);

template void cblas_geam_plus_min<rocblas_half>(rocblas_operation   transA,
                                                rocblas_operation   transB,
                                                rocblas_int         m,
                                                rocblas_int         n,
                                                rocblas_int         k,
                                                const rocblas_half  alpha,
                                                const rocblas_half* A,
                                                rocblas_int         lda,
                                                const rocblas_half* B,
                                                rocblas_int         ldb,
                                                const rocblas_half  beta,
                                                const rocblas_half* C,
                                                rocblas_int         ldc,
                                                rocblas_half*       D,
                                                rocblas_int         ldd);

template void cblas_herkx<rocblas_float_complex, float>(rocblas_fill                 uplo,
                                                        rocblas_operation            transA,
                                                        rocblas_int                  n,
                                                        rocblas_int                  k,
                                                        const rocblas_float_complex* alpha,
                                                        const rocblas_float_complex* A,
                                                        rocblas_int                  lda,
                                                        const rocblas_float_complex* B,
                                                        rocblas_int                  ldb,
                                                        const float*                 beta,
                                                        rocblas_float_complex*       C,
                                                        rocblas_int                  ldc);

template void cblas_herkx<rocblas_double_complex, double>(rocblas_fill                  uplo,
                                                          rocblas_operation             transA,
                                                          rocblas_int                   n,
                                                          rocblas_int                   k,
                                                          const rocblas_double_complex* alpha,
                                                          const rocblas_double_complex* A,
                                                          rocblas_int                   lda,
                                                          const rocblas_double_complex* B,
                                                          rocblas_int                   ldb,
                                                          const double*                 beta,
                                                          rocblas_double_complex*       C,
                                                          rocblas_int                   ldc);
