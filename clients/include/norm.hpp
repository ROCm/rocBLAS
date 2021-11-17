/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once

#include "cblas.h"
#include "norm.hpp"
#include "rocblas/rocblas.h"
#include "rocblas_vector.hpp"
#include "utility.hpp"
#include <cstdio>
#include <limits>
#include <memory>

/* =====================================================================
        Norm check: norm(A-B)/norm(A), evaluate relative error
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Norm check
 */

/* ========================================Norm Check
 * ==================================================== */

/* LAPACK fortran library functionality */
extern "C" {
float  slange_(char* norm_type, int* m, int* n, float* A, int* lda, float* work);
double dlange_(char* norm_type, int* m, int* n, double* A, int* lda, double* work);
float  clange_(char* norm_type, int* m, int* n, rocblas_float_complex* A, int* lda, float* work);
double zlange_(char* norm_type, int* m, int* n, rocblas_double_complex* A, int* lda, double* work);

float  slansy_(char* norm_type, char* uplo, int* n, float* A, int* lda, float* work);
double dlansy_(char* norm_type, char* uplo, int* n, double* A, int* lda, double* work);
float clanhe_(char* norm_type, char* uplo, int* n, rocblas_float_complex* A, int* lda, float* work);
double
    zlanhe_(char* norm_type, char* uplo, int* n, rocblas_double_complex* A, int* lda, double* work);

void saxpy_(int* n, float* alpha, float* x, int* incx, float* y, int* incy);
void daxpy_(int* n, double* alpha, double* x, int* incx, double* y, int* incy);
void caxpy_(int*                   n,
            rocblas_float_complex* alpha,
            rocblas_float_complex* x,
            int*                   incx,
            rocblas_float_complex* y,
            int*                   incy);
void zaxpy_(int*                    n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* x,
            int*                    incx,
            rocblas_double_complex* y,
            int*                    incy);
}

/*! \brief  Overloading: norm check for general Matrix: half/float/doubel/complex */
inline float xlange(char* norm_type, int* m, int* n, float* A, int* lda, float* work)
{
    return slange_(norm_type, m, n, A, lda, work);
}

inline double xlange(char* norm_type, int* m, int* n, double* A, int* lda, double* work)
{
    return dlange_(norm_type, m, n, A, lda, work);
}

inline float
    xlange(char* norm_type, int* m, int* n, rocblas_float_complex* A, int* lda, float* work)
{
    return clange_(norm_type, m, n, A, lda, work);
}

inline double
    xlange(char* norm_type, int* m, int* n, rocblas_double_complex* A, int* lda, double* work)
{
    return zlange_(norm_type, m, n, A, lda, work);
}

inline float xlanhe(char* norm_type, char* uplo, int* n, float* A, int* lda, float* work)
{
    return slansy_(norm_type, uplo, n, A, lda, work);
}

inline double xlanhe(char* norm_type, char* uplo, int* n, double* A, int* lda, double* work)
{
    return dlansy_(norm_type, uplo, n, A, lda, work);
}

inline float
    xlanhe(char* norm_type, char* uplo, int* n, rocblas_float_complex* A, int* lda, float* work)
{
    return clanhe_(norm_type, uplo, n, A, lda, work);
}

inline double
    xlanhe(char* norm_type, char* uplo, int* n, rocblas_double_complex* A, int* lda, double* work)
{
    return zlanhe_(norm_type, uplo, n, A, lda, work);
}

inline void xaxpy(int* n, float* alpha, float* x, int* incx, float* y, int* incy)
{
    return saxpy_(n, alpha, x, incx, y, incy);
}

inline void xaxpy(int* n, double* alpha, double* x, int* incx, double* y, int* incy)
{
    return daxpy_(n, alpha, x, incx, y, incy);
}

inline void xaxpy(int*                   n,
                  rocblas_float_complex* alpha,
                  rocblas_float_complex* x,
                  int*                   incx,
                  rocblas_float_complex* y,
                  int*                   incy)
{
    return caxpy_(n, alpha, x, incx, y, incy);
}

inline void xaxpy(int*                    n,
                  rocblas_double_complex* alpha,
                  rocblas_double_complex* x,
                  int*                    incx,
                  rocblas_double_complex* y,
                  int*                    incy)
{
    return zaxpy_(n, alpha, x, incx, y, incy);
}

template <typename T>
void m_axpy(size_t* N, T* alpha, T* x, int* incx, T* y, int* incy)
{
    for(size_t i = 0; i < *N; i++)
    {
        y[i * (*incy)] = (*alpha) * x[i * (*incx)] + y[i * (*incy)];
    }
}

/* ============== Norm Check for General Matrix ============= */
/*! \brief compare the norm error of two matrices hCPU & hGPU */

// Real
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
double norm_check_general(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    size_t size = N * (size_t)lda;

    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(rocblas_int i = 0; i < N; i++)
    {
        for(rocblas_int j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }

    double      work[1];
    rocblas_int incx  = 1;
    double      alpha = -1.0;

    double cpu_norm = xlange(&norm_type, &M, &N, hCPU_double.data(), &lda, work);
    m_axpy(&size, &alpha, hCPU_double.data(), &incx, hGPU_double.data(), &incx);
    double error = xlange(&norm_type, &M, &N, hGPU_double.data(), &lda, work) / cpu_norm;

    return error;
}

// Complex
template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
double norm_check_general(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    decltype(std::real(*hCPU)) work[1];
    rocblas_int                incx  = 1;
    T                          alpha = -1.0;
    size_t                     size  = N * (size_t)lda;

    double cpu_norm = xlange(&norm_type, &M, &N, hCPU, &lda, work);
    m_axpy(&size, &alpha, hCPU, &incx, hGPU, &incx);
    double error = xlange(&norm_type, &M, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

// For BF16 and half, we convert the results to double first
template <typename T,
          typename VEC,
          std::enable_if_t<std::is_same<T, rocblas_half>{} || std::is_same<T, rocblas_bfloat16>{},
                           int> = 0>
double norm_check_general(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, VEC&& hCPU, T* hGPU)
{
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(rocblas_int i = 0; i < N; i++)
    {
        for(rocblas_int j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = hCPU[idx];
            hGPU_double[idx] = hGPU[idx];
        }
    }

    return norm_check_general<double>(norm_type, M, N, lda, hCPU_double, hGPU_double);
}

/* ============== Norm Check for strided_batched case ============= */
template <typename T, template <typename> class VEC, typename T_hpa>
double norm_check_general(char           norm_type,
                          rocblas_int    M,
                          rocblas_int    N,
                          rocblas_int    lda,
                          rocblas_stride stride_a,
                          VEC<T_hpa>&    hCPU,
                          T*             hGPU,
                          rocblas_int    batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(size_t i = 0; i < batch_count; i++)
    {
        auto index = i * stride_a;

        auto error = norm_check_general(norm_type, M, N, lda, (T_hpa*)hCPU + index, hGPU + index);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

/* ============== Norm Check for batched case ============= */

template <typename T, typename T_hpa>
double norm_check_general(char                      norm_type,
                          rocblas_int               M,
                          rocblas_int               N,
                          rocblas_int               lda,
                          host_batch_vector<T_hpa>& hCPU,
                          host_batch_vector<T>&     hGPU,
                          rocblas_int               batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(rocblas_int i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <typename T>
double norm_check_general(char        norm_type,
                          rocblas_int M,
                          rocblas_int N,
                          rocblas_int lda,
                          T*          hCPU[],
                          T*          hGPU[],
                          rocblas_int batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(rocblas_int i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

/* ============== Norm Check for Symmetric Matrix ============= */
/*! \brief compare the norm error of two Hermitian/symmetric matrices hCPU & hGPU */
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
double norm_check_symmetric(
    char norm_type, char uplo, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    double      work[1];
    rocblas_int incx  = 1;
    double      alpha = -1.0;
    size_t      size  = N * (size_t)lda;

    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(rocblas_int i = 0; i < N; i++)
    {
        for(rocblas_int j = 0; j < N; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }

    double cpu_norm = xlanhe(&norm_type, &uplo, &N, hCPU_double, &lda, work);
    m_axpy(&size, &alpha, hCPU_double.data(), &incx, hGPU_double.data(), &incx);
    double error = xlanhe(&norm_type, &uplo, &N, hGPU_double, &lda, work) / cpu_norm;

    return error;
}

template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
double norm_check_symmetric(
    char norm_type, char uplo, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    decltype(std::real(*hCPU)) work[1];
    rocblas_int                incx  = 1;
    T                          alpha = -1.0;
    size_t                     size  = (size_t)lda * N;

    double cpu_norm = xlanhe(&norm_type, &uplo, &N, hCPU, &lda, work);
    m_axpy(&size, &alpha, hCPU, &incx, hGPU, &incx);
    double error = xlanhe(&norm_type, &uplo, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

template <>
inline double norm_check_symmetric(char          norm_type,
                                   char          uplo,
                                   rocblas_int   N,
                                   rocblas_int   lda,
                                   rocblas_half* hCPU,
                                   rocblas_half* hGPU)
{
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(rocblas_int i = 0; i < N; i++)
    {
        for(rocblas_int j = 0; j < N; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = hCPU[idx];
            hGPU_double[idx] = hGPU[idx];
        }
    }

    return norm_check_symmetric(norm_type, uplo, N, lda, hCPU_double.data(), hGPU_double.data());
}

template <typename T>
double matrix_norm_1(rocblas_int M, rocblas_int N, rocblas_int lda, T* hA_gold, T* hA)
{
    double max_err_scal = 0.0;
    double max_err      = 0.0;
    double err          = 0.0;
    double err_scal     = 0.0;
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M; j++)
        {
            size_t idx = j + i * (size_t)lda;
            err += rocblas_abs((hA_gold[idx] - hA[idx]));
            err_scal += rocblas_abs(hA_gold[idx]);
        }
        max_err_scal = max_err_scal > err_scal ? max_err_scal : err_scal;
        max_err      = max_err > err ? max_err : err;
    }

    return max_err / max_err_scal;
}

template <typename T>
double vector_norm_1(rocblas_int M, rocblas_int incx, T* hx_gold, T* hx)
{
    double max_err_scal = 0.0;
    double max_err      = 0.0;
    for(int i = 0; i < M; i++)
    {
        size_t idx = i * (size_t)incx;
        max_err += rocblas_abs((hx_gold[idx] - hx[idx]));
        max_err_scal += rocblas_abs(hx_gold[idx]);
    }

    return max_err / max_err_scal;
}
