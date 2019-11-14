/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef _NORM_H
#define _NORM_H

#include "cblas.h"
#include "norm.hpp"
#include "rocblas.h"
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
void caxpy_(
    int* n, float* alpha, rocblas_float_complex* x, int* incx, rocblas_float_complex* y, int* incy);
void zaxpy_(int*                    n,
            double*                 alpha,
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

inline void xaxpy(
    int* n, float* alpha, rocblas_float_complex* x, int* incx, rocblas_float_complex* y, int* incy)
{
    return caxpy_(n, alpha, x, incx, y, incy);
}

inline void xaxpy(int*                    n,
                  double*                 alpha,
                  rocblas_double_complex* x,
                  int*                    incx,
                  rocblas_double_complex* y,
                  int*                    incy)
{
    return zaxpy_(n, alpha, x, incx, y, incy);
}

/* ============== Norm Check for General Matrix ============= */
/*! \brief compare the norm error of two matrices hCPU & hGPU */
template <typename T, typename std::enable_if<!is_complex<T>, int>::type = 0>
double norm_check_general(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    host_vector<double> hCPU_double(N * lda);
    host_vector<double> hGPU_double(N * lda);

    for(rocblas_int i = 0; i < N * lda; i++)
    {
        hCPU_double[i] = double(hCPU[i]);
        hGPU_double[i] = double(hGPU[i]);
    }

    double      work[1];
    rocblas_int incx  = 1;
    double      alpha = -1.0;
    rocblas_int size  = lda * N;

    double cpu_norm = xlange(&norm_type, &M, &N, hCPU_double.data(), &lda, work);
    xaxpy(&size, &alpha, hCPU_double.data(), &incx, hGPU_double.data(), &incx);
    double error = xlange(&norm_type, &M, &N, hGPU_double.data(), &lda, work) / cpu_norm;

    return error;
}

template <typename T, typename std::enable_if<is_complex<T>, int>::type = 0>
double norm_check_general(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    decltype(std::real(*hCPU)) work[1];
    rocblas_int                incx  = 1;
    decltype(std::real(*hCPU)) alpha = -1.0f;
    rocblas_int                size  = lda * N;

    double cpu_norm = xlange(&norm_type, &M, &N, hCPU, &lda, work);
    xaxpy(&size, &alpha, hCPU, &incx, hGPU, &incx);
    double error = xlange(&norm_type, &M, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

template <>
inline double norm_check_general(char          norm_type,
                                 rocblas_int   M,
                                 rocblas_int   N,
                                 rocblas_int   lda,
                                 rocblas_half* hCPU,
                                 rocblas_half* hGPU)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    host_vector<double> hCPU_double(N * lda);
    host_vector<double> hGPU_double(N * lda);

    for(rocblas_int i = 0; i < N * lda; i++)
    {
        hCPU_double[i] = hCPU[i];
        hGPU_double[i] = hGPU[i];
    }

    return norm_check_general(norm_type, M, N, lda, hCPU_double.data(), hGPU_double.data());
}

/* ============== Norm Check for strided_batched case ============= */
template <typename T>
double norm_check_general(char           norm_type,
                          rocblas_int    M,
                          rocblas_int    N,
                          rocblas_int    lda,
                          rocblas_stride stride_a,
                          rocblas_int    batch_count,
                          T*             hCPU,
                          T*             hGPU)
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
        auto index = i * stride_a;

        auto error = norm_check_general(norm_type, M, N, lda, hCPU + index, hGPU + index);

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
template <typename T>
double norm_check_general(char           norm_type,
                          rocblas_int    M,
                          rocblas_int    N,
                          rocblas_int    lda,
                          rocblas_int    batch_count,
                          host_vector<T> hCPU[],
                          host_vector<T> hGPU[])
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
                          rocblas_int batch_count,
                          T*          hCPU[],
                          T*          hGPU[])
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
/*! \brief compare the norm error of two hermitian/symmetric matrices hCPU & hGPU */
template <typename T, typename std::enable_if<!is_complex<T>, int>::type = 0>
double norm_check_symmetric(
    char norm_type, char uplo, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    double      work[1];
    rocblas_int incx  = 1;
    double      alpha = -1.0;
    rocblas_int size  = lda * N;

    host_vector<double> hCPU_double(N * lda);
    host_vector<double> hGPU_double(N * lda);

    for(rocblas_int i = 0; i < N * lda; i++)
    {
        hCPU_double[i] = double(hCPU[i]);
        hGPU_double[i] = double(hGPU[i]);
    }

    double cpu_norm = xlanhe(&norm_type, &uplo, &N, hCPU_double, &lda, work);
    xaxpy(&size, &alpha, hCPU_double, &incx, hGPU_double, &incx);
    double error = xlanhe(&norm_type, &uplo, &N, hGPU_double, &lda, work) / cpu_norm;

    return error;
}

template <typename T, typename std::enable_if<is_complex<T>, int>::type = 0>
double norm_check_symmetric(
    char norm_type, char uplo, rocblas_int N, rocblas_int lda, T* hCPU, T* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    decltype(std::real(*hCPU)) work[1];
    rocblas_int                incx  = 1;
    decltype(std::real(*hCPU)) alpha = -1.0;
    rocblas_int                size  = lda * N;

    double cpu_norm = xlanhe(&norm_type, &uplo, &N, hCPU, &lda, work);
    xaxpy(&size, &alpha, hCPU, &incx, hGPU, &incx);
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
    host_vector<double> hCPU_double(N * lda);
    host_vector<double> hGPU_double(N * lda);

    for(rocblas_int i = 0; i < N * lda; i++)
    {
        hCPU_double[i] = hCPU[i];
        hGPU_double[i] = hGPU[i];
    }

    return norm_check_symmetric(norm_type, uplo, N, lda, hCPU_double.data(), hGPU_double.data());
}

#endif
