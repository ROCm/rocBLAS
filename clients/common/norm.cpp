/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "norm.hpp"
#include "cblas.h"
#include "rocblas.h"
#include "rocblas_vector.hpp"
#include "utility.hpp"
#include <cstdio>
#include <limits>
#include <memory>

/* =====================================================================
     README: Norm check: norm(A-B)/norm(A), evaluate relative error
             Numerically, it is recommended by lapack.

    Call lapack fortran routines that do not exsit in cblas library.
    No special header is required. But need to declare
    function prototype

    All the functions are fortran and should append underscore (_) while
    declaring prototype and calling.
    xlange and xaxpy prototype are like following
    =================================================================== */

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

/* ============================Norm Check for General Matrix: float/double/complex template
 * speciliazation ======================================= */

/*! \brief compare the norm error of two matrices hCPU & hGPU */
template <>
double norm_check_general<rocblas_bfloat16>(char              norm_type,
                                            rocblas_int       M,
                                            rocblas_int       N,
                                            rocblas_int       lda,
                                            rocblas_bfloat16* hCPU,
                                            rocblas_bfloat16* hGPU)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    double error_double = std::numeric_limits<double>::quiet_NaN();

    host_vector<float> hCPU_float(N * lda), hGPU_float(N * lda);
    for(rocblas_int i = 0; i < N * lda; i++)
    {
        hCPU_float[i] = float(hCPU[i]);
        hGPU_float[i] = float(hGPU[i]);
    }

    float       work;
    rocblas_int incx  = 1;
    float       alpha = -1.0f;
    rocblas_int size  = lda * N;

    float cpu_norm = slange_(&norm_type, &M, &N, hCPU_float, &lda, &work);
    saxpy_(&size, &alpha, hCPU_float, &incx, hGPU_float, &incx);

    float error_float = slange_(&norm_type, &M, &N, hGPU_float, &lda, &work) / cpu_norm;
    error_double      = double(error_float);

    return error_double;
}

template <>
double norm_check_general<rocblas_half>(char          norm_type,
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

    double error_double = std::numeric_limits<double>::quiet_NaN();

    host_vector<float> hCPU_float(N * lda), hGPU_float(N * lda);
    for(rocblas_int i = 0; i < N * lda; i++)
    {
        hCPU_float[i] = half_to_float(hCPU[i]);
        hGPU_float[i] = half_to_float(hGPU[i]);
    }

    float       work;
    rocblas_int incx  = 1;
    float       alpha = -1.0f;
    rocblas_int size  = lda * N;

    float cpu_norm = slange_(&norm_type, &M, &N, hCPU_float, &lda, &work);
    saxpy_(&size, &alpha, hCPU_float, &incx, hGPU_float, &incx);

    float error_float = slange_(&norm_type, &M, &N, hGPU_float, &lda, &work) / cpu_norm;
    error_double      = double(error_float);

    return error_double;
}

template <>
double norm_check_general<float>(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, float* hCPU, float* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    float       work;
    rocblas_int incx  = 1;
    float       alpha = -1.0f;
    rocblas_int size  = lda * N;

    float cpu_norm = slange_(&norm_type, &M, &N, hCPU, &lda, &work);
    saxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    float error = slange_(&norm_type, &M, &N, hGPU, &lda, &work) / cpu_norm;

    return (double)error;
}

template <>
double norm_check_general<double>(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, double* hCPU, double* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    double      work[1];
    rocblas_int incx  = 1;
    double      alpha = -1.0;
    rocblas_int size  = lda * N;

    double cpu_norm = dlange_(&norm_type, &M, &N, hCPU, &lda, work);
    daxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    double error = dlange_(&norm_type, &M, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

template <>
double norm_check_general<int32_t>(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, int32_t* hCPU, int32_t* hGPU)
{
    // Upconvert int32_t to double and call double version
    host_vector<double> hCPU_double(M * N), hGPU_double(M * N);

    for(int i = 0; i < M * N; i++)
    {
        hCPU_double[i] = double(hCPU[i]);
        hGPU_double[i] = double(hGPU[i]);
    }
    return norm_check_general<double>(norm_type, M, N, lda, hCPU_double, hGPU_double);
}

template <>
double norm_check_general<rocblas_float_complex>(char                   norm_type,
                                                 rocblas_int            M,
                                                 rocblas_int            N,
                                                 rocblas_int            lda,
                                                 rocblas_float_complex* hCPU,
                                                 rocblas_float_complex* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    float       work[1];
    rocblas_int incx  = 1;
    float       alpha = -1.0f;
    rocblas_int size  = lda * N;

    float cpu_norm = clange_(&norm_type, &M, &N, hCPU, &lda, work);
    caxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    float error = clange_(&norm_type, &M, &N, hGPU, &lda, work) / cpu_norm;

    return (double)error;
}

template <>
double norm_check_general<rocblas_double_complex>(char                    norm_type,
                                                  rocblas_int             M,
                                                  rocblas_int             N,
                                                  rocblas_int             lda,
                                                  rocblas_double_complex* hCPU,
                                                  rocblas_double_complex* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    double      work[1];
    rocblas_int incx  = 1;
    double      alpha = -1.0;
    rocblas_int size  = lda * N;

    double cpu_norm = zlange_(&norm_type, &M, &N, hCPU, &lda, work);
    zaxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    double error = zlange_(&norm_type, &M, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

//=====Norm Check for strided_batched matrix
template <>
double norm_check_general<rocblas_bfloat16>(char              norm_type,
                                            rocblas_int       M,
                                            rocblas_int       N,
                                            rocblas_int       lda,
                                            rocblas_int       stride_a,
                                            rocblas_int       batch_count,
                                            rocblas_bfloat16* hCPU,
                                            rocblas_bfloat16* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    rocblas_int        totalsize = N * lda + (batch_count - 1) * stride_a;
    host_vector<float> hCPU_float(totalsize), hGPU_float(totalsize);
    for(rocblas_int i_batch = 0; i_batch < batch_count; i_batch++)
    {
        for(rocblas_int i = 0; i < N * lda; i++)
        {
            auto index        = i + i_batch * stride_a;
            hCPU_float[index] = float(hCPU[index]);
            hGPU_float[index] = float(hGPU[index]);
        }
    }

    float       work;
    rocblas_int incx  = 1;
    float       alpha = -1.0f;
    rocblas_int size  = lda * N;

    double cumulative_error = 0.0;

    for(rocblas_int i = 0; i < batch_count; i++)
    {
        float cpu_norm = slange_(&norm_type, &M, &N, &hCPU_float[i * stride_a], &lda, &work);

        saxpy_(&size, &alpha, &hCPU_float[i * stride_a], &incx, &hGPU_float[i * stride_a], &incx);

        float error
            = slange_(&norm_type, &M, &N, &hGPU_float[i * stride_a], &lda, &work) / cpu_norm;

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

template <>
double norm_check_general<rocblas_half>(char          norm_type,
                                        rocblas_int   M,
                                        rocblas_int   N,
                                        rocblas_int   lda,
                                        rocblas_int   stride_a,
                                        rocblas_int   batch_count,
                                        rocblas_half* hCPU,
                                        rocblas_half* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    rocblas_int        totalsize = N * lda + (batch_count - 1) * stride_a;
    host_vector<float> hCPU_float(totalsize), hGPU_float(totalsize);
    for(rocblas_int i_batch = 0; i_batch < batch_count; i_batch++)
    {
        for(rocblas_int i = 0; i < N * lda; i++)
        {
            auto index        = i + i_batch * stride_a;
            hCPU_float[index] = half_to_float(hCPU[index]);
            hGPU_float[index] = half_to_float(hGPU[index]);
        }
    }

    float       work;
    rocblas_int incx  = 1;
    float       alpha = -1.0f;
    rocblas_int size  = lda * N;

    double cumulative_error = 0.0;

    for(rocblas_int i = 0; i < batch_count; i++)
    {
        float cpu_norm = slange_(&norm_type, &M, &N, &hCPU_float[i * stride_a], &lda, &work);

        saxpy_(&size, &alpha, &hCPU_float[i * stride_a], &incx, &hGPU_float[i * stride_a], &incx);

        float error
            = slange_(&norm_type, &M, &N, &hGPU_float[i * stride_a], &lda, &work) / cpu_norm;

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

//=====Norm Check for strided_batched matrix
template <>
double norm_check_general(char         norm_type,
                          rocblas_int  M,
                          rocblas_int  N,
                          rocblas_int  lda,
                          rocblas_int  stride_a,
                          rocblas_int  batch_count,
                          rocblas_int* hCPU,
                          rocblas_int* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    rocblas_int         totalsize = N * lda + (batch_count - 1) * stride_a;
    host_vector<double> hCPU_double(totalsize), hGPU_double(totalsize);
    for(rocblas_int i_batch = 0; i_batch < batch_count; i_batch++)
    {
        for(rocblas_int i = 0; i < N * lda; i++)
        {
            auto index         = i + i_batch * stride_a;
            hCPU_double[index] = hCPU[index];
            hGPU_double[index] = hGPU[index];
        }
    }

    double      work;
    rocblas_int incx             = 1;
    double      alpha            = -1.0f;
    rocblas_int size             = lda * N;
    double      cumulative_error = 0.0;

    for(rocblas_int i = 0; i < batch_count; i++)
    {
        double cpu_norm = dlange_(&norm_type, &M, &N, &hCPU_double[i * stride_a], &lda, &work);

        daxpy_(&size, &alpha, &hCPU_double[i * stride_a], &incx, &hGPU_double[i * stride_a], &incx);

        double error
            = dlange_(&norm_type, &M, &N, &hGPU_double[i * stride_a], &lda, &work) / cpu_norm;

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

template <>
double norm_check_general<float>(char        norm_type,
                                 rocblas_int M,
                                 rocblas_int N,
                                 rocblas_int lda,
                                 rocblas_int stride_a,
                                 rocblas_int batch_count,
                                 float*      hCPU,
                                 float*      hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    float       work;
    rocblas_int incx  = 1;
    float       alpha = -1.0f;
    rocblas_int size  = lda * N;

    double cumulative_error = 0.0;

    for(int i = 0; i < batch_count; i++)
    {
        float cpu_norm = slange_(&norm_type, &M, &N, &(hCPU[i * stride_a]), &lda, &work);

        saxpy_(&size, &alpha, &(hCPU[i * stride_a]), &incx, &(hGPU[i * stride_a]), &incx);

        float error = slange_(&norm_type, &M, &N, &(hGPU[i * stride_a]), &lda, &work) / cpu_norm;

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

template <>
double norm_check_general<double>(char        norm_type,
                                  rocblas_int M,
                                  rocblas_int N,
                                  rocblas_int lda,
                                  rocblas_int stride_a,
                                  rocblas_int batch_count,
                                  double*     hCPU,
                                  double*     hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double      work;
    rocblas_int incx  = 1;
    double      alpha = -1.0f;
    rocblas_int size  = lda * N;

    double cumulative_error = 0.0;

    for(int i = 0; i < batch_count; i++)
    {
        double cpu_norm = dlange_(&norm_type, &M, &N, &(hCPU[i * stride_a]), &lda, &work);

        daxpy_(&size, &alpha, &(hCPU[i * stride_a]), &incx, &(hGPU[i * stride_a]), &incx);

        double error = dlange_(&norm_type, &M, &N, &(hGPU[i * stride_a]), &lda, &work) / cpu_norm;

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

/* ============================Norm Check for Symmetric Matrix: float/double/complex template
 * speciliazation ======================================= */

/*! \brief compare the norm error of two hermitian/symmetric matrices hCPU & hGPU */

template <>
double norm_check_symmetric<float>(
    char norm_type, char uplo, rocblas_int N, rocblas_int lda, float* hCPU, float* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    float       work[1];
    rocblas_int incx  = 1;
    float       alpha = -1.0f;
    rocblas_int size  = lda * N;

    float cpu_norm = slansy_(&norm_type, &uplo, &N, hCPU, &lda, work);
    saxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    float error = slansy_(&norm_type, &uplo, &N, hGPU, &lda, work) / cpu_norm;

    return (double)error;
}

template <>
double norm_check_symmetric<double>(
    char norm_type, char uplo, rocblas_int N, rocblas_int lda, double* hCPU, double* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    double      work[1];
    rocblas_int incx  = 1;
    double      alpha = -1.0;
    rocblas_int size  = lda * N;

    double cpu_norm = dlansy_(&norm_type, &uplo, &N, hCPU, &lda, work);
    daxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    double error = dlansy_(&norm_type, &uplo, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

template <>
double norm_check_symmetric<rocblas_float_complex>(char                   norm_type,
                                                   char                   uplo,
                                                   rocblas_int            N,
                                                   rocblas_int            lda,
                                                   rocblas_float_complex* hCPU,
                                                   rocblas_float_complex* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    float       work[1];
    rocblas_int incx  = 1;
    float       alpha = -1.0f;
    rocblas_int size  = lda * N;

    float cpu_norm = clanhe_(&norm_type, &uplo, &N, hCPU, &lda, work);
    caxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    float error = clanhe_(&norm_type, &uplo, &N, hGPU, &lda, work) / cpu_norm;

    return (double)error;
}

template <>
double norm_check_symmetric<rocblas_double_complex>(char                    norm_type,
                                                    char                    uplo,
                                                    rocblas_int             N,
                                                    rocblas_int             lda,
                                                    rocblas_double_complex* hCPU,
                                                    rocblas_double_complex* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    double      work[1];
    rocblas_int incx  = 1;
    double      alpha = -1.0;
    rocblas_int size  = lda * N;

    double cpu_norm = zlanhe_(&norm_type, &uplo, &N, hCPU, &lda, work);
    zaxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    double error = zlanhe_(&norm_type, &uplo, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}
