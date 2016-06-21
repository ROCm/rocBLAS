/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_TEMPLATE_FUNCTIONS_H_
#define _ROCBLAS_TEMPLATE_FUNCTIONS_H_

#include <rocblas_types.h>


/*!\file
 * \brief rocblas_template_api.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based NVIDIA GPUs.
 *  This file exposes C++ templated BLAS interface with only the precision templated.
*/

    /*
     * ===========================================================================
     *   READEME: Please follow the naming convention
     *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
     *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
     * ===========================================================================
     */


     template<typename T>
     rocblas_status
     rocblas_scal(rocblas_handle handle,
         rocblas_int n,
         const T *alpha,
         T *x, rocblas_int incx);

     template<typename T>
     rocblas_status
     rocblas_copy(rocblas_handle handle,
         rocblas_int n,
         const T *x, rocblas_int incx,
         T *y, rocblas_int incy);

     template<typename T>
     rocblas_status
     rocblas_swap(rocblas_handle handle,
         rocblas_int n,
         T *x, rocblas_int incx,
         T *y, rocblas_int incy);

     template<typename T>
     rocblas_status
     rocblas_dot(rocblas_handle handle,
         rocblas_int n,
         const T *x, rocblas_int incx,
         const T *y, rocblas_int incy,
         T *result);

     template<typename T1, typename T2>
     rocblas_status
     rocblas_asum(rocblas_handle handle,
         rocblas_int n,
         const T1 *x, rocblas_int incx,
         T2 *result);

     template<typename T1, typename T2>
     rocblas_status
     rocblas_nrm2(rocblas_handle handle,
         rocblas_int n,
         const T1 *x, rocblas_int incx,
         T2 *result);

     template<typename T>
     rocblas_status
     rocblas_amax(rocblas_handle handle,
         rocblas_int n,
         const T *x, rocblas_int incx,
         rocblas_int *result);

     template<typename T>
     rocblas_status
     rocblas_amin(rocblas_handle handle,
         rocblas_int n,
         const T *x, rocblas_int incx,
         rocblas_int *result);

     template<typename T>
     rocblas_status
     rocblas_axpy(rocblas_handle handle,
         rocblas_int n,
         const T *alpha,
         const T *x, rocblas_int incx,
         T *y, rocblas_int incy);

    template<typename T>
    rocblas_status
    rocblas_symv(rocblas_handle handle,
             rocblas_fill uplo,
             rocblas_int n,
             const T *alpha,
             const T *A, rocblas_int lda,
             const T *x, rocblas_int incx,
             const T *beta,
             T *y, rocblas_int incy);

    template<typename T>
    rocblas_status rocblas_gemm(rocblas_handle handle,
        rocblas_operation transA, rocblas_operation transB,
        rocblas_int m, rocblas_int n, rocblas_int k,
        const T *alpha,
        const T *A, rocblas_int lda,
        const T *B, rocblas_int ldb,
        const T *beta,
        T *C, rocblas_int ldc);

    template<typename T>
    rocblas_status rocblas_trsm(rocblas_handle handle,
        rocblas_side side, rocblas_fill uplo,
        rocblas_operation transA, rocblas_diagonal diag,
        rocblas_int m, rocblas_int n,
        const T* alpha,
        const T* A, rocblas_int lda,
        T*       B, rocblas_int ldb);

    template<typename T>
    rocblas_status
    rocblas_trtri(rocblas_handle handle,
        rocblas_fill uplo, rocblas_diagonal diag,
        rocblas_int n,
        T *A, rocblas_int lda);

    template<typename T>
    rocblas_status
    rocblas_trtri_batched_workspace(rocblas_handle handle,
        rocblas_fill uplo, rocblas_diagonal diag,
        rocblas_int n,
        T *A, rocblas_int lda, rocblas_int bsa,
        T *invA, rocblas_int ldinvA, rocblas_int bsinvA,
        rocblas_int batch_count);

#endif  /* _ROCBLAS_TEMPLATE_FUNCTIONS_H_ */
