/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_TEMPLATE_API_
#define _ROCBLAS_TEMPLATE_API_

#include <rocblas_types.h>


/*!\file
 * \brief rocblas_template_api.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based NVIDIA GPUs.
 *  The precision is templated
*/

    /*
     * ===========================================================================
     *   READEME: Please follow the naming convention
     *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
     *     lower case for vector, e.g. vector x, y    GEMV (y = A*x)

     *   rocblas_handle is defined as rocblas_queue (hipStream_t) right now
     *   to allow multiple BLAS routines called in different streams(or called queues)
     *   on the same GPU device. This feature is called streaming supported by CUBLAS,too.
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

#endif  /* _ROCBLAS_TEMPLATE_API_ */
