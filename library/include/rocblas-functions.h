/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCBLAS_FUNCTIONS_H_
#define _ROCBLAS_FUNCTIONS_H_
#include "rocblas-export.h"
#include "rocblas-types.h"

/*!\file
 * \brief rocblas_functions.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based
 * NVIDIA GPUs.
 *  This file exposes C89 BLAS interface
 */

/*
 * ===========================================================================
 *   README: Please follow the naming convention
 *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
 *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
 * ===========================================================================
 */

/*
   ROCBLAS_VA_OPT_PRAGMA(pragma, ...) creates a _Pragma with stringized pragma
   if the trailing argument list is non-empty.

   __VA_OPT__ support is automatically detected if it's available; otherwise,
   the GCC/Clang ##__VA_ARGS__ extension is used to emulate it.
*/
#define ROCBLAS_VA_OPT_3RD_ARG(_1, _2, _3, ...) _3
#define ROCBLAS_VA_OPT_SUPPORTED(...) ROCBLAS_VA_OPT_3RD_ARG(__VA_OPT__(, ), 1, 0, )

#if ROCBLAS_VA_OPT_SUPPORTED(?)
#define ROCBLAS_VA_OPT_PRAGMA(pragma, ...) __VA_OPT__(_Pragma(#pragma))

#else // ROCBLAS_VA_OPT_SUPPORTED

#define ROCBLAS_VA_OPT_COUNT_IMPL(X, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define ROCBLAS_VA_OPT_COUNT(...) \
    ROCBLAS_VA_OPT_COUNT_IMPL(, ##__VA_ARGS__, N, N, N, N, N, N, N, N, N, N, 0)
#define ROCBLAS_VA_OPT_PRAGMA_SELECT0(...)
#define ROCBLAS_VA_OPT_PRAGMA_SELECTN(pragma, ...) _Pragma(#pragma)
#define ROCBLAS_VA_OPT_PRAGMA_IMPL2(pragma, count) ROCBLAS_VA_OPT_PRAGMA_SELECT##count(pragma)
#define ROCBLAS_VA_OPT_PRAGMA_IMPL(pragma, count) ROCBLAS_VA_OPT_PRAGMA_IMPL2(pragma, count)
#define ROCBLAS_VA_OPT_PRAGMA(pragma, ...) \
    ROCBLAS_VA_OPT_PRAGMA_IMPL(pragma, ROCBLAS_VA_OPT_COUNT(__VA_ARGS__))

#endif // ROCBLAS_VA_OPT_SUPPORTED

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

ROCBLAS_EXPORT rocblas_status rocblas_sscal(
    rocblas_handle handle, rocblas_int n, const float* alpha, float* x, rocblas_int incx);

ROCBLAS_EXPORT rocblas_status rocblas_dscal(
    rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx);

ROCBLAS_EXPORT rocblas_status rocblas_cscal(rocblas_handle               handle,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            rocblas_float_complex*       x,
                                            rocblas_int                  incx);

ROCBLAS_EXPORT rocblas_status rocblas_zscal(rocblas_handle                handle,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            rocblas_double_complex*       x,
                                            rocblas_int                   incx);

ROCBLAS_EXPORT rocblas_status rocblas_csscal(rocblas_handle         handle,
                                             rocblas_int            n,
                                             const float*           alpha,
                                             rocblas_float_complex* x,
                                             rocblas_int            incx);

/*! \brief BLAS Level 1 API

    \details
    scal  scales each element of vector x with scalar alpha.

        x := alpha * x

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in x.
    @param[in]
    alpha     device pointer or host pointer for the scalar alpha.
    @param[inout]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.


    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdscal(rocblas_handle          handle,
                                             rocblas_int             n,
                                             const double*           alpha,
                                             rocblas_double_complex* x,
                                             rocblas_int             incx);

ROCBLAS_EXPORT rocblas_status rocblas_sscal_batched(rocblas_handle handle,
                                                    rocblas_int    n,
                                                    const float*   alpha,
                                                    float* const   x[],
                                                    rocblas_int    incx,
                                                    rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dscal_batched(rocblas_handle handle,
                                                    rocblas_int    n,
                                                    const double*  alpha,
                                                    double* const  x[],
                                                    rocblas_int    incx,
                                                    rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cscal_batched(rocblas_handle               handle,
                                                    rocblas_int                  n,
                                                    const rocblas_float_complex* alpha,
                                                    rocblas_float_complex* const x[],
                                                    rocblas_int                  incx,
                                                    rocblas_int                  batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_zscal_batched(rocblas_handle                handle,
                                                    rocblas_int                   n,
                                                    const rocblas_double_complex* alpha,
                                                    rocblas_double_complex* const x[],
                                                    rocblas_int                   incx,
                                                    rocblas_int                   batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csscal_batched(rocblas_handle               handle,
                                                     rocblas_int                  n,
                                                     const float*                 alpha,
                                                     rocblas_float_complex* const x[],
                                                     rocblas_int                  incx,
                                                     rocblas_int                  batch_count);

/*! \brief BLAS Level 1 API
     \details
    scal_batched  scales each element of vector x_i with scalar alpha, for i = 1, ... , batch_count.

         x_i := alpha * x_i

     where (x_i) is the i-th instance of the batch.
    @param[in]
    handle      [rocblas_handle]
                handle to the rocblas library context queue.
    @param[in]
    n           [rocblas_int]
                the number of elements in each x_i.
    @param[in]
    alpha       host pointer or device pointer for the scalar alpha.
    @param[inout]
    x           device array of device pointers storing each vector x_i.
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of each x_i.
    @param[in]
    batch_count [rocblas_int]
                specifies the number of batches in x.
     ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdscal_batched(rocblas_handle                handle,
                                                     rocblas_int                   n,
                                                     const double*                 alpha,
                                                     rocblas_double_complex* const x[],
                                                     rocblas_int                   incx,
                                                     rocblas_int                   batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sscal_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const float*   alpha,
                                                            float*         x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dscal_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const double*  alpha,
                                                            double*        x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cscal_strided_batched(rocblas_handle               handle,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            rocblas_float_complex*       x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_zscal_strided_batched(rocblas_handle                handle,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            rocblas_double_complex*       x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csscal_strided_batched(rocblas_handle         handle,
                                                             rocblas_int            n,
                                                             const float*           alpha,
                                                             rocblas_float_complex* x,
                                                             rocblas_int            incx,
                                                             rocblas_stride         stride_x,
                                                             rocblas_int            batch_count);

/*! \brief BLAS Level 1 API
     \details
    scal_strided_batched  scales each element of vector x_i with scalar alpha, for i = 1, ... , batch_count.

         x_i := alpha * x_i ,

     where (x_i) is the i-th instance of the batch.
     @param[in]
    handle      [rocblas_handle]
                handle to the rocblas library context queue.
    @param[in]
    n           [rocblas_int]
                the number of elements in each x_i.
    @param[in]
    alpha       host pointer or device pointer for the scalar alpha.
    @param[inout]
    x           device pointer to the first vector (x_1) in the batch.
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of x.
    @param[in]
    stride_x    [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stride_x, however the user should
                take care to ensure that stride_x is of appropriate size, for a typical
                case this means stride_x >= n * incx.
    @param[in]
    batch_count [rocblas_int]
                specifies the number of batches in x.
     ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdscal_strided_batched(rocblas_handle          handle,
                                                             rocblas_int             n,
                                                             const double*           alpha,
                                                             rocblas_double_complex* x,
                                                             rocblas_int             incx,
                                                             rocblas_stride          stride_x,
                                                             rocblas_int             batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_scopy(rocblas_handle handle,
                                            rocblas_int    n,
                                            const float*   x,
                                            rocblas_int    incx,
                                            float*         y,
                                            rocblas_int    incy);

ROCBLAS_EXPORT rocblas_status rocblas_dcopy(rocblas_handle handle,
                                            rocblas_int    n,
                                            const double*  x,
                                            rocblas_int    incx,
                                            double*        y,
                                            rocblas_int    incy);

ROCBLAS_EXPORT rocblas_status rocblas_ccopy(rocblas_handle               handle,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            rocblas_float_complex*       y,
                                            rocblas_int                  incy);

/*! \brief BLAS Level 1 API

    \details
    copy  copies each element x[i] into y[i], for  i = 1 , ... , n

        y := x,

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in x to be copied to y.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[out]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zcopy(rocblas_handle                handle,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            rocblas_double_complex*       y,
                                            rocblas_int                   incy);

ROCBLAS_EXPORT rocblas_status rocblas_scopy_batched(rocblas_handle     handle,
                                                    rocblas_int        n,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    float* const       y[],
                                                    rocblas_int        incy,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dcopy_batched(rocblas_handle      handle,
                                                    rocblas_int         n,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    double* const       y[],
                                                    rocblas_int         incy,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ccopy_batched(rocblas_handle                     handle,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    rocblas_float_complex* const       y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 1 API

    \details
    copy_batched copies each element x_i[j] into y_i[j], for  j = 1 , ... , n; i = 1 , ... , batch_count

        y_i := x_i,

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in each x_i to be copied to y_i.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i.
    @param[out]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zcopy_batched(rocblas_handle                      handle,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    rocblas_double_complex* const       y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_scopy_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            float*         y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dcopy_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            double*        y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ccopy_strided_batched(rocblas_handle               handle,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            rocblas_float_complex*       y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stridey,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 1 API

    \details
    copy_strided_batched copies each element x_i[j] into y_i[j], for  j = 1 , ... , n; i = 1 , ... , batch_count

        y_i := x_i,

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in each x_i to be copied to y_i.
    @param[in]
    x         device pointer to the first vector (x_1) in the batch.
    @param[in]
    incx      [rocblas_int]
              specifies the increments for the elements of vectors x_i.
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stride_x, however the user should
                take care to ensure that stride_x is of appropriate size, for a typical
                case this means stride_x >= n * incx.
    @param[out]
    y         device pointer to the first vector (y_1) in the batch.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of vectors y_i.
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1).
                There are no restrictions placed on stride_y, however the user should
                take care to ensure that stride_y is of appropriate size, for a typical
                case this means stride_y >= n * incy. stridey should be non zero.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zcopy_strided_batched(rocblas_handle                handle,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            rocblas_double_complex*       y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stridey,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sdot(rocblas_handle handle,
                                           rocblas_int    n,
                                           const float*   x,
                                           rocblas_int    incx,
                                           const float*   y,
                                           rocblas_int    incy,
                                           float*         result);

ROCBLAS_EXPORT rocblas_status rocblas_ddot(rocblas_handle handle,
                                           rocblas_int    n,
                                           const double*  x,
                                           rocblas_int    incx,
                                           const double*  y,
                                           rocblas_int    incy,
                                           double*        result);

ROCBLAS_EXPORT rocblas_status rocblas_hdot(rocblas_handle      handle,
                                           rocblas_int         n,
                                           const rocblas_half* x,
                                           rocblas_int         incx,
                                           const rocblas_half* y,
                                           rocblas_int         incy,
                                           rocblas_half*       result);

ROCBLAS_EXPORT rocblas_status rocblas_bfdot(rocblas_handle          handle,
                                            rocblas_int             n,
                                            const rocblas_bfloat16* x,
                                            rocblas_int             incx,
                                            const rocblas_bfloat16* y,
                                            rocblas_int             incy,
                                            rocblas_bfloat16*       result);

ROCBLAS_EXPORT rocblas_status rocblas_cdotu(rocblas_handle               handle,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* y,
                                            rocblas_int                  incy,
                                            rocblas_float_complex*       result);

ROCBLAS_EXPORT rocblas_status rocblas_zdotu(rocblas_handle                handle,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* y,
                                            rocblas_int                   incy,
                                            rocblas_double_complex*       result);

ROCBLAS_EXPORT rocblas_status rocblas_cdotc(rocblas_handle               handle,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* y,
                                            rocblas_int                  incy,
                                            rocblas_float_complex*       result);

/*! \brief BLAS Level 1 API

    \details
    dot(u)  performs the dot product of vectors x and y

        result = x * y;

    dotc  performs the dot product of the conjugate of complex vector x and complex vector y

        result = conjugate (x) * y;

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in x and y.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    result
              device pointer or host pointer to store the dot product.
              return is 0.0 if n <= 0.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdotc(rocblas_handle                handle,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* y,
                                            rocblas_int                   incy,
                                            rocblas_double_complex*       result);

ROCBLAS_EXPORT rocblas_status rocblas_sdot_batched(rocblas_handle     handle,
                                                   rocblas_int        n,
                                                   const float* const x[],
                                                   rocblas_int        incx,
                                                   const float* const y[],
                                                   rocblas_int        incy,
                                                   rocblas_int        batch_count,
                                                   float*             result);

ROCBLAS_EXPORT rocblas_status rocblas_ddot_batched(rocblas_handle      handle,
                                                   rocblas_int         n,
                                                   const double* const x[],
                                                   rocblas_int         incx,
                                                   const double* const y[],
                                                   rocblas_int         incy,
                                                   rocblas_int         batch_count,
                                                   double*             result);

ROCBLAS_EXPORT rocblas_status rocblas_hdot_batched(rocblas_handle            handle,
                                                   rocblas_int               n,
                                                   const rocblas_half* const x[],
                                                   rocblas_int               incx,
                                                   const rocblas_half* const y[],
                                                   rocblas_int               incy,
                                                   rocblas_int               batch_count,
                                                   rocblas_half*             result);

ROCBLAS_EXPORT rocblas_status rocblas_bfdot_batched(rocblas_handle                handle,
                                                    rocblas_int                   n,
                                                    const rocblas_bfloat16* const x[],
                                                    rocblas_int                   incx,
                                                    const rocblas_bfloat16* const y[],
                                                    rocblas_int                   incy,
                                                    rocblas_int                   batch_count,
                                                    rocblas_bfloat16*             result);

ROCBLAS_EXPORT rocblas_status rocblas_cdotu_batched(rocblas_handle                     handle,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex* const y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count,
                                                    rocblas_float_complex*             result);

ROCBLAS_EXPORT rocblas_status rocblas_zdotu_batched(rocblas_handle                      handle,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex* const y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int                         batch_count,
                                                    rocblas_double_complex*             result);

ROCBLAS_EXPORT rocblas_status rocblas_cdotc_batched(rocblas_handle                     handle,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex* const y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count,
                                                    rocblas_float_complex*             result);

/*! \brief BLAS Level 1 API

    \details
    dot_batched(u) performs a batch of dot products of vectors x and y

        result_i = x_i * y_i;

    dotc_batched  performs a batch of dot products of the conjugate of complex vector x and complex vector y

        result_i = conjugate (x_i) * y_i;

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors, for i = 1, ..., batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in each x_i and y_i.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch
    @param[inout]
    result
              device array or host array of batch_count size to store the dot products of each batch.
              return 0.0 for each element if n <= 0.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdotc_batched(rocblas_handle                      handle,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex* const y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int                         batch_count,
                                                    rocblas_double_complex*             result);

ROCBLAS_EXPORT rocblas_status rocblas_sdot_strided_batched(rocblas_handle handle,
                                                           rocblas_int    n,
                                                           const float*   x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stridex,
                                                           const float*   y,
                                                           rocblas_int    incy,
                                                           rocblas_stride stridey,
                                                           rocblas_int    batch_count,
                                                           float*         result);

ROCBLAS_EXPORT rocblas_status rocblas_ddot_strided_batched(rocblas_handle handle,
                                                           rocblas_int    n,
                                                           const double*  x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stridex,
                                                           const double*  y,
                                                           rocblas_int    incy,
                                                           rocblas_stride stridey,
                                                           rocblas_int    batch_count,
                                                           double*        result);

ROCBLAS_EXPORT rocblas_status rocblas_hdot_strided_batched(rocblas_handle      handle,
                                                           rocblas_int         n,
                                                           const rocblas_half* x,
                                                           rocblas_int         incx,
                                                           rocblas_stride      stridex,
                                                           const rocblas_half* y,
                                                           rocblas_int         incy,
                                                           rocblas_stride      stridey,
                                                           rocblas_int         batch_count,
                                                           rocblas_half*       result);

ROCBLAS_EXPORT rocblas_status rocblas_bfdot_strided_batched(rocblas_handle          handle,
                                                            rocblas_int             n,
                                                            const rocblas_bfloat16* x,
                                                            rocblas_int             incx,
                                                            rocblas_stride          stridex,
                                                            const rocblas_bfloat16* y,
                                                            rocblas_int             incy,
                                                            rocblas_stride          stridey,
                                                            rocblas_int             batch_count,
                                                            rocblas_bfloat16*       result);

ROCBLAS_EXPORT rocblas_status rocblas_cdotu_strided_batched(rocblas_handle               handle,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            const rocblas_float_complex* y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stridey,
                                                            rocblas_int            batch_count,
                                                            rocblas_float_complex* result);

ROCBLAS_EXPORT rocblas_status rocblas_zdotu_strided_batched(rocblas_handle                handle,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            const rocblas_double_complex* y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stridey,
                                                            rocblas_int             batch_count,
                                                            rocblas_double_complex* result);

ROCBLAS_EXPORT rocblas_status rocblas_cdotc_strided_batched(rocblas_handle               handle,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            const rocblas_float_complex* y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stridey,
                                                            rocblas_int            batch_count,
                                                            rocblas_float_complex* result);

/*! \brief BLAS Level 1 API

    \details
    dot_strided_batched(u)  performs a batch of dot products of vectors x and y

        result_i = x_i * y_i;

    dotc_strided_batched  performs a batch of dot products of the conjugate of complex vector x and complex vector y

        result_i = conjugate (x_i) * y_i;

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors, for i = 1, ..., batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in each x_i and y_i.
    @param[in]
    x         device pointer to the first vector (x_1) in the batch.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1)
    @param[in]
    y         device pointer to the first vector (y_1) in the batch.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch
    @param[inout]
    result
              device array or host array of batch_count size to store the dot products of each batch.
              return 0.0 for each element if n <= 0.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdotc_strided_batched(rocblas_handle                handle,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            const rocblas_double_complex* y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stridey,
                                                            rocblas_int             batch_count,
                                                            rocblas_double_complex* result);

ROCBLAS_EXPORT rocblas_status rocblas_sswap(
    rocblas_handle handle, rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status rocblas_dswap(
    rocblas_handle handle, rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy);

ROCBLAS_EXPORT rocblas_status rocblas_cswap(rocblas_handle         handle,
                                            rocblas_int            n,
                                            rocblas_float_complex* x,
                                            rocblas_int            incx,
                                            rocblas_float_complex* y,
                                            rocblas_int            incy);

/*! \brief BLAS Level 1 API

    \details
    swap  interchanges vectors x and y.

        y := x; x := y

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in x and y.
    @param[inout]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zswap(rocblas_handle          handle,
                                            rocblas_int             n,
                                            rocblas_double_complex* x,
                                            rocblas_int             incx,
                                            rocblas_double_complex* y,
                                            rocblas_int             incy);

ROCBLAS_EXPORT rocblas_status rocblas_sswap_batched(rocblas_handle handle,
                                                    rocblas_int    n,
                                                    float*         x[],
                                                    rocblas_int    incx,
                                                    float*         y[],
                                                    rocblas_int    incy,
                                                    rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dswap_batched(rocblas_handle handle,
                                                    rocblas_int    n,
                                                    double*        x[],
                                                    rocblas_int    incx,
                                                    double*        y[],
                                                    rocblas_int    incy,
                                                    rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cswap_batched(rocblas_handle         handle,
                                                    rocblas_int            n,
                                                    rocblas_float_complex* x[],
                                                    rocblas_int            incx,
                                                    rocblas_float_complex* y[],
                                                    rocblas_int            incy,
                                                    rocblas_int            batch_count);

/*! \brief BLAS Level 1 API

    \details
    swap_batched interchanges vectors x_i and y_i, for i = 1 , ... , batch_count

        y_i := x_i; x_i := y_i

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in each x_i and y_i.
    @param[inout]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zswap_batched(rocblas_handle          handle,
                                                    rocblas_int             n,
                                                    rocblas_double_complex* x[],
                                                    rocblas_int             incx,
                                                    rocblas_double_complex* y[],
                                                    rocblas_int             incy,
                                                    rocblas_int             batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sswap_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            float*         x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            float*         y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dswap_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            double*        x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            double*        y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cswap_strided_batched(rocblas_handle         handle,
                                                            rocblas_int            n,
                                                            rocblas_float_complex* x,
                                                            rocblas_int            incx,
                                                            rocblas_stride         stridex,
                                                            rocblas_float_complex* y,
                                                            rocblas_int            incy,
                                                            rocblas_stride         stridey,
                                                            rocblas_int            batch_count);

/*! \brief BLAS Level 1 API

    \details
    swap_strided_batched interchanges vectors x_i and y_i, for i = 1 , ... , batch_count

        y_i := x_i; x_i := y_i

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in each x_i and y_i.
    @param[inout]
    x         device pointer to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    stridex   [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
              There are no restrictions placed on stride_x, however the user should
              take care to ensure that stride_x is of appropriate size, for a typical
              case this means stride_x >= n * incx.
    @param[inout]
    y         device pointer to the first vector y_1.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    stridey   [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1).
              There are no restrictions placed on stride_x, however the user should
              take care to ensure that stride_y is of appropriate size, for a typical
              case this means stride_y >= n * incy. stridey should be non zero.
     @param[in]
     batch_count [rocblas_int]
                 number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zswap_strided_batched(rocblas_handle          handle,
                                                            rocblas_int             n,
                                                            rocblas_double_complex* x,
                                                            rocblas_int             incx,
                                                            rocblas_stride          stridex,
                                                            rocblas_double_complex* y,
                                                            rocblas_int             incy,
                                                            rocblas_stride          stridey,
                                                            rocblas_int             batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_saxpy(rocblas_handle handle,
                                            rocblas_int    n,
                                            const float*   alpha,
                                            const float*   x,
                                            rocblas_int    incx,
                                            float*         y,
                                            rocblas_int    incy);

ROCBLAS_EXPORT rocblas_status rocblas_daxpy(rocblas_handle handle,
                                            rocblas_int    n,
                                            const double*  alpha,
                                            const double*  x,
                                            rocblas_int    incx,
                                            double*        y,
                                            rocblas_int    incy);

ROCBLAS_EXPORT rocblas_status rocblas_haxpy(rocblas_handle      handle,
                                            rocblas_int         n,
                                            const rocblas_half* alpha,
                                            const rocblas_half* x,
                                            rocblas_int         incx,
                                            rocblas_half*       y,
                                            rocblas_int         incy);

ROCBLAS_EXPORT rocblas_status rocblas_caxpy(rocblas_handle               handle,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            rocblas_float_complex*       y,
                                            rocblas_int                  incy);

/*! \brief BLAS Level 1 API

    \details
    axpy   computes constant alpha multiplied by vector x, plus vector y

        y := alpha * x + y

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in x and y.
    @param[in]
    alpha     device pointer or host pointer to specify the scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[out]
    y         device pointer storing vector y.
    @param[inout]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zaxpy(rocblas_handle                handle,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            rocblas_double_complex*       y,
                                            rocblas_int                   incy);

ROCBLAS_EXPORT rocblas_status rocblas_haxpy_batched(rocblas_handle            handle,
                                                    rocblas_int               n,
                                                    const rocblas_half*       alpha,
                                                    const rocblas_half* const x[],
                                                    rocblas_int               incx,
                                                    rocblas_half* const       y[],
                                                    rocblas_int               incy,
                                                    rocblas_int               batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_saxpy_batched(rocblas_handle     handle,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    float* const       y[],
                                                    rocblas_int        incy,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_daxpy_batched(rocblas_handle      handle,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    double* const       y[],
                                                    rocblas_int         incy,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_caxpy_batched(rocblas_handle                     handle,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    rocblas_float_complex* const       y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 1 API

    \details
    axpy_batched   compute y := alpha * x + y over a set of batched vectors.

    @param[in]
    handle    rocblas_handle
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of x.
    @param[out]
    y         pointer storing vector y on the GPU.
    @param[inout]
    incy      rocblas_int
              specifies the increment for the elements of y.

    @param[in]
    batch_count rocblas_int
              number of instances in the batch

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zaxpy_batched(rocblas_handle                      handle,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    rocblas_double_complex* const       y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_haxpy_strided_batched(rocblas_handle      handle,
                                                            rocblas_int         n,
                                                            const rocblas_half* alpha,
                                                            const rocblas_half* x,
                                                            rocblas_int         incx,
                                                            rocblas_stride      stridex,
                                                            rocblas_half*       y,
                                                            rocblas_int         incy,
                                                            rocblas_stride      stridey,
                                                            rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_saxpy_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const float*   alpha,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            float*         y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_daxpy_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const double*  alpha,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            double*        y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_caxpy_strided_batched(rocblas_handle               handle,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            rocblas_float_complex*       y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stridey,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 1 API

    \details
    axpy_strided_batched   compute y := alpha * x + y over a set of strided batched vectors.

    @param[in]
    handle    rocblas_handle
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of x.
    @param[in]
    stridex   rocblas_stride
              specifies the increment between vectors of x.
    @param[out]
    y         pointer storing vector y on the GPU.
    @param[inout]
    incy      rocblas_int
              specifies the increment for the elements of y.
    @param[in]
    stridey   rocblas_stride
              specifies the increment between vectors of y.

    @param[in]
    batch_count rocblas_int
              number of instances in the batch

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zaxpy_strided_batched(rocblas_handle                handle,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            rocblas_double_complex*       y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stridey,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sasum(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result);

ROCBLAS_EXPORT rocblas_status rocblas_dasum(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result);

ROCBLAS_EXPORT rocblas_status rocblas_scasum(rocblas_handle               handle,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             float*                       result);

/*! \brief BLAS Level 1 API

    \details
    asum computes the sum of the magnitudes of elements of a real vector x,
         or the sum of magnitudes of the real and imaginary parts of elements if x is a complex
   vector

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in x and y.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x. incx must be > 0.
    @param[inout]
    result
              device pointer or host pointer to store the asum product.
              return is 0.0 if n <= 0.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_dzasum(rocblas_handle                handle,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             double*                       result);

ROCBLAS_EXPORT rocblas_status rocblas_sasum_batched(rocblas_handle     handle,
                                                    rocblas_int        n,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    rocblas_int        batch_count,
                                                    float*             results);

ROCBLAS_EXPORT rocblas_status rocblas_dasum_batched(rocblas_handle      handle,
                                                    rocblas_int         n,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    rocblas_int         batch_count,
                                                    double*             results);

ROCBLAS_EXPORT rocblas_status rocblas_scasum_batched(rocblas_handle                     handle,
                                                     rocblas_int                        n,
                                                     const rocblas_float_complex* const x[],
                                                     rocblas_int                        incx,
                                                     rocblas_int                        batch_count,
                                                     float*                             results);

/*! \brief BLAS Level 1 API

    \details
    asum_batched computes the sum of the magnitudes of the elements in a batch of real vectors x_i,
        or the sum of magnitudes of the real and imaginary parts of elements if x_i is a complex
        vector, for i = 1, ..., batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              number of elements in each vector x_i
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i. incx must be > 0.
    @param[out]
    results
              device array or host array of batch_count size for results.
              return is 0.0 if n, incx<=0.
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_dzasum_batched(rocblas_handle                      handle,
                                                     rocblas_int                         n,
                                                     const rocblas_double_complex* const x[],
                                                     rocblas_int                         incx,
                                                     rocblas_int batch_count,
                                                     double*     results);

ROCBLAS_EXPORT rocblas_status rocblas_sasum_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            rocblas_int    batch_count,
                                                            float*         results);

ROCBLAS_EXPORT rocblas_status rocblas_dasum_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            rocblas_int    batch_count,
                                                            double*        results);

ROCBLAS_EXPORT rocblas_status rocblas_scasum_strided_batched(rocblas_handle               handle,
                                                             rocblas_int                  n,
                                                             const rocblas_float_complex* x,
                                                             rocblas_int                  incx,
                                                             rocblas_stride               stridex,
                                                             rocblas_int batch_count,
                                                             float*      results);

/*! \brief BLAS Level 1 API

    \details
    asum_strided_batched computes the sum of the magnitudes of elements of a real vectors x_i,
        or the sum of magnitudes of the real and imaginary parts of elements if x_i is a complex
        vector, for i = 1, ..., batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              number of elements in each vector x_i
    @param[in]
    x         device pointer to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i. incx must be > 0.
    @param[in]
    stridex   [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
              There are no restrictions placed on stride_x, however the user should
              take care to ensure that stride_x is of appropriate size, for a typical
              case this means stride_x >= n * incx.
    @param[out]
    results
              device pointer or host pointer to array for storing contiguous batch_count results.
              return is 0.0 if n, incx<=0.
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_dzasum_strided_batched(rocblas_handle                handle,
                                                             rocblas_int                   n,
                                                             const rocblas_double_complex* x,
                                                             rocblas_int                   incx,
                                                             rocblas_stride                stridex,
                                                             rocblas_int batch_count,
                                                             double*     results);

ROCBLAS_EXPORT rocblas_status rocblas_snrm2(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result);

ROCBLAS_EXPORT rocblas_status rocblas_dnrm2(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result);

ROCBLAS_EXPORT rocblas_status rocblas_scnrm2(rocblas_handle               handle,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             float*                       result);

/*! \brief BLAS Level 1 API

    \details
    nrm2 computes the euclidean norm of a real or complex vector

              result := sqrt( x'*x ) for real vectors
              result := sqrt( x**H*x ) for complex vectors

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in x.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    result
              device pointer or host pointer to store the nrm2 product.
              return is 0.0 if n, incx<=0.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_dznrm2(rocblas_handle                handle,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             double*                       result);

ROCBLAS_EXPORT rocblas_status rocblas_snrm2_batched(rocblas_handle     handle,
                                                    rocblas_int        n,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    rocblas_int        batch_count,
                                                    float*             results);

ROCBLAS_EXPORT rocblas_status rocblas_dnrm2_batched(rocblas_handle      handle,
                                                    rocblas_int         n,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    rocblas_int         batch_count,
                                                    double*             results);

ROCBLAS_EXPORT rocblas_status rocblas_scnrm2_batched(rocblas_handle                     handle,
                                                     rocblas_int                        n,
                                                     const rocblas_float_complex* const x[],
                                                     rocblas_int                        incx,
                                                     rocblas_int                        batch_count,
                                                     float*                             results);

/*! \brief BLAS Level 1 API

    \details
    nrm2_batched computes the euclidean norm over a batch of real or complex vectors

              result := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batch_count
              result := sqrt( x_i**H*x_i ) for complex vectors x, for i = 1, ..., batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              number of elements in each x_i.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i. incx must be > 0.
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch
    @param[out]
    results
              device pointer or host pointer to array of batch_count size for nrm2 results.
              return is 0.0 for each element if n <= 0, incx<=0.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_dznrm2_batched(rocblas_handle                      handle,
                                                     rocblas_int                         n,
                                                     const rocblas_double_complex* const x[],
                                                     rocblas_int                         incx,
                                                     rocblas_int batch_count,
                                                     double*     results);

ROCBLAS_EXPORT rocblas_status rocblas_snrm2_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            rocblas_int    batch_count,
                                                            float*         results);

ROCBLAS_EXPORT rocblas_status rocblas_dnrm2_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            rocblas_int    batch_count,
                                                            double*        results);

ROCBLAS_EXPORT rocblas_status rocblas_scnrm2_strided_batched(rocblas_handle               handle,
                                                             rocblas_int                  n,
                                                             const rocblas_float_complex* x,
                                                             rocblas_int                  incx,
                                                             rocblas_stride               stridex,
                                                             rocblas_int batch_count,
                                                             float*      results);

/*! \brief BLAS Level 1 API

    \details
    nrm2_strided_batched computes the euclidean norm over a batch of real or complex vectors

              := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batch_count
              := sqrt( x_i**H*x_i ) for complex vectors, for i = 1, ..., batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              number of elements in each x_i.
    @param[in]
    x         device pointer to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i. incx must be > 0.
    @param[in]
    stridex   [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
              There are no restrictions placed on stride_x, however the user should
              take care to ensure that stride_x is of appropriate size, for a typical
              case this means stride_x >= n * incx.
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch
    @param[out]
    results
              device pointer or host pointer to array for storing contiguous batch_count results.
              return is 0.0 for each element if n <= 0, incx<=0.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_dznrm2_strided_batched(rocblas_handle                handle,
                                                             rocblas_int                   n,
                                                             const rocblas_double_complex* x,
                                                             rocblas_int                   incx,
                                                             rocblas_stride                stridex,
                                                             rocblas_int batch_count,
                                                             double*     results);

ROCBLAS_EXPORT rocblas_status rocblas_isamax(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_idamax(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_icamax(rocblas_handle               handle,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             rocblas_int*                 result);
/*! \brief BLAS Level 1 API

    \details
    amax finds the first index of the element of maximum magnitude of a vector x.
   vector

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in x.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    result
              device pointer or host pointer to store the amax index.
              return is 0.0 if n, incx<=0.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_izamax(rocblas_handle                handle,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             rocblas_int*                  result);

ROCBLAS_EXPORT rocblas_status rocblas_isamax_batched(rocblas_handle     handle,
                                                     rocblas_int        n,
                                                     const float* const x[],
                                                     rocblas_int        incx,
                                                     rocblas_int        batch_count,
                                                     rocblas_int*       result);
ROCBLAS_EXPORT rocblas_status rocblas_idamax_batched(rocblas_handle      handle,
                                                     rocblas_int         n,
                                                     const double* const x[],
                                                     rocblas_int         incx,
                                                     rocblas_int         batch_count,
                                                     rocblas_int*        result);
ROCBLAS_EXPORT rocblas_status rocblas_icamax_batched(rocblas_handle                     handle,
                                                     rocblas_int                        n,
                                                     const rocblas_float_complex* const x[],
                                                     rocblas_int                        incx,
                                                     rocblas_int                        batch_count,
                                                     rocblas_int*                       result);
/*! \brief BLAS Level 1 API

    \details
     amax_batched finds the first index of the element of maximum magnitude of each vector x_i in a batch, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              number of elements in each vector x_i
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i. incx must be > 0.
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch, must be > 0.
    @param[out]
    result
              device or host array of pointers of batch_count size for results.
              return is 0 if n, incx<=0.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_izamax_batched(rocblas_handle                      handle,
                                                     rocblas_int                         n,
                                                     const rocblas_double_complex* const x[],
                                                     rocblas_int                         incx,
                                                     rocblas_int  batch_count,
                                                     rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_isamax_strided_batched(rocblas_handle handle,
                                                             rocblas_int    n,
                                                             const float*   x,
                                                             rocblas_int    incx,
                                                             rocblas_stride stridex,
                                                             rocblas_int    batch_count,
                                                             rocblas_int*   result);

ROCBLAS_EXPORT rocblas_status rocblas_idamax_strided_batched(rocblas_handle handle,
                                                             rocblas_int    n,
                                                             const double*  x,
                                                             rocblas_int    incx,
                                                             rocblas_stride stridex,
                                                             rocblas_int    batch_count,
                                                             rocblas_int*   result);

ROCBLAS_EXPORT rocblas_status rocblas_icamax_strided_batched(rocblas_handle               handle,
                                                             rocblas_int                  n,
                                                             const rocblas_float_complex* x,
                                                             rocblas_int                  incx,
                                                             rocblas_stride               stridex,
                                                             rocblas_int  batch_count,
                                                             rocblas_int* result);
/*! \brief BLAS Level 1 API

    \details
     amax_strided_batched finds the first index of the element of maximum magnitude of each vector x_i in a batch, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              number of elements in each vector x_i
    @param[in]
    x         device pointer to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i. incx must be > 0.
    @param[in]
    stridex   [rocblas_stride]
              specifies the pointer increment between one x_i and the next x_(i + 1).
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch
    @param[out]
    result
              device or host pointer for storing contiguous batch_count results.
              return is 0 if n <= 0, incx<=0.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_izamax_strided_batched(rocblas_handle                handle,
                                                             rocblas_int                   n,
                                                             const rocblas_double_complex* x,
                                                             rocblas_int                   incx,
                                                             rocblas_stride                stridex,
                                                             rocblas_int  batch_count,
                                                             rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_isamin(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_idamin(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_icamin(rocblas_handle               handle,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* x,
                                             rocblas_int                  incx,
                                             rocblas_int*                 result);
/*! \brief BLAS Level 1 API

    \details
    amin finds the first index of the element of minimum magnitude of a vector x.

   vector

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              the number of elements in x.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    result
              device pointer or host pointer to store the amin index.
              return is 0.0 if n, incx<=0.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_izamin(rocblas_handle                handle,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* x,
                                             rocblas_int                   incx,
                                             rocblas_int*                  result);

ROCBLAS_EXPORT rocblas_status rocblas_isamin_batched(rocblas_handle     handle,
                                                     rocblas_int        n,
                                                     const float* const x[],
                                                     rocblas_int        incx,
                                                     rocblas_int        batch_count,
                                                     rocblas_int*       result);

ROCBLAS_EXPORT rocblas_status rocblas_idamin_batched(rocblas_handle      handle,
                                                     rocblas_int         n,
                                                     const double* const x[],
                                                     rocblas_int         incx,
                                                     rocblas_int         batch_count,
                                                     rocblas_int*        result);

ROCBLAS_EXPORT rocblas_status rocblas_icamin_batched(rocblas_handle                     handle,
                                                     rocblas_int                        n,
                                                     const rocblas_float_complex* const x[],
                                                     rocblas_int                        incx,
                                                     rocblas_int                        batch_count,
                                                     rocblas_int*                       result);

/*! \brief BLAS Level 1 API

    \details
    amin_batched finds the first index of the element of minimum magnitude of each vector x_i in a batch, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              number of elements in each vector x_i
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i. incx must be > 0.
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch, must be > 0.
    @param[out]
    result
              device or host pointers to array of batch_count size for results.
              return is 0 if n, incx<=0.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_izamin_batched(rocblas_handle                      handle,
                                                     rocblas_int                         n,
                                                     const rocblas_double_complex* const x[],
                                                     rocblas_int                         incx,
                                                     rocblas_int  batch_count,
                                                     rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_isamin_strided_batched(rocblas_handle handle,
                                                             rocblas_int    n,
                                                             const float*   x,
                                                             rocblas_int    incx,
                                                             rocblas_stride stridex,
                                                             rocblas_int    batch_count,
                                                             rocblas_int*   result);

ROCBLAS_EXPORT rocblas_status rocblas_idamin_strided_batched(rocblas_handle handle,
                                                             rocblas_int    n,
                                                             const double*  x,
                                                             rocblas_int    incx,
                                                             rocblas_stride stridex,
                                                             rocblas_int    batch_count,
                                                             rocblas_int*   result);

ROCBLAS_EXPORT rocblas_status rocblas_icamin_strided_batched(rocblas_handle               handle,
                                                             rocblas_int                  n,
                                                             const rocblas_float_complex* x,
                                                             rocblas_int                  incx,
                                                             rocblas_stride               stridex,
                                                             rocblas_int  batch_count,
                                                             rocblas_int* result);

/*! \brief BLAS Level 1 API

    \details
     amin_strided_batched finds the first index of the element of minimum magnitude of each vector x_i in a batch, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    n         [rocblas_int]
              number of elements in each vector x_i
    @param[in]
    x         device pointer to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i. incx must be > 0.
    @param[in]
    stridex   [rocblas_stride]
              specifies the pointer increment between one x_i and the next x_(i + 1)
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch
    @param[out]
    result
              device or host pointer to array for storing contiguous batch_count results.
              return is 0 if n <= 0, incx<=0.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_izamin_strided_batched(rocblas_handle                handle,
                                                             rocblas_int                   n,
                                                             const rocblas_double_complex* x,
                                                             rocblas_int                   incx,
                                                             rocblas_stride                stridex,
                                                             rocblas_int  batch_count,
                                                             rocblas_int* result);

ROCBLAS_EXPORT rocblas_status rocblas_srot(rocblas_handle handle,
                                           rocblas_int    n,
                                           float*         x,
                                           rocblas_int    incx,
                                           float*         y,
                                           rocblas_int    incy,
                                           const float*   c,
                                           const float*   s);

ROCBLAS_EXPORT rocblas_status rocblas_drot(rocblas_handle handle,
                                           rocblas_int    n,
                                           double*        x,
                                           rocblas_int    incx,
                                           double*        y,
                                           rocblas_int    incy,
                                           const double*  c,
                                           const double*  s);

ROCBLAS_EXPORT rocblas_status rocblas_crot(rocblas_handle               handle,
                                           rocblas_int                  n,
                                           rocblas_float_complex*       x,
                                           rocblas_int                  incx,
                                           rocblas_float_complex*       y,
                                           rocblas_int                  incy,
                                           const float*                 c,
                                           const rocblas_float_complex* s);

ROCBLAS_EXPORT rocblas_status rocblas_csrot(rocblas_handle         handle,
                                            rocblas_int            n,
                                            rocblas_float_complex* x,
                                            rocblas_int            incx,
                                            rocblas_float_complex* y,
                                            rocblas_int            incy,
                                            const float*           c,
                                            const float*           s);

ROCBLAS_EXPORT rocblas_status rocblas_zrot(rocblas_handle                handle,
                                           rocblas_int                   n,
                                           rocblas_double_complex*       x,
                                           rocblas_int                   incx,
                                           rocblas_double_complex*       y,
                                           rocblas_int                   incy,
                                           const double*                 c,
                                           const rocblas_double_complex* s);

/*! \brief BLAS Level 1 API

    \details
    rot applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
        Scalars c and s may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[in]
    n       [rocblas_int]
            number of elements in the x and y vectors.
    @param[inout]
    x       device pointer storing vector x.
    @param[in]
    incx    [rocblas_int]
            specifies the increment between elements of x.
    @param[inout]
    y       device pointer storing vector y.
    @param[in]
    incy    [rocblas_int]
            specifies the increment between elements of y.
    @param[in]
    c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
    @param[in]
    s       device pointer or host pointer storing scalar sine component of the rotation matrix.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdrot(rocblas_handle          handle,
                                            rocblas_int             n,
                                            rocblas_double_complex* x,
                                            rocblas_int             incx,
                                            rocblas_double_complex* y,
                                            rocblas_int             incy,
                                            const double*           c,
                                            const double*           s);

ROCBLAS_EXPORT rocblas_status rocblas_srot_batched(rocblas_handle handle,
                                                   rocblas_int    n,
                                                   float* const   x[],
                                                   rocblas_int    incx,
                                                   float* const   y[],
                                                   rocblas_int    incy,
                                                   const float*   c,
                                                   const float*   s,
                                                   rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_drot_batched(rocblas_handle handle,
                                                   rocblas_int    n,
                                                   double* const  x[],
                                                   rocblas_int    incx,
                                                   double* const  y[],
                                                   rocblas_int    incy,
                                                   const double*  c,
                                                   const double*  s,
                                                   rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_crot_batched(rocblas_handle               handle,
                                                   rocblas_int                  n,
                                                   rocblas_float_complex* const x[],
                                                   rocblas_int                  incx,
                                                   rocblas_float_complex* const y[],
                                                   rocblas_int                  incy,
                                                   const float*                 c,
                                                   const rocblas_float_complex* s,
                                                   rocblas_int                  batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csrot_batched(rocblas_handle               handle,
                                                    rocblas_int                  n,
                                                    rocblas_float_complex* const x[],
                                                    rocblas_int                  incx,
                                                    rocblas_float_complex* const y[],
                                                    rocblas_int                  incy,
                                                    const float*                 c,
                                                    const float*                 s,
                                                    rocblas_int                  batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_zrot_batched(rocblas_handle                handle,
                                                   rocblas_int                   n,
                                                   rocblas_double_complex* const x[],
                                                   rocblas_int                   incx,
                                                   rocblas_double_complex* const y[],
                                                   rocblas_int                   incy,
                                                   const double*                 c,
                                                   const rocblas_double_complex* s,
                                                   rocblas_int                   batch_count);
/*! \brief BLAS Level 1 API

    \details
    rot_batched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to batched vectors x_i and y_i, for i = 1, ..., batch_count.
        Scalars c and s may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[in]
    n       [rocblas_int]
            number of elements in each x_i and y_i vectors.
    @param[inout]
    x       device array of deivce pointers storing each vector x_i.
    @param[in]
    incx    [rocblas_int]
            specifies the increment between elements of each x_i.
    @param[inout]
    y       device array of device pointers storing each vector y_i.
    @param[in]
    incy    [rocblas_int]
            specifies the increment between elements of each y_i.
    @param[in]
    c       device pointer or host pointer to scalar cosine component of the rotation matrix.
    @param[in]
    s       device pointer or host pointer to scalar sine component of the rotation matrix.
    @param[in]
    batch_count [rocblas_int]
                the number of x and y arrays, i.e. the number of batches.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdrot_batched(rocblas_handle                handle,
                                                    rocblas_int                   n,
                                                    rocblas_double_complex* const x[],
                                                    rocblas_int                   incx,
                                                    rocblas_double_complex* const y[],
                                                    rocblas_int                   incy,
                                                    const double*                 c,
                                                    const double*                 s,
                                                    rocblas_int                   batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_srot_strided_batched(rocblas_handle handle,
                                                           rocblas_int    n,
                                                           float*         x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stride_x,
                                                           float*         y,
                                                           rocblas_int    incy,
                                                           rocblas_stride stride_y,
                                                           const float*   c,
                                                           const float*   s,
                                                           rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_drot_strided_batched(rocblas_handle handle,
                                                           rocblas_int    n,
                                                           double*        x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stride_x,
                                                           double*        y,
                                                           rocblas_int    incy,
                                                           rocblas_stride stride_y,
                                                           const double*  c,
                                                           const double*  s,
                                                           rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_crot_strided_batched(rocblas_handle               handle,
                                                           rocblas_int                  n,
                                                           rocblas_float_complex*       x,
                                                           rocblas_int                  incx,
                                                           rocblas_stride               stride_x,
                                                           rocblas_float_complex*       y,
                                                           rocblas_int                  incy,
                                                           rocblas_stride               stride_y,
                                                           const float*                 c,
                                                           const rocblas_float_complex* s,
                                                           rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csrot_strided_batched(rocblas_handle         handle,
                                                            rocblas_int            n,
                                                            rocblas_float_complex* x,
                                                            rocblas_int            incx,
                                                            rocblas_stride         stride_x,
                                                            rocblas_float_complex* y,
                                                            rocblas_int            incy,
                                                            rocblas_stride         stride_y,
                                                            const float*           c,
                                                            const float*           s,
                                                            rocblas_int            batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_zrot_strided_batched(rocblas_handle                handle,
                                                           rocblas_int                   n,
                                                           rocblas_double_complex*       x,
                                                           rocblas_int                   incx,
                                                           rocblas_stride                stride_x,
                                                           rocblas_double_complex*       y,
                                                           rocblas_int                   incy,
                                                           rocblas_stride                stride_y,
                                                           const double*                 c,
                                                           const rocblas_double_complex* s,
                                                           rocblas_int batch_count);
/*! \brief BLAS Level 1 API

    \details
    rot_strided_batched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to strided batched vectors x_i and y_i, for i = 1, ..., batch_count.
        Scalars c and s may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[in]
    n       [rocblas_int]
            number of elements in each x_i and y_i vectors.
    @param[inout]
    x       device pointer to the first vector x_1.
    @param[in]
    incx    [rocblas_int]
            specifies the increment between elements of each x_i.
    @param[in]
    stride_x [rocblas_stride]
             specifies the increment from the beginning of x_i to the beginning of x_(i+1)
    @param[inout]
    y       device pointer to the first vector y_1.
    @param[in]
    incy    [rocblas_int]
            specifies the increment between elements of each y_i.
    @param[in]
    stride_y [rocblas_stride]
             specifies the increment from the beginning of y_i to the beginning of y_(i+1)
    @param[in]
    c       device pointer or host pointer to scalar cosine component of the rotation matrix.
    @param[in]
    s       device pointer or host pointer to scalar sine component of the rotation matrix.
    @param[in]
    batch_count [rocblas_int]
            the number of x and y arrays, i.e. the number of batches.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdrot_strided_batched(rocblas_handle          handle,
                                                            rocblas_int             n,
                                                            rocblas_double_complex* x,
                                                            rocblas_int             incx,
                                                            rocblas_stride          stride_x,
                                                            rocblas_double_complex* y,
                                                            rocblas_int             incy,
                                                            rocblas_stride          stride_y,
                                                            const double*           c,
                                                            const double*           s,
                                                            rocblas_int             batch_count);

ROCBLAS_EXPORT rocblas_status
    rocblas_srotg(rocblas_handle handle, float* a, float* b, float* c, float* s);

ROCBLAS_EXPORT rocblas_status
    rocblas_drotg(rocblas_handle handle, double* a, double* b, double* c, double* s);

ROCBLAS_EXPORT rocblas_status rocblas_crotg(rocblas_handle         handle,
                                            rocblas_float_complex* a,
                                            rocblas_float_complex* b,
                                            float*                 c,
                                            rocblas_float_complex* s);
/*! \brief BLAS Level 1 API

    \details
    rotg creates the Givens rotation matrix for the vector (a b).
         Scalars c and s and arrays a and b may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.
         If the pointer mode is set to rocblas_pointer_mode_host, this function blocks the CPU until the GPU has finished and the results are available in host memory.
         If the pointer mode is set to rocblas_pointer_mode_device, this function returns immediately and synchronization is required to read the results.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[inout]
    a       device pointer or host pointer to input vector element, overwritten with r.
    @param[inout]
    b       device pointer or host pointer to input vector element, overwritten with z.
    @param[inout]
    c       device pointer or host pointer to cosine element of Givens rotation.
    @param[inout]
    s       device pointer or host pointer sine element of Givens rotation.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zrotg(rocblas_handle          handle,
                                            rocblas_double_complex* a,
                                            rocblas_double_complex* b,
                                            double*                 c,
                                            rocblas_double_complex* s);

ROCBLAS_EXPORT rocblas_status rocblas_srotg_batched(rocblas_handle handle,
                                                    float* const   a[],
                                                    float* const   b[],
                                                    float* const   c[],
                                                    float* const   s[],
                                                    rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_drotg_batched(rocblas_handle handle,
                                                    double* const  a[],
                                                    double* const  b[],
                                                    double* const  c[],
                                                    double* const  s[],
                                                    rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_crotg_batched(rocblas_handle               handle,
                                                    rocblas_float_complex* const a[],
                                                    rocblas_float_complex* const b[],
                                                    float* const                 c[],
                                                    rocblas_float_complex* const s[],
                                                    rocblas_int                  batch_count);
/*! \brief BLAS Level 1 API

    \details
    rotg_batched creates the Givens rotation matrix for the batched vectors (a_i b_i), for i = 1, ..., batch_count.
         a, b, c, and s may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.
         If the pointer mode is set to rocblas_pointer_mode_host, this function blocks the CPU until the GPU has finished and the results are available in host memory.
         If the pointer mode is set to rocblas_pointer_mode_device, this function returns immediately and synchronization is required to read the results.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[inout]
    a       device array of device pointers storing each single input vector element a_i, overwritten with r_i.
    @param[inout]
    b       device array of device pointers storing each single input vector element b_i, overwritten with z_i.
    @param[inout]
    c       device array of device pointers storing each cosine element of Givens rotation for the batch.
    @param[inout]
    s       device array of device pointers storing each sine element of Givens rotation for the batch.
    @param[in]
    batch_count [rocblas_int]
                number of batches (length of arrays a, b, c, and s).

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zrotg_batched(rocblas_handle                handle,
                                                    rocblas_double_complex* const a[],
                                                    rocblas_double_complex* const b[],
                                                    double* const                 c[],
                                                    rocblas_double_complex* const s[],
                                                    rocblas_int                   batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_srotg_strided_batched(rocblas_handle handle,
                                                            float*         a,
                                                            rocblas_stride stride_a,
                                                            float*         b,
                                                            rocblas_stride stride_b,
                                                            float*         c,
                                                            rocblas_stride stride_c,
                                                            float*         s,
                                                            rocblas_stride stride_s,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_drotg_strided_batched(rocblas_handle handle,
                                                            double*        a,
                                                            rocblas_stride stride_a,
                                                            double*        b,
                                                            rocblas_stride stride_b,
                                                            double*        c,
                                                            rocblas_stride stride_c,
                                                            double*        s,
                                                            rocblas_stride stride_s,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_crotg_strided_batched(rocblas_handle         handle,
                                                            rocblas_float_complex* a,
                                                            rocblas_stride         stride_a,
                                                            rocblas_float_complex* b,
                                                            rocblas_stride         stride_b,
                                                            float*                 c,
                                                            rocblas_stride         stride_c,
                                                            rocblas_float_complex* s,
                                                            rocblas_stride         stride_s,
                                                            rocblas_int            batch_count);
/*! \brief BLAS Level 1 API

    \details
    rotg_strided_batched creates the Givens rotation matrix for the strided batched vectors (a_i b_i), for i = 1, ..., batch_count.
         a, b, c, and s may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.
         If the pointer mode is set to rocblas_pointer_mode_host, this function blocks the CPU until the GPU has finished and the results are available in host memory.
         If the pointer mode is set to rocblas_pointer_mode_device, this function returns immediately and synchronization is required to read the results.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[inout]
    a       device strided_batched pointer or host strided_batched pointer to first single input vector element a_1, overwritten with r.
    @param[in]
    stride_a [rocblas_stride]
             distance between elements of a in batch (distance between a_i and a_(i + 1))
    @param[inout]
    b       device strided_batched pointer or host strided_batched pointer to first single input vector element b_1, overwritten with z.
    @param[in]
    stride_b [rocblas_stride]
             distance between elements of b in batch (distance between b_i and b_(i + 1))
    @param[inout]
    c       device strided_batched pointer or host strided_batched pointer to first cosine element of Givens rotations c_1.
    @param[in]
    stride_c [rocblas_stride]
             distance between elements of c in batch (distance between c_i and c_(i + 1))
    @param[inout]
    s       device strided_batched pointer or host strided_batched pointer to sine element of Givens rotations s_1.
    @param[in]
    stride_s [rocblas_stride]
             distance between elements of s in batch (distance between s_i and s_(i + 1))
    @param[in]
    batch_count [rocblas_int]
                number of batches (length of arrays a, b, c, and s).

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zrotg_strided_batched(rocblas_handle          handle,
                                                            rocblas_double_complex* a,
                                                            rocblas_stride          stride_a,
                                                            rocblas_double_complex* b,
                                                            rocblas_stride          stride_b,
                                                            double*                 c,
                                                            rocblas_stride          stride_c,
                                                            rocblas_double_complex* s,
                                                            rocblas_stride          stride_s,
                                                            rocblas_int             batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_srotm(rocblas_handle handle,
                                            rocblas_int    n,
                                            float*         x,
                                            rocblas_int    incx,
                                            float*         y,
                                            rocblas_int    incy,
                                            const float*   param);
/*! \brief BLAS Level 1 API

    \details
    rotm applies the modified Givens rotation matrix defined by param to vectors x and y.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[in]
    n       [rocblas_int]
            number of elements in the x and y vectors.
    @param[inout]
    x       device pointer storing vector x.
    @param[in]
    incx    [rocblas_int]
            specifies the increment between elements of x.
    @param[inout]
    y       device pointer storing vector y.
    @param[in]
    incy    [rocblas_int]
            specifies the increment between elements of y.
    @param[in]
    param   device vector or host vector of 5 elements defining the rotation.
            param[0] = flag
            param[1] = H11
            param[2] = H21
            param[3] = H12
            param[4] = H22
            The flag parameter defines the form of H:
            flag = -1 => H = ( H11 H12 H21 H22 )
            flag =  0 => H = ( 1.0 H12 H21 1.0 )
            flag =  1 => H = ( H11 1.0 -1.0 H22 )
            flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
            param may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_drotm(rocblas_handle handle,
                                            rocblas_int    n,
                                            double*        x,
                                            rocblas_int    incx,
                                            double*        y,
                                            rocblas_int    incy,
                                            const double*  param);

ROCBLAS_EXPORT rocblas_status rocblas_srotm_batched(rocblas_handle     handle,
                                                    rocblas_int        n,
                                                    float* const       x[],
                                                    rocblas_int        incx,
                                                    float* const       y[],
                                                    rocblas_int        incy,
                                                    const float* const param[],
                                                    rocblas_int        batch_count);

/*! \brief BLAS Level 1 API

    \details
    rotm_batched applies the modified Givens rotation matrix defined by param_i to batched vectors x_i and y_i, for i = 1, ..., batch_count.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[in]
    n       [rocblas_int]
            number of elements in the x and y vectors.
    @param[inout]
    x       device array of device pointers storing each vector x_i.
    @param[in]
    incx    [rocblas_int]
            specifies the increment between elements of each x_i.
    @param[inout]
    y       device array of device pointers storing each vector y_1.
    @param[in]
    incy    [rocblas_int]
            specifies the increment between elements of each y_i.
    @param[in]
    param   device array of device vectors of 5 elements defining the rotation.
            param[0] = flag
            param[1] = H11
            param[2] = H21
            param[3] = H12
            param[4] = H22
            The flag parameter defines the form of H:
            flag = -1 => H = ( H11 H12 H21 H22 )
            flag =  0 => H = ( 1.0 H12 H21 1.0 )
            flag =  1 => H = ( H11 1.0 -1.0 H22 )
            flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
            param may ONLY be stored on the device for the batched version of this function.
    @param[in]
    batch_count [rocblas_int]
                the number of x and y arrays, i.e. the number of batches.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_drotm_batched(rocblas_handle      handle,
                                                    rocblas_int         n,
                                                    double* const       x[],
                                                    rocblas_int         incx,
                                                    double* const       y[],
                                                    rocblas_int         incy,
                                                    const double* const param[],
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_srotm_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            float*         x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            float*         y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stride_y,
                                                            const float*   param,
                                                            rocblas_stride stride_param,
                                                            rocblas_int    batch_count);

/*! \brief BLAS Level 1 API

    \details
    rotm_strided_batched applies the modified Givens rotation matrix defined by param_i to strided batched vectors x_i and y_i, for i = 1, ..., batch_count

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[in]
    n       [rocblas_int]
            number of elements in the x and y vectors.
    @param[inout]
    x       device pointer pointing to first strided batched vector x_1.
    @param[in]
    incx    [rocblas_int]
            specifies the increment between elements of each x_i.
    @param[in]
    stride_x [rocblas_stride]
             specifies the increment between the beginning of x_i and x_(i + 1)
    @param[inout]
    y       device pointer pointing to first strided batched vector y_1.
    @param[in]
    incy    [rocblas_int]
            specifies the increment between elements of each y_i.
    @param[in]
    stride_y [rocblas_stride]
             specifies the increment between the beginning of y_i and y_(i + 1)
    @param[in]
    param   device pointer pointing to first array of 5 elements defining the rotation (param_1).
            param[0] = flag
            param[1] = H11
            param[2] = H21
            param[3] = H12
            param[4] = H22
            The flag parameter defines the form of H:
            flag = -1 => H = ( H11 H12 H21 H22 )
            flag =  0 => H = ( 1.0 H12 H21 1.0 )
            flag =  1 => H = ( H11 1.0 -1.0 H22 )
            flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
            param may ONLY be stored on the device for the strided_batched version of this function.
    @param[in]
    stride_param [rocblas_stride]
                 specifies the increment between the beginning of param_i and param_(i + 1)
    @param[in]
    batch_count [rocblas_int]
                the number of x and y arrays, i.e. the number of batches.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_drotm_strided_batched(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            double*        x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            double*        y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stride_y,
                                                            const double*  param,
                                                            rocblas_stride stride_param,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_srotmg(
    rocblas_handle handle, float* d1, float* d2, float* x1, const float* y1, float* param);

/*! \brief BLAS Level 1 API

    \details
    rotmg creates the modified Givens rotation matrix for the vector (d1 * x1, d2 * y1).
          Parameters may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.
          If the pointer mode is set to rocblas_pointer_mode_host, this function blocks the CPU until the GPU has finished and the results are available in host memory.
          If the pointer mode is set to rocblas_pointer_mode_device, this function returns immediately and synchronization is required to read the results.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[inout]
    d1      device pointer or host pointer to input scalar that is overwritten.
    @param[inout]
    d2      device pointer or host pointer to input scalar that is overwritten.
    @param[inout]
    x1      device pointer or host pointer to input scalar that is overwritten.
    @param[in]
    y1      device pointer or host pointer to input scalar.
    @param[out]
    param   device vector or host vector of 5 elements defining the rotation.
            param[0] = flag
            param[1] = H11
            param[2] = H21
            param[3] = H12
            param[4] = H22
            The flag parameter defines the form of H:
            flag = -1 => H = ( H11 H12 H21 H22 )
            flag =  0 => H = ( 1.0 H12 H21 1.0 )
            flag =  1 => H = ( H11 1.0 -1.0 H22 )
            flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
            param may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_drotmg(
    rocblas_handle handle, double* d1, double* d2, double* x1, const double* y1, double* param);

ROCBLAS_EXPORT rocblas_status rocblas_srotmg_batched(rocblas_handle     handle,
                                                     float* const       d1[],
                                                     float* const       d2[],
                                                     float* const       x1[],
                                                     const float* const y1[],
                                                     float* const       param[],
                                                     rocblas_int        batch_count);

/*! \brief BLAS Level 1 API

    \details
    rotmg_batched creates the modified Givens rotation matrix for the batched vectors (d1_i * x1_i, d2_i * y1_i), for i = 1, ..., batch_count.
          Parameters may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.
          If the pointer mode is set to rocblas_pointer_mode_host, this function blocks the CPU until the GPU has finished and the results are available in host memory.
          If the pointer mode is set to rocblas_pointer_mode_device, this function returns immediately and synchronization is required to read the results.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[inout]
    d1      device batched array or host batched array of input scalars that is overwritten.
    @param[inout]
    d2      device batched array or host batched array of input scalars that is overwritten.
    @param[inout]
    x1      device batched array or host batched array of input scalars that is overwritten.
    @param[in]
    y1      device batched array or host batched array of input scalars.
    @param[out]
    param   device batched array or host batched array of vectors of 5 elements defining the rotation.
            param[0] = flag
            param[1] = H11
            param[2] = H21
            param[3] = H12
            param[4] = H22
            The flag parameter defines the form of H:
            flag = -1 => H = ( H11 H12 H21 H22 )
            flag =  0 => H = ( 1.0 H12 H21 1.0 )
            flag =  1 => H = ( H11 1.0 -1.0 H22 )
            flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
            param may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.
    @param[in]
    batch_count [rocblas_int]
                the number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_drotmg_batched(rocblas_handle      handle,
                                                     double* const       d1[],
                                                     double* const       d2[],
                                                     double* const       x1[],
                                                     const double* const y1[],
                                                     double* const       param[],
                                                     rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_srotmg_strided_batched(rocblas_handle handle,
                                                             float*         d1,
                                                             rocblas_stride stride_d1,
                                                             float*         d2,
                                                             rocblas_stride stride_d2,
                                                             float*         x1,
                                                             rocblas_stride stride_x1,
                                                             const float*   y1,
                                                             rocblas_stride stride_y1,
                                                             float*         param,
                                                             rocblas_stride stride_param,
                                                             rocblas_int    batch_count);

/*! \brief BLAS Level 1 API

    \details
    rotmg_strided_batched creates the modified Givens rotation matrix for the strided batched vectors (d1_i * x1_i, d2_i * y1_i), for i = 1, ..., batch_count.
          Parameters may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.
          If the pointer mode is set to rocblas_pointer_mode_host, this function blocks the CPU until the GPU has finished and the results are available in host memory.
          If the pointer mode is set to rocblas_pointer_mode_device, this function returns immediately and synchronization is required to read the results.

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.
    @param[inout]
    d1      device strided_batched array or host strided_batched array of input scalars that is overwritten.
    @param[in]
    stride_d1 [rocblas_stride]
              specifies the increment between the beginning of d1_i and d1_(i+1)
    @param[inout]
    d2      device strided_batched array or host strided_batched array of input scalars that is overwritten.
    @param[in]
    stride_d2 [rocblas_stride]
              specifies the increment between the beginning of d2_i and d2_(i+1)
    @param[inout]
    x1      device strided_batched array or host strided_batched array of input scalars that is overwritten.
    @param[in]
    stride_x1 [rocblas_stride]
              specifies the increment between the beginning of x1_i and x1_(i+1)
    @param[in]
    y1      device strided_batched array or host strided_batched array of input scalars.
    @param[in]
    stride_y1 [rocblas_stride]
              specifies the increment between the beginning of y1_i and y1_(i+1)
    @param[out]
    param   device strided_batched array or host strided_batched array of vectors of 5 elements defining the rotation.
            param[0] = flag
            param[1] = H11
            param[2] = H21
            param[3] = H12
            param[4] = H22
            The flag parameter defines the form of H:
            flag = -1 => H = ( H11 H12 H21 H22 )
            flag =  0 => H = ( 1.0 H12 H21 1.0 )
            flag =  1 => H = ( H11 1.0 -1.0 H22 )
            flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
            param may be stored in either host or device memory, location is specified by calling rocblas_set_pointer_mode.
    @param[in]
    stride_param [rocblas_stride]
                 specifies the increment between the beginning of param_i and param_(i + 1)
    @param[in]
    batch_count [rocblas_int]
                the number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_drotmg_strided_batched(rocblas_handle handle,
                                                             double*        d1,
                                                             rocblas_stride stride_d1,
                                                             double*        d2,
                                                             rocblas_stride stride_d2,
                                                             double*        x1,
                                                             rocblas_stride stride_x1,
                                                             const double*  y1,
                                                             rocblas_stride stride_y1,
                                                             double*        param,
                                                             rocblas_stride stride_param,
                                                             rocblas_int    batch_count);

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

ROCBLAS_EXPORT rocblas_status rocblas_sgbmv(rocblas_handle    handle,
                                            rocblas_operation trans,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            rocblas_int       kl,
                                            rocblas_int       ku,
                                            const float*      alpha,
                                            const float*      A,
                                            rocblas_int       lda,
                                            const float*      x,
                                            rocblas_int       incx,
                                            const float*      beta,
                                            float*            y,
                                            rocblas_int       incy);

ROCBLAS_EXPORT rocblas_status rocblas_dgbmv(rocblas_handle    handle,
                                            rocblas_operation trans,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            rocblas_int       kl,
                                            rocblas_int       ku,
                                            const double*     alpha,
                                            const double*     A,
                                            rocblas_int       lda,
                                            const double*     x,
                                            rocblas_int       incx,
                                            const double*     beta,
                                            double*           y,
                                            rocblas_int       incy);

ROCBLAS_EXPORT rocblas_status rocblas_cgbmv(rocblas_handle               handle,
                                            rocblas_operation            trans,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            rocblas_int                  kl,
                                            rocblas_int                  ku,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       y,
                                            rocblas_int                  incy);

/*! \brief BLAS Level 2 API

    \details
    xGBMV performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n banded matrix with kl sub-diagonals and ku super-diagonals.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not
    @param[in]
    m         [rocblas_int]
              number of rows of matrix A
    @param[in]
    n         [rocblas_int]
              number of columns of matrix A
    @param[in]
    kl        [rocblas_int]
              number of sub-diagonals of A
    @param[in]
    ku        [rocblas_int]
              number of super-diagonals of A
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
        A     device pointer storing banded matrix A.
              Leading (kl + ku + 1) by n part of the matrix contains the coefficients
              of the banded matrix. The leading diagonal resides in row (ku + 1) with
              the first super-diagonal above on the RHS of row ku. The first sub-diagonal
              resides below on the LHS of row ku + 2. This propogates up and down across
              sub/super-diagonals.
                Ex: (m = n = 7; ku = 2, kl = 2)
                1 2 3 0 0 0 0             0 0 3 3 3 3 3
                4 1 2 3 0 0 0             0 2 2 2 2 2 2
                5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
                0 5 4 1 2 3 0             4 4 4 4 4 4 0
                0 0 5 4 1 2 0             5 5 5 5 5 0 0
                0 0 0 5 4 1 2             0 0 0 0 0 0 0
                0 0 0 0 5 4 1             0 0 0 0 0 0 0
              Note that the empty elements which don't correspond to data will not
              be referenced.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. Must be >= (kl + ku + 1)
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgbmv(rocblas_handle                handle,
                                            rocblas_operation             trans,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            rocblas_int                   kl,
                                            rocblas_int                   ku,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       y,
                                            rocblas_int                   incy);

ROCBLAS_EXPORT rocblas_status rocblas_sgbmv_batched(rocblas_handle     handle,
                                                    rocblas_operation  trans,
                                                    rocblas_int        m,
                                                    rocblas_int        n,
                                                    rocblas_int        kl,
                                                    rocblas_int        ku,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    const float*       beta,
                                                    float* const       y[],
                                                    rocblas_int        incy,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgbmv_batched(rocblas_handle      handle,
                                                    rocblas_operation   trans,
                                                    rocblas_int         m,
                                                    rocblas_int         n,
                                                    rocblas_int         kl,
                                                    rocblas_int         ku,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    const double*       beta,
                                                    double* const       y[],
                                                    rocblas_int         incy,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgbmv_batched(rocblas_handle                     handle,
                                                    rocblas_operation                  trans,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    rocblas_int                        kl,
                                                    rocblas_int                        ku,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex* const       y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xGBMV_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n banded matrix with kl sub-diagonals and ku super-diagonals,
    for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not
    @param[in]
    m         [rocblas_int]
              number of rows of each matrix A_i
    @param[in]
    n         [rocblas_int]
              number of columns of each matrix A_i
    @param[in]
    kl        [rocblas_int]
              number of sub-diagonals of each A_i
    @param[in]
    ku        [rocblas_int]
              number of super-diagonals of each A_i
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
        A     device array of device pointers storing each banded matrix A_i.
              Leading (kl + ku + 1) by n part of the matrix contains the coefficients
              of the banded matrix. The leading diagonal resides in row (ku + 1) with
              the first super-diagonal above on the RHS of row ku. The first sub-diagonal
              resides below on the LHS of row ku + 2. This propogates up and down across
              sub/super-diagonals.
                Ex: (m = n = 7; ku = 2, kl = 2)
                1 2 3 0 0 0 0             0 0 3 3 3 3 3
                4 1 2 3 0 0 0             0 2 2 2 2 2 2
                5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
                0 5 4 1 2 3 0             4 4 4 4 4 4 0
                0 0 5 4 1 2 0             5 5 5 5 5 0 0
                0 0 0 5 4 1 2             0 0 0 0 0 0 0
                0 0 0 0 5 4 1             0 0 0 0 0 0 0
              Note that the empty elements which don't correspond to data will not
              be referenced.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. Must be >= (kl + ku + 1)
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    batch_count [rocblas_int]
                specifies the number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgbmv_batched(rocblas_handle                      handle,
                                                    rocblas_operation                   trans,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    rocblas_int                         kl,
                                                    rocblas_int                         ku,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex* const       y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgbmv_strided_batched(rocblas_handle    handle,
                                                            rocblas_operation trans,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            rocblas_int       kl,
                                                            rocblas_int       ku,
                                                            const float*      alpha,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            const float*      x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            const float*      beta,
                                                            float*            y,
                                                            rocblas_int       incy,
                                                            rocblas_stride    stride_y,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgbmv_strided_batched(rocblas_handle    handle,
                                                            rocblas_operation trans,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            rocblas_int       kl,
                                                            rocblas_int       ku,
                                                            const double*     alpha,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            const double*     x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            const double*     beta,
                                                            double*           y,
                                                            rocblas_int       incy,
                                                            rocblas_stride    stride_y,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgbmv_strided_batched(rocblas_handle               handle,
                                                            rocblas_operation            trans,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            rocblas_int                  kl,
                                                            rocblas_int                  ku,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stride_y,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    xGBMV_STRIDED_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n banded matrix with kl sub-diagonals and ku super-diagonals,
    for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not
    @param[in]
    m         [rocblas_int]
              number of rows of matrix A
    @param[in]
    n         [rocblas_int]
              number of columns of matrix A
    @param[in]
    kl        [rocblas_int]
              number of sub-diagonals of A
    @param[in]
    ku        [rocblas_int]
              number of super-diagonals of A
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
        A     device pointer to first banded matrix (A_1).
              Leading (kl + ku + 1) by n part of the matrix contains the coefficients
              of the banded matrix. The leading diagonal resides in row (ku + 1) with
              the first super-diagonal above on the RHS of row ku. The first sub-diagonal
              resides below on the LHS of row ku + 2. This propogates up and down across
              sub/super-diagonals.
                Ex: (m = n = 7; ku = 2, kl = 2)
                1 2 3 0 0 0 0             0 0 3 3 3 3 3
                4 1 2 3 0 0 0             0 2 2 2 2 2 2
                5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
                0 5 4 1 2 3 0             4 4 4 4 4 4 0
                0 0 5 4 1 2 0             5 5 5 5 5 0 0
                0 0 0 5 4 1 2             0 0 0 0 0 0 0
                0 0 0 0 5 4 1             0 0 0 0 0 0 0
              Note that the empty elements which don't correspond to data will not
              be referenced.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. Must be >= (kl + ku + 1)
    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         device pointer to first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1)
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer to first vector (y_1).
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (x_i+1)
    @param[in]
    batch_count [rocblas_int]
                specifies the number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgbmv_strided_batched(rocblas_handle                handle,
                                                            rocblas_operation             trans,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            rocblas_int                   kl,
                                                            rocblas_int                   ku,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stride_y,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgemv(rocblas_handle    handle,
                                            rocblas_operation trans,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            const float*      alpha,
                                            const float*      A,
                                            rocblas_int       lda,
                                            const float*      x,
                                            rocblas_int       incx,
                                            const float*      beta,
                                            float*            y,
                                            rocblas_int       incy);

ROCBLAS_EXPORT rocblas_status rocblas_dgemv(rocblas_handle    handle,
                                            rocblas_operation trans,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            const double*     alpha,
                                            const double*     A,
                                            rocblas_int       lda,
                                            const double*     x,
                                            rocblas_int       incx,
                                            const double*     beta,
                                            double*           y,
                                            rocblas_int       incy);

ROCBLAS_EXPORT rocblas_status rocblas_cgemv(rocblas_handle               handle,
                                            rocblas_operation            trans,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       y,
                                            rocblas_int                  incy);

/*! \brief BLAS Level 2 API

    \details
    xGEMV performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not
    @param[in]
    m         [rocblas_int]
              number of rows of matrix A
    @param[in]
    n         [rocblas_int]
              number of columns of matrix A
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgemv(rocblas_handle                handle,
                                            rocblas_operation             trans,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       y,
                                            rocblas_int                   incy);

ROCBLAS_EXPORT rocblas_status rocblas_sgemv_batched(rocblas_handle     handle,
                                                    rocblas_operation  trans,
                                                    rocblas_int        m,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    const float*       beta,
                                                    float* const       y[],
                                                    rocblas_int        incy,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgemv_batched(rocblas_handle      handle,
                                                    rocblas_operation   trans,
                                                    rocblas_int         m,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    const double*       beta,
                                                    double* const       y[],
                                                    rocblas_int         incy,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgemv_batched(rocblas_handle                     handle,
                                                    rocblas_operation                  trans,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex* const       y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xGEMV_BATCHED performs a batch of matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batch_count.

    @param[in]
    handle      [rocblas_handle]
                handle to the rocblas library context queue.
    @param[in]
    trans       [rocblas_operation]
                indicates whether matrices A_i are tranposed (conjugated) or not
    @param[in]
    m           [rocblas_int]
                number of rows of each matrix A_i
    @param[in]
    n           [rocblas_int]
                number of columns of each matrix A_i
    @param[in]
    alpha       device pointer or host pointer to scalar alpha.
    @param[in]
    A           device array of device pointers storing each matrix A_i.
    @param[in]
    lda         [rocblas_int]
                specifies the leading dimension of each matrix A_i.
    @param[in]
    x           device array of device pointers storing each vector x_i.
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of each vector x_i.
    @param[in]
    beta        device pointer or host pointer to scalar beta.
    @param[inout]
    y           device array of device pointers storing each vector y_i.
    @param[in]
    incy        [rocblas_int]
                specifies the increment for the elements of each vector y_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgemv_batched(rocblas_handle                      handle,
                                                    rocblas_operation                   trans,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex* const       y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgemv_strided_batched(rocblas_handle    handle,
                                                            rocblas_operation transA,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            const float*      alpha,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    strideA,
                                                            const float*      x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stridex,
                                                            const float*      beta,
                                                            float*            y,
                                                            rocblas_int       incy,
                                                            rocblas_stride    stridey,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgemv_strided_batched(rocblas_handle    handle,
                                                            rocblas_operation transA,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            const double*     alpha,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    strideA,
                                                            const double*     x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stridex,
                                                            const double*     beta,
                                                            double*           y,
                                                            rocblas_int       incy,
                                                            rocblas_stride    stridey,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgemv_strided_batched(rocblas_handle               handle,
                                                            rocblas_operation            transA,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               strideA,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stridey,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    xGEMV_STRIDED_BATCHED performs a batch of matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batch_count.

    @param[in]
    handle      [rocblas_handle]
                handle to the rocblas library context queue.
    @param[in]
    transA      [rocblas_operation]
                indicates whether matrices A_i are tranposed (conjugated) or not
    @param[in]
    m           [rocblas_int]
                number of rows of matrices A_i
    @param[in]
    n           [rocblas_int]
                number of columns of matrices A_i
    @param[in]
    alpha       device pointer or host pointer to scalar alpha.
    @param[in]
    A           device pointer to the first matrix (A_1) in the batch.
    @param[in]
    lda         [rocblas_int]
                specifies the leading dimension of matrices A_i.
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x           device pointer to the first vector (x_1) in the batch.
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of vectors x_i.
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stride_x, however the user should
                take care to ensure that stride_x is of appropriate size. When trans equals rocblas_operation_none
                this typically means stride_x >= n * incx, otherwise stride_x >= m * incx.
    @param[in]
    beta        device pointer or host pointer to scalar beta.
    @param[inout]
    y           device pointer to the first vector (y_1) in the batch.
    @param[in]
    incy        [rocblas_int]
                specifies the increment for the elements of vectors y_i.
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1).
                There are no restrictions placed on stride_y, however the user should
                take care to ensure that stride_y is of appropriate size. When trans equals rocblas_operation_none
                this typically means stride_y >= m * incy, otherwise stride_y >= n * incy. stridey should be non zero.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgemv_strided_batched(rocblas_handle                handle,
                                                            rocblas_operation             transA,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                strideA,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stridey,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chbmv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_int                  n,
                                            rocblas_int                  k,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       y,
                                            rocblas_int                  incy);

/*! \brief BLAS Level 2 API

    \details
    xHBMV performs the matrix-vector operations

        y := alpha*A*x + beta*y

    where alpha and beta are scalars, x and y are n element vectors and A is an
    n by n hermitian band matrix, with k super-diagonals.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: The upper triangular part of A is being supplied.
              rocblas_fill_lower: The lower triangular part of A is being supplied.
    @param[in]
    n         [rocblas_int]
              the order of the matrix A.
    @param[in]
    k         [rocblas_int]
              the number of super-diagonals of the matrix A. Must be >= 0.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device pointer storing matrix A. Of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The leading (k + 1) by n part of A must contain the upper
                triangular band part of the hermitian matrix, with the leading
                diagonal in row (k + 1), the first super-diagonal on the RHS
                of row k, etc.
                The top left k by x triangle of A will not be referenced.
                    Ex (upper, lda = n = 4, k = 1):
                    A                             Represented matrix
                    (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
                    (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)

              if uplo == rocblas_fill_lower:
                The leading (k + 1) by n part of A must contain the lower
                triangular band part of the hermitian matrix, with the leading
                diagonal in row (1), the first sub-diagonal on the LHS of
                row 2, etc.
                The bottom right k by k triangle of A will not be referenced.
                    Ex (lower, lda = 2, n = 4, k = 1):
                    A                               Represented matrix
                    (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
                    (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
                                                    (0, 0) (6, 8) (3, 0) (7,-7)
                                                    (0, 0) (0, 0) (7, 7) (4, 0)

              As a hermitian matrix, the imaginary part of the main diagonal
              of A will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. must be >= k + 1
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhbmv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_int                   n,
                                            rocblas_int                   k,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       y,
                                            rocblas_int                   incy);

ROCBLAS_EXPORT rocblas_status rocblas_chbmv_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_int                        n,
                                                    rocblas_int                        k,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex* const       y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHBMV_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n hermitian band matrix with k super-diagonals, for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: The upper triangular part of each A_i is being supplied.
              rocblas_fill_lower: The lower triangular part of each A_i is being supplied.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    k         [rocblas_int]
              the number of super-diagonals of each matrix A_i. Must be >= 0.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device array of device pointers storing each matrix_i A of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The leading (k + 1) by n part of each A_i must contain the upper
                triangular band part of the hermitian matrix, with the leading
                diagonal in row (k + 1), the first super-diagonal on the RHS
                of row k, etc.
                The top left k by x triangle of each A_i will not be referenced.
                    Ex (upper, lda = n = 4, k = 1):
                    A                             Represented matrix
                    (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
                    (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)

              if uplo == rocblas_fill_lower:
                The leading (k + 1) by n part of each A_i must contain the lower
                triangular band part of the hermitian matrix, with the leading
                diagonal in row (1), the first sub-diagonal on the LHS of
                row 2, etc.
                The bottom right k by k triangle of each A_i will not be referenced.
                    Ex (lower, lda = 2, n = 4, k = 1):
                    A                               Represented matrix
                    (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
                    (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
                                                    (0, 0) (6, 8) (3, 0) (7,-7)
                                                    (0, 0) (0, 0) (7, 7) (4, 0)

              As a hermitian matrix, the imaginary part of the main diagonal
              of each A_i will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. must be >= max(1, n)
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhbmv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_int                         n,
                                                    rocblas_int                         k,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex* const       y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chbmv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_int                  n,
                                                            rocblas_int                  k,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stride_y,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHBMV_STRIDED_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n hermitian band matrix with k super-diagonals, for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: The upper triangular part of each A_i is being supplied.
              rocblas_fill_lower: The lower triangular part of each A_i is being supplied.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    k         [rocblas_int]
              the number of super-diagonals of each matrix A_i. Must be >= 0.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device array pointing to the first matrix A_1. Each A_i is of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The leading (k + 1) by n part of each A_i must contain the upper
                triangular band part of the hermitian matrix, with the leading
                diagonal in row (k + 1), the first super-diagonal on the RHS
                of row k, etc.
                The top left k by x triangle of each A_i will not be referenced.
                    Ex (upper, lda = n = 4, k = 1):
                    A                             Represented matrix
                    (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
                    (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)

              if uplo == rocblas_fill_lower:
                The leading (k + 1) by n part of each A_i must contain the lower
                triangular band part of the hermitian matrix, with the leading
                diagonal in row (1), the first sub-diagonal on the LHS of
                row 2, etc.
                The bottom right k by k triangle of each A_i will not be referenced.
                    Ex (lower, lda = 2, n = 4, k = 1):
                    A                               Represented matrix
                    (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
                    (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
                                                    (0, 0) (6, 8) (3, 0) (7,-7)
                                                    (0, 0) (0, 0) (7, 7) (4, 0)

              As a hermitian matrix, the imaginary part of the main diagonal
              of each A_i will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. must be >= max(1, n)
    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         device array pointing to the first vector y_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1)
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array pointing to the first vector y_1.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhbmv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_int                   n,
                                                            rocblas_int                   k,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stride_y,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chemv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       y,
                                            rocblas_int                  incy);

/*! \brief BLAS Level 2 API

    \details
    xHEMV performs one of the matrix-vector operations

        y := alpha*A*x + beta*y

    where alpha and beta are scalars, x and y are n element vectors and A is an
    n by n hermitian matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of the hermitian matrix A is supplied.
              rocblas_fill_lower: the lower triangular part of the hermitian matrix A is supplied.
    @param[in]
    n         [rocblas_int]
              the order of the matrix A.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device pointer storing matrix A. Of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular part of A must contain
                the upper triangular part of a hermitian matrix. The lower
                triangular part of A will not be referenced.
              if uplo == rocblas_fill_lower:
                The lower triangular part of A must contain
                the lower triangular part of a hermitian matrix. The upper
                triangular part of A will not be referenced.
              As a hermitian matrix, the imaginary part of the main diagonal
              of A will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. must be >= max(1, n)
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhemv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       y,
                                            rocblas_int                   incy);

ROCBLAS_EXPORT rocblas_status rocblas_chemv_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex* const       y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHEMV_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n hermitian matrix, for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of the hermitian matrix A is supplied.
              rocblas_fill_lower: the lower triangular part of the hermitian matrix A is supplied.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device array of device pointers storing each matrix A_i of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular part of each A_i must contain
                the upper triangular part of a hermitian matrix. The lower
                triangular part of each A_i will not be referenced.
              if uplo == rocblas_fill_lower:
                The lower triangular part of each A_i must contain
                the lower triangular part of a hermitian matrix. The upper
                triangular part of each A_i will not be referenced.
              As a hermitian matrix, the imaginary part of the main diagonal
              of each A_i will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. must be >= max(1, n)
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhemv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex* const       y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chemv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stride_y,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHEMV_STRIDED_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n hermitian matrix, for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of the hermitian matrix A is supplied.
              rocblas_fill_lower: the lower triangular part of the hermitian matrix A is supplied.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device array of device pointers storing each matrix A_i of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular part of each A_i must contain
                the upper triangular part of a hermitian matrix. The lower
                triangular part of each A_i will not be referenced.
              if uplo == rocblas_fill_lower:
                The lower triangular part of each A_i must contain
                the lower triangular part of a hermitian matrix. The upper
                triangular part of each A_i will not be referenced.
              As a hermitian matrix, the imaginary part of the main diagonal
              of each A_i will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. must be >= max(1, n)
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhemv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stride_y,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cher(rocblas_handle               handle,
                                           rocblas_fill                 uplo,
                                           rocblas_int                  n,
                                           const float*                 alpha,
                                           const rocblas_float_complex* x,
                                           rocblas_int                  incx,
                                           rocblas_float_complex*       A,
                                           rocblas_int                  lda);

/*! \brief BLAS Level 2 API

    \details
    xHER performs the matrix-vector operations

        A := A + alpha*x*x**H

    where alpha is a real scalar, x is a vector, and A is an
    n by n hermitian matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in A.
              rocblas_fill_lower: The lower triangular part of A is supplied in A.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[inout]
    A         device pointer storing the specified triangular portion of
              the hermitian matrix A. Of size (lda * n).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the hermitian matrix A is supplied. The lower
                triangluar portion will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the hermitian matrix A is supplied. The upper
                triangular portion will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. Must be at least max(1, n).
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zher(rocblas_handle                handle,
                                           rocblas_fill                  uplo,
                                           rocblas_int                   n,
                                           const double*                 alpha,
                                           const rocblas_double_complex* x,
                                           rocblas_int                   incx,
                                           rocblas_double_complex*       A,
                                           rocblas_int                   lda);

ROCBLAS_EXPORT rocblas_status rocblas_cher_batched(rocblas_handle                     handle,
                                                   rocblas_fill                       uplo,
                                                   rocblas_int                        n,
                                                   const float*                       alpha,
                                                   const rocblas_float_complex* const x[],
                                                   rocblas_int                        incx,
                                                   rocblas_float_complex* const       A[],
                                                   rocblas_int                        lda,
                                                   rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHER_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in A.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in A.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[inout]
    A         device array of device pointers storing the specified triangular portion of
              each hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each hermitian matrix A_i is supplied. The lower triangular portion
                of each A_i will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each hermitian matrix A_i is supplied. The upper triangular portion
                of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. Must be at least max(1, n).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zher_batched(rocblas_handle                      handle,
                                                   rocblas_fill                        uplo,
                                                   rocblas_int                         n,
                                                   const double*                       alpha,
                                                   const rocblas_double_complex* const x[],
                                                   rocblas_int                         incx,
                                                   rocblas_double_complex* const       A[],
                                                   rocblas_int                         lda,
                                                   rocblas_int                         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cher_strided_batched(rocblas_handle               handle,
                                                           rocblas_fill                 uplo,
                                                           rocblas_int                  n,
                                                           const float*                 alpha,
                                                           const rocblas_float_complex* x,
                                                           rocblas_int                  incx,
                                                           rocblas_stride               stride_x,
                                                           rocblas_float_complex*       A,
                                                           rocblas_int                  lda,
                                                           rocblas_stride               stride_A,
                                                           rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHER_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n hermitian matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in A.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in A.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[inout]
    A         device array of device pointers storing the specified triangular portion of
              each hermitian matrix A_i. Points to the first matrix (A_1).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each hermitian matrix A_i is supplied. The lower triangular
                portion of each A_i will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each hermitian matrix A_i is supplied. The upper triangular
                portion of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zher_strided_batched(rocblas_handle                handle,
                                                           rocblas_fill                  uplo,
                                                           rocblas_int                   n,
                                                           const double*                 alpha,
                                                           const rocblas_double_complex* x,
                                                           rocblas_int                   incx,
                                                           rocblas_stride                stride_x,
                                                           rocblas_double_complex*       A,
                                                           rocblas_int                   lda,
                                                           rocblas_stride                stride_A,
                                                           rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cher2(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* y,
                                            rocblas_int                  incy,
                                            rocblas_float_complex*       A,
                                            rocblas_int                  lda);

/*! \brief BLAS Level 2 API

    \details
    xHER2 performs the matrix-vector operations

        A := A + alpha*x*y**H + conj(alpha)*y*x**H

    where alpha is a complex scalar, x and y are vectors, and A is an
    n by n hermitian matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied.
              rocblas_fill_lower: The lower triangular part of A is supplied.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    A         device pointer storing the specified triangular portion of
              the hermitian matrix A. Of size (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the hermitian matrix A is supplied. The lower triangular
                portion of A will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the hermitian matrix A is supplied. The upper triangular
                portion of A will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. Must be at least max(lda, 1).
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zher2(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* y,
                                            rocblas_int                   incy,
                                            rocblas_double_complex*       A,
                                            rocblas_int                   lda);

ROCBLAS_EXPORT rocblas_status rocblas_cher2_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex* const y[],
                                                    rocblas_int                        incy,
                                                    rocblas_float_complex* const       A[],
                                                    rocblas_int                        lda,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHER2_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n hermitian matrix for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[inout]
    A         device array of device pointers storing the specified triangular portion of
              each hermitian matrix A_i of size (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each hermitian matrix A_i is supplied. The lower triangular
                portion of each A_i will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each hermitian matrix A_i is supplied. The upper triangular
                portion of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. Must be at least max(lda, 1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zher2_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex* const y[],
                                                    rocblas_int                         incy,
                                                    rocblas_double_complex* const       A[],
                                                    rocblas_int                         lda,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cher2_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            const rocblas_float_complex* y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stride_y,
                                                            rocblas_float_complex*       A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            rocblas_int batch_count);
/*! \brief BLAS Level 2 API

    \details
    xHER2_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n hermitian matrix for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              specifies the stride between the beginning of one vector (x_i) and the next (x_i+1).
    @param[in]
    y         device pointer pointing to the first vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    stride_y  [rocblas_stride]
              specifies the stride between the beginning of one vector (y_i) and the next (y_i+1).
    @param[inout]
    A         device pointer pointing to the first matrix (A_1). Stores the specified triangular portion of
              each hermitian matrix A_i.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each hermitian matrix A_i is supplied. The lower triangular
                portion of each A_i will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each hermitian matrix A_i is supplied. The upper triangular
                portion of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. Must be at least max(lda, 1).
    @param[in]
    stride_A  [rocblas_stride]
              specifies the stride between the beginning of one matrix (A_i) and the next (A_i+1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zher2_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            const rocblas_double_complex* y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stride_y,
                                                            rocblas_double_complex*       A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chpmv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* AP,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       y,
                                            rocblas_int                  incy);

/*! \brief BLAS Level 2 API

    \details
    xHPMV performs the matrix-vector operation

        y := alpha*A*x + beta*y

    where alpha and beta are scalars, x and y are n element vectors and A is an
    n by n hermitian matrix, supplied in packed form (see description below).

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of the hermitian matrix A is supplied in AP.
              rocblas_fill_lower: the lower triangular part of the hermitian matrix A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the order of the matrix A, must be >= 0.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    AP        device pointer storing the packed version of the specified triangular portion of
              the hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (3, 2)
                        (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
                        (3,-2) (5, 1) (6, 0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (3, 2)
                        (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
                        (3,-2) (5, 1) (6, 0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhpmv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* AP,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       y,
                                            rocblas_int                   incy);

ROCBLAS_EXPORT rocblas_status rocblas_chpmv_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const AP[],
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex* const       y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHPMV_BATCHED performs the matrix-vector operation

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n hermitian matrix, supplied in packed form (see description below),
    for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of each hermitian matrix A_i is supplied in AP.
              rocblas_fill_lower: the lower triangular part of each hermitian matrix A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    AP      device pointer of device pointers storing the packed version of the specified triangular
            portion of each hermitian matrix A_i. Each A_i is of at least size ((n * (n + 1)) / 2).
            if uplo == rocblas_fill_upper:
            The upper triangular portion of each hermitian matrix A_i is supplied.
            The matrix is compacted so that each AP_i contains the triangular portion column-by-column
            so that:
            AP(0) = A(0,0)
            AP(1) = A(0,1)
            AP(2) = A(1,1), etc.
                Ex: (rocblas_fill_upper; n = 3)
                    (1, 0) (2, 1) (3, 2)
                    (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
                    (3,-2) (5, 1) (6, 0)
        if uplo == rocblas_fill_lower:
            The lower triangular portion of each hermitian matrix A_i is supplied.
            The matrix is compacted so that each AP_i contains the triangular portion column-by-column
            so that:
            AP(0) = A(0,0)
            AP(1) = A(1,0)
            AP(2) = A(2,1), etc.
                Ex: (rocblas_fill_lower; n = 3)
                    (1, 0) (2, 1) (3, 2)
                    (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
                    (3,-2) (5, 1) (6, 0)
        Note that the imaginary part of the diagonal elements are not accessed and are assumed
        to be 0.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhpmv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const AP[],
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex* const       y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chpmv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* AP,
                                                            rocblas_stride               stride_A,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stride_y,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHPMV_STRIDED_BATCHED performs the matrix-vector operation

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n hermitian matrix, supplied in packed form (see description below),
    for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of each hermitian matrix A_i is supplied in AP.
              rocblas_fill_lower: the lower triangular part of each hermitian matrix A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    AP        device pointer pointing to the beginning of the first matrix (AP_1). Stores the packed
              version of the specified triangular portion of each hermitian matrix AP_i of size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that each AP_i contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (3, 2)
                        (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
                        (3,-2) (5, 1) (6, 0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that each AP_i contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (3, 2)
                        (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
                        (3,-2) (5, 1) (6, 0)
        Note that the imaginary part of the diagonal elements are not accessed and are assumed
        to be 0.
    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (AP_i) and the next one (AP_i+1).
    @param[in]
    x         device array pointing to the beginning of the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array pointing to the beginning of the first vector (y_1).
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhpmv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* AP,
                                                            rocblas_stride                stride_A,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stride_y,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chpr(rocblas_handle               handle,
                                           rocblas_fill                 uplo,
                                           rocblas_int                  n,
                                           const float*                 alpha,
                                           const rocblas_float_complex* x,
                                           rocblas_int                  incx,
                                           rocblas_float_complex*       AP);

/*! \brief BLAS Level 2 API

    \details
    xHPR performs the matrix-vector operations

        A := A + alpha*x*x**H

    where alpha is a real scalar, x is a vector, and A is an
    n by n hermitian matrix, supplied in packed form.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in AP.
              rocblas_fill_lower: The lower triangular part of A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              the hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhpr(rocblas_handle                handle,
                                           rocblas_fill                  uplo,
                                           rocblas_int                   n,
                                           const double*                 alpha,
                                           const rocblas_double_complex* x,
                                           rocblas_int                   incx,
                                           rocblas_double_complex*       AP);

ROCBLAS_EXPORT rocblas_status rocblas_chpr_batched(rocblas_handle                     handle,
                                                   rocblas_fill                       uplo,
                                                   rocblas_int                        n,
                                                   const float*                       alpha,
                                                   const rocblas_float_complex* const x[],
                                                   rocblas_int                        incx,
                                                   rocblas_float_complex* const       AP[],
                                                   rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHPR_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhpr_batched(rocblas_handle                      handle,
                                                   rocblas_fill                        uplo,
                                                   rocblas_int                         n,
                                                   const double*                       alpha,
                                                   const rocblas_double_complex* const x[],
                                                   rocblas_int                         incx,
                                                   rocblas_double_complex* const       AP[],
                                                   rocblas_int                         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chpr_strided_batched(rocblas_handle               handle,
                                                           rocblas_fill                 uplo,
                                                           rocblas_int                  n,
                                                           const float*                 alpha,
                                                           const rocblas_float_complex* x,
                                                           rocblas_int                  incx,
                                                           rocblas_stride               stride_x,
                                                           rocblas_float_complex*       AP,
                                                           rocblas_stride               stride_A,
                                                           rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHPR_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each hermitian matrix A_i. Points to the first matrix (A_1).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhpr_strided_batched(rocblas_handle                handle,
                                                           rocblas_fill                  uplo,
                                                           rocblas_int                   n,
                                                           const double*                 alpha,
                                                           const rocblas_double_complex* x,
                                                           rocblas_int                   incx,
                                                           rocblas_stride                stride_x,
                                                           rocblas_double_complex*       AP,
                                                           rocblas_stride                stride_A,
                                                           rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chpr2(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* y,
                                            rocblas_int                  incy,
                                            rocblas_float_complex*       AP);

/*! \brief BLAS Level 2 API

    \details
    xHPR2 performs the matrix-vector operations

        A := A + alpha*x*y**H + conj(alpha)*y*x**H

    where alpha is a complex scalar, x and y are vectors, and A is an
    n by n hermitian matrix, supplied in packed form.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in AP.
              rocblas_fill_lower: The lower triangular part of A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              the hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhpr2(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* y,
                                            rocblas_int                   incy,
                                            rocblas_double_complex*       AP);

ROCBLAS_EXPORT rocblas_status rocblas_chpr2_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex* const y[],
                                                    rocblas_int                        incy,
                                                    rocblas_float_complex* const       AP[],
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHPR2_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhpr2_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex* const y[],
                                                    rocblas_int                         incy,
                                                    rocblas_double_complex* const       AP[],
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chpr2_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            const rocblas_float_complex* y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stride_y,
                                                            rocblas_float_complex*       AP,
                                                            rocblas_stride               stride_A,
                                                            rocblas_int batch_count);
/*! \brief BLAS Level 2 API

    \details
    xHPR2_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[in]
    y         device pointer pointing to the first vector (y_1).
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1).
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each hermitian matrix A_i. Points to the first matrix (A_1).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhpr2_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            const rocblas_double_complex* y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stride_y,
                                                            rocblas_double_complex*       AP,
                                                            rocblas_stride                stride_A,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strmv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            const float*      A,
                                            rocblas_int       lda,
                                            float*            x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_dtrmv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            const double*     A,
                                            rocblas_int       lda,
                                            double*           x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_ctrmv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            transA,
                                            rocblas_diagonal             diag,
                                            rocblas_int                  m,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            rocblas_float_complex*       x,
                                            rocblas_int                  incx);

/*! \brief BLAS Level 2 API

    \details
    trmv performs one of the matrix-vector operations

         x = A*x or x = A**T*x,

    where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix.

    The vector x is overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of A. m >= 0.

    @param[in]
    A         device pointer storing matrix A,
              of dimension ( lda, m )

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
              lda = max( 1, m ).

    @param[in]
    x         device pointer storing vector x.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrmv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             transA,
                                            rocblas_diagonal              diag,
                                            rocblas_int                   m,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            rocblas_double_complex*       x,
                                            rocblas_int                   incx);

ROCBLAS_EXPORT rocblas_status rocblas_strmv_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_operation   transA,
                                                    rocblas_diagonal    diag,
                                                    rocblas_int         m,
                                                    const float* const* A,
                                                    rocblas_int         lda,
                                                    float* const*       x,
                                                    rocblas_int         incx,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrmv_batched(rocblas_handle       handle,
                                                    rocblas_fill         uplo,
                                                    rocblas_operation    transA,
                                                    rocblas_diagonal     diag,
                                                    rocblas_int          m,
                                                    const double* const* A,
                                                    rocblas_int          lda,
                                                    double* const*       x,
                                                    rocblas_int          incx,
                                                    rocblas_int          batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrmv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   transA,
                                                    rocblas_diagonal                    diag,
                                                    rocblas_int                         m,
                                                    const rocblas_float_complex* const* A,
                                                    rocblas_int                         lda,
                                                    rocblas_float_complex* const*       x,
                                                    rocblas_int                         incx,
                                                    rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    trmv_batched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batch_count

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)

    The vectors x_i are overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of matrices A_i. m >= 0.

    @param[in]
    A         device pointer storing pointer of matrices A_i,
              of dimension ( lda, m )

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A_i.
              lda >= max( 1, m ).

    @param[in]
    x         device pointer storing vectors x_i.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of vectors x_i.

    @param[in]
    batch_count [rocblas_int]
              The number of batched matrices/vectors.


    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrmv_batched(rocblas_handle                       handle,
                                                    rocblas_fill                         uplo,
                                                    rocblas_operation                    transA,
                                                    rocblas_diagonal                     diag,
                                                    rocblas_int                          m,
                                                    const rocblas_double_complex* const* A,
                                                    rocblas_int                          lda,
                                                    rocblas_double_complex* const*       x,
                                                    rocblas_int                          incx,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strmv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stridea,
                                                            float*            x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stridex,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrmv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stridea,
                                                            double*           x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stridex,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrmv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            transA,
                                                            rocblas_diagonal             diag,
                                                            rocblas_int                  m,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stridea,
                                                            rocblas_float_complex*       x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    trmv_strided_batched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batch_count

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
    with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).

    The vectors x_i are overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of matrices A_i. m >= 0.

    @param[in]
    A         device pointer of the matrix A_0,
              of dimension ( lda, m )

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A_i.
              lda >= max( 1, m ).

    @param[in]
    stride_a  [rocblas_stride]
              stride from the start of one A_i matrix to the next A_{i + 1}

    @param[in]
    x         device pointer storing the vector x_0.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of one vector x.

    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one x_i vector to the next x_{i + 1}

    @param[in]
    batch_count [rocblas_int]
              The number of batched matrices/vectors.


    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrmv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             transA,
                                                            rocblas_diagonal              diag,
                                                            rocblas_int                   m,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stridea,
                                                            rocblas_double_complex*       x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_stpmv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            const float*      A,
                                            float*            x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_dtpmv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            const double*     A,
                                            double*           x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_ctpmv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            transA,
                                            rocblas_diagonal             diag,
                                            rocblas_int                  m,
                                            const rocblas_float_complex* A,
                                            rocblas_float_complex*       x,
                                            rocblas_int                  incx);
/*! \brief BLAS Level 2 API

    \details
    tpmv performs one of the matrix-vector operations

         x = A*x or x = A**T*x,

    where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix, supplied in the pack form.

    The vector x is overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of A. m >= 0.

    @param[in]
    A       device pointer storing matrix A,
            of dimension at leat ( m * ( m + 1 ) / 2 ).
          Before entry with uplo = rocblas_fill_upper, the array A
          must contain the upper triangular matrix packed sequentially,
          column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
          a_{0,1} and a_{1, 1} respectively, and so on.
          Before entry with uplo = rocblas_fill_lower, the array A
          must contain the lower triangular matrix packed sequentially,
          column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
          a_{1,0} and a_{2,0} respectively, and so on.
          Note that when DIAG = rocblas_diagonal_unit, the diagonal elements of A are
          not referenced, but are assumed to be unity.

    @param[in]
    x       device pointer storing vector x.

    @param[in]
    incx    [rocblas_int]
            specifies the increment for the elements of x. incx must not be zero.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ztpmv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             transA,
                                            rocblas_diagonal              diag,
                                            rocblas_int                   m,
                                            const rocblas_double_complex* A,
                                            rocblas_double_complex*       x,
                                            rocblas_int                   incx);

ROCBLAS_EXPORT rocblas_status rocblas_stpmv_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_operation   transA,
                                                    rocblas_diagonal    diag,
                                                    rocblas_int         m,
                                                    const float* const* A,
                                                    float* const*       x,
                                                    rocblas_int         incx,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtpmv_batched(rocblas_handle       handle,
                                                    rocblas_fill         uplo,
                                                    rocblas_operation    transA,
                                                    rocblas_diagonal     diag,
                                                    rocblas_int          m,
                                                    const double* const* A,
                                                    double* const*       x,
                                                    rocblas_int          incx,
                                                    rocblas_int          batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctpmv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   transA,
                                                    rocblas_diagonal                    diag,
                                                    rocblas_int                         m,
                                                    const rocblas_float_complex* const* A,
                                                    rocblas_float_complex* const*       x,
                                                    rocblas_int                         incx,
                                                    rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    tpmv_batched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batch_count

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)

    The vectors x_i are overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of matrices A_i. m >= 0.

    @param[in]
    A         device pointer storing pointer of matrices A_i,
              of dimension ( lda, m )

    @param[in]
    x         device pointer storing vectors x_i.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of vectors x_i.

    @param[in]
    batch_count [rocblas_int]
              The number of batched matrices/vectors.


    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztpmv_batched(rocblas_handle                       handle,
                                                    rocblas_fill                         uplo,
                                                    rocblas_operation                    transA,
                                                    rocblas_diagonal                     diag,
                                                    rocblas_int                          m,
                                                    const rocblas_double_complex* const* A,
                                                    rocblas_double_complex* const*       x,
                                                    rocblas_int                          incx,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_stpmv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            const float*      A,
                                                            rocblas_stride    stridea,
                                                            float*            x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stridex,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtpmv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            const double*     A,
                                                            rocblas_stride    stridea,
                                                            double*           x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stridex,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctpmv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            transA,
                                                            rocblas_diagonal             diag,
                                                            rocblas_int                  m,
                                                            const rocblas_float_complex* A,
                                                            rocblas_stride               stridea,
                                                            rocblas_float_complex*       x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            rocblas_int batch_count);
/*! \brief BLAS Level 2 API

    \details
    tpmv_strided_batched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batch_count

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
    with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).

    The vectors x_i are overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of matrices A_i. m >= 0.

    @param[in]
    A         device pointer of the matrix A_0,
              of dimension ( lda, m )

    @param[in]
    stride_a  [rocblas_stride]
              stride from the start of one A_i matrix to the next A_{i + 1}

    @param[in]
    x         device pointer storing the vector x_0.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of one vector x.

    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one x_i vector to the next x_{i + 1}

    @param[in]
    batch_count [rocblas_int]
              The number of batched matrices/vectors.


    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ztpmv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             transA,
                                                            rocblas_diagonal              diag,
                                                            rocblas_int                   m,
                                                            const rocblas_double_complex* A,
                                                            rocblas_stride                stridea,
                                                            rocblas_double_complex*       x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_stbmv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation trans,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            rocblas_int       k,
                                            const float*      A,
                                            rocblas_int       lda,
                                            float*            x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_dtbmv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation trans,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            rocblas_int       k,
                                            const double*     A,
                                            rocblas_int       lda,
                                            double*           x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_ctbmv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            trans,
                                            rocblas_diagonal             diag,
                                            rocblas_int                  m,
                                            rocblas_int                  k,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            rocblas_float_complex*       x,
                                            rocblas_int                  incx);

/*! \brief BLAS Level 2 API

    \details
    xTBMV performs one of the matrix-vector operations

        x := A*x      or
        x := A**T*x   or
        x := A**H*x,

    x is a vectors and A is a banded m by m matrix (see description below).

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: A is an upper banded triangular matrix.
              rocblas_fill_lower: A is a  lower banded triangular matrix.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not.
    @param[in]
    diag      [rocblas_diagonal]
              rocblas_diagonal_unit: The main diagonal of A is assumed to consist of only
                                     1's and is not referenced.
              rocblas_diagonal_non_unit: No assumptions are made of A's main diagonal.
    @param[in]
    m         [rocblas_int]
              the number of rows and columns of the matrix represented by A.
    @param[in]
    k         [rocblas_int]
              if uplo == rocblas_fill_upper, k specifies the number of super-diagonals
              of the matrix A.
              if uplo == rocblas_fill_lower, k specifies the number of sub-diagonals
              of the matrix A.
              k must satisfy k > 0 && k < lda.
    @param[in]
    A         device pointer storing banded triangular matrix A.
              if uplo == rocblas_fill_upper:
                The matrix represented is an upper banded triangular matrix
                with the main diagonal and k super-diagonals, everything
                else can be assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the k'th
                row, the first super diagonal resides on the RHS of the k-1'th row, etc,
                with the k'th diagonal on the RHS of the 0'th row.
                   Ex: (rocblas_fill_upper; m = 5; k = 2)
                      1 6 9 0 0              0 0 9 8 7
                      0 2 7 8 0              0 6 7 8 9
                      0 0 3 8 7     ---->    1 2 3 4 5
                      0 0 0 4 9              0 0 0 0 0
                      0 0 0 0 5              0 0 0 0 0
              if uplo == rocblas_fill_lower:
                The matrix represnted is a lower banded triangular matrix
                with the main diagonal and k sub-diagonals, everything else can be
                assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the 0'th row,
                working up to the k'th diagonal residing on the LHS of the k'th row.
                   Ex: (rocblas_fill_lower; m = 5; k = 2)
                      1 0 0 0 0              1 2 3 4 5
                      6 2 0 0 0              6 7 8 9 0
                      9 7 3 0 0     ---->    9 8 7 0 0
                      0 8 8 4 0              0 0 0 0 0
                      0 0 7 9 5              0 0 0 0 0
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. lda must satisfy lda > k.
    @param[inout]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztbmv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             trans,
                                            rocblas_diagonal              diag,
                                            rocblas_int                   m,
                                            rocblas_int                   k,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            rocblas_double_complex*       x,
                                            rocblas_int                   incx);

ROCBLAS_EXPORT rocblas_status rocblas_stbmv_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_operation  trans,
                                                    rocblas_diagonal   diag,
                                                    rocblas_int        m,
                                                    rocblas_int        k,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    float* const       x[],
                                                    rocblas_int        incx,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtbmv_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_operation   trans,
                                                    rocblas_diagonal    diag,
                                                    rocblas_int         m,
                                                    rocblas_int         k,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    double* const       x[],
                                                    rocblas_int         incx,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctbmv_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_operation                  trans,
                                                    rocblas_diagonal                   diag,
                                                    rocblas_int                        m,
                                                    rocblas_int                        k,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        da,
                                                    rocblas_float_complex* const       x[],
                                                    rocblas_int                        incx,
                                                    rocblas_int                        batch_count);
/*! \brief BLAS Level 2 API

    \details
    xTBMV_BATCHED performs one of the matrix-vector operations

        x_i := A_i*x_i      or
        x_i := A_i**T*x_i   or
        x_i := A_i**H*x_i,

    where (A_i, x_i) is the i-th instance of the batch.
    x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: each A_i is an upper banded triangular matrix.
              rocblas_fill_lower: each A_i is a  lower banded triangular matrix.
    @param[in]
    trans     [rocblas_operation]
              indicates whether each matrix A_i is tranposed (conjugated) or not.
    @param[in]
    diag      [rocblas_diagonal]
              rocblas_diagonal_unit: The main diagonal of each A_i is assumed to consist of only
                                     1's and is not referenced.
              rocblas_diagonal_non_unit: No assumptions are made of each A_i's main diagonal.
    @param[in]
    m         [rocblas_int]
              the number of rows and columns of the matrix represented by each A_i.
    @param[in]
    k         [rocblas_int]
              if uplo == rocblas_fill_upper, k specifies the number of super-diagonals
              of each matrix A_i.
              if uplo == rocblas_fill_lower, k specifies the number of sub-diagonals
              of each matrix A_i.
              k must satisfy k > 0 && k < lda.
    @param[in]
    A         device array of device pointers storing each banded triangular matrix A_i.
              if uplo == rocblas_fill_upper:
                The matrix represented is an upper banded triangular matrix
                with the main diagonal and k super-diagonals, everything
                else can be assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the k'th
                row, the first super diagonal resides on the RHS of the k-1'th row, etc,
                with the k'th diagonal on the RHS of the 0'th row.
                   Ex: (rocblas_fill_upper; m = 5; k = 2)
                      1 6 9 0 0              0 0 9 8 7
                      0 2 7 8 0              0 6 7 8 9
                      0 0 3 8 7     ---->    1 2 3 4 5
                      0 0 0 4 9              0 0 0 0 0
                      0 0 0 0 5              0 0 0 0 0
              if uplo == rocblas_fill_lower:
                The matrix represnted is a lower banded triangular matrix
                with the main diagonal and k sub-diagonals, everything else can be
                assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the 0'th row,
                working up to the k'th diagonal residing on the LHS of the k'th row.
                   Ex: (rocblas_fill_lower; m = 5; k = 2)
                      1 0 0 0 0              1 2 3 4 5
                      6 2 0 0 0              6 7 8 9 0
                      9 7 3 0 0     ---->    9 8 7 0 0
                      0 8 8 4 0              0 0 0 0 0
                      0 0 7 9 5              0 0 0 0 0
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. lda must satisfy lda > k.
    @param[inout]
    x         device array of device pointer storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ztbmv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   trans,
                                                    rocblas_diagonal                    diag,
                                                    rocblas_int                         m,
                                                    rocblas_int                         k,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    rocblas_double_complex* const       x[],
                                                    rocblas_int                         incx,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_stbmv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation trans,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            rocblas_int       k,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            float*            x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtbmv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation trans,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            rocblas_int       k,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            double*           x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctbmv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            trans,
                                                            rocblas_diagonal             diag,
                                                            rocblas_int                  m,
                                                            rocblas_int                  k,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            rocblas_float_complex*       x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            rocblas_int batch_count);
/*! \brief BLAS Level 2 API

    \details
    xTBMV_STRIDED_BATCHED performs one of the matrix-vector operations

        x_i := A_i*x_i      or
        x_i := A_i**T*x_i   or
        x_i := A_i**H*x_i,

    where (A_i, x_i) is the i-th instance of the batch.
    x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: each A_i is an upper banded triangular matrix.
              rocblas_fill_lower: each A_i is a  lower banded triangular matrix.
    @param[in]
    trans     [rocblas_operation]
              indicates whether each matrix A_i is tranposed (conjugated) or not.
    @param[in]
    diag      [rocblas_diagonal]
              rocblas_diagonal_unit: The main diagonal of each A_i is assumed to consist of only
                                     1's and is not referenced.
              rocblas_diagonal_non_unit: No assumptions are made of each A_i's main diagonal.
    @param[in]
    m         [rocblas_int]
              the number of rows and columns of the matrix represented by each A_i.
    @param[in]
    k         [rocblas_int]
              if uplo == rocblas_fill_upper, k specifies the number of super-diagonals
              of each matrix A_i.
              if uplo == rocblas_fill_lower, k specifies the number of sub-diagonals
              of each matrix A_i.
              k must satisfy k > 0 && k < lda.
    @param[in]
    A         device array to the first matrix A_i of the batch. Stores each banded triangular matrix A_i.
              if uplo == rocblas_fill_upper:
                The matrix represented is an upper banded triangular matrix
                with the main diagonal and k super-diagonals, everything
                else can be assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the k'th
                row, the first super diagonal resides on the RHS of the k-1'th row, etc,
                with the k'th diagonal on the RHS of the 0'th row.
                   Ex: (rocblas_fill_upper; m = 5; k = 2)
                      1 6 9 0 0              0 0 9 8 7
                      0 2 7 8 0              0 6 7 8 9
                      0 0 3 8 7     ---->    1 2 3 4 5
                      0 0 0 4 9              0 0 0 0 0
                      0 0 0 0 5              0 0 0 0 0
              if uplo == rocblas_fill_lower:
                The matrix represnted is a lower banded triangular matrix
                with the main diagonal and k sub-diagonals, everything else can be
                assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the 0'th row,
                working up to the k'th diagonal residing on the LHS of the k'th row.
                   Ex: (rocblas_fill_lower; m = 5; k = 2)
                      1 0 0 0 0              1 2 3 4 5
                      6 2 0 0 0              6 7 8 9 0
                      9 7 3 0 0     ---->    9 8 7 0 0
                      0 8 8 4 0              0 0 0 0 0
                      0 0 7 9 5              0 0 0 0 0
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. lda must satisfy lda > k.
    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one A_i matrix to the next A_(i + 1).
    @param[inout]
    x         device array to the first vector x_i of the batch.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one x_i matrix to the next x_(i + 1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ztbmv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             trans,
                                                            rocblas_diagonal              diag,
                                                            rocblas_int                   m,
                                                            rocblas_int                   k,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            rocblas_double_complex*       x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_stbsv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const float*      A,
                                            rocblas_int       lda,
                                            float*            x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_dtbsv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const double*     A,
                                            rocblas_int       lda,
                                            double*           x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_ctbsv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            transA,
                                            rocblas_diagonal             diag,
                                            rocblas_int                  n,
                                            rocblas_int                  k,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            rocblas_float_complex*       x,
                                            rocblas_int                  incx);
/*! \brief BLAS Level 2 API

    \details
    tbsv solves

         A*x = b or A**T*x = b or A**H*x = b,

    where x and b are vectors and A is a banded triangular matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]
               rocblas_operation_none: Solves A*x = b
               rocblas_operation_transpose: Solves A**T*x = b
               rocblas_operation_conjugate_transpose: Solves A**H*x = b

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular (i.e. the diagonal elements
                                       of A are not used in computations).
            rocblas_diagonal_non_unit: A is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of b. n >= 0.
    @param[in]
    k         [rocblas_int]
              if(uplo == rocblas_fill_upper)
                k specifies the number of super-diagonals of A.
              if(uplo == rocblas_fill_lower)
                k specifies the number of sub-diagonals of A.
              k >= 0.

    @param[in]
    A         device pointer storing the matrix A in banded format.

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
              lda >= (k + 1).

    @param[inout]
    x         device pointer storing input vector b. Overwritten by the output vector x.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ztbsv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             transA,
                                            rocblas_diagonal              diag,
                                            rocblas_int                   n,
                                            rocblas_int                   k,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            rocblas_double_complex*       x,
                                            rocblas_int                   incx);

ROCBLAS_EXPORT rocblas_status rocblas_stbsv_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_operation  transA,
                                                    rocblas_diagonal   diag,
                                                    rocblas_int        n,
                                                    rocblas_int        k,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    float* const       x[],
                                                    rocblas_int        incx,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtbsv_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_operation   transA,
                                                    rocblas_diagonal    diag,
                                                    rocblas_int         n,
                                                    rocblas_int         k,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    double* const       x[],
                                                    rocblas_int         incx,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctbsv_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_operation                  transA,
                                                    rocblas_diagonal                   diag,
                                                    rocblas_int                        n,
                                                    rocblas_int                        k,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    rocblas_float_complex* const       x[],
                                                    rocblas_int                        incx,
                                                    rocblas_int                        batch_count);
/*! \brief BLAS Level 2 API

    \details
    tbsv_batched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a banded triangular matrix,
    for i = [1, batch_count].

    The input vectors b_i are overwritten by the output vectors x_i.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]
               rocblas_operation_none: Solves A_i*x_i = b_i
               rocblas_operation_transpose: Solves A_i**T*x_i = b_i
               rocblas_operation_conjugate_transpose: Solves A_i**H*x_i = b_i

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
                                       of each A_i are not used in computations).
            rocblas_diagonal_non_unit: each A_i is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of each b_i. n >= 0.
    @param[in]
    k         [rocblas_int]
              if(uplo == rocblas_fill_upper)
                k specifies the number of super-diagonals of each A_i.
              if(uplo == rocblas_fill_lower)
                k specifies the number of sub-diagonals of each A_i.
              k >= 0.

    @param[in]
    A         device vector of device pointers storing each matrix A_i in banded format.

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
              lda >= (k + 1).

    @param[inout]
    x         device vector of device pointers storing each input vector b_i. Overwritten by each output
              vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ztbsv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   transA,
                                                    rocblas_diagonal                    diag,
                                                    rocblas_int                         n,
                                                    rocblas_int                         k,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    rocblas_double_complex* const       x[],
                                                    rocblas_int                         incx,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_stbsv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       n,
                                                            rocblas_int       k,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            float*            x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtbsv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       n,
                                                            rocblas_int       k,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            double*           x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctbsv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            transA,
                                                            rocblas_diagonal             diag,
                                                            rocblas_int                  n,
                                                            rocblas_int                  k,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            rocblas_float_complex*       x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    tbsv_strided_batched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a banded triangular matrix,
    for i = [1, batch_count].

    The input vectors b_i are overwritten by the output vectors x_i.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]
               rocblas_operation_none: Solves A_i*x_i = b_i
               rocblas_operation_transpose: Solves A_i**T*x_i = b_i
               rocblas_operation_conjugate_transpose: Solves A_i**H*x_i = b_i

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
                                       of each A_i are not used in computations).
            rocblas_diagonal_non_unit: each A_i is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of each b_i. n >= 0.
    @param[in]
    k         [rocblas_int]
              if(uplo == rocblas_fill_upper)
                k specifies the number of super-diagonals of each A_i.
              if(uplo == rocblas_fill_lower)
                k specifies the number of sub-diagonals of each A_i.
              k >= 0.

    @param[in]
    A         device pointer pointing to the first banded matrix A_1.

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
              lda >= (k + 1).
    @param[in]
    stride_A  [rocblas_stride]
              specifies the distance between the start of one matrix (A_i) and the next (A_i+1).

    @param[inout]
    x         device pointer pointing to the first input vector b_1. Overwritten by output vectors x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              specifies the distance between the start of one vector (x_i) and the next (x_i+1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztbsv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             transA,
                                                            rocblas_diagonal              diag,
                                                            rocblas_int                   n,
                                                            rocblas_int                   k,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            rocblas_double_complex*       x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strsv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            const float*      A,
                                            rocblas_int       lda,
                                            float*            x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_dtrsv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            const double*     A,
                                            rocblas_int       lda,
                                            double*           x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_ctrsv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            transA,
                                            rocblas_diagonal             diag,
                                            rocblas_int                  m,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            rocblas_float_complex*       x,
                                            rocblas_int                  incx);

/*! \brief BLAS Level 2 API

    \details
    trsv solves

         A*x = b or A**T*x = b,

    where x and b are vectors and A is a triangular matrix.

    The vector x is overwritten on b.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of b. m >= 0.

    @param[in]
    A         device pointer storing matrix A,
              of dimension ( lda, m )

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
              lda = max( 1, m ).

    @param[in]
    x         device pointer storing vector x.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrsv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             transA,
                                            rocblas_diagonal              diag,
                                            rocblas_int                   m,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            rocblas_double_complex*       x,
                                            rocblas_int                   incx);

ROCBLAS_EXPORT rocblas_status rocblas_strsv_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_operation  transA,
                                                    rocblas_diagonal   diag,
                                                    rocblas_int        m,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    float* const       x[],
                                                    rocblas_int        incx,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrsv_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_operation   transA,
                                                    rocblas_diagonal    diag,
                                                    rocblas_int         m,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    double* const       x[],
                                                    rocblas_int         incx,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrsv_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_operation                  transA,
                                                    rocblas_diagonal                   diag,
                                                    rocblas_int                        m,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    rocblas_float_complex* const       x[],
                                                    rocblas_int                        incx,
                                                    rocblas_int                        batch_count);
/*! \brief BLAS Level 2 API

    \details
    trsv_batched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i,

    where (A_i, x_i, b_i) is the i-th instance of the batch.
    x_i and b_i are vectors and A_i is an
    m by m triangular matrix.

    The vector x is overwritten on b.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of b. m >= 0.

    @param[in]
    A         device array of device pointers storing each matrix A_i.

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
              lda = max(1, m)

    @param[in]
    x         device array of device pointers storing each vector x_i.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ztrsv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   transA,
                                                    rocblas_diagonal                    diag,
                                                    rocblas_int                         m,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    rocblas_double_complex* const       x[],
                                                    rocblas_int                         incx,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strsv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            float*            x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrsv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            double*           x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrsv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            transA,
                                                            rocblas_diagonal             diag,
                                                            rocblas_int                  m,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            rocblas_float_complex*       x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    trsv_strided_batched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i,

    where (A_i, x_i, b_i) is the i-th instance of the batch.
    x_i and b_i are vectors and A_i is an m by m triangular matrix, for i = 1, ..., batch_count.

    The vector x is overwritten on b.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of each b_i. m >= 0.

    @param[in]
    A         device pointer to the first matrix (A_1) in the batch, of dimension ( lda, m )

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one A_i matrix to the next A_(i + 1)

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
              lda = max( 1, m ).

    @param[in, out]
    x         device pointer to the first vector (x_1) in the batch.

    @param[in]
    stride_x [rocblas_stride]
             stride from the start of one x_i vector to the next x_(i + 1)

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrsv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             transA,
                                                            rocblas_diagonal              diag,
                                                            rocblas_int                   m,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            rocblas_double_complex*       x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_stpsv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       n,
                                            const float*      AP,
                                            float*            x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_dtpsv(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       n,
                                            const double*     AP,
                                            double*           x,
                                            rocblas_int       incx);

ROCBLAS_EXPORT rocblas_status rocblas_ctpsv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            transA,
                                            rocblas_diagonal             diag,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* AP,
                                            rocblas_float_complex*       x,
                                            rocblas_int                  incx);

/*! \brief BLAS Level 2 API

    \details
    TPSV solves

         A*x = b or A**T*x = b, or A**H*x = b,

    where x and b are vectors and A is a triangular matrix stored in the packed format.

    The input vector b is overwritten by the output vector x.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_none: Solves A*x = b
            rocblas_operation_transpose: Solves A**T*x = b
            rocblas_operation_conjugate_transpose: Solves A**H*x = b

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular (i.e. the diagonal elements
                                       of A are not used in computations).
            rocblas_diagonal_non_unit: A is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of b. n >= 0.

    @param[in]
    AP        device pointer storing the packed version of matrix A,
              of dimension >= (n * (n + 1) / 2)

    @param[inout]
    x         device pointer storing vector b on input, overwritten by x on output.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztpsv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             transA,
                                            rocblas_diagonal              diag,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* AP,
                                            rocblas_double_complex*       x,
                                            rocblas_int                   incx);

ROCBLAS_EXPORT rocblas_status rocblas_stpsv_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_operation  transA,
                                                    rocblas_diagonal   diag,
                                                    rocblas_int        n,
                                                    const float* const AP[],
                                                    float* const       x[],
                                                    rocblas_int        incx,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtpsv_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_operation   transA,
                                                    rocblas_diagonal    diag,
                                                    rocblas_int         n,
                                                    const double* const AP[],
                                                    double* const       x[],
                                                    rocblas_int         incx,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctpsv_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_operation                  transA,
                                                    rocblas_diagonal                   diag,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex* const AP[],
                                                    rocblas_float_complex* const       x[],
                                                    rocblas_int                        incx,
                                                    rocblas_int                        batch_count);
/*! \brief BLAS Level 2 API

    \details
    TPSV_BATCHED solves

         A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
    for i in [1, batch_count].

    The input vectors b_i are overwritten by the output vectors x_i.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  each A_i is an upper triangular matrix.
            rocblas_fill_lower:  each A_i is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_none: Solves A*x = b
            rocblas_operation_transpose: Solves A**T*x = b
            rocblas_operation_conjugate_transpose: Solves A**H*x = b

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
                                       of each A_i are not used in computations).
            rocblas_diagonal_non_unit: each A_i is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of each b_i. n >= 0.

    @param[in]
    AP        device array of device pointers storing the packed versions of each matrix A_i,
              of dimension >= (n * (n + 1) / 2)

    @param[inout]
    x         device array of device pointers storing each input vector b_i, overwritten by x_i on output.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    batch_count [rocblas_int]
                specifies the number of instances in the batch.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ztpsv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   transA,
                                                    rocblas_diagonal                    diag,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex* const AP[],
                                                    rocblas_double_complex* const       x[],
                                                    rocblas_int                         incx,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_stpsv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       n,
                                                            const float*      AP,
                                                            rocblas_stride    stride_A,
                                                            float*            x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtpsv_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       n,
                                                            const double*     AP,
                                                            rocblas_stride    stride_A,
                                                            double*           x,
                                                            rocblas_int       incx,
                                                            rocblas_stride    stride_x,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctpsv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            transA,
                                                            rocblas_diagonal             diag,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* AP,
                                                            rocblas_stride               stride_A,
                                                            rocblas_float_complex*       x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 2 API

    \details
    TPSV_STRIDED_BATCHED solves

         A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
    for i in [1, batch_count].

    The input vectors b_i are overwritten by the output vectors x_i.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  each A_i is an upper triangular matrix.
            rocblas_fill_lower:  each A_i is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_none: Solves A*x = b
            rocblas_operation_transpose: Solves A**T*x = b
            rocblas_operation_conjugate_transpose: Solves A**H*x = b

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
                                       of each A_i are not used in computations).
            rocblas_diagonal_non_unit: each A_i is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of each b_i. n >= 0.

    @param[in]
    AP        device pointer pointing to the first packed matrix A_1,
              of dimension >= (n * (n + 1) / 2)

    @param[in]
    stride_A  [rocblas_stride]
              stride from the beginning of one packed matrix (AP_i) and the next (AP_i+1).

    @param[inout]
    x         device pointer pointing to the first input vector b_1. Overwritten by each x_i on output.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the beginning of one vector (x_i) and the next (x_i+1).
    @param[in]
    batch_count [rocblas_int]
                specifies the number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztpsv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             transA,
                                                            rocblas_diagonal              diag,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* AP,
                                                            rocblas_stride                stride_A,
                                                            rocblas_double_complex*       x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssymv(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            const float*   alpha,
                                            const float*   A,
                                            rocblas_int    lda,
                                            const float*   x,
                                            rocblas_int    incx,
                                            const float*   beta,
                                            float*         y,
                                            rocblas_int    incy);

ROCBLAS_EXPORT rocblas_status rocblas_dsymv(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            const double*  alpha,
                                            const double*  A,
                                            rocblas_int    lda,
                                            const double*  x,
                                            rocblas_int    incx,
                                            const double*  beta,
                                            double*        y,
                                            rocblas_int    incy);

ROCBLAS_EXPORT rocblas_status rocblas_csymv(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       y,
                                            rocblas_int                  incy);

/*! \brief BLAS Level 2 API

    \details
    xSYMV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A should contain an upper or lower triangular n by n symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
    @param[in]
    alpha
              specifies the scalar alpha
    @param[in]
    A         pointer storing matrix A on the GPU
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A
    @param[in]
    x         pointer storing vector x on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x
    @param[in]
    beta      specifies the scalar beta
    @param[out]
    y         pointer storing vector y on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsymv(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       y,
                                            rocblas_int                   incy);

ROCBLAS_EXPORT rocblas_status rocblas_ssymv_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    const float*       beta,
                                                    float*             y[],
                                                    rocblas_int        incy,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsymv_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    const double*       beta,
                                                    double*             y[],
                                                    rocblas_int         incy,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csymv_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex*             y[],
                                                    rocblas_int                        incy,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 2 API

    \details
    xSYMV_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.
    A a should contain an upper or lower triangular symmetric matrix
    and the opposing triangular part of A is not referenced

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         device array of device pointers storing each matrix A_i
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each matrix A_i
    @param[in]
    x         device array of device pointers storing each vector x_i
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         device array of device pointers storing each vector y_i
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsymv_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex*             y[],
                                                    rocblas_int                         incy,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssymv_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            const float*   alpha,
                                                            const float*   A,
                                                            rocblas_int    lda,
                                                            rocblas_stride strideA,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            const float*   beta,
                                                            float*         y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsymv_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            const double*  alpha,
                                                            const double*  A,
                                                            rocblas_int    lda,
                                                            rocblas_stride strideA,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            const double*  beta,
                                                            double*        y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csymv_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               strideA,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stridey,
                                                            rocblas_int batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSYMV_strided_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.
    A a should contain an upper or lower triangular symmetric matrix
    and the opposing triangular part of A is not referenced

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         Device pointer to the first matrix A_1 on the GPU
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each matrix A_i
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         Device pointer to the first vector x_1 on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stride_x, however the user should
                take care to ensure that stridex is of appropriate size.
                This typically means stridex >= n * incx. stridex should be non zero.
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         Device pointer to the first vector y_1 on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1).
                There are no restrictions placed on stride_y, however the user should
                take care to ensure that stridey is of appropriate size.
                This typically means stridey >= n * incy. stridey should be non zero.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsymv_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                strideA,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stridey,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sspmv(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            const float*   alpha,
                                            const float*   A,
                                            const float*   x,
                                            rocblas_int    incx,
                                            const float*   beta,
                                            float*         y,
                                            rocblas_int    incy);
/*! \brief BLAS Level 2 API

    \details
    xSPMV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A should contain an upper or lower triangular n by n packed symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
    @param[in]
    alpha
              specifies the scalar alpha
    @param[in]
    A         pointer storing matrix A on the GPU
    @param[in]
    x         pointer storing vector x on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x
    @param[in]
    beta      specifies the scalar beta
    @param[out]
    y         pointer storing vector y on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_dspmv(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            const double*  alpha,
                                            const double*  A,
                                            const double*  x,
                                            rocblas_int    incx,
                                            const double*  beta,
                                            double*        y,
                                            rocblas_int    incy);

ROCBLAS_EXPORT rocblas_status rocblas_sspmv_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    const float*       beta,
                                                    float*             y[],
                                                    rocblas_int        incy,
                                                    rocblas_int        batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSPMV_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.
    A should contain an upper or lower triangular n by n packed symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         device array of device pointers storing each matrix A_i
    @param[in]
    x         device array of device pointers storing each vector x_i
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         device array of device pointers storing each vector y_i
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_dspmv_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    const double*       beta,
                                                    double*             y[],
                                                    rocblas_int         incy,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sspmv_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            const float*   alpha,
                                                            const float*   A,
                                                            rocblas_stride strideA,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            const float*   beta,
                                                            float*         y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSPMV_strided_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.
    A should contain an upper or lower triangular n by n packed symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         Device pointer to the first matrix A_1 on the GPU
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         Device pointer to the first vector x_1 on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stridex, however the user should
                take care to ensure that stridex is of appropriate size.
                This typically means stridex >= n * incx. stridex should be non zero.
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         Device pointer to the first vector y_1 on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1).
                There are no restrictions placed on stridey, however the user should
                take care to ensure that stridey is of appropriate size.
                This typically means stridey >= n * incy. stridey should be non zero.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_dspmv_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            const double*  alpha,
                                                            const double*  A,
                                                            rocblas_stride strideA,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            const double*  beta,
                                                            double*        y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssbmv(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            rocblas_int    k,
                                            const float*   alpha,
                                            const float*   A,
                                            rocblas_int    lda,
                                            const float*   x,
                                            rocblas_int    incx,
                                            const float*   beta,
                                            float*         y,
                                            rocblas_int    incy);

/*! \brief BLAS Level 2 API

    \details
    xSBMV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A should contain an upper or lower triangular n by n symmetric banded matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
    @param[in]
    k         [rocblas_int]
              specifies the number of sub- and super-diagonals
    @param[in]
    alpha
              specifies the scalar alpha
    @param[in]
    A         pointer storing matrix A on the GPU
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of matrix A
    @param[in]
    x         pointer storing vector x on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x
    @param[in]
    beta      specifies the scalar beta
    @param[out]
    y         pointer storing vector y on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_dsbmv(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            rocblas_int    k,
                                            const double*  alpha,
                                            const double*  A,
                                            rocblas_int    lda,
                                            const double*  x,
                                            rocblas_int    incx,
                                            const double*  beta,
                                            double*        y,
                                            rocblas_int    incy);

ROCBLAS_EXPORT rocblas_status rocblas_dsbmv_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_int         n,
                                                    rocblas_int         k,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    const double*       beta,
                                                    double*             y[],
                                                    rocblas_int         incy,
                                                    rocblas_int         batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSBMV_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric banded matrix, for i = 1, ..., batch_count.
    A should contain an upper or lower triangular n by n symmetric banded matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    k         [rocblas_int]
              specifies the number of sub- and super-diagonals
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         device array of device pointers storing each matrix A_i
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each matrix A_i
    @param[in]
    x         device array of device pointers storing each vector x_i
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         device array of device pointers storing each vector y_i
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_ssbmv_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_int        n,
                                                    rocblas_int        k,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    const float*       beta,
                                                    float*             y[],
                                                    rocblas_int        incy,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssbmv_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            rocblas_int    k,
                                                            const float*   alpha,
                                                            const float*   A,
                                                            rocblas_int    lda,
                                                            rocblas_stride strideA,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            const float*   beta,
                                                            float*         y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

/*! \brief BLAS Level 2 API

    \details
    xSBMV_strided_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric banded matrix, for i = 1, ..., batch_count.
    A should contain an upper or lower triangular n by n symmetric banded matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    k         [rocblas_int]
              specifies the number of sub- and super-diagonals
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         Device pointer to the first matrix A_1 on the GPU
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each matrix A_i
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         Device pointer to the first vector x_1 on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stridex, however the user should
                take care to ensure that stridex is of appropriate size.
                This typically means stridex >= n * incx. stridex should be non zero.
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         Device pointer to the first vector y_1 on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1).
                There are no restrictions placed on stridey, however the user should
                take care to ensure that stridey is of appropriate size.
                This typically means stridey >= n * incy. stridey should be non zero.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_dsbmv_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            rocblas_int    k,
                                                            const double*  alpha,
                                                            const double*  A,
                                                            rocblas_int    lda,
                                                            rocblas_stride strideA,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            const double*  beta,
                                                            double*        y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sger(rocblas_handle handle,
                                           rocblas_int    m,
                                           rocblas_int    n,
                                           const float*   alpha,
                                           const float*   x,
                                           rocblas_int    incx,
                                           const float*   y,
                                           rocblas_int    incy,
                                           float*         A,
                                           rocblas_int    lda);

ROCBLAS_EXPORT rocblas_status rocblas_dger(rocblas_handle handle,
                                           rocblas_int    m,
                                           rocblas_int    n,
                                           const double*  alpha,
                                           const double*  x,
                                           rocblas_int    incx,
                                           const double*  y,
                                           rocblas_int    incy,
                                           double*        A,
                                           rocblas_int    lda);

ROCBLAS_EXPORT rocblas_status rocblas_cgeru(rocblas_handle               handle,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* y,
                                            rocblas_int                  incy,
                                            rocblas_float_complex*       A,
                                            rocblas_int                  lda);

ROCBLAS_EXPORT rocblas_status rocblas_zgeru(rocblas_handle                handle,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* y,
                                            rocblas_int                   incy,
                                            rocblas_double_complex*       A,
                                            rocblas_int                   lda);

ROCBLAS_EXPORT rocblas_status rocblas_cgerc(rocblas_handle               handle,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* y,
                                            rocblas_int                  incy,
                                            rocblas_float_complex*       A,
                                            rocblas_int                  lda);
/*! \brief BLAS Level 2 API

    \details
    xGER,xGERU,xGERC performs the matrix-vector operations

        A := A + alpha*x*y**T , OR
        A := A + alpha*x*y**H for xGERC

    where alpha is a scalar, x and y are vectors, and A is an
    m by n matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    m         [rocblas_int]
              the number of rows of the matrix A.
    @param[in]
    n         [rocblas_int]
              the number of columns of the matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zgerc(rocblas_handle                handle,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* y,
                                            rocblas_int                   incy,
                                            rocblas_double_complex*       A,
                                            rocblas_int                   lda);

ROCBLAS_EXPORT rocblas_status rocblas_sger_batched(rocblas_handle     handle,
                                                   rocblas_int        m,
                                                   rocblas_int        n,
                                                   const float*       alpha,
                                                   const float* const x[],
                                                   rocblas_int        incx,
                                                   const float* const y[],
                                                   rocblas_int        incy,
                                                   float* const       A[],
                                                   rocblas_int        lda,
                                                   rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dger_batched(rocblas_handle      handle,
                                                   rocblas_int         m,
                                                   rocblas_int         n,
                                                   const double*       alpha,
                                                   const double* const x[],
                                                   rocblas_int         incx,
                                                   const double* const y[],
                                                   rocblas_int         incy,
                                                   double* const       A[],
                                                   rocblas_int         lda,
                                                   rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgeru_batched(rocblas_handle                     handle,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex* const y[],
                                                    rocblas_int                        incy,
                                                    rocblas_float_complex* const       A[],
                                                    rocblas_int                        lda,
                                                    rocblas_int                        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_zgeru_batched(rocblas_handle                      handle,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex* const y[],
                                                    rocblas_int                         incy,
                                                    rocblas_double_complex* const       A[],
                                                    rocblas_int                         lda,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgerc_batched(rocblas_handle                     handle,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex* const y[],
                                                    rocblas_int                        incy,
                                                    rocblas_float_complex* const       A[],
                                                    rocblas_int                        lda,
                                                    rocblas_int                        batch_count);
/*! \brief BLAS Level 2 API

    \details
    xGER,xGERU,xGERC_BATCHED performs a batch of the matrix-vector operations

        A := A + alpha*x*y**T , OR
        A := A + alpha*x*y**H for xGERC

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha is a scalar, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    m         [rocblas_int]
              the number of rows of each matrix A_i.
    @param[in]
    n         [rocblas_int]
              the number of columns of eaceh matrix A_i.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i.
    @param[inout]
    A         device array of device pointers storing each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zgerc_batched(rocblas_handle                      handle,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex* const y[],
                                                    rocblas_int                         incy,
                                                    rocblas_double_complex* const       A[],
                                                    rocblas_int                         lda,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sger_strided_batched(rocblas_handle handle,
                                                           rocblas_int    m,
                                                           rocblas_int    n,
                                                           const float*   alpha,
                                                           const float*   x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stridex,
                                                           const float*   y,
                                                           rocblas_int    incy,
                                                           rocblas_stride stridey,
                                                           float*         A,
                                                           rocblas_int    lda,
                                                           rocblas_stride strideA,
                                                           rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dger_strided_batched(rocblas_handle handle,
                                                           rocblas_int    m,
                                                           rocblas_int    n,
                                                           const double*  alpha,
                                                           const double*  x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stridex,
                                                           const double*  y,
                                                           rocblas_int    incy,
                                                           rocblas_stride stridey,
                                                           double*        A,
                                                           rocblas_int    lda,
                                                           rocblas_stride strideA,
                                                           rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgeru_strided_batched(rocblas_handle               handle,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            const rocblas_float_complex* y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stridey,
                                                            rocblas_float_complex*       A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               strideA,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_zgeru_strided_batched(rocblas_handle                handle,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            const rocblas_double_complex* y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stridey,
                                                            rocblas_double_complex*       A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                strideA,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgerc_strided_batched(rocblas_handle               handle,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            const rocblas_float_complex* y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stridey,
                                                            rocblas_float_complex*       A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               strideA,
                                                            rocblas_int batch_count);
/*! \brief BLAS Level 2 API

    \details
    xGERC,xGERU,xGERC_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**T, OR
        A_i := A_i + alpha*x_i*y_i**H  for xGERC

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha is a scalar, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    m         [rocblas_int]
              the number of rows of each matrix A_i.
    @param[in]
    n         [rocblas_int]
              the number of columns of each matrix A_i.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer to the first vector (x_1) in the batch.
    @param[in]
    incx      [rocblas_int]
              specifies the increments for the elements of each vector x_i.
    @param[in]
    stridex   [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
              There are no restrictions placed on stride_x, however the user should
              take care to ensure that stride_x is of appropriate size, for a typical
              case this means stride_x >= m * incx.
    @param[inout]
    y         device pointer to the first vector (y_1) in the batch.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i.
    @param[in]
    stridey   [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1).
              There are no restrictions placed on stride_y, however the user should
              take care to ensure that stride_y is of appropriate size, for a typical
              case this means stride_y >= n * incy.
    @param[inout]
    A         device pointer to the first matrix (A_1) in the batch.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zgerc_strided_batched(rocblas_handle                handle,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            const rocblas_double_complex* y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stridey,
                                                            rocblas_double_complex*       A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                strideA,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sspr(rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           rocblas_int    n,
                                           const float*   alpha,
                                           const float*   x,
                                           rocblas_int    incx,
                                           float*         AP);

ROCBLAS_EXPORT rocblas_status rocblas_dspr(rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           rocblas_int    n,
                                           const double*  alpha,
                                           const double*  x,
                                           rocblas_int    incx,
                                           double*        AP);

ROCBLAS_EXPORT rocblas_status rocblas_cspr(rocblas_handle               handle,
                                           rocblas_fill                 uplo,
                                           rocblas_int                  n,
                                           const rocblas_float_complex* alpha,
                                           const rocblas_float_complex* x,
                                           rocblas_int                  incx,
                                           rocblas_float_complex*       AP);
/*! \brief BLAS Level 2 API

    \details
    xSPR performs the matrix-vector operations

        A := A + alpha*x*x**T

    where alpha is a scalar, x is a vector, and A is an
    n by n symmetric matrix, supplied in packed form.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in AP.
              rocblas_fill_lower: The lower triangular part of A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the symmetric matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the symmetric matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zspr(rocblas_handle                handle,
                                           rocblas_fill                  uplo,
                                           rocblas_int                   n,
                                           const rocblas_double_complex* alpha,
                                           const rocblas_double_complex* x,
                                           rocblas_int                   incx,
                                           rocblas_double_complex*       AP);

ROCBLAS_EXPORT rocblas_status rocblas_sspr_batched(rocblas_handle     handle,
                                                   rocblas_fill       uplo,
                                                   rocblas_int        n,
                                                   const float*       alpha,
                                                   const float* const x[],
                                                   rocblas_int        incx,
                                                   float* const       AP[],
                                                   rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dspr_batched(rocblas_handle      handle,
                                                   rocblas_fill        uplo,
                                                   rocblas_int         n,
                                                   const double*       alpha,
                                                   const double* const x[],
                                                   rocblas_int         incx,
                                                   double* const       AP[],
                                                   rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cspr_batched(rocblas_handle                     handle,
                                                   rocblas_fill                       uplo,
                                                   rocblas_int                        n,
                                                   const rocblas_float_complex*       alpha,
                                                   const rocblas_float_complex* const x[],
                                                   rocblas_int                        incx,
                                                   rocblas_float_complex* const       AP[],
                                                   rocblas_int                        batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSPR_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**T

    where alpha is a scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zspr_batched(rocblas_handle                      handle,
                                                   rocblas_fill                        uplo,
                                                   rocblas_int                         n,
                                                   const rocblas_double_complex*       alpha,
                                                   const rocblas_double_complex* const x[],
                                                   rocblas_int                         incx,
                                                   rocblas_double_complex* const       AP[],
                                                   rocblas_int                         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sspr_strided_batched(rocblas_handle handle,
                                                           rocblas_fill   uplo,
                                                           rocblas_int    n,
                                                           const float*   alpha,
                                                           const float*   x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stride_x,
                                                           float*         AP,
                                                           rocblas_stride stride_A,
                                                           rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dspr_strided_batched(rocblas_handle handle,
                                                           rocblas_fill   uplo,
                                                           rocblas_int    n,
                                                           const double*  alpha,
                                                           const double*  x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stride_x,
                                                           double*        AP,
                                                           rocblas_stride stride_A,
                                                           rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cspr_strided_batched(rocblas_handle               handle,
                                                           rocblas_fill                 uplo,
                                                           rocblas_int                  n,
                                                           const rocblas_float_complex* alpha,
                                                           const rocblas_float_complex* x,
                                                           rocblas_int                  incx,
                                                           rocblas_stride               stride_x,
                                                           rocblas_float_complex*       AP,
                                                           rocblas_stride               stride_A,
                                                           rocblas_int batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSPR_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**T

    where alpha is a scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              each symmetric matrix A_i. Points to the first A_1.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zspr_strided_batched(rocblas_handle                handle,
                                                           rocblas_fill                  uplo,
                                                           rocblas_int                   n,
                                                           const rocblas_double_complex* alpha,
                                                           const rocblas_double_complex* x,
                                                           rocblas_int                   incx,
                                                           rocblas_stride                stride_x,
                                                           rocblas_double_complex*       AP,
                                                           rocblas_stride                stride_A,
                                                           rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sspr2(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            const float*   alpha,
                                            const float*   x,
                                            rocblas_int    incx,
                                            const float*   y,
                                            rocblas_int    incy,
                                            float*         AP);
/*! \brief BLAS Level 2 API

    \details
    xSPR2 performs the matrix-vector operation

        A := A + alpha*x*y**T + alpha*y*x**T

    where alpha is a scalar, x and y are vectors, and A is an
    n by n symmetric matrix, supplied in packed form.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in AP.
              rocblas_fill_lower: The lower triangular part of A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the symmetric matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the symmetric matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(n) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_dspr2(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            const double*  alpha,
                                            const double*  x,
                                            rocblas_int    incx,
                                            const double*  y,
                                            rocblas_int    incy,
                                            double*        AP);

ROCBLAS_EXPORT rocblas_status rocblas_sspr2_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    const float* const y[],
                                                    rocblas_int        incy,
                                                    float* const       AP[],
                                                    rocblas_int        batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSPR2_BATCHED performs the matrix-vector operation

        A_i := A_i + alpha*x_i*y_i**T + alpha*y*x**T

    where alpha is a scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(n) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_dspr2_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    const double* const y[],
                                                    rocblas_int         incy,
                                                    double* const       AP[],
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sspr2_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            const float*   alpha,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            const float*   y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stride_y,
                                                            float*         AP,
                                                            rocblas_stride stride_A,
                                                            rocblas_int    batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSPR_STRIDED_BATCHED performs the matrix-vector operation

        A_i := A_i + alpha*x_i*x_i**T

    where alpha is a scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[in]
    y         device pointer pointing to the first vector (y_1).
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1).
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              each symmetric matrix A_i. Points to the first A_1.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(n) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_dspr2_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            const double*  alpha,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            const double*  y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stride_y,
                                                            double*        AP,
                                                            rocblas_stride stride_A,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyr(rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           rocblas_int    n,
                                           const float*   alpha,
                                           const float*   x,
                                           rocblas_int    incx,
                                           float*         A,
                                           rocblas_int    lda);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr(rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           rocblas_int    n,
                                           const double*  alpha,
                                           const double*  x,
                                           rocblas_int    incx,
                                           double*        A,
                                           rocblas_int    lda);

ROCBLAS_EXPORT rocblas_status rocblas_csyr(rocblas_handle               handle,
                                           rocblas_fill                 uplo,
                                           rocblas_int                  n,
                                           const rocblas_float_complex* alpha,
                                           const rocblas_float_complex* x,
                                           rocblas_int                  incx,
                                           rocblas_float_complex*       A,
                                           rocblas_int                  lda);
/*! \brief BLAS Level 2 API

    \details
    xSYR performs the matrix-vector operations

        A := A + alpha*x*x**T

    where alpha is a scalar, x is a vector, and A is an
    n by n symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced

    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[inout]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsyr(rocblas_handle                handle,
                                           rocblas_fill                  uplo,
                                           rocblas_int                   n,
                                           const rocblas_double_complex* alpha,
                                           const rocblas_double_complex* x,
                                           rocblas_int                   incx,
                                           rocblas_double_complex*       A,
                                           rocblas_int                   lda);

ROCBLAS_EXPORT rocblas_status rocblas_ssyr_batched(rocblas_handle     handle,
                                                   rocblas_fill       uplo,
                                                   rocblas_int        n,
                                                   const float*       alpha,
                                                   const float* const x[],
                                                   rocblas_int        incx,
                                                   float* const       A[],
                                                   rocblas_int        lda,
                                                   rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr_batched(rocblas_handle      handle,
                                                   rocblas_fill        uplo,
                                                   rocblas_int         n,
                                                   const double*       alpha,
                                                   const double* const x[],
                                                   rocblas_int         incx,
                                                   double* const       A[],
                                                   rocblas_int         lda,
                                                   rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyr_batched(rocblas_handle                     handle,
                                                   rocblas_fill                       uplo,
                                                   rocblas_int                        n,
                                                   const rocblas_float_complex*       alpha,
                                                   const rocblas_float_complex* const x[],
                                                   rocblas_int                        incx,
                                                   rocblas_float_complex* const       A[],
                                                   rocblas_int                        lda,
                                                   rocblas_int                        batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSYR_batched performs a batch of matrix-vector operations

        A[i] := A[i] + alpha*x[i]*x[i]**T

    where alpha is a scalar, x is an array of vectors, and A is an array of
    n by n symmetric matrices, for i = 1 , ... , batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[inout]
    A         device array of device pointers storing each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsyr_batched(rocblas_handle                      handle,
                                                   rocblas_fill                        uplo,
                                                   rocblas_int                         n,
                                                   const rocblas_double_complex*       alpha,
                                                   const rocblas_double_complex* const x[],
                                                   rocblas_int                         incx,
                                                   rocblas_double_complex* const       A[],
                                                   rocblas_int                         lda,
                                                   rocblas_int                         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyr_strided_batched(rocblas_handle handle,
                                                           rocblas_fill   uplo,
                                                           rocblas_int    n,
                                                           const float*   alpha,
                                                           const float*   x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stridex,
                                                           float*         A,
                                                           rocblas_int    lda,
                                                           rocblas_stride strideA,
                                                           rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr_strided_batched(rocblas_handle handle,
                                                           rocblas_fill   uplo,
                                                           rocblas_int    n,
                                                           const double*  alpha,
                                                           const double*  x,
                                                           rocblas_int    incx,
                                                           rocblas_stride stridex,
                                                           double*        A,
                                                           rocblas_int    lda,
                                                           rocblas_stride strideA,
                                                           rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyr_strided_batched(rocblas_handle               handle,
                                                           rocblas_fill                 uplo,
                                                           rocblas_int                  n,
                                                           const rocblas_float_complex* alpha,
                                                           const rocblas_float_complex* x,
                                                           rocblas_int                  incx,
                                                           rocblas_stride               stridex,
                                                           rocblas_float_complex*       A,
                                                           rocblas_int                  lda,
                                                           rocblas_stride               strideA,
                                                           rocblas_int batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSYR_strided_batched performs the matrix-vector operations

        A[i] := A[i] + alpha*x[i]*x[i]**T

    where alpha is a scalar, vectors, and A is an array of
    n by n symmetric matrices, for i = 1 , ... , batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stridex   [rocblas_stride]
              specifies the pointer increment between vectors (x_i) and (x_i+1).
    @param[inout]
    A         device pointer to the first matrix A_1.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    strideA   [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsyr_strided_batched(rocblas_handle                handle,
                                                           rocblas_fill                  uplo,
                                                           rocblas_int                   n,
                                                           const rocblas_double_complex* alpha,
                                                           const rocblas_double_complex* x,
                                                           rocblas_int                   incx,
                                                           rocblas_stride                stridex,
                                                           rocblas_double_complex*       A,
                                                           rocblas_int                   lda,
                                                           rocblas_stride                strideA,
                                                           rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyr2(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            const float*   alpha,
                                            const float*   x,
                                            rocblas_int    incx,
                                            const float*   y,
                                            rocblas_int    incy,
                                            float*         A,
                                            rocblas_int    lda);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr2(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            const double*  alpha,
                                            const double*  x,
                                            rocblas_int    incx,
                                            const double*  y,
                                            rocblas_int    incy,
                                            double*        A,
                                            rocblas_int    lda);

ROCBLAS_EXPORT rocblas_status rocblas_csyr2(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            const rocblas_float_complex* y,
                                            rocblas_int                  incy,
                                            rocblas_float_complex*       A,
                                            rocblas_int                  lda);
/*! \brief BLAS Level 2 API

    \details
    xSYR2 performs the matrix-vector operations

        A := A + alpha*x*y**T + alpha*y*x**T

    where alpha is a scalar, x and y are vectors, and A is an
    n by n symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced

    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsyr2(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            const rocblas_double_complex* y,
                                            rocblas_int                   incy,
                                            rocblas_double_complex*       A,
                                            rocblas_int                   lda);

ROCBLAS_EXPORT rocblas_status rocblas_ssyr2_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    const float* const y[],
                                                    rocblas_int        incy,
                                                    float* const       A[],
                                                    rocblas_int        lda,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr2_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    const double* const y[],
                                                    rocblas_int         incy,
                                                    double* const       A[],
                                                    rocblas_int         lda,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyr2_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    const rocblas_float_complex* const y[],
                                                    rocblas_int                        incy,
                                                    rocblas_float_complex* const       A[],
                                                    rocblas_int                        lda,
                                                    rocblas_int                        batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSYR2_BATCHED performs a batch of matrix-vector operations

        A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T

    where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
    n by n symmetric matrix, for i = 1 , ... , batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[inout]
    A         device array of device pointers storing each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsyr2_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    const rocblas_double_complex* const y[],
                                                    rocblas_int                         incy,
                                                    rocblas_double_complex* const       A[],
                                                    rocblas_int                         lda,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyr2_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            const float*   alpha,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            const float*   y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            float*         A,
                                                            rocblas_int    lda,
                                                            rocblas_stride strideA,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr2_strided_batched(rocblas_handle handle,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    n,
                                                            const double*  alpha,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            const double*  y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            double*        A,
                                                            rocblas_int    lda,
                                                            rocblas_stride strideA,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyr2_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stridex,
                                                            const rocblas_float_complex* y,
                                                            rocblas_int                  incy,
                                                            rocblas_stride               stridey,
                                                            rocblas_float_complex*       A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               strideA,
                                                            rocblas_int batch_count);
/*! \brief BLAS Level 2 API

    \details
    xSYR2_STRIDED_BATCHED the matrix-vector operations

        A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T

    where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
    n by n symmetric matrices, for i = 1 , ... , batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stridex   [rocblas_stride]
              specifies the pointer increment between vectors (x_i) and (x_i+1).
    @param[in]
    y         device pointer to the first vector y_1.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    stridey   [rocblas_stride]
              specifies the pointer increment between vectors (y_i) and (y_i+1).
    @param[inout]
    A         device pointer to the first matrix A_1.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    strideA   [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsyr2_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stridex,
                                                            const rocblas_double_complex* y,
                                                            rocblas_int                   incy,
                                                            rocblas_stride                stridey,
                                                            rocblas_double_complex*       A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                strideA,
                                                            rocblas_int batch_count);

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

ROCBLAS_EXPORT rocblas_status rocblas_chemm(rocblas_handle               handle,
                                            rocblas_side                 side,
                                            rocblas_fill                 uplo,
                                            rocblas_int                  n,
                                            rocblas_int                  k,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* B,
                                            rocblas_int                  ldb,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       C,
                                            rocblas_int                  ldc);

/*! \brief BLAS Level 3 API

    \details

    hemm performs one of the matrix-matrix operations:

    C := alpha*A*B + beta*C if side == rocblas_side_left,
    C := alpha*B*A + beta*C if side == rocblas_side_right,

    where alpha and beta are scalars, B and C are m by n matrices, and
    A is a Hermitian matrix stored as either upper or lower.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side  [rocblas_side]
            rocblas_side_left:      C := alpha*A*B + beta*C
            rocblas_side_right:     C := alpha*B*A + beta*C

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix
            rocblas_fill_lower:  A is a  lower triangular matrix

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B and C. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B and C. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A and B are not referenced.

    @param[in]
    A       pointer storing matrix A on the GPU.
            A is m by m if side == rocblas_side_left
            A is n by n if side == rocblas_side_right
            Only the upper/lower triangular part is accessed.
            The imaginary component of the diagonal elements is not used.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            otherwise lda >= max( 1, n ).

    @param[in]
    B       pointer storing matrix B on the GPU.
            Matrix dimension is m by n

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B. ldb >= max( 1, m )

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       pointer storing matrix C on the GPU.
            Matrix dimension is m by n

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, m )

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhemm(rocblas_handle                handle,
                                            rocblas_side                  side,
                                            rocblas_fill                  uplo,
                                            rocblas_int                   n,
                                            rocblas_int                   k,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* B,
                                            rocblas_int                   ldb,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       C,
                                            rocblas_int                   ldc);

ROCBLAS_EXPORT rocblas_status rocblas_chemm_batched(rocblas_handle                     handle,
                                                    rocblas_side                       side,
                                                    rocblas_fill                       uplo,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex* const B[],
                                                    rocblas_int                        ldb,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex* const       C[],
                                                    rocblas_int                        ldc,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 3 API

    \details

    hemm_batched performs a batch of the matrix-matrix operations:

    C_i := alpha*A_i*B_i + beta*C_i if side == rocblas_side_left,
    C_i := alpha*B_i*A_i + beta*C_i if side == rocblas_side_right,

    where alpha and beta are scalars, B_i and C_i are m by n matrices, and
    A_i is a Hermitian matrix stored as either upper or lower.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side  [rocblas_side]
            rocblas_side_left:      C_i := alpha*A_i*B_i + beta*C_i
            rocblas_side_right:     C_i := alpha*B_i*A_i + beta*C_i

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix
            rocblas_fill_lower:  A_i is a  lower triangular matrix

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B_i and C_i. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B_i and C_i. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A_i and B_i are not referenced.

    @param[in]
    A       device array of device pointers storing each matrix A_i on the GPU.
            A_i is m by m if side == rocblas_side_left
            A_i is n by n if side == rocblas_side_right
            Only the upper/lower triangular part is accessed.
            The imaginary component of the diagonal elements is not used.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            otherwise lda >= max( 1, n ).

    @param[in]
    B       device array of device pointers storing each matrix B_i on the GPU.
            Matrix dimension is m by n

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i. ldb >= max( 1, m )

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C_i need not be set before entry.

    @param[in]
    C       device array of device pointers storing each matrix C_i on the GPU.
            Matrix dimension is m by n

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C_i. ldc >= max( 1, m )

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhemm_batched(rocblas_handle                      handle,
                                                    rocblas_side                        side,
                                                    rocblas_fill                        uplo,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex* const B[],
                                                    rocblas_int                         ldb,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex* const       C[],
                                                    rocblas_int                         ldc,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_chemm_strided_batched(rocblas_handle               handle,
                                                            rocblas_side                 side,
                                                            rocblas_fill                 uplo,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            const rocblas_float_complex* B,
                                                            rocblas_int                  ldb,
                                                            rocblas_stride               stride_B,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       C,
                                                            rocblas_int                  ldc,
                                                            rocblas_stride               stride_C,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    hemm_strided_batched performs a batch of the matrix-matrix operations:

    C_i := alpha*A_i*B_i + beta*C_i if side == rocblas_side_left,
    C_i := alpha*B_i*A_i + beta*C_i if side == rocblas_side_right,

    where alpha and beta are scalars, B_i and C_i are m by n matrices, and
    A_i is a Hermitian matrix stored as either upper or lower.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side  [rocblas_side]
            rocblas_side_left:      C_i := alpha*A_i*B_i + beta*C_i
            rocblas_side_right:     C_i := alpha*B_i*A_i + beta*C_i

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix
            rocblas_fill_lower:  A_i is a  lower triangular matrix

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B_i and C_i. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B_i and C_i. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A_i and B_i are not referenced.

    @param[in]
    A       device pointer to first matrix A_1
            A_i is m by m if side == rocblas_side_left
            A_i is n by n if side == rocblas_side_right
            Only the upper/lower triangular part is accessed.
            The imaginary component of the diagonal elements is not used.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            otherwise lda >= max( 1, n ).

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    B       device pointer to first matrix B_1 of dimension (ldb, n) on the GPU

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i.
            if side = rocblas_operation_none,  ldb >= max( 1, m ),
            otherwise ldb >= max( 1, n ).

    @param[in]
    stride_B  [rocblas_stride]
              stride from the start of one matrix (B_i) and the next one (B_i+1)

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C        device pointer to first matrix C_1 of dimension (ldc, n) on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, m )

    @param[inout]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zhemm_strided_batched(rocblas_handle                handle,
                                                            rocblas_side                  side,
                                                            rocblas_fill                  uplo,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            const rocblas_double_complex* B,
                                                            rocblas_int                   ldb,
                                                            rocblas_stride                stride_B,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       C,
                                                            rocblas_int                   ldc,
                                                            rocblas_stride                stride_C,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cherk(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            transA,
                                            rocblas_int                  n,
                                            rocblas_int                  k,
                                            const float*                 alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const float*                 beta,
                                            rocblas_float_complex*       C,
                                            rocblas_int                  ldc);

/*! \brief BLAS Level 3 API

    \details

    herk performs one of the matrix-matrix operations for a Hermitian rank-k update

    C := alpha*op( A )*op( A )^H + beta*C

    where  alpha and beta are scalars, op(A) is an n by k matrix, and
    C is a n x n Hermitian matrix stored as either upper or lower.

        op( A ) = A, and A is n by k if transA == rocblas_operation_none
        op( A ) = A^H and A is k by n if transA == rocblas_operation_conjugate_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C is an upper triangular matrix
            rocblas_fill_lower:  C is a  lower triangular matrix

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_conjugate_transpose:  op(A) = A^H
            rocblas_operation_none:                 op(A) = A

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       pointer storing matrix A on the GPU.
            Martrix dimension is ( lda, k ) when if transA = rocblas_operation_none, otherwise (lda, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if transA = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       pointer storing matrix C on the GPU.
            The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zherk(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             transA,
                                            rocblas_int                   n,
                                            rocblas_int                   k,
                                            const double*                 alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const double*                 beta,
                                            rocblas_double_complex*       C,
                                            rocblas_int                   ldc);

ROCBLAS_EXPORT rocblas_status rocblas_cherk_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_operation                  transA,
                                                    rocblas_int                        n,
                                                    rocblas_int                        k,
                                                    const float*                       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const float*                       beta,
                                                    rocblas_float_complex* const       C[],
                                                    rocblas_int                        ldc,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 3 API

    \details

    herk_batched performs a batch of the matrix-matrix operations for a Hermitian rank-k update

    C_i := alpha*op( A_i )*op( A_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A) is an n by k matrix, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.

        op( A_i ) = A_i, and A_i is n by k if transA == rocblas_operation_none
        op( A_i ) = A_i^H and A_i is k by n if transA == rocblas_operation_conjugate_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_conjugate_transpose: op(A) = A^H
            rocblas_operation_none:                op(A) = A

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       device array of device pointers storing each matrix_i A of dimension (lda, k)
            when transA is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if transA = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       device array of device pointers storing each matrix C_i on the GPU.
            The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zherk_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   transA,
                                                    rocblas_int                         n,
                                                    rocblas_int                         k,
                                                    const double*                       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const double*                       beta,
                                                    rocblas_double_complex* const       C[],
                                                    rocblas_int                         ldc,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cherk_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            transA,
                                                            rocblas_int                  n,
                                                            rocblas_int                  k,
                                                            const float*                 alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            const float*                 beta,
                                                            rocblas_float_complex*       C,
                                                            rocblas_int                  ldc,
                                                            rocblas_stride               stride_C,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    herk_strided_batched performs a batch of the matrix-matrix operations for a Hermitian rank-k update

    C_i := alpha*op( A_i )*op( A_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A) is an n by k matrix, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.

        op( A_i ) = A_i, and A_i is n by k if transA == rocblas_operation_none
        op( A_i ) = A_i^H and A_i is k by n if transA == rocblas_operation_conjugate_transpose


    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_conjugate_transpose: op(A) = A^H
            rocblas_operation_none:                op(A) = A

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
            when transA is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if transA = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       Device pointer to the first matrix C_1 on the GPU.
            The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    @param[inout]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zherk_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             transA,
                                                            rocblas_int                   n,
                                                            rocblas_int                   k,
                                                            const double*                 alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            const double*                 beta,
                                                            rocblas_double_complex*       C,
                                                            rocblas_int                   ldc,
                                                            rocblas_stride                stride_C,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cher2k(rocblas_handle               handle,
                                             rocblas_fill                 uplo,
                                             rocblas_operation            trans,
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

/*! \brief BLAS Level 3 API

    \details

    her2k performs one of the matrix-matrix operations for a Hermitian rank-2k update

    C := alpha*op( A )*op( B )^H + conj(alpha)*op( B )*op( A )^H + beta*C

    where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
    C is a n x n Hermitian matrix stored as either upper or lower.

        op( A ) = A, op( B ) = B, and A and B are n by k if trans == rocblas_operation_none
        op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == rocblas_operation_conjugate_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C is an upper triangular matrix
            rocblas_fill_lower:  C is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_conjugate_transpose:  op( A ) = A^H, op( B ) = B^H
            rocblas_operation_none:                 op( A ) = A, op( B ) = B

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       pointer storing matrix A on the GPU.
            Martrix dimension is ( lda, k ) when if trans = rocblas_operation_none, otherwise (lda, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).
    @param[in]
    B       pointer storing matrix B on the GPU.
            Martrix dimension is ( ldb, k ) when if trans = rocblas_operation_none, otherwise (ldb, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).
    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       pointer storing matrix C on the GPU.
            The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zher2k(rocblas_handle                handle,
                                             rocblas_fill                  uplo,
                                             rocblas_operation             trans,
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

ROCBLAS_EXPORT rocblas_status rocblas_cher2k_batched(rocblas_handle                     handle,
                                                     rocblas_fill                       uplo,
                                                     rocblas_operation                  trans,
                                                     rocblas_int                        n,
                                                     rocblas_int                        k,
                                                     const rocblas_float_complex*       alpha,
                                                     const rocblas_float_complex* const A[],
                                                     rocblas_int                        lda,
                                                     const rocblas_float_complex* const B[],
                                                     rocblas_int                        ldb,
                                                     const float*                       beta,
                                                     rocblas_float_complex* const       C[],
                                                     rocblas_int                        ldc,
                                                     rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    her2k_batched performs a batch of the matrix-matrix operations for a Hermitian rank-2k update

    C_i := alpha*op( A_i )*op( B_i )^H + conj(alpha)*op( B_i )*op( A_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == rocblas_operation_none
        op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == rocblas_operation_conjugate_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_conjugate_transpose: op(A) = A^H
            rocblas_operation_none:                op(A) = A

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       device array of device pointers storing each matrix_i A of dimension (lda, k)
            when trans is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).
    @param[in]
    B       device array of device pointers storing each matrix_i B of dimension (ldb, k)
            when trans is rocblas_operation_none, otherwise of dimension (ldb, n)

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).
    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       device array of device pointers storing each matrix C_i on the GPU.
            The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zher2k_batched(rocblas_handle                      handle,
                                                     rocblas_fill                        uplo,
                                                     rocblas_operation                   trans,
                                                     rocblas_int                         n,
                                                     rocblas_int                         k,
                                                     const rocblas_double_complex*       alpha,
                                                     const rocblas_double_complex* const A[],
                                                     rocblas_int                         lda,
                                                     const rocblas_double_complex* const B[],
                                                     rocblas_int                         ldb,
                                                     const double*                       beta,
                                                     rocblas_double_complex* const       C[],
                                                     rocblas_int                         ldc,
                                                     rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cher2k_strided_batched(rocblas_handle               handle,
                                                             rocblas_fill                 uplo,
                                                             rocblas_operation            trans,
                                                             rocblas_int                  n,
                                                             rocblas_int                  k,
                                                             const rocblas_float_complex* alpha,
                                                             const rocblas_float_complex* A,
                                                             rocblas_int                  lda,
                                                             rocblas_stride               stride_A,
                                                             const rocblas_float_complex* B,
                                                             rocblas_int                  ldb,
                                                             rocblas_stride               stride_B,
                                                             const float*                 beta,
                                                             rocblas_float_complex*       C,
                                                             rocblas_int                  ldc,
                                                             rocblas_stride               stride_C,
                                                             rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    her2k_strided_batched performs a batch of the matrix-matrix operations for a Hermitian rank-2k update

    C_i := alpha*op( A_i )*op( B_i )^H + conj(alpha)*op( B_i )*op( A_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == rocblas_operation_none
        op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == rocblas_operation_conjugate_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_conjugate_transpose: op( A_i ) = A_i^H, op( B_i ) = B_i^H
            rocblas_operation_none:                op( A_i ) = A_i, op( B_i ) = B_i

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
            when trans is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    B       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
            when trans is rocblas_operation_none, otherwise of dimension (ldb, n)

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).

    @param[in]
    stride_B  [rocblas_stride]
              stride from the start of one matrix (B_i) and the next one (B_i+1)

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       Device pointer to the first matrix C_1 on the GPU.
            The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    @param[inout]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zher2k_strided_batched(rocblas_handle                handle,
                                                             rocblas_fill                  uplo,
                                                             rocblas_operation             trans,
                                                             rocblas_int                   n,
                                                             rocblas_int                   k,
                                                             const rocblas_double_complex* alpha,
                                                             const rocblas_double_complex* A,
                                                             rocblas_int                   lda,
                                                             rocblas_stride                stride_A,
                                                             const rocblas_double_complex* B,
                                                             rocblas_int                   ldb,
                                                             rocblas_stride                stride_B,
                                                             const double*                 beta,
                                                             rocblas_double_complex*       C,
                                                             rocblas_int                   ldc,
                                                             rocblas_stride                stride_C,
                                                             rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cherkx(rocblas_handle               handle,
                                             rocblas_fill                 uplo,
                                             rocblas_operation            trans,
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

/*! \brief BLAS Level 3 API

    \details

    herkx performs one of the matrix-matrix operations for a Hermitian rank-k update

    C := alpha*op( A )*op( B )^H + beta*C

    where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
    C is a n x n Hermitian matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermetian.


        op( A ) = A, op( B ) = B, and A and B are n by k if trans == rocblas_operation_none
        op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == rocblas_operation_conjugate_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C is an upper triangular matrix
            rocblas_fill_lower:  C is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_conjugate_transpose:  op( A ) = A^H, op( B ) = B^H
            rocblas_operation_none:                 op( A ) = A, op( B ) = B

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       pointer storing matrix A on the GPU.
            Martrix dimension is ( lda, k ) when if trans = rocblas_operation_none, otherwise (lda, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).
    @param[in]
    B       pointer storing matrix B on the GPU.
            Martrix dimension is ( ldb, k ) when if trans = rocblas_operation_none, otherwise (ldb, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).
    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       pointer storing matrix C on the GPU.
            The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zherkx(rocblas_handle                handle,
                                             rocblas_fill                  uplo,
                                             rocblas_operation             trans,
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

ROCBLAS_EXPORT rocblas_status rocblas_cherkx_batched(rocblas_handle                     handle,
                                                     rocblas_fill                       uplo,
                                                     rocblas_operation                  trans,
                                                     rocblas_int                        n,
                                                     rocblas_int                        k,
                                                     const rocblas_float_complex*       alpha,
                                                     const rocblas_float_complex* const A[],
                                                     rocblas_int                        lda,
                                                     const rocblas_float_complex* const B[],
                                                     rocblas_int                        ldb,
                                                     const float*                       beta,
                                                     rocblas_float_complex* const       C[],
                                                     rocblas_int                        ldc,
                                                     rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    herkx_batched performs a batch of the matrix-matrix operations for a Hermitian rank-k update

    C_i := alpha*op( A_i )*op( B_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermetian.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == rocblas_operation_none
        op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == rocblas_operation_conjugate_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_conjugate_transpose: op(A) = A^H
            rocblas_operation_none:                op(A) = A

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       device array of device pointers storing each matrix_i A of dimension (lda, k)
            when trans is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    B       device array of device pointers storing each matrix_i B of dimension (ldb, k)
            when trans is rocblas_operation_none, otherwise of dimension (ldb, n)

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       device array of device pointers storing each matrix C_i on the GPU.
            The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zherkx_batched(rocblas_handle                      handle,
                                                     rocblas_fill                        uplo,
                                                     rocblas_operation                   trans,
                                                     rocblas_int                         n,
                                                     rocblas_int                         k,
                                                     const rocblas_double_complex*       alpha,
                                                     const rocblas_double_complex* const A[],
                                                     rocblas_int                         lda,
                                                     const rocblas_double_complex* const B[],
                                                     rocblas_int                         ldb,
                                                     const double*                       beta,
                                                     rocblas_double_complex* const       C[],
                                                     rocblas_int                         ldc,
                                                     rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cherkx_strided_batched(rocblas_handle               handle,
                                                             rocblas_fill                 uplo,
                                                             rocblas_operation            trans,
                                                             rocblas_int                  n,
                                                             rocblas_int                  k,
                                                             const rocblas_float_complex* alpha,
                                                             const rocblas_float_complex* A,
                                                             rocblas_int                  lda,
                                                             rocblas_stride               stride_A,
                                                             const rocblas_float_complex* B,
                                                             rocblas_int                  ldb,
                                                             rocblas_stride               stride_B,
                                                             const float*                 beta,
                                                             rocblas_float_complex*       C,
                                                             rocblas_int                  ldc,
                                                             rocblas_stride               stride_C,
                                                             rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    herkx_strided_batched performs a batch of the matrix-matrix operations for a Hermitian rank-k update

    C_i := alpha*op( A_i )*op( B_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermetian.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == rocblas_operation_none
        op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == rocblas_operation_conjugate_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_conjugate_transpose: op( A_i ) = A_i^H, op( B_i ) = B_i^H
            rocblas_operation_none:                op( A_i ) = A_i, op( B_i ) = B_i

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
            when trans is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    B       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
            when trans is rocblas_operation_none, otherwise of dimension (ldb, n)

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).

    @param[in]
    stride_B  [rocblas_stride]
              stride from the start of one matrix (B_i) and the next one (B_i+1)

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       Device pointer to the first matrix C_1 on the GPU.
            The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    @param[inout]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zherkx_strided_batched(rocblas_handle                handle,
                                                             rocblas_fill                  uplo,
                                                             rocblas_operation             trans,
                                                             rocblas_int                   n,
                                                             rocblas_int                   k,
                                                             const rocblas_double_complex* alpha,
                                                             const rocblas_double_complex* A,
                                                             rocblas_int                   lda,
                                                             rocblas_stride                stride_A,
                                                             const rocblas_double_complex* B,
                                                             rocblas_int                   ldb,
                                                             rocblas_stride                stride_B,
                                                             const double*                 beta,
                                                             rocblas_double_complex*       C,
                                                             rocblas_int                   ldc,
                                                             rocblas_stride                stride_C,
                                                             rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssymm(rocblas_handle handle,
                                            rocblas_side   side,
                                            rocblas_fill   uplo,
                                            rocblas_int    m,
                                            rocblas_int    n,
                                            const float*   alpha,
                                            const float*   A,
                                            rocblas_int    lda,
                                            const float*   B,
                                            rocblas_int    ldb,
                                            const float*   beta,
                                            float*         C,
                                            rocblas_int    ldc);

ROCBLAS_EXPORT rocblas_status rocblas_dsymm(rocblas_handle handle,
                                            rocblas_side   side,
                                            rocblas_fill   uplo,
                                            rocblas_int    m,
                                            rocblas_int    n,
                                            const double*  alpha,
                                            const double*  A,
                                            rocblas_int    lda,
                                            const double*  B,
                                            rocblas_int    ldb,
                                            const double*  beta,
                                            double*        C,
                                            rocblas_int    ldc);

ROCBLAS_EXPORT rocblas_status rocblas_csymm(rocblas_handle               handle,
                                            rocblas_side                 side,
                                            rocblas_fill                 uplo,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* B,
                                            rocblas_int                  ldb,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       C,
                                            rocblas_int                  ldc);

/*! \brief BLAS Level 3 API

    \details

    symm performs one of the matrix-matrix operations:

    C := alpha*A*B + beta*C if side == rocblas_side_left,
    C := alpha*B*A + beta*C if side == rocblas_side_right,

    where alpha and beta are scalars, B and C are m by n matrices, and
    A is a symmetric matrix stored as either upper or lower.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side  [rocblas_side]
            rocblas_side_left:      C := alpha*A*B + beta*C
            rocblas_side_right:     C := alpha*B*A + beta*C

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix
            rocblas_fill_lower:  A is a  lower triangular matrix

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B and C. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B and C. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A and B are not referenced.

    @param[in]
    A       pointer storing matrix A on the GPU.
            A is m by m if side == rocblas_side_left
            A is n by n if side == rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            otherwise lda >= max( 1, n ).

    @param[in]
    B       pointer storing matrix B on the GPU.
            Matrix dimension is m by n

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B. ldb >= max( 1, m )

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       pointer storing matrix C on the GPU.
            Matrix dimension is m by n

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, m )

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsymm(rocblas_handle                handle,
                                            rocblas_side                  side,
                                            rocblas_fill                  uplo,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* B,
                                            rocblas_int                   ldb,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       C,
                                            rocblas_int                   ldc);

ROCBLAS_EXPORT rocblas_status rocblas_ssymm_batched(rocblas_handle     handle,
                                                    rocblas_side       side,
                                                    rocblas_fill       uplo,
                                                    rocblas_int        m,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    const float* const B[],
                                                    rocblas_int        ldb,
                                                    const float*       beta,
                                                    float* const       C[],
                                                    rocblas_int        ldc,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsymm_batched(rocblas_handle      handle,
                                                    rocblas_side        side,
                                                    rocblas_fill        uplo,
                                                    rocblas_int         m,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    const double* const B[],
                                                    rocblas_int         ldb,
                                                    const double*       beta,
                                                    double* const       C[],
                                                    rocblas_int         ldc,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csymm_batched(rocblas_handle                     handle,
                                                    rocblas_side                       side,
                                                    rocblas_fill                       uplo,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex* const B[],
                                                    rocblas_int                        ldb,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex* const       C[],
                                                    rocblas_int                        ldc,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 3 API

    \details

    symm_batched performs a batch of the matrix-matrix operations:

    C_i := alpha*A_i*B_i + beta*C_i if side == rocblas_side_left,
    C_i := alpha*B_i*A_i + beta*C_i if side == rocblas_side_right,

    where alpha and beta are scalars, B_i and C_i are m by n matrices, and
    A_i is a symmetric matrix stored as either upper or lower.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side  [rocblas_side]
            rocblas_side_left:      C_i := alpha*A_i*B_i + beta*C_i
            rocblas_side_right:     C_i := alpha*B_i*A_i + beta*C_i

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix
            rocblas_fill_lower:  A_i is a  lower triangular matrix

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B_i and C_i. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B_i and C_i. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A_i and B_i are not referenced.

    @param[in]
    A       device array of device pointers storing each matrix A_i on the GPU.
            A_i is m by m if side == rocblas_side_left
            A_i is n by n if side == rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            otherwise lda >= max( 1, n ).

    @param[in]
    B       device array of device pointers storing each matrix B_i on the GPU.
            Matrix dimension is m by n

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i. ldb >= max( 1, m )

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C_i need not be set before entry.

    @param[in]
    C       device array of device pointers storing each matrix C_i on the GPU.
            Matrix dimension is m by n

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C_i. ldc >= max( 1, m )

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsymm_batched(rocblas_handle                      handle,
                                                    rocblas_side                        side,
                                                    rocblas_fill                        uplo,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex* const B[],
                                                    rocblas_int                         ldb,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex* const       C[],
                                                    rocblas_int                         ldc,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssymm_strided_batched(rocblas_handle handle,
                                                            rocblas_side   side,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    m,
                                                            rocblas_int    n,
                                                            const float*   alpha,
                                                            const float*   A,
                                                            rocblas_int    lda,
                                                            rocblas_stride stride_A,
                                                            const float*   B,
                                                            rocblas_int    ldb,
                                                            rocblas_stride stride_B,
                                                            const float*   beta,
                                                            float*         C,
                                                            rocblas_int    ldc,
                                                            rocblas_stride stride_C,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsymm_strided_batched(rocblas_handle handle,
                                                            rocblas_side   side,
                                                            rocblas_fill   uplo,
                                                            rocblas_int    m,
                                                            rocblas_int    n,
                                                            const double*  alpha,
                                                            const double*  A,
                                                            rocblas_int    lda,
                                                            rocblas_stride stride_A,
                                                            const double*  B,
                                                            rocblas_int    ldb,
                                                            rocblas_stride stride_B,
                                                            const double*  beta,
                                                            double*        C,
                                                            rocblas_int    ldc,
                                                            rocblas_stride stride_C,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csymm_strided_batched(rocblas_handle               handle,
                                                            rocblas_side                 side,
                                                            rocblas_fill                 uplo,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            const rocblas_float_complex* B,
                                                            rocblas_int                  ldb,
                                                            rocblas_stride               stride_B,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       C,
                                                            rocblas_int                  ldc,
                                                            rocblas_stride               stride_C,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    symm_strided_batched performs a batch of the matrix-matrix operations:

    C_i := alpha*A_i*B_i + beta*C_i if side == rocblas_side_left,
    C_i := alpha*B_i*A_i + beta*C_i if side == rocblas_side_right,

    where alpha and beta are scalars, B_i and C_i are m by n matrices, and
    A_i is a symmetric matrix stored as either upper or lower.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side  [rocblas_side]
            rocblas_side_left:      C_i := alpha*A_i*B_i + beta*C_i
            rocblas_side_right:     C_i := alpha*B_i*A_i + beta*C_i

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix
            rocblas_fill_lower:  A_i is a  lower triangular matrix

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B_i and C_i. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B_i and C_i. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A_i and B_i are not referenced.

    @param[in]
    A       device pointer to first matrix A_1
            A_i is m by m if side == rocblas_side_left
            A_i is n by n if side == rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            otherwise lda >= max( 1, n ).

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    B       device pointer to first matrix B_1 of dimension (ldb, n) on the GPU.

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i. ldb >= max( 1, m )

    @param[in]
    stride_B  [rocblas_stride]
              stride from the start of one matrix (B_i) and the next one (B_i+1)
    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C        device pointer to first matrix C_1 of dimension (ldc, n) on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, m ).

    @param[inout]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsymm_strided_batched(rocblas_handle                handle,
                                                            rocblas_side                  side,
                                                            rocblas_fill                  uplo,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            const rocblas_double_complex* B,
                                                            rocblas_int                   ldb,
                                                            rocblas_stride                stride_B,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       C,
                                                            rocblas_int                   ldc,
                                                            rocblas_stride                stride_C,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyrk(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const float*      alpha,
                                            const float*      A,
                                            rocblas_int       lda,
                                            const float*      beta,
                                            float*            C,
                                            rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_dsyrk(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const double*     alpha,
                                            const double*     A,
                                            rocblas_int       lda,
                                            const double*     beta,
                                            double*           C,
                                            rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_csyrk(rocblas_handle               handle,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            transA,
                                            rocblas_int                  n,
                                            rocblas_int                  k,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       C,
                                            rocblas_int                  ldc);
/*! \brief BLAS Level 3 API

    \details

    syrk performs one of the matrix-matrix operations for a symmetric rank-k update

    C := alpha*op( A )*op( A )^T + beta*C

    where  alpha and beta are scalars, op(A) is an n by k matrix, and
    C is a symmetric n x n matrix stored as either upper or lower.

        op( A ) = A, and A is n by k if transA == rocblas_operation_none
        op( A ) = A^T and A is k by n if transA == rocblas_operation_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C is an upper triangular matrix
            rocblas_fill_lower:  C is a  lower triangular matrix

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_transpose:      op(A) = A^T
            rocblas_operation_none:           op(A) = A

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       pointer storing matrix A on the GPU.
            Martrix dimension is ( lda, k ) when if transA = rocblas_operation_none, otherwise (lda, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if transA = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       pointer storing matrix C on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_zsyrk(rocblas_handle                handle,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             transA,
                                            rocblas_int                   n,
                                            rocblas_int                   k,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       C,
                                            rocblas_int                   ldc);

ROCBLAS_EXPORT rocblas_status rocblas_ssyrk_batched(rocblas_handle     handle,
                                                    rocblas_fill       uplo,
                                                    rocblas_operation  transA,
                                                    rocblas_int        n,
                                                    rocblas_int        k,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    const float*       beta,
                                                    float* const       C[],
                                                    rocblas_int        ldc,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyrk_batched(rocblas_handle      handle,
                                                    rocblas_fill        uplo,
                                                    rocblas_operation   transA,
                                                    rocblas_int         n,
                                                    rocblas_int         k,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    const double*       beta,
                                                    double* const       C[],
                                                    rocblas_int         ldc,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyrk_batched(rocblas_handle                     handle,
                                                    rocblas_fill                       uplo,
                                                    rocblas_operation                  transA,
                                                    rocblas_int                        n,
                                                    rocblas_int                        k,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex* const       C[],
                                                    rocblas_int                        ldc,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 3 API

    \details

    syrk_batched performs a batch of the matrix-matrix operations for a symmetric rank-k update

    C_i := alpha*op( A_i )*op( A_i )^T + beta*C_i

    where  alpha and beta are scalars, op(A_i) is an n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.

        op( A_i ) = A_i, and A_i is n by k if transA == rocblas_operation_none
        op( A_i ) = A^T and A_i is k by n if transA == rocblas_operation_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_transpose:      op(A) = A^T
            rocblas_operation_none:           op(A) = A

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       device array of device pointers storing each matrix_i A of dimension (lda, k)
            when transA is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if transA = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       device array of device pointers storing each matrix C_i on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsyrk_batched(rocblas_handle                      handle,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   transA,
                                                    rocblas_int                         n,
                                                    rocblas_int                         k,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex* const       C[],
                                                    rocblas_int                         ldc,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyrk_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_int       n,
                                                            rocblas_int       k,
                                                            const float*      alpha,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            const float*      beta,
                                                            float*            C,
                                                            rocblas_int       ldc,
                                                            rocblas_stride    stride_C,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyrk_strided_batched(rocblas_handle    handle,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_int       n,
                                                            rocblas_int       k,
                                                            const double*     alpha,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_A,
                                                            const double*     beta,
                                                            double*           C,
                                                            rocblas_int       ldc,
                                                            rocblas_stride    stride_C,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyrk_strided_batched(rocblas_handle               handle,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            transA,
                                                            rocblas_int                  n,
                                                            rocblas_int                  k,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_A,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       C,
                                                            rocblas_int                  ldc,
                                                            rocblas_stride               stride_C,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    syrk_strided_batched performs a batch of the matrix-matrix operations for a symmetric rank-k update

    C_i := alpha*op( A_i )*op( A_i )^T + beta*C_i

    where  alpha and beta are scalars, op(A_i) is an n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.

        op( A_i ) = A_i, and A_i is n by k if transA == rocblas_operation_none
        op( A_i ) = A^T and A_i is k by n if transA == rocblas_operation_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_transpose:      op(A) = A^T
            rocblas_operation_none:           op(A) = A

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
            when transA is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if transA = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       Device pointer to the first matrix C_1 on the GPU. on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    @param[inout]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsyrk_strided_batched(rocblas_handle                handle,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             transA,
                                                            rocblas_int                   n,
                                                            rocblas_int                   k,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_A,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       C,
                                                            rocblas_int                   ldc,
                                                            rocblas_stride                stride_C,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyr2k(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation trans,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const float*      alpha,
                                             const float*      A,
                                             rocblas_int       lda,
                                             const float*      B,
                                             rocblas_int       ldb,
                                             const float*      beta,
                                             float*            C,
                                             rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr2k(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation trans,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const double*     alpha,
                                             const double*     A,
                                             rocblas_int       lda,
                                             const double*     B,
                                             rocblas_int       ldb,
                                             const double*     beta,
                                             double*           C,
                                             rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_csyr2k(rocblas_handle               handle,
                                             rocblas_fill                 uplo,
                                             rocblas_operation            trans,
                                             rocblas_int                  n,
                                             rocblas_int                  k,
                                             const rocblas_float_complex* alpha,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  lda,
                                             const rocblas_float_complex* B,
                                             rocblas_int                  ldb,
                                             const rocblas_float_complex* beta,
                                             rocblas_float_complex*       C,
                                             rocblas_int                  ldc);

/*! \brief BLAS Level 3 API

    \details

    syr2k performs one of the matrix-matrix operations for a symmetric rank-2k update

    C := alpha*(op( A )*op( B )^T + op( B )*op( A )^T) + beta*C

    where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
    C is a symmetric n x n matrix stored as either upper or lower.

        op( A ) = A, op( B ) = B, and A and B are n by k if trans == rocblas_operation_none
        op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == rocblas_operation_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C is an upper triangular matrix
            rocblas_fill_lower:  C is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_transpose:      op( A ) = A^T, op( B ) = B^T
            rocblas_operation_none:           op( A ) = A, op( B ) = B

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A) and op(B). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       pointer storing matrix A on the GPU.
            Martrix dimension is ( lda, k ) when if trans = rocblas_operation_none, otherwise (lda, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).
    @param[in]
    B       pointer storing matrix B on the GPU.
            Martrix dimension is ( ldb, k ) when if trans = rocblas_operation_none, otherwise (ldb, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).
    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       pointer storing matrix C on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsyr2k(rocblas_handle                handle,
                                             rocblas_fill                  uplo,
                                             rocblas_operation             trans,
                                             rocblas_int                   n,
                                             rocblas_int                   k,
                                             const rocblas_double_complex* alpha,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   lda,
                                             const rocblas_double_complex* B,
                                             rocblas_int                   ldb,
                                             const rocblas_double_complex* beta,
                                             rocblas_double_complex*       C,
                                             rocblas_int                   ldc);

ROCBLAS_EXPORT rocblas_status rocblas_ssyr2k_batched(rocblas_handle     handle,
                                                     rocblas_fill       uplo,
                                                     rocblas_operation  trans,
                                                     rocblas_int        n,
                                                     rocblas_int        k,
                                                     const float*       alpha,
                                                     const float* const A[],
                                                     rocblas_int        lda,
                                                     const float* const B[],
                                                     rocblas_int        ldb,
                                                     const float*       beta,
                                                     float* const       C[],
                                                     rocblas_int        ldc,
                                                     rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr2k_batched(rocblas_handle      handle,
                                                     rocblas_fill        uplo,
                                                     rocblas_operation   trans,
                                                     rocblas_int         n,
                                                     rocblas_int         k,
                                                     const double*       alpha,
                                                     const double* const A[],
                                                     rocblas_int         lda,
                                                     const double* const B[],
                                                     rocblas_int         ldb,
                                                     const double*       beta,
                                                     double* const       C[],
                                                     rocblas_int         ldc,
                                                     rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyr2k_batched(rocblas_handle                     handle,
                                                     rocblas_fill                       uplo,
                                                     rocblas_operation                  trans,
                                                     rocblas_int                        n,
                                                     rocblas_int                        k,
                                                     const rocblas_float_complex*       alpha,
                                                     const rocblas_float_complex* const A[],
                                                     rocblas_int                        lda,
                                                     const rocblas_float_complex* const B[],
                                                     rocblas_int                        ldb,
                                                     const rocblas_float_complex*       beta,
                                                     rocblas_float_complex* const       C[],
                                                     rocblas_int                        ldc,
                                                     rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    syr2k_batched performs a batch of the matrix-matrix operations for a symmetric rank-2k update

    C_i := alpha*(op( A_i )*op( B_i )^T + op( B_i )*op( A_i )^T) + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == rocblas_operation_none
        op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == rocblas_operation_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_transpose:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
            rocblas_operation_none:           op( A_i ) = A_i, op( B_i ) = B_i

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       device array of device pointers storing each matrix_i A of dimension (lda, k)
            when trans is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).
    @param[in]
    B       device array of device pointers storing each matrix_i B of dimension (ldb, k)
            when trans is rocblas_operation_none, otherwise of dimension (ldb, n)
    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).
    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       device array of device pointers storing each matrix C_i on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsyr2k_batched(rocblas_handle                      handle,
                                                     rocblas_fill                        uplo,
                                                     rocblas_operation                   trans,
                                                     rocblas_int                         n,
                                                     rocblas_int                         k,
                                                     const rocblas_double_complex*       alpha,
                                                     const rocblas_double_complex* const A[],
                                                     rocblas_int                         lda,
                                                     const rocblas_double_complex* const B[],
                                                     rocblas_int                         ldb,
                                                     const rocblas_double_complex*       beta,
                                                     rocblas_double_complex* const       C[],
                                                     rocblas_int                         ldc,
                                                     rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyr2k_strided_batched(rocblas_handle    handle,
                                                             rocblas_fill      uplo,
                                                             rocblas_operation trans,
                                                             rocblas_int       n,
                                                             rocblas_int       k,
                                                             const float*      alpha,
                                                             const float*      A,
                                                             rocblas_int       lda,
                                                             rocblas_stride    stride_A,
                                                             const float*      B,
                                                             rocblas_int       ldb,
                                                             rocblas_stride    stride_B,
                                                             const float*      beta,
                                                             float*            C,
                                                             rocblas_int       ldc,
                                                             rocblas_stride    stride_C,
                                                             rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyr2k_strided_batched(rocblas_handle    handle,
                                                             rocblas_fill      uplo,
                                                             rocblas_operation trans,
                                                             rocblas_int       n,
                                                             rocblas_int       k,
                                                             const double*     alpha,
                                                             const double*     A,
                                                             rocblas_int       lda,
                                                             rocblas_stride    stride_A,
                                                             const double*     B,
                                                             rocblas_int       ldb,
                                                             rocblas_stride    stride_B,
                                                             const double*     beta,
                                                             double*           C,
                                                             rocblas_int       ldc,
                                                             rocblas_stride    stride_C,
                                                             rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyr2k_strided_batched(rocblas_handle               handle,
                                                             rocblas_fill                 uplo,
                                                             rocblas_operation            trans,
                                                             rocblas_int                  n,
                                                             rocblas_int                  k,
                                                             const rocblas_float_complex* alpha,
                                                             const rocblas_float_complex* A,
                                                             rocblas_int                  lda,
                                                             rocblas_stride               stride_A,
                                                             const rocblas_float_complex* B,
                                                             rocblas_int                  ldb,
                                                             rocblas_stride               stride_B,
                                                             const rocblas_float_complex* beta,
                                                             rocblas_float_complex*       C,
                                                             rocblas_int                  ldc,
                                                             rocblas_stride               stride_C,
                                                             rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    syr2k_strided_batched performs a batch of the matrix-matrix operations for a symmetric rank-2k update

    C_i := alpha*(op( A_i )*op( B_i )^T + op( B_i )*op( A_i )^T) + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == rocblas_operation_none
        op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == rocblas_operation_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_transpose:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
            rocblas_operation_none:           op( A_i ) = A_i, op( B_i ) = B_i

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
            when trans is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    B       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
            when trans is rocblas_operation_none, otherwise of dimension (ldb, n)

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).

    @param[in]
    stride_B  [rocblas_stride]
              stride from the start of one matrix (B_i) and the next one (B_i+1)

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       Device pointer to the first matrix C_1 on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    @param[inout]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsyr2k_strided_batched(rocblas_handle                handle,
                                                             rocblas_fill                  uplo,
                                                             rocblas_operation             trans,
                                                             rocblas_int                   n,
                                                             rocblas_int                   k,
                                                             const rocblas_double_complex* alpha,
                                                             const rocblas_double_complex* A,
                                                             rocblas_int                   lda,
                                                             rocblas_stride                stride_A,
                                                             const rocblas_double_complex* B,
                                                             rocblas_int                   ldb,
                                                             rocblas_stride                stride_B,
                                                             const rocblas_double_complex* beta,
                                                             rocblas_double_complex*       C,
                                                             rocblas_int                   ldc,
                                                             rocblas_stride                stride_C,
                                                             rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyrkx(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation trans,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const float*      alpha,
                                             const float*      A,
                                             rocblas_int       lda,
                                             const float*      B,
                                             rocblas_int       ldb,
                                             const float*      beta,
                                             float*            C,
                                             rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_dsyrkx(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation trans,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const double*     alpha,
                                             const double*     A,
                                             rocblas_int       lda,
                                             const double*     B,
                                             rocblas_int       ldb,
                                             const double*     beta,
                                             double*           C,
                                             rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_csyrkx(rocblas_handle               handle,
                                             rocblas_fill                 uplo,
                                             rocblas_operation            trans,
                                             rocblas_int                  n,
                                             rocblas_int                  k,
                                             const rocblas_float_complex* alpha,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  lda,
                                             const rocblas_float_complex* B,
                                             rocblas_int                  ldb,
                                             const rocblas_float_complex* beta,
                                             rocblas_float_complex*       C,
                                             rocblas_int                  ldc);

/*! \brief BLAS Level 3 API

    \details

    syrkx performs one of the matrix-matrix operations for a symmetric rank-k update

    C := alpha*op( A )*op( B )^T + beta*C

    where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
    C is a symmetric n x n matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be symmetric.

        op( A ) = A, op( B ) = B, and A and B are n by k if trans == rocblas_operation_none
        op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == rocblas_operation_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C is an upper triangular matrix
            rocblas_fill_lower:  C is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_transpose:      op( A ) = A^T, op( B ) = B^T
            rocblas_operation_none:           op( A ) = A, op( B ) = B

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A) and op(B). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       pointer storing matrix A on the GPU.
            Martrix dimension is ( lda, k ) when if trans = rocblas_operation_none, otherwise (lda, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    B       pointer storing matrix B on the GPU.
            Martrix dimension is ( ldb, k ) when if trans = rocblas_operation_none, otherwise (ldb, n)
            only the upper/lower triangular part is accessed.

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       pointer storing matrix C on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsyrkx(rocblas_handle                handle,
                                             rocblas_fill                  uplo,
                                             rocblas_operation             trans,
                                             rocblas_int                   n,
                                             rocblas_int                   k,
                                             const rocblas_double_complex* alpha,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   lda,
                                             const rocblas_double_complex* B,
                                             rocblas_int                   ldb,
                                             const rocblas_double_complex* beta,
                                             rocblas_double_complex*       C,
                                             rocblas_int                   ldc);

ROCBLAS_EXPORT rocblas_status rocblas_ssyrkx_batched(rocblas_handle     handle,
                                                     rocblas_fill       uplo,
                                                     rocblas_operation  trans,
                                                     rocblas_int        n,
                                                     rocblas_int        k,
                                                     const float*       alpha,
                                                     const float* const A[],
                                                     rocblas_int        lda,
                                                     const float* const B[],
                                                     rocblas_int        ldb,
                                                     const float*       beta,
                                                     float* const       C[],
                                                     rocblas_int        ldc,
                                                     rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyrkx_batched(rocblas_handle      handle,
                                                     rocblas_fill        uplo,
                                                     rocblas_operation   trans,
                                                     rocblas_int         n,
                                                     rocblas_int         k,
                                                     const double*       alpha,
                                                     const double* const A[],
                                                     rocblas_int         lda,
                                                     const double* const B[],
                                                     rocblas_int         ldb,
                                                     const double*       beta,
                                                     double* const       C[],
                                                     rocblas_int         ldc,
                                                     rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyrkx_batched(rocblas_handle                     handle,
                                                     rocblas_fill                       uplo,
                                                     rocblas_operation                  trans,
                                                     rocblas_int                        n,
                                                     rocblas_int                        k,
                                                     const rocblas_float_complex*       alpha,
                                                     const rocblas_float_complex* const A[],
                                                     rocblas_int                        lda,
                                                     const rocblas_float_complex* const B[],
                                                     rocblas_int                        ldb,
                                                     const rocblas_float_complex*       beta,
                                                     rocblas_float_complex* const       C[],
                                                     rocblas_int                        ldc,
                                                     rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    syrkx_batched performs a batch of the matrix-matrix operations for a symmetric rank-k update

    C_i := alpha*op( A_i )*op( B_i )^T + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A_i )*op( B_i )^T will be symmetric.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == rocblas_operation_none
        op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == rocblas_operation_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_transpose:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
            rocblas_operation_none:           op( A_i ) = A_i, op( B_i ) = B_i

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       device array of device pointers storing each matrix_i A of dimension (lda, k)
            when trans is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    B       device array of device pointers storing each matrix_i B of dimension (ldb, k)
            when trans is rocblas_operation_none, otherwise of dimension (ldb, n)

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       device array of device pointers storing each matrix C_i on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    @param[in]
    batch_count [rocblas_int]
            number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsyrkx_batched(rocblas_handle                      handle,
                                                     rocblas_fill                        uplo,
                                                     rocblas_operation                   trans,
                                                     rocblas_int                         n,
                                                     rocblas_int                         k,
                                                     const rocblas_double_complex*       alpha,
                                                     const rocblas_double_complex* const A[],
                                                     rocblas_int                         lda,
                                                     const rocblas_double_complex* const B[],
                                                     rocblas_int                         ldb,
                                                     const rocblas_double_complex*       beta,
                                                     rocblas_double_complex* const       C[],
                                                     rocblas_int                         ldc,
                                                     rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ssyrkx_strided_batched(rocblas_handle    handle,
                                                             rocblas_fill      uplo,
                                                             rocblas_operation trans,
                                                             rocblas_int       n,
                                                             rocblas_int       k,
                                                             const float*      alpha,
                                                             const float*      A,
                                                             rocblas_int       lda,
                                                             rocblas_stride    stride_A,
                                                             const float*      B,
                                                             rocblas_int       ldb,
                                                             rocblas_stride    stride_B,
                                                             const float*      beta,
                                                             float*            C,
                                                             rocblas_int       ldc,
                                                             rocblas_stride    stride_C,
                                                             rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dsyrkx_strided_batched(rocblas_handle    handle,
                                                             rocblas_fill      uplo,
                                                             rocblas_operation trans,
                                                             rocblas_int       n,
                                                             rocblas_int       k,
                                                             const double*     alpha,
                                                             const double*     A,
                                                             rocblas_int       lda,
                                                             rocblas_stride    stride_A,
                                                             const double*     B,
                                                             rocblas_int       ldb,
                                                             rocblas_stride    stride_B,
                                                             const double*     beta,
                                                             double*           C,
                                                             rocblas_int       ldc,
                                                             rocblas_stride    stride_C,
                                                             rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_csyrkx_strided_batched(rocblas_handle               handle,
                                                             rocblas_fill                 uplo,
                                                             rocblas_operation            trans,
                                                             rocblas_int                  n,
                                                             rocblas_int                  k,
                                                             const rocblas_float_complex* alpha,
                                                             const rocblas_float_complex* A,
                                                             rocblas_int                  lda,
                                                             rocblas_stride               stride_A,
                                                             const rocblas_float_complex* B,
                                                             rocblas_int                  ldb,
                                                             rocblas_stride               stride_B,
                                                             const rocblas_float_complex* beta,
                                                             rocblas_float_complex*       C,
                                                             rocblas_int                  ldc,
                                                             rocblas_stride               stride_C,
                                                             rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    syrkx_strided_batched performs a batch of the matrix-matrix operations for a symmetric rank-k update

    C_i := alpha*op( A_i )*op( B_i )^T + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A_i )*op( B_i )^T will be symmetric.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == rocblas_operation_none
        op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == rocblas_operation_transpose

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  C_i is an upper triangular matrix
            rocblas_fill_lower:  C_i is a  lower triangular matrix

    @param[in]
    trans  [rocblas_operation]
            rocblas_operation_transpose:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
            rocblas_operation_none:           op( A_i ) = A_i, op( B_i ) = B_i

    @param[in]
    n       [rocblas_int]
            n specifies the number of rows and columns of C_i. n >= 0.

    @param[in]
    k       [rocblas_int]
            k specifies the number of columns of op(A). k >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and A need not be set before
            entry.

    @param[in]
    A       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
            when trans is rocblas_operation_none, otherwise of dimension (lda, n)

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A_i.
            if trans = rocblas_operation_none,  lda >= max( 1, n ),
            otherwise lda >= max( 1, k ).

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    B       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
            when trans is rocblas_operation_none, otherwise of dimension (ldb, n)

    @param[in]
    ldb     [rocblas_int]
            ldb specifies the first dimension of B_i.
            if trans = rocblas_operation_none,  ldb >= max( 1, n ),
            otherwise ldb >= max( 1, k ).

    @param[in]
    stride_B  [rocblas_stride]
              stride from the start of one matrix (B_i) and the next one (B_i+1)

    @param[in]
    beta
            beta specifies the scalar beta. When beta is
            zero then C need not be set before entry.

    @param[in]
    C       Device pointer to the first matrix C_1 on the GPU.

    @param[in]
    ldc    [rocblas_int]
           ldc specifies the first dimension of C. ldc >= max( 1, n ).

    @param[inout]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zsyrkx_strided_batched(rocblas_handle                handle,
                                                             rocblas_fill                  uplo,
                                                             rocblas_operation             trans,
                                                             rocblas_int                   n,
                                                             rocblas_int                   k,
                                                             const rocblas_double_complex* alpha,
                                                             const rocblas_double_complex* A,
                                                             rocblas_int                   lda,
                                                             rocblas_stride                stride_A,
                                                             const rocblas_double_complex* B,
                                                             rocblas_int                   ldb,
                                                             rocblas_stride                stride_B,
                                                             const rocblas_double_complex* beta,
                                                             rocblas_double_complex*       C,
                                                             rocblas_int                   ldc,
                                                             rocblas_stride                stride_C,
                                                             rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strmm(rocblas_handle    handle,
                                            rocblas_side      side,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            const float*      alpha,
                                            const float*      A,
                                            rocblas_int       lda,
                                            float*            B,
                                            rocblas_int       ldb);

ROCBLAS_EXPORT rocblas_status rocblas_dtrmm(rocblas_handle    handle,
                                            rocblas_side      side,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            const double*     alpha,
                                            const double*     A,
                                            rocblas_int       lda,
                                            double*           B,
                                            rocblas_int       ldb);

ROCBLAS_EXPORT rocblas_status rocblas_ctrmm(rocblas_handle               handle,
                                            rocblas_side                 side,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            transA,
                                            rocblas_diagonal             diag,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            rocblas_float_complex*       B,
                                            rocblas_int                  ldb);

/*! \brief BLAS Level 3 API

    \details

    trmm performs one of the matrix-matrix operations

    B := alpha*op( A )*B,   or   B := alpha*B*op( A )

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side    [rocblas_side]
            Specifies whether op(A) multiplies B from the left or right as follows:
            rocblas_side_left:       B := alpha*op( A )*B.
            rocblas_side_right:      B := alpha*B*op( A ).

    @param[in]
    uplo    [rocblas_fill]
            Specifies whether the matrix A is an uppoer or lower triangular matrix as follows:
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            Specifies the form of op(A) to be used in the matrix multiplication as follows:
            rocblas_operation_none:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.

    @param[in]
    diag    [rocblas_diagonal]
            Specifies whether or not A is unit triangular as follows:
            rocblas_diagonal_unit:      A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    A       Device pointer to matrix A on the GPU.
            A has dimension ( lda, k ), where k is m
            when  side == rocblas_side_left  and
            is  n  when  side == rocblas_side_right.

        When uplo == rocblas_fill_upper the  leading  k by k
        upper triangular part of the array  A must contain the upper
        triangular matrix  and the strictly lower triangular part of
        A is not referenced.

        When uplo == rocblas_fill_lower the  leading  k by k
        lower triangular part of the array  A must contain the lower
        triangular matrix  and the strictly upper triangular part of
        A is not referenced.

        Note that when  diag == rocblas_diagonal_unit  the diagonal elements of
        A  are not referenced either,  but are assumed to be  unity.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if side == rocblas_side_left,  lda >= max( 1, m ),
            if side == rocblas_side_right, lda >= max( 1, n ).

    @param[in]
    B       Device pointer to the first matrix B_0 on the GPU.
            On entry,  the leading  m by n part of the array  B must
           contain the matrix  B,  and  on exit  is overwritten  by the
           transformed matrix.

    @param[in]
    ldb    [rocblas_int]
           ldb specifies the first dimension of B. ldb >= max( 1, m ).

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrmm(rocblas_handle                handle,
                                            rocblas_side                  side,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             transA,
                                            rocblas_diagonal              diag,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            rocblas_double_complex*       B,
                                            rocblas_int                   ldb);

ROCBLAS_EXPORT rocblas_status rocblas_strmm_batched(rocblas_handle     handle,
                                                    rocblas_side       side,
                                                    rocblas_fill       uplo,
                                                    rocblas_operation  transa,
                                                    rocblas_diagonal   diag,
                                                    rocblas_int        m,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const a[],
                                                    rocblas_int        lda,
                                                    float* const       c[],
                                                    rocblas_int        ldc,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrmm_batched(rocblas_handle      handle,
                                                    rocblas_side        side,
                                                    rocblas_fill        uplo,
                                                    rocblas_operation   transa,
                                                    rocblas_diagonal    diag,
                                                    rocblas_int         m,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const a[],
                                                    rocblas_int         lda,
                                                    double* const       c[],
                                                    rocblas_int         ldc,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrmm_batched(rocblas_handle                     handle,
                                                    rocblas_side                       side,
                                                    rocblas_fill                       uplo,
                                                    rocblas_operation                  transa,
                                                    rocblas_diagonal                   diag,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const a[],
                                                    rocblas_int                        lda,
                                                    rocblas_float_complex* const       c[],
                                                    rocblas_int                        ldc,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 3 API

    \details

    trmm_batched performs one of the batched matrix-matrix operations

    B_i := alpha*op( A_i )*B_i,   or   B_i := alpha*B_i*op( A_i )  for i = 0, 1, ... batch_count -1

    where  alpha  is a scalar,  B_i  is an m by n matrix,  A_i  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A_i )  is one  of

        op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side    [rocblas_side]
            Specifies whether op(A_i) multiplies B_i from the left or right as follows:
            rocblas_side_left:       B_i := alpha*op( A_i )*B_i.
            rocblas_side_right:      B_i := alpha*B_i*op( A_i ).

    @param[in]
    uplo    [rocblas_fill]
            Specifies whether the matrix A is an uppoer or lower triangular matrix as follows:
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            Specifies the form of op(A_i) to be used in the matrix multiplication as follows:
            rocblas_operation_none:    op(A_i) = A_i.
            rocblas_operation_transpose:      op(A_i) = A_i^T.
            rocblas_operation_conjugate_transpose:  op(A_i) = A_i^H.

    @param[in]
    diag    [rocblas_diagonal]
            Specifies whether or not A_i is unit triangular as follows:
            rocblas_diagonal_unit:      A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B_i. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B_i. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A_i is not referenced and B_i need not be set before
            entry.

    @param[in]
    A       Device array of device pointers storing each matrix A_i on the GPU.
            Each A_i is of dimension ( lda, k ), where k is m
            when  side == rocblas_side_left  and
            is  n  when  side == rocblas_side_right.

        When uplo == rocblas_fill_upper the  leading  k by k
        upper triangular part of the array  A must contain the upper
        triangular matrix  and the strictly lower triangular part of
        A is not referenced.

        When uplo == rocblas_fill_lower the  leading  k by k
        lower triangular part of the array  A must contain the lower
        triangular matrix  and the strictly upper triangular part of
        A is not referenced.

        Note that when  diag == rocblas_diagonal_unit  the diagonal elements of
        A_i  are not referenced either,  but are assumed to be  unity.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if side == rocblas_side_left,  lda >= max( 1, m ),
            if side == rocblas_side_right, lda >= max( 1, n ).

    @param[in]
    B       device array of device pointers storing each matrix B_i on the GPU.
            On entry,  the leading  m by n part of the array  B_i must
           contain the matrix  B_i,  and  on exit  is overwritten  by the
           transformed matrix.

    @param[in]
    ldb    [rocblas_int]
           ldb specifies the first dimension of B_i. ldb >= max( 1, m ).

    @param[in]
    batch_count [rocblas_int]
                number of instances i in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrmm_batched(rocblas_handle                      handle,
                                                    rocblas_side                        side,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   transa,
                                                    rocblas_diagonal                    diag,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const a[],
                                                    rocblas_int                         lda,
                                                    rocblas_double_complex* const       c[],
                                                    rocblas_int                         ldc,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strmm_strided_batched(rocblas_handle    handle,
                                                            rocblas_side      side,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transa,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            const float*      alpha,
                                                            const float*      a,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_a,
                                                            float*            c,
                                                            rocblas_int       ldc,
                                                            rocblas_stride    stride_c,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrmm_strided_batched(rocblas_handle    handle,
                                                            rocblas_side      side,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transa,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            const double*     alpha,
                                                            const double*     a,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_a,
                                                            double*           c,
                                                            rocblas_int       ldc,
                                                            rocblas_stride    stride_c,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrmm_strided_batched(rocblas_handle               handle,
                                                            rocblas_side                 side,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            transa,
                                                            rocblas_diagonal             diag,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* a,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_a,
                                                            rocblas_float_complex*       c,
                                                            rocblas_int                  ldc,
                                                            rocblas_stride               stride_c,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details

    trmm_strided_batched performs one of the strided_batched matrix-matrix operations

    B_i := alpha*op( A_i )*B_i,   or   B_i := alpha*B_i*op( A_i )  for i = 0, 1, ... batch_count -1

    where  alpha  is a scalar,  B_i  is an m by n matrix,  A_i  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A_i )  is one  of

        op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side    [rocblas_side]
            Specifies whether op(A_i) multiplies B_i from the left or right as follows:
            rocblas_side_left:       B_i := alpha*op( A_i )*B_i.
            rocblas_side_right:      B_i := alpha*B_i*op( A_i ).

    @param[in]
    uplo    [rocblas_fill]
            Specifies whether the matrix A is an uppoer or lower triangular matrix as follows:
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            Specifies the form of op(A_i) to be used in the matrix multiplication as follows:
            rocblas_operation_none:    op(A_i) = A_i.
            rocblas_operation_transpose:      op(A_i) = A_i^T.
            rocblas_operation_conjugate_transpose:  op(A_i) = A_i^H.

    @param[in]
    diag    [rocblas_diagonal]
            Specifies whether or not A_i is unit triangular as follows:
            rocblas_diagonal_unit:      A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B_i. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B_i. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            zero then A_i is not referenced and B_i need not be set before
            entry.

    @param[in]
    A       Device pointer to the first matrix A_0 on the GPU.
            Each A_i is of dimension ( lda, k ), where k is m
            when  side == rocblas_side_left  and
            is  n  when  side == rocblas_side_right.

        When uplo == rocblas_fill_upper the  leading  k by k
        upper triangular part of the array  A must contain the upper
        triangular matrix  and the strictly lower triangular part of
        A is not referenced.

        When uplo == rocblas_fill_lower the  leading  k by k
        lower triangular part of the array  A must contain the lower
        triangular matrix  and the strictly upper triangular part of
        A is not referenced.

        Note that when  diag == rocblas_diagonal_unit  the diagonal elements of
        A_i  are not referenced either,  but are assumed to be  unity.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if side == rocblas_side_left,  lda >= max( 1, m ),
            if side == rocblas_side_right, lda >= max( 1, n ).

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    B       Device pointer to the first matrix B_0 on the GPU.
            On entry,  the leading  m by n part of the array  B_i must
           contain the matrix  B_i,  and  on exit  is overwritten  by the
           transformed matrix.

    @param[in]
    ldb    [rocblas_int]
           ldb specifies the first dimension of B_i. ldb >= max( 1, m ).

           @param[in]
    stride_B  [rocblas_stride]
              stride from the start of one matrix (B_i) and the next one (B_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances i in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrmm_strided_batched(rocblas_handle                handle,
                                                            rocblas_side                  side,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             transa,
                                                            rocblas_diagonal              diag,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* a,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_a,
                                                            rocblas_double_complex*       c,
                                                            rocblas_int                   ldc,
                                                            rocblas_stride                stride_c,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strtri(rocblas_handle   handle,
                                             rocblas_fill     uplo,
                                             rocblas_diagonal diag,
                                             rocblas_int      n,
                                             const float*     A,
                                             rocblas_int      lda,
                                             float*           invA,
                                             rocblas_int      ldinvA);

ROCBLAS_EXPORT rocblas_status rocblas_dtrtri(rocblas_handle   handle,
                                             rocblas_fill     uplo,
                                             rocblas_diagonal diag,
                                             rocblas_int      n,
                                             const double*    A,
                                             rocblas_int      lda,
                                             double*          invA,
                                             rocblas_int      ldinvA);

ROCBLAS_EXPORT rocblas_status rocblas_ctrtri(rocblas_handle               handle,
                                             rocblas_fill                 uplo,
                                             rocblas_diagonal             diag,
                                             rocblas_int                  n,
                                             const rocblas_float_complex* A,
                                             rocblas_int                  lda,
                                             rocblas_float_complex*       invA,
                                             rocblas_int                  ldinvA);

/*! \brief BLAS Level 3 API

    \details
    trtri  compute the inverse of a matrix A, namely, invA

        and write the result into invA;

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    diag      [rocblas_diagonal]
              = 'rocblas_diagonal_non_unit', A is non-unit triangular;
              = 'rocblas_diagonal_unit', A is unit triangular;
    @param[in]
    n         [rocblas_int]
              size of matrix A and invA
    @param[in]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[out]
    invA      device pointer storing matrix invA.
    @param[in]
    ldinvA    [rocblas_int]
              specifies the leading dimension of invA.

********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrtri(rocblas_handle                handle,
                                             rocblas_fill                  uplo,
                                             rocblas_diagonal              diag,
                                             rocblas_int                   n,
                                             const rocblas_double_complex* A,
                                             rocblas_int                   lda,
                                             rocblas_double_complex*       invA,
                                             rocblas_int                   ldinvA);

ROCBLAS_EXPORT rocblas_status rocblas_strtri_batched(rocblas_handle     handle,
                                                     rocblas_fill       uplo,
                                                     rocblas_diagonal   diag,
                                                     rocblas_int        n,
                                                     const float* const A[],
                                                     rocblas_int        lda,
                                                     float*             invA[],
                                                     rocblas_int        ldinvA,
                                                     rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrtri_batched(rocblas_handle      handle,
                                                     rocblas_fill        uplo,
                                                     rocblas_diagonal    diag,
                                                     rocblas_int         n,
                                                     const double* const A[],
                                                     rocblas_int         lda,
                                                     double*             invA[],
                                                     rocblas_int         ldinvA,
                                                     rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrtri_batched(rocblas_handle                     handle,
                                                     rocblas_fill                       uplo,
                                                     rocblas_diagonal                   diag,
                                                     rocblas_int                        n,
                                                     const rocblas_float_complex* const A[],
                                                     rocblas_int                        lda,
                                                     rocblas_float_complex*             invA[],
                                                     rocblas_int                        ldinvA,
                                                     rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details
    trtri_batched  compute the inverse of A_i and write into invA_i where
                   A_i and invA_i are the i-th matrices in the batch,
                   for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
    @param[in]
    diag      [rocblas_diagonal]
              = 'rocblas_diagonal_non_unit', A is non-unit triangular;
              = 'rocblas_diagonal_unit', A is unit triangular;
    @param[in]
    n         [rocblas_int]
    @param[in]
    A         device array of device pointers storing each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[out]
    invA      device array of device pointers storing the inverse of each matrix A_i.
              Partial inplace operation is supported, see below.
              If UPLO = 'U', the leading N-by-N upper triangular part of the invA will store
              the inverse of the upper triangular matrix, and the strictly lower
              triangular part of invA is cleared.
              If UPLO = 'L', the leading N-by-N lower triangular part of the invA will store
              the inverse of the lower triangular matrix, and the strictly upper
              triangular part of invA is cleared.
    @param[in]
    ldinvA    [rocblas_int]
              specifies the leading dimension of each invA_i.
    @param[in]
    batch_count [rocblas_int]
              numbers of matrices in the batch
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrtri_batched(rocblas_handle                      handle,
                                                     rocblas_fill                        uplo,
                                                     rocblas_diagonal                    diag,
                                                     rocblas_int                         n,
                                                     const rocblas_double_complex* const A[],
                                                     rocblas_int                         lda,
                                                     rocblas_double_complex*             invA[],
                                                     rocblas_int                         ldinvA,
                                                     rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strtri_strided_batched(rocblas_handle   handle,
                                                             rocblas_fill     uplo,
                                                             rocblas_diagonal diag,
                                                             rocblas_int      n,
                                                             const float*     A,
                                                             rocblas_int      lda,
                                                             rocblas_stride   stride_a,
                                                             float*           invA,
                                                             rocblas_int      ldinvA,
                                                             rocblas_stride   stride_invA,
                                                             rocblas_int      batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrtri_strided_batched(rocblas_handle   handle,
                                                             rocblas_fill     uplo,
                                                             rocblas_diagonal diag,
                                                             rocblas_int      n,
                                                             const double*    A,
                                                             rocblas_int      lda,
                                                             rocblas_stride   stride_a,
                                                             double*          invA,
                                                             rocblas_int      ldinvA,
                                                             rocblas_stride   stride_invA,
                                                             rocblas_int      batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrtri_strided_batched(rocblas_handle               handle,
                                                             rocblas_fill                 uplo,
                                                             rocblas_diagonal             diag,
                                                             rocblas_int                  n,
                                                             const rocblas_float_complex* A,
                                                             rocblas_int                  lda,
                                                             rocblas_stride               stride_a,
                                                             rocblas_float_complex*       invA,
                                                             rocblas_int                  ldinvA,
                                                             rocblas_stride stride_invA,
                                                             rocblas_int    batch_count);

/*! \brief BLAS Level 3 API

    \details
    trtri_strided_batched compute the inverse of A_i and write into invA_i where
                   A_i and invA_i are the i-th matrices in the batch,
                   for i = 1, ..., batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
    @param[in]
    diag      [rocblas_diagonal]
              = 'rocblas_diagonal_non_unit', A is non-unit triangular;
              = 'rocblas_diagonal_unit', A is unit triangular;
    @param[in]
    n         [rocblas_int]
    @param[in]
    A         device pointer pointing to address of first matrix A_1.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A.
    @param[in]
    stride_a  [rocblas_stride]
             "batch stride a": stride from the start of one A_i matrix to the next A_(i + 1).
    @param[out]
    invA      device pointer storing the inverses of each matrix A_i.
              Partial inplace operation is supported, see below.
              If UPLO = 'U', the leading N-by-N upper triangular part of the invA will store
              the inverse of the upper triangular matrix, and the strictly lower
              triangular part of invA is cleared.
              If UPLO = 'L', the leading N-by-N lower triangular part of the invA will store
              the inverse of the lower triangular matrix, and the strictly upper
              triangular part of invA is cleared.
    @param[in]
    ldinvA    [rocblas_int]
              specifies the leading dimension of each invA_i.
    @param[in]
    stride_invA  [rocblas_stride]
                 "batch stride invA": stride from the start of one invA_i matrix to the next invA_(i + 1).
    @param[in]
    batch_count  [rocblas_int]
                 numbers of matrices in the batch
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrtri_strided_batched(rocblas_handle                handle,
                                                             rocblas_fill                  uplo,
                                                             rocblas_diagonal              diag,
                                                             rocblas_int                   n,
                                                             const rocblas_double_complex* A,
                                                             rocblas_int                   lda,
                                                             rocblas_stride                stride_a,
                                                             rocblas_double_complex*       invA,
                                                             rocblas_int                   ldinvA,
                                                             rocblas_stride stride_invA,
                                                             rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strsm(rocblas_handle    handle,
                                            rocblas_side      side,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            const float*      alpha,
                                            const float*      A,
                                            rocblas_int       lda,
                                            float*            B,
                                            rocblas_int       ldb);

ROCBLAS_EXPORT rocblas_status rocblas_dtrsm(rocblas_handle    handle,
                                            rocblas_side      side,
                                            rocblas_fill      uplo,
                                            rocblas_operation transA,
                                            rocblas_diagonal  diag,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            const double*     alpha,
                                            const double*     A,
                                            rocblas_int       lda,
                                            double*           B,
                                            rocblas_int       ldb);

ROCBLAS_EXPORT rocblas_status rocblas_ctrsm(rocblas_handle               handle,
                                            rocblas_side                 side,
                                            rocblas_fill                 uplo,
                                            rocblas_operation            transA,
                                            rocblas_diagonal             diag,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            rocblas_float_complex*       B,
                                            rocblas_int                  ldb);

/*! \brief BLAS Level 3 API

    \details

    trsm solves

        op(A)*X = alpha*B or  X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices,
    A is triangular matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    The matrix X is overwritten on B.

    Note about memory allocation:
    When trsm is launched with a k evenly divisible by the internal block size of 128,
    and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
    memory found in the handle to increase overall performance. This memory can be managed by using
    the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
    used for temporary storage will default to 1 MB and may result in chunking, which in turn may
    reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
    to the desired chunk of right hand sides to be used at a time.

    (where k is m when rocblas_side_left and is n when rocblas_side_right)

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    side    [rocblas_side]
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B. n >= 0.

    @param[in]
    alpha
            device pointer or host pointer specifying the scalar alpha. When alpha is
            &zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    A       device pointer storing matrix A.
            of dimension ( lda, k ), where k is m
            when  rocblas_side_left  and
            is  n  when  rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).

    @param[in,out]
    B       device pointer storing matrix B.

    @param[in]
    ldb    [rocblas_int]
           ldb specifies the first dimension of B. ldb >= max( 1, m ).

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrsm(rocblas_handle                handle,
                                            rocblas_side                  side,
                                            rocblas_fill                  uplo,
                                            rocblas_operation             transA,
                                            rocblas_diagonal              diag,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            rocblas_double_complex*       B,
                                            rocblas_int                   ldb);

ROCBLAS_EXPORT rocblas_status rocblas_strsm_batched(rocblas_handle     handle,
                                                    rocblas_side       side,
                                                    rocblas_fill       uplo,
                                                    rocblas_operation  transA,
                                                    rocblas_diagonal   diag,
                                                    rocblas_int        m,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    float*             B[],
                                                    rocblas_int        ldb,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrsm_batched(rocblas_handle      handle,
                                                    rocblas_side        side,
                                                    rocblas_fill        uplo,
                                                    rocblas_operation   transA,
                                                    rocblas_diagonal    diag,
                                                    rocblas_int         m,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    double*             B[],
                                                    rocblas_int         ldb,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrsm_batched(rocblas_handle                     handle,
                                                    rocblas_side                       side,
                                                    rocblas_fill                       uplo,
                                                    rocblas_operation                  transA,
                                                    rocblas_diagonal                   diag,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    rocblas_float_complex*             B[],
                                                    rocblas_int                        ldb,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 3 API
    \details
    trsm_batched performs the following batched operation:

        op(A_i)*X_i = alpha*B_i or  X_i*op(A_i) = alpha*B_i, for i = 1, ..., batch_count.

    where alpha is a scalar, X and B are batched m by n matrices,
    A is triangular batched matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    Each matrix X_i is overwritten on B_i for i = 1, ..., batch_count.

    Note about memory allocation:
    When trsm is launched with a k evenly divisible by the internal block size of 128,
    and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
    memory found in the handle to increase overall performance. This memory can be managed by using
    the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
    used for temporary storage will default to 1 MB and may result in chunking, which in turn may
    reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
    to the desired chunk of right hand sides to be used at a time.
    (where k is m when rocblas_side_left and is n when rocblas_side_right)

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    side    [rocblas_side]
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.
    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  each A_i is an upper triangular matrix.
            rocblas_fill_lower:  each A_i is a  lower triangular matrix.
    @param[in]
    transA  [rocblas_operation]
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.
    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  each A_i is not assumed to be unit triangular.
    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of each B_i. m >= 0.
    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of each B_i. n >= 0.
    @param[in]
    alpha
            device pointer or host pointer specifying the scalar alpha. When alpha is
            &zero then A is not referenced and B need not be set before
            entry.
    @param[in]
    A       device array of device pointers storing each matrix A_i on the GPU.
            Matricies are of dimension ( lda, k ), where k is m
            when  rocblas_side_left  and is  n  when  rocblas_side_right
            only the upper/lower triangular part is accessed.
    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of each A_i.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).
    @param[in,out]
    B       device array of device pointers storing each matrix B_i on the GPU.
    @param[in]
    ldb    [rocblas_int]
           ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
    @param[in]
    batch_count [rocblas_int]
                number of trsm operatons in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrsm_batched(rocblas_handle                      handle,
                                                    rocblas_side                        side,
                                                    rocblas_fill                        uplo,
                                                    rocblas_operation                   transA,
                                                    rocblas_diagonal                    diag,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    rocblas_double_complex*             B[],
                                                    rocblas_int                         ldb,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strsm_strided_batched(rocblas_handle    handle,
                                                            rocblas_side      side,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            const float*      alpha,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_a,
                                                            float*            B,
                                                            rocblas_int       ldb,
                                                            rocblas_stride    stride_b,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrsm_strided_batched(rocblas_handle    handle,
                                                            rocblas_side      side,
                                                            rocblas_fill      uplo,
                                                            rocblas_operation transA,
                                                            rocblas_diagonal  diag,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            const double*     alpha,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_a,
                                                            double*           B,
                                                            rocblas_int       ldb,
                                                            rocblas_stride    stride_b,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ctrsm_strided_batched(rocblas_handle               handle,
                                                            rocblas_side                 side,
                                                            rocblas_fill                 uplo,
                                                            rocblas_operation            transA,
                                                            rocblas_diagonal             diag,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_a,
                                                            rocblas_float_complex*       B,
                                                            rocblas_int                  ldb,
                                                            rocblas_stride               stride_b,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 3 API
    \details
    trsm_srided_batched performs the following strided batched operation:

        op(A_i)*X_i = alpha*B_i or  X_i*op(A_i) = alpha*B_i, for i = 1, ..., batch_count.

    where alpha is a scalar, X and B are strided batched m by n matrices,
    A is triangular strided batched matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    Each matrix X_i is overwritten on B_i for i = 1, ..., batch_count.

    Note about memory allocation:
    When trsm is launched with a k evenly divisible by the internal block size of 128,
    and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
    memory found in the handle to increase overall performance. This memory can be managed by using
    the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
    used for temporary storage will default to 1 MB and may result in chunking, which in turn may
    reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
    to the desired chunk of right hand sides to be used at a time.
    (where k is m when rocblas_side_left and is n when rocblas_side_right)
    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    side    [rocblas_side]
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.
    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  each A_i is an upper triangular matrix.
            rocblas_fill_lower:  each A_i is a  lower triangular matrix.
    @param[in]
    transA  [rocblas_operation]
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.
    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  each A_i is not assumed to be unit triangular.
    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of each B_i. m >= 0.
    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of each B_i. n >= 0.
    @param[in]
    alpha
            device pointer or host pointer specifying the scalar alpha. When alpha is
            &zero then A is not referenced and B need not be set before
            entry.
    @param[in]
    A       device pointer pointing to the first matrix A_1.
            of dimension ( lda, k ), where k is m
            when  rocblas_side_left  and
            is  n  when  rocblas_side_right
            only the upper/lower triangular part is accessed.
    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of each A_i.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).
    @param[in]
    stride_a [rocblas_stride]
             stride from the start of one A_i matrix to the next A_(i + 1).
    @param[in,out]
    B       device pointer pointing to the first matrix B_1.
    @param[in]
    ldb    [rocblas_int]
           ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
    @param[in]
    stride_b [rocblas_stride]
             stride from the start of one B_i matrix to the next B_(i + 1).
    @param[in]
    batch_count [rocblas_int]
                number of trsm operatons in the batch.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_ztrsm_strided_batched(rocblas_handle                handle,
                                                            rocblas_side                  side,
                                                            rocblas_fill                  uplo,
                                                            rocblas_operation             transA,
                                                            rocblas_diagonal              diag,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_a,
                                                            rocblas_double_complex*       B,
                                                            rocblas_int                   ldb,
                                                            rocblas_stride                stride_b,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgemm(rocblas_handle    handle,
                                            rocblas_operation transA,
                                            rocblas_operation transB,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const float*      alpha,
                                            const float*      A,
                                            rocblas_int       lda,
                                            const float*      B,
                                            rocblas_int       ldb,
                                            const float*      beta,
                                            float*            C,
                                            rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_dgemm(rocblas_handle    handle,
                                            rocblas_operation transA,
                                            rocblas_operation transB,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const double*     alpha,
                                            const double*     A,
                                            rocblas_int       lda,
                                            const double*     B,
                                            rocblas_int       ldb,
                                            const double*     beta,
                                            double*           C,
                                            rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_hgemm(rocblas_handle      handle,
                                            rocblas_operation   transA,
                                            rocblas_operation   transB,
                                            rocblas_int         m,
                                            rocblas_int         n,
                                            rocblas_int         k,
                                            const rocblas_half* alpha,
                                            const rocblas_half* A,
                                            rocblas_int         lda,
                                            const rocblas_half* B,
                                            rocblas_int         ldb,
                                            const rocblas_half* beta,
                                            rocblas_half*       C,
                                            rocblas_int         ldc);

ROCBLAS_EXPORT rocblas_status rocblas_cgemm(rocblas_handle               handle,
                                            rocblas_operation            transA,
                                            rocblas_operation            transB,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            rocblas_int                  k,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* B,
                                            rocblas_int                  ldb,
                                            const rocblas_float_complex* beta,
                                            rocblas_float_complex*       C,
                                            rocblas_int                  ldc);

/*! \brief BLAS Level 3 API

    \details
    xGEMM performs one of the matrix-matrix operations

        C = alpha*op( A )*op( B ) + beta*C,

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A )
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B )
    @param[in]
    m         [rocblas_int]
              number or rows of matrices op( A ) and C
    @param[in]
    n         [rocblas_int]
              number of columns of matrices op( B ) and C
    @param[in]
    k         [rocblas_int]
              number of columns of matrix op( A ) and number of rows of matrix op( B )
    @param[in]
    alpha     device pointer or host pointer specifying the scalar alpha.
    @param[in]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    B         device pointer storing matrix B.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of B.
    @param[in]
    beta      device pointer or host pointer specifying the scalar beta.
    @param[in, out]
    C         device pointer storing matrix C on the GPU.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgemm(rocblas_handle                handle,
                                            rocblas_operation             transA,
                                            rocblas_operation             transB,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            rocblas_int                   k,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* B,
                                            rocblas_int                   ldb,
                                            const rocblas_double_complex* beta,
                                            rocblas_double_complex*       C,
                                            rocblas_int                   ldc);

ROCBLAS_EXPORT rocblas_status rocblas_sgemm_batched(rocblas_handle     handle,
                                                    rocblas_operation  transA,
                                                    rocblas_operation  transB,
                                                    rocblas_int        m,
                                                    rocblas_int        n,
                                                    rocblas_int        k,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    const float* const B[],
                                                    rocblas_int        ldb,
                                                    const float*       beta,
                                                    float* const       C[],
                                                    rocblas_int        ldc,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgemm_batched(rocblas_handle      handle,
                                                    rocblas_operation   transA,
                                                    rocblas_operation   transB,
                                                    rocblas_int         m,
                                                    rocblas_int         n,
                                                    rocblas_int         k,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    const double* const B[],
                                                    rocblas_int         ldb,
                                                    const double*       beta,
                                                    double* const       C[],
                                                    rocblas_int         ldc,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_hgemm_batched(rocblas_handle            handle,
                                                    rocblas_operation         transA,
                                                    rocblas_operation         transB,
                                                    rocblas_int               m,
                                                    rocblas_int               n,
                                                    rocblas_int               k,
                                                    const rocblas_half*       alpha,
                                                    const rocblas_half* const A[],
                                                    rocblas_int               lda,
                                                    const rocblas_half* const B[],
                                                    rocblas_int               ldb,
                                                    const rocblas_half*       beta,
                                                    rocblas_half* const       C[],
                                                    rocblas_int               ldc,
                                                    rocblas_int               batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgemm_batched(rocblas_handle                     handle,
                                                    rocblas_operation                  transA,
                                                    rocblas_operation                  transB,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    rocblas_int                        k,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex* const B[],
                                                    rocblas_int                        ldb,
                                                    const rocblas_float_complex*       beta,
                                                    rocblas_float_complex* const       C[],
                                                    rocblas_int                        ldc,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 3 API
     \details
    xGEMM_BATCHED performs one of the batched matrix-matrix operations
         C_i = alpha*op( A_i )*op( B_i ) + beta*C_i, for i = 1, ..., batch_count.
     where op( X ) is one of
         op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
     alpha and beta are scalars, and A, B and C are strided batched matrices, with
    op( A ) an m by k by batch_count strided_batched matrix,
    op( B ) an k by n by batch_count strided_batched matrix and
    C an m by n by batch_count strided_batched matrix.
    @param[in]
    handle    [rocblas_handle
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A )
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B )
    @param[in]
    m         [rocblas_int]
              matrix dimention m.
    @param[in]
    n         [rocblas_int]
              matrix dimention n.
    @param[in]
    k         [rocblas_int]
              matrix dimention k.
    @param[in]
    alpha     device pointer or host pointer specifying the scalar alpha.
    @param[in]
    A         device array of device pointers storing each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    B         device array of device pointers storing each matrix B_i.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of each B_i.
    @param[in]
    beta      device pointer or host pointer specifying the scalar beta.
    @param[in, out]
    C         device array of device pointers storing each matrix C_i.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of each C_i.
    @param[in]
    batch_count
              [rocblas_int]
              number of gemm operations in the batch
     ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgemm_batched(rocblas_handle                      handle,
                                                    rocblas_operation                   transa,
                                                    rocblas_operation                   transb,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    rocblas_int                         k,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex* const B[],
                                                    rocblas_int                         ldb,
                                                    const rocblas_double_complex*       beta,
                                                    rocblas_double_complex* const       C[],
                                                    rocblas_int                         ldc,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgemm_strided_batched(rocblas_handle    handle,
                                                            rocblas_operation transA,
                                                            rocblas_operation transB,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            rocblas_int       k,
                                                            const float*      alpha,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_a,
                                                            const float*      B,
                                                            rocblas_int       ldb,
                                                            rocblas_stride    stride_b,
                                                            const float*      beta,
                                                            float*            C,
                                                            rocblas_int       ldc,
                                                            rocblas_stride    stride_c,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgemm_strided_batched(rocblas_handle    handle,
                                                            rocblas_operation transA,
                                                            rocblas_operation transB,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            rocblas_int       k,
                                                            const double*     alpha,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_a,
                                                            const double*     B,
                                                            rocblas_int       ldb,
                                                            rocblas_stride    stride_b,
                                                            const double*     beta,
                                                            double*           C,
                                                            rocblas_int       ldc,
                                                            rocblas_stride    stride_c,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_hgemm_strided_batched(rocblas_handle      handle,
                                                            rocblas_operation   transA,
                                                            rocblas_operation   transB,
                                                            rocblas_int         m,
                                                            rocblas_int         n,
                                                            rocblas_int         k,
                                                            const rocblas_half* alpha,
                                                            const rocblas_half* A,
                                                            rocblas_int         lda,
                                                            rocblas_stride      stride_a,
                                                            const rocblas_half* B,
                                                            rocblas_int         ldb,
                                                            rocblas_stride      stride_b,
                                                            const rocblas_half* beta,
                                                            rocblas_half*       C,
                                                            rocblas_int         ldc,
                                                            rocblas_stride      stride_c,
                                                            rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_hgemm_kernel_name(rocblas_handle      handle,
                                                        rocblas_operation   transA,
                                                        rocblas_operation   transB,
                                                        rocblas_int         m,
                                                        rocblas_int         n,
                                                        rocblas_int         k,
                                                        const rocblas_half* alpha,
                                                        const rocblas_half* A,
                                                        rocblas_int         lda,
                                                        rocblas_stride      stride_a,
                                                        const rocblas_half* B,
                                                        rocblas_int         ldb,
                                                        rocblas_stride      stride_b,
                                                        const rocblas_half* beta,
                                                        rocblas_half*       C,
                                                        rocblas_int         ldc,
                                                        rocblas_stride      stride_c,
                                                        rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgemm_kernel_name(rocblas_handle    handle,
                                                        rocblas_operation transA,
                                                        rocblas_operation transB,
                                                        rocblas_int       m,
                                                        rocblas_int       n,
                                                        rocblas_int       k,
                                                        const float*      alpha,
                                                        const float*      A,
                                                        rocblas_int       lda,
                                                        rocblas_stride    stride_a,
                                                        const float*      B,
                                                        rocblas_int       ldb,
                                                        rocblas_stride    stride_b,
                                                        const float*      beta,
                                                        float*            C,
                                                        rocblas_int       ldc,
                                                        rocblas_stride    stride_c,
                                                        rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgemm_kernel_name(rocblas_handle    handle,
                                                        rocblas_operation transA,
                                                        rocblas_operation transB,
                                                        rocblas_int       m,
                                                        rocblas_int       n,
                                                        rocblas_int       k,
                                                        const double*     alpha,
                                                        const double*     A,
                                                        rocblas_int       lda,
                                                        rocblas_stride    stride_a,
                                                        const double*     B,
                                                        rocblas_int       ldb,
                                                        rocblas_stride    stride_b,
                                                        const double*     beta,
                                                        double*           C,
                                                        rocblas_int       ldc,
                                                        rocblas_stride    stride_c,
                                                        rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgemm_strided_batched(rocblas_handle               handle,
                                                            rocblas_operation            transa,
                                                            rocblas_operation            transb,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            rocblas_int                  k,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_a,
                                                            const rocblas_float_complex* B,
                                                            rocblas_int                  ldb,
                                                            rocblas_stride               stride_b,
                                                            const rocblas_float_complex* beta,
                                                            rocblas_float_complex*       C,
                                                            rocblas_int                  ldc,
                                                            rocblas_stride               stride_c,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details
    xGEMM_STRIDED_BATCHED performs one of the strided batched matrix-matrix operations

        C_i = alpha*op( A_i )*op( B_i ) + beta*C_i, for i = 1, ..., batch_count.

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are strided batched matrices, with
    op( A ) an m by k by batch_count strided_batched matrix,
    op( B ) an k by n by batch_count strided_batched matrix and
    C an m by n by batch_count strided_batched matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A )
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B )
    @param[in]
    m         [rocblas_int]
              matrix dimention m.
    @param[in]
    n         [rocblas_int]
              matrix dimention n.
    @param[in]
    k         [rocblas_int]
              matrix dimention k.
    @param[in]
    alpha     device pointer or host pointer specifying the scalar alpha.
    @param[in]
    A         device pointer pointing to the first matrix A_1.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    stride_a  [rocblas_stride]
              stride from the start of one A_i matrix to the next A_(i + 1).
    @param[in]
    B         device pointer pointing to the first matrix B_1.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of each B_i.
    @param[in]
    stride_b  [rocblas_stride]
              stride from the start of one B_i matrix to the next B_(i + 1).
    @param[in]
    beta      device pointer or host pointer specifying the scalar beta.
    @param[in, out]
    C         device pointer pointing to the first matrix C_1.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of each C_i.
    @param[in]
    stride_c  [rocblas_stride]
              stride from the start of one C_i matrix to the next C_(i + 1).
    @param[in]
    batch_count
              [rocblas_int]
              number of gemm operatons in the batch

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgemm_strided_batched(rocblas_handle                handle,
                                                            rocblas_operation             transa,
                                                            rocblas_operation             transb,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            rocblas_int                   k,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_a,
                                                            const rocblas_double_complex* B,
                                                            rocblas_int                   ldb,
                                                            rocblas_stride                stride_b,
                                                            const rocblas_double_complex* beta,
                                                            rocblas_double_complex*       C,
                                                            rocblas_int                   ldc,
                                                            rocblas_stride                stride_c,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sdgmm(rocblas_handle handle,
                                            rocblas_side   side,
                                            rocblas_int    m,
                                            rocblas_int    n,
                                            const float*   A,
                                            rocblas_int    lda,
                                            const float*   x,
                                            rocblas_int    incx,
                                            float*         C,
                                            rocblas_int    ldc);

ROCBLAS_EXPORT rocblas_status rocblas_ddgmm(rocblas_handle handle,
                                            rocblas_side   side,
                                            rocblas_int    m,
                                            rocblas_int    n,
                                            const double*  A,
                                            rocblas_int    lda,
                                            const double*  x,
                                            rocblas_int    incx,
                                            double*        C,
                                            rocblas_int    ldc);

ROCBLAS_EXPORT rocblas_status rocblas_cdgmm(rocblas_handle               handle,
                                            rocblas_side                 side,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* x,
                                            rocblas_int                  incx,
                                            rocblas_float_complex*       C,
                                            rocblas_int                  ldc);

/*! \brief BLAS Level 3 API

    \details
    xDGMM performs one of the matrix-matrix operations

        C = A * diag(x) if side == rocblas_side_right
        C = diag(x) * A if side == rocblas_side_left

    where C and A are m by n dimensional matrices. diag( x ) is a diagonal matrix
    and x is vector of dimension n if side == rocblas_side_right and dimension m
    if side == rocblas_side_left.


    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    side      [rocblas_side]
              specifies the side of diag(x)
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment between values of x
    @param[in, out]
    C         device pointer storing matrix C.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdgmm(rocblas_handle                handle,
                                            rocblas_side                  side,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* x,
                                            rocblas_int                   incx,
                                            rocblas_double_complex*       C,
                                            rocblas_int                   ldc);

ROCBLAS_EXPORT rocblas_status rocblas_sdgmm_batched(rocblas_handle     handle,
                                                    rocblas_side       side,
                                                    rocblas_int        m,
                                                    rocblas_int        n,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    const float* const x[],
                                                    rocblas_int        incx,
                                                    float* const       C[],
                                                    rocblas_int        ldc,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ddgmm_batched(rocblas_handle      handle,
                                                    rocblas_side        side,
                                                    rocblas_int         m,
                                                    rocblas_int         n,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    const double* const x[],
                                                    rocblas_int         incx,
                                                    double* const       C[],
                                                    rocblas_int         ldc,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cdgmm_batched(rocblas_handle                     handle,
                                                    rocblas_side                       side,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex* const x[],
                                                    rocblas_int                        incx,
                                                    rocblas_float_complex* const       C[],
                                                    rocblas_int                        ldc,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 3 API

    \details
    xDGMM_batched performs one of the batched matrix-matrix operations

        C_i = A_i * diag(x_i) for i = 0, 1, ... batch_count-1 if side == rocblas_side_right
        C_i = diag(x_i) * A_i for i = 0, 1, ... batch_count-1 if side == rocblas_side_left

    where C_i and A_i are m by n dimensional matrices. diag(x_i) is a diagonal matrix
    and x_i is vector of dimension n if side == rocblas_side_right and dimension m
    if side == rocblas_side_left.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    side      [rocblas_side]
              specifies the side of diag(x)
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    A         device array of device pointers storing each matrix A_i on the GPU.
              Each A_i is of dimension ( lda, n )
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A_i.
    @param[in]
    x         device array of device pointers storing each vector x_i on the GPU.
              Each x_i is of dimension n if side == rocblas_side_right and dimension
              m if side == rocblas_side_left
    @param[in]
    incx      [rocblas_int]
              specifies the increment between values of x_i
    @param[in, out]
    C         device array of device pointers storing each matrix C_i on the GPU.
              Each C_i is of dimension ( ldc, n ).
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdgmm_batched(rocblas_handle                      handle,
                                                    rocblas_side                        side,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex* const x[],
                                                    rocblas_int                         incx,
                                                    rocblas_double_complex* const       C[],
                                                    rocblas_int                         ldc,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sdgmm_strided_batched(rocblas_handle handle,
                                                            rocblas_side   side,
                                                            rocblas_int    m,
                                                            rocblas_int    n,
                                                            const float*   A,
                                                            rocblas_int    lda,
                                                            rocblas_stride stride_a,
                                                            const float*   x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            float*         C,
                                                            rocblas_int    ldc,
                                                            rocblas_stride stride_c,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_ddgmm_strided_batched(rocblas_handle handle,
                                                            rocblas_side   side,
                                                            rocblas_int    m,
                                                            rocblas_int    n,
                                                            const double*  A,
                                                            rocblas_int    lda,
                                                            rocblas_stride stride_a,
                                                            const double*  x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stride_x,
                                                            double*        C,
                                                            rocblas_int    ldc,
                                                            rocblas_stride stride_c,
                                                            rocblas_int    batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cdgmm_strided_batched(rocblas_handle               handle,
                                                            rocblas_side                 side,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_a,
                                                            const rocblas_float_complex* x,
                                                            rocblas_int                  incx,
                                                            rocblas_stride               stride_x,
                                                            rocblas_float_complex*       C,
                                                            rocblas_int                  ldc,
                                                            rocblas_stride               stride_c,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details
    xDGMM_strided_batched performs one of the batched matrix-matrix operations

        C_i = A_i * diag(x_i)   if side == rocblas_side_right   for i = 0, 1, ... batch_count-1
        C_i = diag(x_i) * A_i   if side == rocblas_side_left    for i = 0, 1, ... batch_count-1

    where C_i and A_i are m by n dimensional matrices. diag(x_i) is a diagonal matrix
    and x_i is vector of dimension n if side == rocblas_side_right and dimension m
    if side == rocblas_side_left.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    side      [rocblas_side]
              specifies the side of diag(x)
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    A         device pointer to the first matrix A_0 on the GPU.
              Each A_i is of dimension ( lda, n )
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         pointer to the first vector x_0 on the GPU.
              Each x_i is of dimension n if side == rocblas_side_right and dimension
              m if side == rocblas_side_left
    @param[in]
    incx      [rocblas_int]
              specifies the increment between values of x
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector(x_i) and the next one (x_i+1)
    @param[in, out]
    C         device pointer to the first matrix C_0 on the GPU.
              Each C_i is of dimension ( ldc, n ).
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C.
    @param[in]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances i in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zdgmm_strided_batched(rocblas_handle                handle,
                                                            rocblas_side                  side,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_a,
                                                            const rocblas_double_complex* x,
                                                            rocblas_int                   incx,
                                                            rocblas_stride                stride_x,
                                                            rocblas_double_complex*       C,
                                                            rocblas_int                   ldc,
                                                            rocblas_stride                stride_c,
                                                            rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgeam(rocblas_handle    handle,
                                            rocblas_operation transA,
                                            rocblas_operation transB,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            const float*      alpha,
                                            const float*      A,
                                            rocblas_int       lda,
                                            const float*      beta,
                                            const float*      B,
                                            rocblas_int       ldb,
                                            float*            C,
                                            rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_dgeam(rocblas_handle    handle,
                                            rocblas_operation transA,
                                            rocblas_operation transB,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            const double*     alpha,
                                            const double*     A,
                                            rocblas_int       lda,
                                            const double*     beta,
                                            const double*     B,
                                            rocblas_int       ldb,
                                            double*           C,
                                            rocblas_int       ldc);

ROCBLAS_EXPORT rocblas_status rocblas_cgeam(rocblas_handle               handle,
                                            rocblas_operation            transA,
                                            rocblas_operation            transB,
                                            rocblas_int                  m,
                                            rocblas_int                  n,
                                            const rocblas_float_complex* alpha,
                                            const rocblas_float_complex* A,
                                            rocblas_int                  lda,
                                            const rocblas_float_complex* beta,
                                            const rocblas_float_complex* B,
                                            rocblas_int                  ldb,
                                            rocblas_float_complex*       C,
                                            rocblas_int                  ldc);

/*! \brief BLAS Level 3 API

    \details
    xGEAM performs one of the matrix-matrix operations

        C = alpha*op( A ) + beta*op( B ),

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by n matrix, op( B ) an m by n matrix, and C an m by n matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A )
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B )
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    alpha     device pointer or host pointer specifying the scalar alpha.
    @param[in]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    beta      device pointer or host pointer specifying the scalar beta.
    @param[in]
    B         device pointer storing matrix B.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of B.
    @param[in, out]
    C         device pointer storing matrix C.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgeam(rocblas_handle                handle,
                                            rocblas_operation             transA,
                                            rocblas_operation             transB,
                                            rocblas_int                   m,
                                            rocblas_int                   n,
                                            const rocblas_double_complex* alpha,
                                            const rocblas_double_complex* A,
                                            rocblas_int                   lda,
                                            const rocblas_double_complex* beta,
                                            const rocblas_double_complex* B,
                                            rocblas_int                   ldb,
                                            rocblas_double_complex*       C,
                                            rocblas_int                   ldc);

ROCBLAS_EXPORT rocblas_status rocblas_sgeam_batched(rocblas_handle     handle,
                                                    rocblas_operation  transA,
                                                    rocblas_operation  transB,
                                                    rocblas_int        m,
                                                    rocblas_int        n,
                                                    const float*       alpha,
                                                    const float* const A[],
                                                    rocblas_int        lda,
                                                    const float*       beta,
                                                    const float* const B[],
                                                    rocblas_int        ldb,
                                                    float* const       C[],
                                                    rocblas_int        ldc,
                                                    rocblas_int        batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgeam_batched(rocblas_handle      handle,
                                                    rocblas_operation   transA,
                                                    rocblas_operation   transB,
                                                    rocblas_int         m,
                                                    rocblas_int         n,
                                                    const double*       alpha,
                                                    const double* const A[],
                                                    rocblas_int         lda,
                                                    const double*       beta,
                                                    const double* const B[],
                                                    rocblas_int         ldb,
                                                    double* const       C[],
                                                    rocblas_int         ldc,
                                                    rocblas_int         batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgeam_batched(rocblas_handle                     handle,
                                                    rocblas_operation                  transA,
                                                    rocblas_operation                  transB,
                                                    rocblas_int                        m,
                                                    rocblas_int                        n,
                                                    const rocblas_float_complex*       alpha,
                                                    const rocblas_float_complex* const A[],
                                                    rocblas_int                        lda,
                                                    const rocblas_float_complex*       beta,
                                                    const rocblas_float_complex* const B[],
                                                    rocblas_int                        ldb,
                                                    rocblas_float_complex* const       C[],
                                                    rocblas_int                        ldc,
                                                    rocblas_int                        batch_count);

/*! \brief BLAS Level 3 API

    \details
    xGEAM_batched performs one of the batched matrix-matrix operations

        C_i = alpha*op( A_i ) + beta*op( B_i )  for i = 0, 1, ... batch_count - 1

    where alpha and beta are scalars, and op(A_i), op(B_i) and C_i are m by n matrices
    and op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A )
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B )
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    alpha     device pointer or host pointer specifying the scalar alpha.
    @param[in]
    A         device array of device pointers storing each matrix A_i on the GPU.
              Each A_i is of dimension ( lda, k ), where k is m
              when  transA == rocblas_operation_none and
              is  n  when  transA == rocblas_operation_transpose.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    beta      device pointer or host pointer specifying the scalar beta.
    @param[in]
    B         device array of device pointers storing each matrix B_i on the GPU.
              Each B_i is of dimension ( ldb, k ), where k is m
              when  transB == rocblas_operation_none and
              is  n  when  transB == rocblas_operation_transpose.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of B.
    @param[in, out]
    C         device array of device pointers storing each matrix C_i on the GPU.
              Each C_i is of dimension ( ldc, n ).
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C.

    @param[in]
    batch_count [rocblas_int]
                number of instances i in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgeam_batched(rocblas_handle                      handle,
                                                    rocblas_operation                   transA,
                                                    rocblas_operation                   transB,
                                                    rocblas_int                         m,
                                                    rocblas_int                         n,
                                                    const rocblas_double_complex*       alpha,
                                                    const rocblas_double_complex* const A[],
                                                    rocblas_int                         lda,
                                                    const rocblas_double_complex*       beta,
                                                    const rocblas_double_complex* const B[],
                                                    rocblas_int                         ldb,
                                                    rocblas_double_complex* const       C[],
                                                    rocblas_int                         ldc,
                                                    rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_sgeam_strided_batched(rocblas_handle    handle,
                                                            rocblas_operation transA,
                                                            rocblas_operation transB,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            const float*      alpha,
                                                            const float*      A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_a,
                                                            const float*      beta,
                                                            const float*      B,
                                                            rocblas_int       ldb,
                                                            rocblas_stride    stride_b,
                                                            float*            C,
                                                            rocblas_int       ldc,
                                                            rocblas_stride    stride_c,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dgeam_strided_batched(rocblas_handle    handle,
                                                            rocblas_operation transA,
                                                            rocblas_operation transB,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            const double*     alpha,
                                                            const double*     A,
                                                            rocblas_int       lda,
                                                            rocblas_stride    stride_a,
                                                            const double*     beta,
                                                            const double*     B,
                                                            rocblas_int       ldb,
                                                            rocblas_stride    stride_b,
                                                            double*           C,
                                                            rocblas_int       ldc,
                                                            rocblas_stride    stride_c,
                                                            rocblas_int       batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_cgeam_strided_batched(rocblas_handle               handle,
                                                            rocblas_operation            transA,
                                                            rocblas_operation            transB,
                                                            rocblas_int                  m,
                                                            rocblas_int                  n,
                                                            const rocblas_float_complex* alpha,
                                                            const rocblas_float_complex* A,
                                                            rocblas_int                  lda,
                                                            rocblas_stride               stride_a,
                                                            const rocblas_float_complex* beta,
                                                            const rocblas_float_complex* B,
                                                            rocblas_int                  ldb,
                                                            rocblas_stride               stride_b,
                                                            rocblas_float_complex*       C,
                                                            rocblas_int                  ldc,
                                                            rocblas_stride               stride_c,
                                                            rocblas_int batch_count);

/*! \brief BLAS Level 3 API

    \details
    xGEAM_strided_batched performs one of the batched matrix-matrix operations

        C_i = alpha*op( A_i ) + beta*op( B_i )  for i = 0, 1, ... batch_count - 1

    where alpha and beta are scalars, and op(A_i), op(B_i) and C_i are m by n matrices
    and op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A )

    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B )

    @param[in]
    m         [rocblas_int]
              matrix dimension m.

    @param[in]
    n         [rocblas_int]
              matrix dimension n.

    @param[in]
    alpha     device pointer or host pointer specifying the scalar alpha.

    @param[in]
    A         device pointer to the first matrix A_0 on the GPU.
              Each A_i is of dimension ( lda, k ), where k is m
              when  transA == rocblas_operation_none and
              is  n  when  transA == rocblas_operation_transpose.

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)

    @param[in]
    beta      device pointer or host pointer specifying the scalar beta.

    @param[in]
    B         pointer to the first matrix B_0 on the GPU.
              Each B_i is of dimension ( ldb, k ), where k is m
              when  transB == rocblas_operation_none and
              is  n  when  transB == rocblas_operation_transpose.

    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of B.

    @param[in]
    stride_B  [rocblas_stride]
              stride from the start of one matrix (B_i) and the next one (B_i+1)

    @param[in, out]
    C         pointer to the first matrix C_0 on the GPU.
              Each C_i is of dimension ( ldc, n ).

    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C.

    @param[in]
    stride_C  [rocblas_stride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    @param[in]
    batch_count [rocblas_int]
                number of instances i in the batch.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_zgeam_strided_batched(rocblas_handle                handle,
                                                            rocblas_operation             transA,
                                                            rocblas_operation             transB,
                                                            rocblas_int                   m,
                                                            rocblas_int                   n,
                                                            const rocblas_double_complex* alpha,
                                                            const rocblas_double_complex* A,
                                                            rocblas_int                   lda,
                                                            rocblas_stride                stride_a,
                                                            const rocblas_double_complex* beta,
                                                            const rocblas_double_complex* B,
                                                            rocblas_int                   ldb,
                                                            rocblas_stride                stride_b,
                                                            rocblas_double_complex*       C,
                                                            rocblas_int                   ldc,
                                                            rocblas_stride                stride_c,
                                                            rocblas_int batch_count);

/*
 * ===========================================================================
 *    BLAS extensions
 * ===========================================================================
 */

/*! \brief BLAS EX API

    \details
    GEMM_EX performs one of the matrix-matrix operations

        D = alpha*op( A )*op( B ) + beta*C,

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B, C, and D are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C and D are m by n matrices.

    Supported types are as follows:
        - rocblas_datatype_f64_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f32_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r =
   compute_type
        - rocblas_datatype_bf16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r =
   compute_type
        - rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r = c_type = d_type =
   compute_type
        - rocblas_datatype_f32_c  = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f64_c  = a_type = b_type = c_type = d_type = compute_type

    Below are restrictions for rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r =
   c_type = d_type = compute_type:
        - k must be a multiple of 4
        - lda must be a multiple of 4 if transA == rocblas_operation_transpose
        - ldb must be a multiple of 4 if transB == rocblas_operation_none
        - for transA == rocblas_operation_transpose or transB == rocblas_operation_none the matrices
   A and B must
          have each 4 consecutive values in the k dimension packed. This packing can be achieved
   with the following
          pseudo-code. The code assumes the original matrices are in A and B, and the packed
   matrices are A_packed
          and B_packed. The size of the A_packed matrix is the same as the size of the A matrix, and
   the size of
          the B_packed matrix is the same as the size of the B matrix.

    @code
    if(transA == rocblas_operation_none)
    {
        int nb = 4;
        for(int i_m = 0; i_m < m; i_m++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                A_packed[i_k % nb + (i_m + (i_k / nb) * lda) * nb] = A[i_m + i_k * lda];
            }
        }
    }
    else
    {
        A_packed = A;
    }
    if(transB == rocblas_operation_transpose)
    {
        int nb = 4;
        for(int i_n = 0; i_n < m; i_n++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                B_packed[i_k % nb + (i_n + (i_k / nb) * lda) * nb] = B[i_n + i_k * lda];
            }
        }
    }
    else
    {
        B_packed = B;
    }
    @endcode

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A ).
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B ).
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    k         [rocblas_int]
              matrix dimension k.
    @param[in]
    alpha     [const void *]
              device pointer or host pointer specifying the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         [void *]
              device pointer storing matrix A.
    @param[in]
    a_type    [rocblas_datatype]
              specifies the datatype of matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    b         [void *]
              device pointer storing matrix B.
    @param[in]
    b_type    [rocblas_datatype]
              specifies the datatype of matrix B.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of B.
    @param[in]
    beta      [const void *]
              device pointer or host pointer specifying the scalar beta. Same datatype as compute_type.
    @param[in]
    c         [void *]
              device pointer storing matrix C.
    @param[in]
    c_type    [rocblas_datatype]
              specifies the datatype of matrix C.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C.
    @param[out]
    d         [void *]
              device pointer storing matrix D.
    @param[in]
    d_type    [rocblas_datatype]
              specifies the datatype of matrix D.
    @param[in]
    ldd       [rocblas_int]
              specifies the leading dimension of D.
    @param[in]
    compute_type
              [rocblas_datatype]
              specifies the datatype of computation.
    @param[in]
    algo      [rocblas_gemm_algo]
              enumerant specifying the algorithm type.
    @param[in]
    solution_index
              [int32_t]
              reserved for future use.
    @param[in]
    flags     [uint32_t]
              reserved for future use.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_gemm_ex(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const void*       alpha,
                                              const void*       a,
                                              rocblas_datatype  a_type,
                                              rocblas_int       lda,
                                              const void*       b,
                                              rocblas_datatype  b_type,
                                              rocblas_int       ldb,
                                              const void*       beta,
                                              const void*       c,
                                              rocblas_datatype  c_type,
                                              rocblas_int       ldc,
                                              void*             d,
                                              rocblas_datatype  d_type,
                                              rocblas_int       ldd,
                                              rocblas_datatype  compute_type,
                                              rocblas_gemm_algo algo,
                                              int32_t           solution_index,
                                              uint32_t          flags);

/* For backward compatiblity, unused workspace_size and workspace arguments are ignored */
// clang-format off
#define rocblas_gemm_ex(handle,         \
                        transA,         \
                        transB,         \
                        m,              \
                        n,              \
                        k,              \
                        alpha,          \
                        a,              \
                        a_type,         \
                        lda,            \
                        b,              \
                        b_type,         \
                        ldb,            \
                        beta,           \
                        c,              \
                        c_type,         \
                        ldc,            \
                        d,              \
                        d_type,         \
                        ldd,            \
                        compute_type,   \
                        algo,           \
                        solution_index, \
                        flags,          \
                        ...)            \
                        ROCBLAS_VA_OPT_PRAGMA(GCC warning "rocblas_gemm_ex: The workspace_size and workspace arguments are obsolete, and will be ignored", __VA_ARGS__) \
        rocblas_gemm_ex(handle,         \
                        transA,         \
                        transB,         \
                        m,              \
                        n,              \
                        k,              \
                        alpha,          \
                        a,              \
                        a_type,         \
                        lda,            \
                        b,              \
                        b_type,         \
                        ldb,            \
                        beta,           \
                        c,              \
                        c_type,         \
                        ldc,            \
                        d,              \
                        d_type,         \
                        ldd,            \
                        compute_type,   \
                        algo,           \
                        solution_index, \
                        flags)
// clang-format on

/*! \brief BLAS EX API
    \details
    GEMM_BATCHED_EX performs one of the batched matrix-matrix operations
        D_i = alpha*op(A_i)*op(B_i) + beta*C_i, for i = 1, ..., batch_count.
    where op( X ) is one of
        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
    alpha and beta are scalars, and A, B, C, and D are batched pointers to matrices, with
    op( A ) an m by k by batch_count batched matrix,
    op( B ) a k by n by batch_count batched matrix and
    C and D are m by n by batch_count batched matrices.
    The batched matrices are an array of pointers to matrices.
    The number of pointers to matrices is batch_count.
    Supported types are as follows:
        - rocblas_datatype_f64_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f32_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r =
   compute_type
        - rocblas_datatype_bf16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r =
   compute_type
        - rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r = c_type = d_type =
   compute_type
        - rocblas_datatype_f32_c  = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f64_c  = a_type = b_type = c_type = d_type = compute_type
    Below are restrictions for rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r =
   c_type = d_type = compute_type:
        - k must be a multiple of 4
        - lda must be a multiple of 4 if transA == rocblas_operation_transpose
        - ldb must be a multiple of 4 if transB == rocblas_operation_none
        - for transA == rocblas_operation_transpose or transB == rocblas_operation_none the matrices
   A and B must
          have each 4 consecutive values in the k dimension packed. This packing can be achieved
   with the following
          pseudo-code. The code assumes the original matrices are in A and B, and the packed
   matrices are A_packed
          and B_packed. The size of the A_packed matrix is the same as the size of the A matrix, and
   the size of
          the B_packed matrix is the same as the size of the B matrix.
    @code
    if(transA == rocblas_operation_none)
    {
        int nb = 4;
        for(int i_m = 0; i_m < m; i_m++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                A_packed[i_k % nb + (i_m + (i_k / nb) * lda) * nb] = A[i_m + i_k * lda];
            }
        }
    }
    else
    {
        A_packed = A;
    }
    if(transB == rocblas_operation_transpose)
    {
        int nb = 4;
        for(int i_n = 0; i_n < m; i_n++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                B_packed[i_k % nb + (i_n + (i_k / nb) * lda) * nb] = B[i_n + i_k * lda];
            }
        }
    }
    else
    {
        B_packed = B;
    }
    @endcode
    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A ).
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B ).
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    k         [rocblas_int]
              matrix dimension k.
    @param[in]
    alpha     [const void *]
              device pointer or host pointer specifying the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         [void *]
              device pointer storing array of pointers to each matrix A_i.
    @param[in]
    a_type    [rocblas_datatype]
              specifies the datatype of each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    b         [void *]
              device pointer storing array of pointers to each matrix B_i.
    @param[in]
    b_type    [rocblas_datatype]
              specifies the datatype of each matrix B_i.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of each B_i.
    @param[in]
    beta      [const void *]
              device pointer or host pointer specifying the scalar beta. Same datatype as compute_type.
    @param[in]
    c         [void *]
              device array of device pointers to each matrix C_i.
    @param[in]
    c_type    [rocblas_datatype]
              specifies the datatype of each matrix C_i.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of each C_i.
    @param[out]
    d         [void *]
              device array of device pointers to each matrix D_i.
    @param[in]
    d_type    [rocblas_datatype]
              specifies the datatype of each matrix D_i.
    @param[in]
    ldd       [rocblas_int]
              specifies the leading dimension of each D_i.
    @param[in]
    batch_count
              [rocblas_int]
              number of gemm operations in the batch.
    @param[in]
    compute_type
              [rocblas_datatype]
              specifies the datatype of computation.
    @param[in]
    algo      [rocblas_gemm_algo]
              enumerant specifying the algorithm type.
    @param[in]
    solution_index
              [int32_t]
              reserved for future use.
    @param[in]
    flags     [uint32_t]
              reserved for future use.
    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_gemm_batched_ex(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_operation transB,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const void*       alpha,
                                                      const void*       a,
                                                      rocblas_datatype  a_type,
                                                      rocblas_int       lda,
                                                      const void*       b,
                                                      rocblas_datatype  b_type,
                                                      rocblas_int       ldb,
                                                      const void*       beta,
                                                      const void*       c,
                                                      rocblas_datatype  c_type,
                                                      rocblas_int       ldc,
                                                      void*             d,
                                                      rocblas_datatype  d_type,
                                                      rocblas_int       ldd,
                                                      rocblas_int       batch_count,
                                                      rocblas_datatype  compute_type,
                                                      rocblas_gemm_algo algo,
                                                      int32_t           solution_index,
                                                      uint32_t          flags);

/*! \brief BLAS EX API

    \details
    GEMM_STRIDED_BATCHED_EX performs one of the strided_batched matrix-matrix operations

        D_i = alpha*op(A_i)*op(B_i) + beta*C_i, for i = 1, ..., batch_count

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B, C, and D are strided_batched matrices, with
    op( A ) an m by k by batch_count strided_batched matrix,
    op( B ) a k by n by batch_count strided_batched matrix and
    C and D are m by n by batch_count strided_batched matrices.

    The strided_batched matrices are multiple matrices separated by a constant stride.
    The number of matrices is batch_count.

    Supported types are as follows:
        - rocblas_datatype_f64_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f32_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r =
   compute_type
        - rocblas_datatype_bf16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r =
   compute_type
        - rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r = c_type = d_type =
   compute_type
        - rocblas_datatype_f32_c  = a_type = b_type = c_type = d_type = compute_type
        - rocblas_datatype_f64_c  = a_type = b_type = c_type = d_type = compute_type

    Below are restrictions for rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r =
   c_type = d_type = compute_type:
        - k must be a multiple of 4
        - lda must be a multiple of 4 if transA == rocblas_operation_transpose
        - ldb must be a multiple of 4 if transB == rocblas_operation_none
        - for transA == rocblas_operation_transpose or transB == rocblas_operation_none the matrices
   A and B must
          have each 4 consecutive values in the k dimension packed. This packing can be achieved
   with the following
          pseudo-code. The code assumes the original matrices are in A and B, and the packed
   matrices are A_packed
          and B_packed. The size of the A_packed matrix is the same as the size of the A matrix, and
   the size of
          the B_packed matrix is the same as the size of the B matrix.

    @code
    if(transA == rocblas_operation_none)
    {
        int nb = 4;
        for(int i_m = 0; i_m < m; i_m++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                A_packed[i_k % nb + (i_m + (i_k / nb) * lda) * nb] = A[i_m + i_k * lda];
            }
        }
    }
    else
    {
        A_packed = A;
    }
    if(transB == rocblas_operation_transpose)
    {
        int nb = 4;
        for(int i_n = 0; i_n < m; i_n++)
        {
            for(int i_k = 0; i_k < k; i_k++)
            {
                B_packed[i_k % nb + (i_n + (i_k / nb) * lda) * nb] = B[i_n + i_k * lda];
            }
        }
    }
    else
    {
        B_packed = B;
    }
    @endcode

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A ).
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B ).
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    k         [rocblas_int]
              matrix dimension k.
    @param[in]
    alpha     [const void *]
              device pointer or host pointer specifying the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         [void *]
              device pointer pointing to first matrix A_1.
    @param[in]
    a_type    [rocblas_datatype]
              specifies the datatype of each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    stride_a  [rocblas_stride]
              specifies stride from start of one A_i matrix to the next A_(i + 1).
    @param[in]
    b         [void *]
              device pointer pointing to first matrix B_1.
    @param[in]
    b_type    [rocblas_datatype]
              specifies the datatype of each matrix B_i.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of each B_i.
    @param[in]
    stride_b  [rocblas_stride]
              specifies stride from start of one B_i matrix to the next B_(i + 1).
    @param[in]
    beta      [const void *]
              device pointer or host pointer specifying the scalar beta. Same datatype as compute_type.
    @param[in]
    c         [void *]
              device pointer pointing to first matrix C_1.
    @param[in]
    c_type    [rocblas_datatype]
              specifies the datatype of each matrix C_i.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of each C_i.
    @param[in]
    stride_c  [rocblas_stride]
              specifies stride from start of one C_i matrix to the next C_(i + 1).
    @param[out]
    d         [void *]
              device pointer storing each matrix D_i.
    @param[in]
    d_type    [rocblas_datatype]
              specifies the datatype of each matrix D_i.
    @param[in]
    ldd       [rocblas_int]
              specifies the leading dimension of each D_i.
    @param[in]
    stride_d  [rocblas_stride]
              specifies stride from start of one D_i matrix to the next D_(i + 1).
    @param[in]
    batch_count
              [rocblas_int]
              number of gemm operations in the batch.
    @param[in]
    compute_type
              [rocblas_datatype]
              specifies the datatype of computation.
    @param[in]
    algo      [rocblas_gemm_algo]
              enumerant specifying the algorithm type.
    @param[in]
    solution_index
              [int32_t]
              reserved for future use.
    @param[in]
    flags     [uint32_t]
              reserved for future use.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_gemm_strided_batched_ex(rocblas_handle    handle,
                                                              rocblas_operation transA,
                                                              rocblas_operation transB,
                                                              rocblas_int       m,
                                                              rocblas_int       n,
                                                              rocblas_int       k,
                                                              const void*       alpha,
                                                              const void*       a,
                                                              rocblas_datatype  a_type,
                                                              rocblas_int       lda,
                                                              rocblas_stride    stride_a,
                                                              const void*       b,
                                                              rocblas_datatype  b_type,
                                                              rocblas_int       ldb,
                                                              rocblas_stride    stride_b,
                                                              const void*       beta,
                                                              const void*       c,
                                                              rocblas_datatype  c_type,
                                                              rocblas_int       ldc,
                                                              rocblas_stride    stride_c,
                                                              void*             d,
                                                              rocblas_datatype  d_type,
                                                              rocblas_int       ldd,
                                                              rocblas_stride    stride_d,
                                                              rocblas_int       batch_count,
                                                              rocblas_datatype  compute_type,
                                                              rocblas_gemm_algo algo,
                                                              int32_t           solution_index,
                                                              uint32_t          flags);

/* For backward compatiblity, unused workspace_size and workspace arguments are ignored */
// clang-format off
#define rocblas_gemm_strided_batched_ex(handle,         \
                                        transA,         \
                                        transB,        \
                                        m,              \
                                        n,              \
                                        k,              \
                                        alpha,          \
                                        a,              \
                                        a_type,         \
                                        lda,            \
                                        stride_a,       \
                                        b,              \
                                        b_type,         \
                                        ldb,            \
                                        stride_b,       \
                                        beta,           \
                                        c,              \
                                        c_type,         \
                                        ldc,            \
                                        stride_c,       \
                                        d,              \
                                        d_type,         \
                                        ldd,            \
                                        stride_d,       \
                                        batch_count,    \
                                        compute_type,   \
                                        algo,           \
                                        solution_index, \
                                        flags,          \
                                        ...)            \
                                        ROCBLAS_VA_OPT_PRAGMA(GCC warning "rocblas_gemm_strided_batched_ex: The workspace_size and workspace arguments are obsolete, and will be ignored", __VA_ARGS__) \
        rocblas_gemm_strided_batched_ex(handle,         \
                                        transA,         \
                                        transB,        \
                                        m,              \
                                        n,              \
                                        k,              \
                                        alpha,          \
                                        a,              \
                                        a_type,         \
                                        lda,            \
                                        stride_a,       \
                                        b,              \
                                        b_type,         \
                                        ldb,            \
                                        stride_b,       \
                                        beta,           \
                                        c,              \
                                        c_type,         \
                                        ldc,            \
                                        stride_c,       \
                                        d,              \
                                        d_type,         \
                                        ldd,            \
                                        stride_d,       \
                                        batch_count,    \
                                        compute_type,   \
                                        algo,           \
                                        solution_index, \
                                        flags)

// clang-format on

/*! BLAS EX API

    \details
    TRSM_EX solves

        op(A)*X = alpha*B or X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices,
    A is triangular matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    The matrix X is overwritten on B.

    TRSM_EX gives the user the ability to reuse the invA matrix between runs.
    If invA == NULL, rocblas_trsm_ex will automatically calculate invA on every run.

    Setting up invA:
    The accepted invA matrix consists of the packed 128x128 inverses of the diagonal blocks of
    matrix A, followed by any smaller diagonal block that remains.
    To set up invA it is recommended that rocblas_trtri_batched be used with matrix A as the input.

    Device memory of size 128 x k should be allocated for invA ahead of time, where k is m when
    rocblas_side_left and is n when rocblas_side_right. The actual number of elements in invA
    should be passed as invA_size.

    To begin, rocblas_trtri_batched must be called on the full 128x128 sized diagonal blocks of
    matrix A. Below are the restricted parameters:
      - n = 128
      - ldinvA = 128
      - stride_invA = 128x128
      - batch_count = k / 128,

    Then any remaining block may be added:
      - n = k % 128
      - invA = invA + stride_invA * previous_batch_count
      - ldinvA = 128
      - batch_count = 1

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.

    @param[in]
    side    [rocblas_side]
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of B. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of B. n >= 0.

    @param[in]
    alpha   [void *]
            device pointer or host pointer specifying the scalar alpha. When alpha is
            &zero then A is not referenced, and B need not be set before
            entry.

    @param[in]
    A       [void *]
            device pointer storing matrix A.
            of dimension ( lda, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).

    @param[in, out]
    B       [void *]
            device pointer storing matrix B.
            B is of dimension ( ldb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B, and on exit is
            overwritten by the solution matrix X.

    @param[in]
    ldb    [rocblas_int]
           ldb specifies the first dimension of B. ldb >= max( 1, m ).

    @param[in]
    invA    [void *]
            device pointer storing the inverse diagonal blocks of A.
            invA is of dimension ( ld_invA, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right.
            ld_invA must be equal to 128.

    @param[in]
    invA_size [rocblas_int]
            invA_size specifies the number of elements of device memory in invA.

    @param[in]
    compute_type [rocblas_datatype]
            specifies the datatype of computation

    ********************************************************************/

ROCBLAS_EXPORT rocblas_status rocblas_trsm_ex(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const void*       alpha,
                                              const void*       A,
                                              rocblas_int       lda,
                                              void*             B,
                                              rocblas_int       ldb,
                                              const void*       invA,
                                              rocblas_int       invA_size,
                                              rocblas_datatype  compute_type);

/* For backward compatiblity, option, x_temp_size and x_temp_workspace arguments are ignored */
// clang-format off
#define rocblas_trsm_ex(handle,       \
                        side,         \
                        uplo,         \
                        transA,       \
                        diag,         \
                        m,            \
                        n,            \
                        alpha,        \
                        A,            \
                        lda,          \
                        B,            \
                        ldb,          \
                        invA,         \
                        invA_size,    \
                        compute_type, \
                        ...)          \
                        ROCBLAS_VA_OPT_PRAGMA(GCC warning "rocblas_trsm_ex: The option, x_temp_size and x_temp_workspace arguments are obsolete, and will be ignored", __VA_ARGS__) \
        rocblas_trsm_ex(handle,       \
                        side,         \
                        uplo,         \
                        transA,       \
                        diag,         \
                        m,            \
                        n,            \
                        alpha,        \
                        A,            \
                        lda,          \
                        B,            \
                        ldb,          \
                        invA,         \
                        invA_size,    \
                        compute_type)
// clang-format on

/*! BLAS EX API

    \details
    TRSM_BATCHED_EX solves

        op(A_i)*X_i = alpha*B_i or X_i*op(A_i) = alpha*B_i,

    for i = 1, ..., batch_count; and where alpha is a scalar, X and B are arrays of m by n matrices,
    A is an array of triangular matrix and each op(A_i) is one of

        op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.

    Each matrix X_i is overwritten on B_i.

    TRSM_EX gives the user the ability to reuse the invA matrix between runs.
    If invA == NULL, rocblas_trsm_batched_ex will automatically calculate each invA_i on every run.

    Setting up invA:
    Each accepted invA_i matrix consists of the packed 128x128 inverses of the diagonal blocks of
    matrix A_i, followed by any smaller diagonal block that remains.
    To set up each invA_i it is recommended that rocblas_trtri_batched be used with matrix A_i as the input.
    invA is an array of pointers of batch_count length holding each invA_i.

    Device memory of size 128 x k should be allocated for each invA_i ahead of time, where k is m when
    rocblas_side_left and is n when rocblas_side_right. The actual number of elements in each invA_i
    should be passed as invA_size.

    To begin, rocblas_trtri_batched must be called on the full 128x128 sized diagonal blocks of each
    matrix A_i. Below are the restricted parameters:
      - n = 128
      - ldinvA = 128
      - stride_invA = 128x128
      - batch_count = k / 128,

    Then any remaining block may be added:
      - n = k % 128
      - invA = invA + stride_invA * previous_batch_count
      - ldinvA = 128
      - batch_count = 1

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.

    @param[in]
    side    [rocblas_side]
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  each A_i is an upper triangular matrix.
            rocblas_fill_lower:  each A_i is a lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  each A_i is not assumed to be unit triangular.

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of each B_i. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of each B_i. n >= 0.

    @param[in]
    alpha   [void *]
            device pointer or host pointer alpha specifying the scalar alpha. When alpha is
            &zero then A is not referenced, and B need not be set before
            entry.

    @param[in]
    A       [void *]
            device array of device pointers storing each matrix A_i.
            each A_i is of dimension ( lda, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of each A_i.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).

    @param[in, out]
    B       [void *]
            device array of device pointers storing each matrix B_i.
            each B_i is of dimension ( ldb, n ).
            Before entry, the leading m by n part of the array B_i must
            contain the right-hand side matrix B_i, and on exit is
            overwritten by the solution matrix X_i

    @param[in]
    ldb    [rocblas_int]
           ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).

    @param[in]
    batch_count [rocblas_int]
            specifies how many batches.

    @param[in]
    invA    [void *]
            device array of device pointers storing the inverse diagonal blocks of each A_i.
            each invA_i is of dimension ( ld_invA, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right.
            ld_invA must be equal to 128.

    @param[in]
    invA_size [rocblas_int]
            invA_size specifies the number of elements of device memory in each invA_i.

    @param[in]
    compute_type [rocblas_datatype]
            specifies the datatype of computation

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_trsm_batched_ex(rocblas_handle    handle,
                                                      rocblas_side      side,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const void*       alpha,
                                                      const void*       A,
                                                      rocblas_int       lda,
                                                      void*             B,
                                                      rocblas_int       ldb,
                                                      rocblas_int       batch_count,
                                                      const void*       invA,
                                                      rocblas_int       invA_size,
                                                      rocblas_datatype  compute_type);

/*! BLAS EX API

    \details
    TRSM_STRIDED_BATCHED_EX solves

        op(A_i)*X_i = alpha*B_i or X_i*op(A_i) = alpha*B_i,

    for i = 1, ..., batch_count; and where alpha is a scalar, X and B are strided batched m by n matrices,
    A is a strided batched triangular matrix and op(A_i) is one of

        op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.

    Each matrix X_i is overwritten on B_i.

    TRSM_EX gives the user the ability to reuse each invA_i matrix between runs.
    If invA == NULL, rocblas_trsm_batched_ex will automatically calculate each invA_i on every run.

    Setting up invA:
    Each accepted invA_i matrix consists of the packed 128x128 inverses of the diagonal blocks of
    matrix A_i, followed by any smaller diagonal block that remains.
    To set up invA_i it is recommended that rocblas_trtri_batched be used with matrix A_i as the input.
    invA is a contiguous piece of memory holding each invA_i.

    Device memory of size 128 x k should be allocated for each invA_i ahead of time, where k is m when
    rocblas_side_left and is n when rocblas_side_right. The actual number of elements in each invA_i
    should be passed as invA_size.

    To begin, rocblas_trtri_batched must be called on the full 128x128 sized diagonal blocks of each
    matrix A_i. Below are the restricted parameters:
      - n = 128
      - ldinvA = 128
      - stride_invA = 128x128
      - batch_count = k / 128,

    Then any remaining block may be added:
      - n = k % 128
      - invA = invA + stride_invA * previous_batch_count
      - ldinvA = 128
      - batch_count = 1

    @param[in]
    handle  [rocblas_handle]
            handle to the rocblas library context queue.

    @param[in]
    side    [rocblas_side]
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  each A_i is an upper triangular matrix.
            rocblas_fill_lower:  each A_i is a lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  each A_i is not assumed to be unit triangular.

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of each B_i. m >= 0.

    @param[in]
    n       [rocblas_int]
            n specifies the number of columns of each B_i. n >= 0.

    @param[in]
    alpha   [void *]
            device pointer or host pointer specifying the scalar alpha. When alpha is
            &zero then A is not referenced, and B need not be set before
            entry.

    @param[in]
    A       [void *]
            device pointer storing matrix A.
            of dimension ( lda, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     [rocblas_int]
            lda specifies the first dimension of A.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).

    @param[in]
    stride_A [rocblas_stride]
            The stride between each A matrix.

    @param[in, out]
    B       [void *]
            device pointer pointing to first matrix B_i.
            each B_i is of dimension ( ldb, n ).
            Before entry, the leading m by n part of each array B_i must
            contain the right-hand side of matrix B_i, and on exit is
            overwritten by the solution matrix X_i.

    @param[in]
    ldb    [rocblas_int]
           ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).

    @param[in]
    stride_B [rocblas_stride]
            The stride between each B_i matrix.

    @param[in]
    batch_count [rocblas_int]
            specifies how many batches.

    @param[in]
    invA    [void *]
            device pointer storing the inverse diagonal blocks of each A_i.
            invA points to the first invA_1.
            each invA_i is of dimension ( ld_invA, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right.
            ld_invA must be equal to 128.

    @param[in]
    invA_size [rocblas_int]
            invA_size specifies the number of elements of device memory in each invA_i.

    @param[in]
    stride_invA [rocblas_stride]
            The stride between each invA matrix.

    @param[in]
    compute_type [rocblas_datatype]
            specifies the datatype of computation

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_trsm_strided_batched_ex(rocblas_handle    handle,
                                                              rocblas_side      side,
                                                              rocblas_fill      uplo,
                                                              rocblas_operation transA,
                                                              rocblas_diagonal  diag,
                                                              rocblas_int       m,
                                                              rocblas_int       n,
                                                              const void*       alpha,
                                                              const void*       A,
                                                              rocblas_int       lda,
                                                              rocblas_stride    stride_A,
                                                              void*             B,
                                                              rocblas_int       ldb,
                                                              rocblas_stride    stride_B,
                                                              rocblas_int       batch_count,
                                                              const void*       invA,
                                                              rocblas_int       invA_size,
                                                              rocblas_stride    stride_invA,
                                                              rocblas_datatype  compute_type);

/*! BLAS Auxiliary API

    \details
    rocblas_status_to_string

    Returns string representing rocblas_status value

    @param[in]
    status  [rocblas_status]
            rocBLAS status to convert to string
*/

ROCBLAS_EXPORT const char* rocblas_status_to_string(rocblas_status status);

/* \brief Initialize rocBLAS, to avoid costly startup time at the first call.
*/

ROCBLAS_EXPORT void rocblas_initialize(void);

/*
 * ===========================================================================
 *    build information
 * ===========================================================================
 */

/*! \brief   loads char* buf with the rocblas library version. size_t len
    is the maximum length of char* buf.
    \details

    @param[in, out]
    buf             pointer to buffer for version string

    @param[in]
    len             length of buf

 ******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_version_string(char* buf, size_t len);

/*
 * ===========================================================================
 *    device memory allocation
 * ===========================================================================
 */

/*! \brief
    \details
    Indicates that subsequent rocBLAS kernel calls should collect the optimal device memory size in bytes for their given kernel arguments,
    and keep track of the maximum.
    Each kernel call can reuse temporary device memory on the same stream, so the maximum is collected.
    Returns rocblas_status_size_query_mismatch if another size query is already in progress; returns rocblas_status_success otherwise.
    @param[in]
    handle          rocblas handle
 ******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_start_device_memory_size_query(rocblas_handle handle);

/*! \brief
    \details
    Stops collecting optimal device memory size information
    Returns rocblas_status_size_query_mismatch if a collection is not underway; rocblas_status_invalid_handle if handle is nullptr;
    rocblas_status_invalid_pointer if size is nullptr; rocblas_status_success otherwise
    @param[in]
    handle          rocblas handle
    @param[out]
    size             maximum of the optimal sizes collected
 ******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_stop_device_memory_size_query(rocblas_handle handle,
                                                                    size_t*        size);

/*! \brief
    \details
    Gets the current device memory size for the handle
    Returns rocblas_status_invalid_handle if handle is nullptr; rocblas_status_invalid_pointer if size is nullptr; rocblas_status_success otherwise
    @param[in]
    handle          rocblas handle
    @param[out]
    size             current device memory size for the handle
 ******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_device_memory_size(rocblas_handle handle, size_t* size);

/*! \brief
    \details
    Changes the size of allocated device memory at runtime.

    Any previously allocated device memory is freed.

    If size > 0 sets the device memory size to the specified size (in bytes)
    If size == 0 frees the memory allocated so far, and lets rocBLAS manage device memory in the future, expanding it when necessary
    Returns rocblas_status_invalid_handle if handle is nullptr; rocblas_status_invalid_pointer if size is nullptr; rocblas_status_success otherwise
    @param[in]
    handle          rocblas handle
    @param[in]
    size             size of allocated device memory
 ******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_set_device_memory_size(rocblas_handle handle, size_t size);

/*! \brief
    \details
    Returns true when device memory in handle is managed by rocBLAS
    @param[in]
    handle          rocblas handle
 ******************************************************************************/
ROCBLAS_EXPORT bool rocblas_is_managing_device_memory(rocblas_handle handle);

/*! \brief
    \details
    Abort function which safely flushes all IO
 ******************************************************************************/
ROCBLAS_EXPORT void rocblas_abort(void) __attribute__((__noreturn__));

#ifdef __cplusplus
}
#endif

#endif /* _ROCBLAS_FUNCTIONS_H_ */
