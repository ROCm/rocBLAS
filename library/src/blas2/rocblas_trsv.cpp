/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"

#include "status.h"
#include "definitions.h"
#include "trsv_device.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

template <typename T, const rocblas_int NB_X>
__global__ void gemvc_kernel_device_pointer(rocblas_operation transA,
                                            rocblas_int m,
                                            rocblas_int n,
                                            const T* alpha,
                                            const T* __restrict__ A,
                                            rocblas_int lda,
                                            const T* __restrict__ x,
                                            rocblas_int incx,
                                            const T* beta,
                                            T* y,
                                            rocblas_int incy)
{
    gemvc_device<T, NB_X>(m, n, *alpha, A, lda, x, incx, *beta, y, incy);
}

/*! \brief BLAS Level 2 API

    \details
    trsv solves

         A*x = alpha*b or A**T*x = alpha*b,

    where x and b are vectors and A is a triangular matrix.

    The vector x is overwritten on b.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.

    @param[in]
    uplo    rocblas_fill.
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     rocblas_operation

    @param[in]
    diag    rocblas_diagonal.
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         rocblas_int
              m specifies the number of rows of b. m >= 0.

//    @param[in]
//    n         rocblas_int
//              n specifies the number of rows of b. n >= 0.

    @param[in]
    alpha
              specifies the scalar alpha.

    @param[in]
    A         pointer storing matrix A on the GPU,
              of dimension ( lda, m )

    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
              lda = max( 1, m ).

    @param[in]
    x         pointer storing vector x on the GPU.

    @param[in]
    incx      specifies the increment for the elements of x.

    ********************************************************************/

template <typename T, rocblas_int BLOCK>
rocblas_status rocblas_trsv_template(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
//                                     rocblas_int n,
                                     const T* alpha,
                                     const T* A,
                                     rocblas_int lda,
                                     const T* x,
                                     rocblas_int incx)
{
    if(handle == nullptr)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocblas_Xtrsv"),
                  uplo,
                  transA,
                  diag,
                  m,
                  *alpha,
                  (const void*&)A,
                  lda,
                  (const void*&)x,
                  incx);

        std::string uplo_letter   = rocblas_fill_letter(uplo);
        std::string transA_letter = rocblas_transpose_letter(transA);
        std::string diag_letter   = rocblas_diag_letter(diag);

        log_bench(handle,
                  "./rocblas-bench -f trsv -r",
                  replaceX<T>("X"),
                  "--uplo",
                  uplo_letter,
                  "--transposeA",
                  transA_letter,
                  "--diag",
                  diag_letter,
                  "-m",
                  m,
//                  "-n",
//                  n,
                  "--alpha",
                  *alpha,
                  "--lda",
                  lda,
                  "--incx",
                  incx);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocblas_Xtrsv"),
                  uplo,
                  transA,
                  diag,
                  m,
                  (const void*&)alpha,
                  (const void*&)A,
                  lda,
                  (const void*&)x,
                  incx);
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if(alpha == nullptr)
        return rocblas_status_invalid_pointer;
    else if(nullptr == A)
        return rocblas_status_invalid_pointer;
    else if(nullptr == x)
        return rocblas_status_invalid_pointer;
    else if(m < 0)
        return rocblas_status_invalid_size;
    else if(lda < m || lda < 1)
        return rocblas_status_invalid_size;
    else if(0 == incx)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(m == 0 /*|| n == 0*/)
        return rocblas_status_success;


}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocblas_strsv(rocblas_handle handle,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
   //                                     rocblas_int n,
                                        const T* alpha,
                                        const T* A,
                                        rocblas_int lda,
                                        const T* x,
                                        rocblas_int incx)
{
    return rocblas_trsv_template<float, STRSV_BLOCK>(   //not defined?? find STRSM_BLOCK
        handle, uplo, transA, diag, m, /*n,*/ alpha, A, lda, x, incx);
}

extern "C" rocblas_status rocblas_dtrsv(rocblas_handle handle,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
   //                                     rocblas_int n,
                                        const T* alpha,
                                        const T* A,
                                        rocblas_int lda,
                                        const T* x,
                                        rocblas_int incx)
{
    return rocblas_gemv_template<double, STRSV_BLOCK>(
        handle, uplo, transA, diag, m, /*n,*/ alpha, A, lda, x, incx);
}
