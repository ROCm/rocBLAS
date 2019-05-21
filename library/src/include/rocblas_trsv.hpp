/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCBLAS_TRSV_HPP_
#define _ROCBLAS_TRSV_HPP_

#include "rocblas.h"
#include "status.h"

/* ============================================================================================ */

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

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsv_template(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
                                     const T* A,
                                     rocblas_int lda,
                                     T* x,
                                     rocblas_int incx);

template <rocblas_int BLOCK, typename T>
rocblas_status rocblas_trsv_ex_template(rocblas_handle handle,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
                                        const T* A,
                                        rocblas_int lda,
                                        T* x,
                                        rocblas_int incx,
                                        const T* invA,
                                        rocblas_int ldInvA,
                                        const size_t* x_temp_size,
                                        T* x_temp);
#endif // _ROCBLAS_TRSV_HPP_
