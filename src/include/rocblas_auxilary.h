/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_AUXILARY_H_
#define _ROCBLAS_AUXILARY_H_

#include <rocblas_types.h>

/*!\file
 * \brief rocblas_auxilary.h provides auxilary functions in rocblas
*/



#define CHECK_ERROR(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
    }


    /* ============================================================================================ */
    /*! \brief  indicates whether the pointer is on the host or device. currently HIP API can only recoginize the input ptr on deive or not
    can not recoginize it is on host or not */
    rocblas_pointer_type rocblas_get_pointer_type(void *ptr);

#ifdef __cplusplus
extern "C" {
#endif

    /*! \brief   create rocblas handle called before any rocblas library routines*/
    rocblas_status rocblas_create(rocblas_handle *handle);

    /*! \brief   release rocblas handle, will implicitly synchronize host and device */
    rocblas_status rocblas_destroy(rocblas_handle handle);


    /* ============================================================================================ */
    // synchronouse Functions
    /*! \brief   copy a vector hx of length n on host to a vector dx on device */
    rocblas_status rocblas_set_vector(rocblas_int n,
                                      rocblas_int elem_size,
                                      const void *hx, rocblas_int incx,
                                      void *dx, rocblas_int incy);

    /*! \brief   copy a vector dx of length n on device to a vector hx on host */
    rocblas_status rocblas_get_vector(rocblas_int n,
                                      rocblas_int elem_size,
                                      const void *dx, rocblas_int incx,
                                      void *hx, rocblas_int incy);


    /*! \brief   copy row*column part of A on host to row*column part of B on device. Both A and B are in column-major */
    rocblas_status rocblas_set_matrix(rocblas_int row, rocblas_int column,
                                      rocblas_int elem_size,
                                      const void *A, rocblas_int lda,
                                      void *B, rocblas_int ldb);


    /*! \brief   copy row*column part of A on device to row*column part of B on host. Both A and B are in column-major */
    rocblas_status rocblas_get_matrix(rocblas_int row, rocblas_int column,
                                      rocblas_int elem_size,
                                      const void *A, rocblas_int lda,
                                      void *B, rocblas_int ldb);

    /* ============================================================================================ */
    // asynchronouse Functions
    /*! \brief   copy a vector hx of length n on host to a vector dx on device. done asynchronously */
    rocblas_status rocblas_set_vector_async(rocblas_int n,
                                      rocblas_int elem_size,
                                      const void *hx, rocblas_int incx,
                                      void *dx, rocblas_int incy, hipStream_t stream);


    /*! \brief   copy a vector dx of length n on device to a vector hx on host. done asynchronously */
    rocblas_status rocblas_get_vector_async(rocblas_int n,
                                      rocblas_int elem_size,
                                      const void *dx, rocblas_int incx,
                                      void *hx, rocblas_int incy, hipStream_t stream);


    /*! \brief   copy row*column part of A on host to row*column part of B on device. Both A and B are in column-major. done asynchronously */
    rocblas_status rocblas_set_matrix_async(rocblas_int row, rocblas_int column,
                                      rocblas_int elem_size,
                                      const void *A, rocblas_int lda,
                                      void *B, rocblas_int ldb, hipStream_t stream);


    /*! \brief   copy row*column part of A on device to row*column part of B on host. Both A and B are in column-major. done asynchronously */
    rocblas_status rocblas_get_matrix_async(rocblas_int row, rocblas_int column,
                                      rocblas_int elem_size,
                                      const void *A, rocblas_int lda,
                                      void *B, rocblas_int ldb, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  /* _ROCBLAS_AUXILARY_H_ */
