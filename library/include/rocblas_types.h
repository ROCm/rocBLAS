/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/*! \file
 * \brief rocblas_types.h defines data types used by rocblas
 */

#pragma once
#ifndef _ROCBLAS_TYPES_H_
#define _ROCBLAS_TYPES_H_

#include <stddef.h>
#include <stdint.h>
#include <hip/hip_vector_types.h>


// integer type
/*! \brief To specify whether int32 or int64 is used
 */
#if defined( rocblas_ILP64 )
typedef int64_t rocblas_int;
#else
typedef int32_t rocblas_int;
#endif
// complex type
typedef float2  rocblas_float_complex;
typedef double2 rocblas_double_complex;
// half type TODO put name of half here
typedef float    rocblas_half;
typedef float2   rocblas_half_complex;

typedef struct _rocblas_handle * rocblas_handle;

#ifdef __cplusplus
extern "C" {
#endif

    /* ============================================================================================ */

    /*! parameter constants.
     *  numbering is consistent with CBLAS, ACML and most standard C BLAS libraries
     */

    /*! \brief Used to specify whether the matrix is in row major or column major storage format. */
    typedef enum rocblas_order_{
        rocblas_order_row_major         = 101,
        rocblas_order_column_major      = 102
    } rocblas_order;


    /*! \brief Used to specify whether the matrix is to be transposed or not. */
    typedef enum rocblas_operation_ {
        rocblas_operation_none                = 111, /**< Operate with the matrix. */
        rocblas_operation_transpose           = 112, /**< Operate with the transpose of the matrix. */
        rocblas_operation_conjugate_transpose = 113  /**< Operate with the conjugate transpose of the matrix. */
    } rocblas_operation;

    /*! \brief Used by the Hermitian, symmetric and triangular matrix
     * routines to specify whether the upper or lower triangle is being referenced.
     */
    typedef enum rocblas_fill_ {
        rocblas_fill_upper = 121,               /**< Upper triangle. */
        rocblas_fill_lower = 122,               /**< Lower triangle. */
        rocblas_fill_full  = 123
    } rocblas_fill;


    /*! \brief It is used by the triangular matrix routines to specify whether the
     * matrix is unit triangular.
     */
    typedef enum rocblas_diagonal_ {
        rocblas_diagonal_non_unit  = 131,           /**< Non-unit triangular. */
        rocblas_diagonal_unit      = 132,          /**< Unit triangular. */
    } rocblas_diagonal;


    /*! \brief Indicates the side matrix A is located relative to matrix B during multiplication. */
    typedef enum rocblas_side_ {
        rocblas_side_left  = 141,        /**< Multiply general matrix by symmetric,
                                   Hermitian or triangular matrix on the left. */
        rocblas_side_right = 142,        /**< Multiply general matrix by symmetric,
                                   Hermitian or triangular matrix on the right. */
        rocblas_side_both  = 143
    } rocblas_side;




    /* ============================================================================================ */
    /**
     *   @brief rocblas status codes definition
     */
    typedef enum rocblas_status_ {
        rocblas_status_success          = 0, /**< success */
        rocblas_status_invalid_handle   = 1, /**< handle not initialized, invalid or null */
        rocblas_status_not_implemented  = 2, /**< function is not implemented */
        rocblas_status_invalid_pointer  = 3, /**< invalid pointer parameter */
        rocblas_status_invalid_size     = 3, /**< invalid size parameter */
        rocblas_status_memory_error     = 4, /**< failed internal memory allocation, copy or dealloc */
        rocblas_status_internal_error   = 5, /**< other internal library failure */
    } rocblas_status;


    /*! \brief Indicates the precision width of data stored in a blas type. */
    typedef enum rocblas_precision_ {
      rocblas_precision_half            = 150,
      rocblas_precision_single          = 151,
      rocblas_precision_double          = 152,
      rocblas_precision_complex_half    = 153,
      rocblas_precision_complex_single  = 154,
      rocblas_precision_complex_double  = 155
    } rocblas_precision;


    /*! \brief Indicates the pointer is device pointer or host pointer */
    typedef enum rocblas_mem_location_ {
      rocblas_mem_location_host   = 0,
      rocblas_mem_location_device = 1
  } rocblas_mem_location;

#ifdef __cplusplus
}
#endif

#endif
