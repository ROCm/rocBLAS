/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/*!\file
 * \brief rocblas-types.h defines public types to be consummed by the library
 * The types are agnostic to the underlying runtime used by the library
 */

#pragma once
#ifndef _ROCBLAS_TYPES_H_
#define _ROCBLAS_TYPES_H_

#include <stddef.h>
#include <stdint.h>


/*! \file
 * \brief rocblas_types.h defines data types used by rocblas
 */


    /*! \brief To specify whether int32 or int64 is used
     */


    #if defined( rocblas_ILP64 )
    typedef int64_t rocblas_int;
    #else
    typedef int32_t rocblas_int;
    #endif

    typedef void* rocblas_control;

#ifdef __cplusplus
extern "C" {
#endif

    /* ============================================================================================ */

    /*! parameter constants.
     *  numbering is consistent with CBLAS, ACML and most standard C BLAS libraries
     */

    /*! \brief Used to specify whether the matrix is in row major or column major storage format. */
    typedef enum rocblas_order_{
        rocblas_row_major         = 101,
        rocblas_column_major      = 102
    } rocblas_order;


    /*! \brief Used to specify whether the matrix is to be transposed or not. */
    typedef enum rocblas_trans_ {
        rocblas_no_trans   = 111,           /**< Operate with the matrix. */
        rocblas_trans      = 112,           /**< Operate with the transpose of the matrix. */
        rocblas_conj_trans = 113            /**< Operate with the conjugate transpose of
                                         the matrix. */
    } rocblas_transpose;

    /*! \brief Used by the Hermitian, symmetric and triangular matrix
     * routines to specify whether the upper or lower triangle is being referenced.
     */
    typedef enum rocblas_uplo_ {
        rocblas_upper = 121,               /**< Upper triangle. */
        rocblas_lower = 122,               /**< Lower triangle. */
        rocblas_full  = 123
    } rocblas_uplo;


    /*! \brief It is used by the triangular matrix routines to specify whether the
     * matrix is unit triangular.
     */
    typedef enum rocblas_diag_ {
        rocblas_non_unit  = 131,           /**< Non-unit triangular. */
        rocblas_unit      = 132,          /**< Unit triangular. */
    } rocblas_diag;


    /*! \brief Indicates the side matrix A is located relative to matrix B during multiplication. */
    typedef enum rocblas_side_ {
        rocblas_left  = 141,        /**< Multiply general matrix by symmetric,
                                   Hermitian or triangular matrix on the left. */
        rocblas_right = 142,        /**< Multiply general matrix by symmetric,
                                   Hermitian or triangular matrix on the right. */
        rocblas_both  = 143
    } rocblas_side;




    /* ============================================================================================ */


    /*! \brief Indicates the precision width of data stored in a blas type. */
    typedef enum rocblas_precision_ {
      rocblas_single_ = 151,
      rocblas_double_ = 152,
      rocblas_single_complex_ = 153,
      rocblas_double_complex_ = 154
    } rocblas_precision;


    /*! \brief Indicates the pointer is device pointer or host pointer */
    typedef enum rocblas_get_pointer_location_ {
      DEVICE_POINTER = 1,
      HOST_POINTER = 0
  } rocblas_get_pointer_location;


#ifdef __cplusplus
}
#endif



/* ============================================================================================ */

/*! \brief Structure to encapsulate dense matrix/vector/scalar data to rocblas API.
 * \details Able to store multiple matrices (or vectors, scalars)
 * to facilitate high-performance batched oprations;
 * gracefully becomes a 'normal' matrix when num_matrices == 1.
 * \verbatim
    rocblas V2: Given a column major matrix with M, N, lda, nontranspose
    This matrix is represented in rocblas as:
    num_rows = M
    num_cols = N
    row_stride = 1
    col_stride = ldX
    num_matrices = 1
   \endverbatim
 * rocblas API represents scalars as rocblas_matrix with num_rows = 1 and num_cols = 1.
 * rocblas API represents vectors as rocblas_matrix with num_rows = 1 or  num_cols = 1.
 *
 * \note It is the users responsibility to allocate/deallocate buffers
 * \note Traditional matrix fields not explicitely represented within this structure
 * \li \b transpose layout
 * \li \b row/column major layout
 * \attention There has been significant debate about changing the matrix meta data below from host scalar values
 * into batched scalar values by changing their types to rocblasScalar.  The advantage is that we
 * could then process batched matricies of arbitrary row, column and stride values.  The problem is
 * that both host and device need access to this data, which would introduce mapping calls.  The host needs
 * the data to figure out how to form launch parameters, and the device needs access to be able to
 * handle matrix tail cases properly.  This may have reasonable performance on APU's, but the performance
 * impact on discrete devices could be significant.  For now, we keep the num_rows, num_cols and strides as a size_t on host
 */
typedef struct rocblas_matrix_ {

    /*! \brief Buffer that holds the matrix data.
     * \details Polymorphic pointer for the library.  If rocblas is compiled with BUILD_CLVERSION < 200,
     * value will be will be treated as a pointer allocated with clCreateBuffer().  If
     * BUILD_CLVERSION >= 200 then this will be treated as a pointer allocated with clSVMalloc()
     * For batched matrices, this buffer contains the packed values of all the matrices.
     */
    void* data;

    /*! Precision of the data.
     */
    rocblas_precision precision;

    /*! \brief This offset is added to the cl_mem location on device to define beginning of the data in the cl_mem buffers
     * \details Usually used to define the start a smaller submatrix in larger matrix allocation block.
     * This same offset is applied to every matrix in a batch
     */
    size_t offset;

    /*! \brief Number of rows in each matrix.
     * \details For batched matrices, this is a constant property of each 'matrix', where each matrix has the same number
     *  of rows
     */
    size_t num_rows;

    /*! \brief Number of columns in each matrix.
    * \details For batched matrices, this is a constant property of each 'matrix', where each matrix has the same number
    *  of columns
     */
    size_t num_cols;

     /*! Number of matrices stored in the buffer; a single matrix would have num_matrices == 1.
     * \pre num_matrices > 0
      */
    size_t num_matrices;

    /*! \brief Stride to consecutive rows in each matrix.
     * \details ptr += row_stride would point to same column, same matrix, next row.
     * For column-major matrix, row_stride = 1.
     */
    size_t row_stride;

    /*! \brief Stride to consecutive columns in each matrix.
     * \details ptr += col_stride would point to same row, same matrix, next column.
     * For row-major matrix, col_stride = 1.
     */
    size_t col_stride;

     /*! \bried Stride to consectutive matrices.
      * \details ptr += matrix_stride would point to same row, same column, next matrix
      * \pre row_major: batch_stride >= num_rows * row_stride
      * \pre column_major: batch_stride >= num_cols * col_stride
      */
    size_t matrix_stride;

} rocblas_matrix;



#endif
