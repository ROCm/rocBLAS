/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/*!\file
 * \brief aBLAS-types.h defines public types to be consummed by the library
 * The types are agnostic to the underlying runtime used by the library
 */

#pragma once
#ifndef _ABLAS_TYPES_H_
#define _ABLAS_TYPES_H_

#include <stdint.h>
#include <ablas_hip.h>


/*! \file
 * \brief ablas_types.h defines data types used by ablas
 */


#ifdef __cplusplus
extern "C" {
#endif

    /* ============================================================================================ */

    /*! parameter constants. 
     *  numbering is consistent with CBLAS, ACML and most standard C BLAS libraries 
     */

    /*! \brief Used to specify whether the matrix is in row major or column major storage format. */
    typedef enum ablas_order_{
        ablas_row_major         = 101,
        ablas_column_major      = 102
    } ablas_order;


    /*! \brief Used to specify whether the matrix is to be transposed or not. */
    typedef enum ablas_trans_ {
        ablas_no_trans   = 111,           /**< Operate with the matrix. */
        ablas_trans      = 112,           /**< Operate with the transpose of the matrix. */
        ablas_conj_trans = 113            /**< Operate with the conjugate transpose of
                                         the matrix. */
    } ablas_transpose;

    /*! \brief Used by the Hermitian, symmetric and triangular matrix
     * routines to specify whether the upper or lower triangle is being referenced.
     */
    typedef enum ablas_uplo_ {
        ablas_upper = 121,               /**< Upper triangle. */
        ablas_lower = 122,               /**< Lower triangle. */
        ablas_full  = 123
    } ablas_uplo;


    /*! \brief It is used by the triangular matrix routines to specify whether the
     * matrix is unit triangular.
     */
    typedef enum ablas_diag_ {
        ablas_non_unit  = 131           /**< Non-unit triangular. */
        ablas_unit      = 132,          /**< Unit triangular. */
    } ablas_diag;


    /*! \brief Indicates the side matrix A is located relative to matrix B during multiplication. */
    typedef enum ablas_side_ {
        ablas_left  = 141,        /**< Multiply general matrix by symmetric,
                                   Hermitian or triangular matrix on the left. */
        ablas_right = 142,        /**< Multiply general matrix by symmetric,
                                   Hermitian or triangular matrix on the right. */
        ablas_both  = 143
    } ablas_side;



    /* ============================================================================================ */

    /*! \brief To specify whether int32 or int64 is used
     */
    #if defined(ABLAS_ILP64)
    typedef int64_t ablas_int;
    #else
    typedef int32_t ablas_int;
    #endif

    /*! \brief  HIP & CUDA both use float2/double2 to define complex number  
     */
    typedef float2 ablas_float_complex
    typedef double2 ablas_double_complex
    // A Lot TODO about complex

#ifdef __cplusplus
}
#endif



/* ============================================================================================ */

/*! \brief Structure to encapsulate dense matrix/vector/scalar data to aBLAS API.
 * \details Able to store multiple matrices (or vectors, scalars)
 * to facilitate high-performance batched oprations;
 * gracefully becomes a 'normal' matrix when num_matrices == 1.
 * \verbatim
    ablas V2: Given a column major matrix with M, N, lda, nontranspose
    This matrix is represented in aBLAS as:
    num_rows = M
    num_cols = N
    row_stride = 1
    col_stride = ldX
    num_matrices = 1
   \endverbatim
 * aBLAS API represents scalars as ablas_matrix with num_rows = 1 and num_cols = 1.
 * aBLAS API represents vectors as ablas_matrix with num_rows = 1 or  num_cols = 1.
 *
 * \note It is the users responsibility to allocate/deallocate buffers
 * \note Traditional matrix fields not explicitely represented within this structure
 * \li \b transpose layout
 * \li \b row/column major layout
 * \attention There has been significant debate about changing the matrix meta data below from host scalar values
 * into batched scalar values by changing their types to ablasScalar.  The advantage is that we
 * could then process batched matricies of arbitrary row, column and stride values.  The problem is
 * that both host and device need access to this data, which would introduce mapping calls.  The host needs
 * the data to figure out how to form launch parameters, and the device needs access to be able to
 * handle matrix tail cases properly.  This may have reasonable performance on APU's, but the performance
 * impact on discrete devices could be significant.  For now, we keep the num_rows, num_cols and strides as a size_t on host
 */
typedef struct ablas_matrix_ {

    /*! \brief Buffer that holds the matrix data.
     * \details Polymorphic pointer for the library.  If aBLAS is compiled with BUILD_CLVERSION < 200,
     * value will be will be treated as a pointer allocated with clCreateBuffer().  If
     * BUILD_CLVERSION >= 200 then this will be treated as a pointer allocated with clSVMalloc()
     * For batched matrices, this buffer contains the packed values of all the matrices.
     */
    void* data;

    /*! Precision of the data.
     */
    ablas_precision precision;

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

} ablas_matrix;



#endif
