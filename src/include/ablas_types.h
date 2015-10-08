/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

/*! \file
 * \brief aBLAS-types.h defines public types to be consummed by the library
 * The types are agnostic to the underlying runtime used by the library
 */

#pragma once
#ifndef _ABLAS_TYPES_H_
#define _ABLAS_TYPES_H_

/*! \brief An enumeration to describe the precision of data pointed by a
 * particular instance of a struct
 * \remarks This impllies that aBLAS can support mixed precision operations
 */
typedef enum ablas_precision_ {
    ablas_single_real,
    ablas_double_real,
    ablas_single_complex,
    ablas_double_complex,
} ablas_precision;

/*! \brief Used by the Hermitian, symmetric and triangular matrix
 * routines to specify whether the upper or lower triangle is being referenced.
 */
typedef enum ablas_uplo_ {
    ablas_upper,               /**< Upper triangle. */
    ablas_lower                /**< Lower triangle. */
} ablas_uplo;

/*! \brief It is used by the triangular matrix routines to specify whether the
 * matrix is unit triangular.
 */
typedef enum ablas_diag_ {
    ablas_unit,               /**< Unit triangular. */
    ablas_non_unit             /**< Non-unit triangular. */
} ablas_diag;

/*! \brief Indicates the side matrix A is located relative to matrix B during multiplication. */
typedef enum ablas_side_ {
    ablas_left,        /**< Multiply general matrix by symmetric,
                               Hermitian or triangular matrix on the left. */
    ablas_right        /**< Multiply general matrix by symmetric,
                               Hermitian or triangular matrix on the right. */
} ablas_side;


/*! \brief Structure to encapsulate dense matrix/vector/scalar data to aBLAS API.
 * \details Able to store multiple matrices (or vectors, scalars)
 * to facilitate high-performance batched oprations;
 * gracefully becomes a 'normal' matrix when num_matrices == 1.
 * \verbatim
    clBLAS V2: Given a column major matrix with M, N, lda, nontranspose
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
