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
 * \brief clBLAS-types.h defines public types to be consummed by the library
 * The types are agnostic to the underlying runtime used by the library
 */

#pragma once
#ifndef _CL_BLAS_TYPES_H_
#define _CL_BLAS_TYPES_H_

/*! \brief An enumeration to describe the precision of data pointed by a
 * particular instance of a struct
 * \remarks This impllies that clBLAS can support mixed precision operations
 */
typedef enum clblasPrecision_ {
    clblasSingleReal,
    clblasDoubleReal,
    clblasSingleComplex,
    clblasDoubleComplex,
} clblasPrecision;

/*! \brief Used by the Hermitian, symmetric and triangular matrix
 * routines to specify whether the upper or lower triangle is being referenced.
 */
typedef enum clblasUplo_ {
    clblasUpper,               /**< Upper triangle. */
    clblasLower                /**< Lower triangle. */
} clblasUplo;

/*! \brief It is used by the triangular matrix routines to specify whether the
 * matrix is unit triangular.
 */
typedef enum clblasDiag_ {
    clblasUnit,               /**< Unit triangular. */
    clblasNonUnit             /**< Non-unit triangular. */
} clblasDiag;

/*! \brief Indicates the side matrix A is located relative to matrix B during multiplication. */
typedef enum clblasSide_ {
    clblasLeft,        /**< Multiply general matrix by symmetric,
                               Hermitian or triangular matrix on the left. */
    clblasRight        /**< Multiply general matrix by symmetric,
                               Hermitian or triangular matrix on the right. */
} clblasSide;


/*! \brief Structure to encapsulate scalar data to clBLAS API
 * \details This stores data in a struct of arrays (SoA) model.  This should help performance
 * for batched operation, and gracefully become a 'normal' struct when batch_size == 1
 * \note It is the users responsibility to allocate/deallocate OpenCL buffers
 */
typedef struct clblasScalar_
{
    /*! Polymorphic pointer for the library.  If clBLAS is compiled with BUILD_CLVERSION < 200,
     * value will be will be treated as allocated with clCreateBuffer().  If
     * BUILD_CLVERSION >= 200 then this will be treated as allocated with clSVMalloc()
     */
    void* device_scalar;

    /*! This describes the precision of the data pointed to by value
     */
    clblasPrecision precision;

    /*! This offset is added to the cl_mem locations on device to define beginning of the data in the cl_mem buffers
     */
    size_t offset;

    /*! This is the number of scalar values stored in the value buffer
     */
    size_t batch_size;

    /*! This is the distance between scalars in value; batch_stride >= 1
     * Packed scalars would have a batch_stride of 1
     */
    size_t batch_stride;
} clblasScalar;

/*! \brief Structure to encapsulate dense vector data to clBLAS API
 * \details This stores data in a struct of arrays (SoA) model.  This should help performance
 * for batched operation, and gracefully become a 'normal' struct when batch_size == 1
 * \note It is the users responsibility to allocate/deallocate OpenCL buffers
 */
typedef struct clblasVector_
{
    /*! Polymorphic pointer for the library.  If clBLAS is compiled with BUILD_CLVERSION < 200,
     * value will be will be treated as a pointer allocated with clCreateBuffer().  If
     * BUILD_CLVERSION >= 200 then this will be treated as a pointer allocated with clSVMalloc()
     */
    void* device_vector;

    /*! This describes the precision of the data pointed to by value
     */
    clblasPrecision precision;

    /*! \brief This offset is added to the cl_mem location on device to define beginning of the data in the cl_mem buffers
     * \details Usually used to define the start a smaller subvector in larger vector allocation block.
     * This same offset is applied to every vector in a batch
     */
    size_t offset;

    size_t length;  /*!< Length of an individual vector */

    /*! \brief Stride to consecutive elements in vector
     * \details For packed vectors, stride == 1
     */
    size_t stride;

    /*! This is the number of vectors stored in the values buffer; a single vector would have batch_size == 1
     */
    size_t batch_size;

    /*! This is the distance between vectors in values; batch_stride >= num_values
     * Packed vectors would have a batch_stride == num_values
     */
    size_t batch_stride;
} clblasVector;

/*! \brief Structure to encapsulate dense matrix data to clBLAS API
 * \details This stores data in a struct of arrays (SoA) model.  This should help performance
 * for batched operation, and gracefully become a 'normal' struct when batch_size == 1
 * \verbatim
    clBlas V2: Given a column major matrix with M, N, lda, nontranspose
    This matrix is represented in clBlas V3 as:
    num_rows = M
    num_cols = N
    row_stride = 1
    col_stride = ldX
   \endverbatim

 * \note It is the users responsibility to allocate/deallocate OpenCL buffers
 * \note Traditional matrix fields not explicitely represented within this structure
 * \li \b transpose layout
 * \li \b row/column major layout
 * \attention There has been significant debate about changing the matrix meta data below from host scalar values
 * into batched scalar values by changing their types to clblasScalar.  The advantage is that we
 * could then process batched matricies of arbitrary row, column and stride values.  The problem is
 * that both host and device need access to this data, which would introduce mapping calls.  The host needs
 * the data to figure out how to form launch parameters, and the device needs access to be able to
 * handle matrix tail cases properly.  This may have reasonable performance on APU's, but the performance
 * impact on discrete devices could be significant.  For now, we keep the num_rows, num_cols and strides as a size_t on host
 */
typedef struct clblasMatrix_
{
    /*! \brief An OpenCL buffer that holds the values of all the matrix data
     * \details Polymorphic pointer for the library.  If clBLAS is compiled with BUILD_CLVERSION < 200,
     * value will be will be treated as a pointer allocated with clCreateBuffer().  If
     * BUILD_CLVERSION >= 200 then this will be treated as a pointer allocated with clSVMalloc()
     * For batched matrices, this buffer contains the packed values of all the matrices.
     */
    void* device_matrix;

    /*! This describes the precision of the data pointed to by value
     */
    clblasPrecision precision;

    /*! \brief This offset is added to the cl_mem location on device to define beginning of the data in the cl_mem buffers
     * \details Usually used to define the start a smaller submatrix in larger matrix allocation block.
     * This same offset is applied to every matrix in a batch
     */
    size_t offset_values;

    /*! \brief Number of elements in a matrix row
     * \details For batched matrices, this is a constant property of each 'matrix', where each matrix has the same number
     *  of rows
     */
    size_t num_rows;

    /*! \brief Number of elements in a matrix column
    * \details For batched matrices, this is a constant property of each 'matrix', where each matrix has the same number
    *  of columns
     */
    size_t num_cols;

    /*! \brief Stride to consecutive elements in a matrix row
     * \details For an example of a packed matrices, in 'C' family row major
     * languages this is num_rows, for Fortran family languages this is 1
     */
    size_t row_stride;

    /*! \brief Stride to consecutive elements in a matrix column
     * \details For an example of a packed matrices, in 'C' family row major
     * languages this is 1, for Fortran family languages this is num_rows
     */
    size_t col_stride;

     /*! This is the number of vectors stored in the values buffer; a single vector would have batch_size == 1
     * \pre batch_size > 0
      */
    size_t batch_size;

     /*! This is the distance between the start of matrices in values
      * \pre row_major: batch_stride >= num_rows * row_stride
      * \pre column_major: batch_stride >= num_cols * col_stride
      */
    size_t batch_stride;

} clblasMatrix;

#endif
