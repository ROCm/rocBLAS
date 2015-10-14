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
 * \brief ablas.h defines 'C' compatible callable functions and types that
 * call into the library
 * \details The minimum compiler versions the library should support
 * ( These compilers have solid C++11 support):
 * - Visual Studio 2013 and up
 * - GCC 4.8 and up
 * - Clang 3.4 and up
 */

#pragma once
#ifndef _ABLAS_H_
#define _ABLAS_H_

#include <stdbool.h>

/*!
 * CMake-generated file to define export related preprocessor macros, including
 * ABLAS_EXPORT and ABLAS_DEPRECATED
*/
#include "ablas_export.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "ablas_types.h"

/*! Define ABLAS_USE_OPENCL to build library for OpenCL
 */
#if defined( ABLAS_USE_OPENCL )
  #include "ablas_opencl.h"
#else
  // Boltzman headers to be included here
  #include "ablas_hsa.h"
#endif

/*!
* \defgroup STATE Back-end agnostic state
* \brief Functions to create or modify library state
*/

/*!
 * \defgroup STATE-SINGLE single device
 * \brief These functions only concern themselves with single device functionality
 *
 * \ingroup STATE
 */
/**@{*/

/*!
* \brief Enable/Disable asynchronous behavior for ablas
*
* \param[in] control  A valid clsparseControl created with ablasCreateControl
* \param[in] async  True to enable immediate return, false to block execution until event completion
*
* \ingroup STATE-SINGLE
*
* \returns \b ablasSuccess
*/
ABLAS_EXPORT ablas_status
ablas_enable_async( ablas_control control, bool async );
/**@}*/

/*!
 * \defgroup BLAS3 Level 3 BLAS
 * \brief Functions executing dense order /f$ O(N^3) /f$ linear algebra
 */

 /*!
  * \defgroup BLAS3-SINGLE single device
  * \brief single device Level 3 BLAS API
  *
  * \ingroup BLAS3
  */
/**@{*/


/*! \brief Refactored ablas API
 * \details Generic matrix-matrix multiplication. These pointers are not denoting arrays.  The batch processing is specified inside of these
 * structs with batch_size
 * \f$ c \leftarrow \alpha o (a \ast b) + \beta o c \f$
 *
 * operator 'o' represent the entrywise (Hadamard) product.
 * scalar o scalar
 * scalar o vector
 * scalar o matrix
 * vector o vector
 * vector o matrix
 * matrix o matrix
 *
 * The general equation can be simplified by the terms being either ZERO or IDENTITY.
 *
 * GEMM (L3)
 * alpha - scalar, vector or matrix
 * a - matrix
 * b - matrix
 * beta - scalar, vector or matrix
 * c - matrix
 *
 * GEMV (L2)
 * alpha - scalar or vector
 * a - matrix
 * b - vector
 * beta - scalar or vector
 * c - vector
 *
 * AXPY (L1)
 * alpha - scalar or vector
 * a - vector
 * b - IDENTITY
 * beta - ZERO
 * c - vector
 *
 * SDOT (L1)
 * alpha - IDENTITY
 * a - vector
 * b - vector
 * beta - ZERO
 * c - scalar
 *
 * SCAL (L1)
 * alpha - ZERO
 * a - ZERO
 * b - ZERO
 * beta - scalar, vector or matrix
 * c - scalar, vector or matrix
 *
 *
 * \param[in] alpha  Scalar value to be multiplied into the product of A * B
 * \param[in] a  Source matrix
 * \param[in] b  Source matrix
 * \param[in] beta  Scalar value to be multiplied into the matrix C on read
 * \param[in,out] c  Destination matrix
 * \param[in,out] control  ablas state object
 */
ABLAS_EXPORT ablas_status
ablas_gemm(
  const ablas_matrix *alpha,
  const ablas_matrix *a,
  const ablas_matrix *b,
  const ablas_matrix *beta,
        ablas_matrix *c,
        ablas_control *control );
/**@}*/

// Example of older ablas API from v2.x.x
// ABLAS_DEPRECATED ablasStatus
// ablasSgemm(
//     ablasOrder order,
//     ablasTranspose transA,
//     ablasTranspose transB,
//     size_t M,
//     size_t N,
//     size_t K,
//     cl_float alpha,
//     const cl_mem A,
//     size_t offA,
//     size_t lda,
//     const cl_mem B,
//     size_t offB,
//     size_t ldb,
//     cl_float beta,
//     cl_mem C,
//     size_t offC,
//     size_t ldc,
//     cl_uint numCommandQueues,
//     cl_command_queue *commandQueues,
//     cl_uint numEventsInWaitList,
//     const cl_event *eventWaitList,
//     cl_event *events);


/**@{*/
ABLAS_EXPORT ablas_status
clblas_2_ablas_init_matrix(
    const ablas_precision *precision,
    const ablas_order *order,
    const ablas_transpose *trans,
    const ablas_uplo *uplo,
    const ablas_diag *diag,
    const size_t *M,
    const size_t *N,
    const size_t *K,
    size_t offset,
    size_t ldX,
    void *X,
    ablas_matrix *ablas_X
);
/**@}*/

#ifdef __cplusplus
}      // extern C
#endif

#endif // _ABLAS_H_
