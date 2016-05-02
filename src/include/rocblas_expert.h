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
 * \brief rocblas.h defines 'C' compatible callable functions and types that
 * call into the library
 * \details The minimum compiler versions the library should support
 * ( These compilers have solid C++11 support):
 * - Visual Studio 2013 and up
 * - GCC 4.8 and up
 * - Clang 3.4 and up
 */

#pragma once
#ifndef _ROCBLAS_EXPERT_H_
#define _ROCBLAS_EXPERT_H_

#include <stdbool.h>


    /*
     * ===========================================================================
     *   READEME: This set of API is supposed to be used by expert users
     *     who are sensitive to performance and want more control over their computation
     * ===========================================================================
     */


/*!
 * CMake-generated file to define export related preprocessor macros, including
 * ROCBLAS_EXPORT and ROCBLAS_DEPRECATED
*/
//#include "rocblas_export.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "rocblas_types.h"


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
* \brief Enable/Disable asynchronous behavior for rocblas
*
* \param[in] control  A valid clsparseControl created with rocblasCreateControl
* \param[in] async  True to enable immediate return, false to block execution until event completion
*
* \ingroup STATE-SINGLE
*
* \returns \b rocblasSuccess
*/

//rocblas_status
//rocblas_enable_async( rocblas_control control, bool async );
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


/*! \brief Refactored rocblas API
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
 * \param[in] alpha  Scalar value to be multiplied into the product of A * B
 * \param[in] a  Source matrix
 * \param[in] b  Source matrix
 * \param[in] beta  Scalar value to be multiplied into the matrix C on read
 * \param[in,out] c  Destination matrix
 * \param[in,out] control  rocblas state object
 */
rocblas_status
rocblas_gemm(
  const rocblas_matrix *alpha,
  const rocblas_matrix *a,
  const rocblas_matrix *b,
  const rocblas_matrix *beta,
        rocblas_matrix *c,
        rocblas_control *control );
/**@}*/

;

#ifdef __cplusplus
}      // extern C
#endif

#endif // _ROCBLAS_EXPERT_H_
