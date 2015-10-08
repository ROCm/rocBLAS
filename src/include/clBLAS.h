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
 * \brief clBLAS.h defines 'C' compatible callable functions and types that
 * call into the library
 * \details The minimum compiler versions the library should support
 * ( These compilers have solid C++11 support):
 * - Visual Studio 2013 and up
 * - GCC 4.8 and up
 * - Clang 3.4 and up
 */

#pragma once
#ifndef _CL_BLAS_H_
#define _CL_BLAS_H_

#include <stdbool.h>

/*!
 * CMake-generated file to define export related preprocessor macros, including
 * CLBLAS_EXPORT and CLBLAS_DEPRECATED
*/
#include "clblas_export.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "clBLAS-types.h"

/*! Define CLBLAS_USE_OPENCL to build library for OpenCL
 */
#if defined( CLBLAS_USE_OPENCL )
  #include "clBLAS-opencl.h"
#else
  // Boltzman headers to be included here
  #include "clBLAS-hsa.h"
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
* \brief Enable/Disable asynchronous behavior for clBLAS
*
* \param[in] control  A valid clsparseControl created with clblasCreateControl
* \param[in] async  True to enable immediate return, false to block execution until event completion
*
* \ingroup STATE-SINGLE
*
* \returns \b clblasSuccess
*/
CLBLAS_EXPORT clblasStatus
  clblasEnableAsync( clblasControl control, bool async );
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


/*! \brief Refactored clBLAS API
 * \details These pointers are not denoting arrays.  The batch processing is specified inside of these
 * structs with batch_size, \f$ C \leftarrow \alpha \ast A \ast B + \beta \ast C \f$
 * \param[in] alpha  Scalar value to be multiplied into the product of A * B
 * \param[in] a  Source matrix
 * \param[in] b  Source matrix
 * \param[in] beta  Scalar value to be multiplied into the matrix C on read
 * \param[in,out] c  Destination matrix
 * \param[in] control  clBLAS state object
 */
CLBLAS_EXPORT clblasStatus
  clblasGemm(  const clblasScalar* alpha,
                const clblasMatrix* a,
                const clblasMatrix* b,
                const clblasScalar* beta,
                clblasMatrix* c,
                clblasControl control );
/**@}*/

// Example of older clBLAS API from v2.x.x
// CLBLAS_DEPRECATED clblasStatus
// clblasSgemm(
//     clblasOrder order,
//     clblasTranspose transA,
//     clblasTranspose transB,
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

#ifdef __cplusplus
}      // extern C
#endif

#endif // _CL_BLAS_H_
