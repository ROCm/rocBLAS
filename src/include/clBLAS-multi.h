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
 * \brief Defines 'C' compatible callable functions and types that
 * call into the library with multiple device support
 * \details The minimum compiler versions the library should support
 * ( These compilers have solid C++11 support):
 * - Visual Studio 2013 and up
 * - GCC 4.8 and up
 * - Clang 3.4 and up
 * \note We would only ship the *-multi files when we have a multi-gpu solution.
 * We should concentrate on getting maximum performance on a single gpu first before
 * we start exporting multi-gpu API'.  That is why the multi-gpu API's are seperated
 * out in a separate header.
 */

#pragma once
#ifndef _CL_BLAS_MULTI_H_
#define _CL_BLAS_MULTI_H_

#include "clBLAS.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Define CLBLAS_USE_OPENCL to build library for OpenCL
 */
#if defined( CLBLAS_USE_OPENCL )
  #include "clBLAS-opencl-multi.h"
#else
  // Boltzman headers to be included here
  #include "clBLAS-hsa-multi.h"
#endif

/*!
 * \defgroup STATE-MULTI multi device
 * \brief These functions work with multi-device configurations
 *
 * \ingroup STATE
 */
/**@{*/

/*!
* \brief Enable/Disable asynchronous behavior for mutli-device clBLAS
*
* \param[in] control  A valid clblasControlMulti created with clblasCreateControl
* \param[in] async  Pass true to enable immediate return, false to block execution until all devices
* in clblasControlMulti have completed
*
* \ingroup STATE-MULTI
*
* \returns \b clblasSuccess
*/
CLBLAS_EXPORT clblasStatus
  clblasEnableAsyncMulti( clblasControlMulti control, bool async );
/**@}*/

/*!
 * \defgroup BLAS3-MULTI multi device
 *
 * \brief multi device Level 3 BLAS API
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
 * \ingroup BLAS3-MULTI
 */
CLBLAS_EXPORT clblasStatus
  clblasGemmMulti( const clblasScalar* alpha,
              const clblasMatrix* a,
              const clblasMatrix* b,
              const clblasScalar* beta,
              clblasMatrix* c,
              clblasControlMulti control );
/**@}*/

#ifdef __cplusplus
}      // extern C
#endif

#endif // _CL_BLAS_MULTI_H_
