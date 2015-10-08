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
 * \brief clBLAS-opencl.h defines 'C' compatible callable functions and types that
 * are specific to an opencl runtime implementation
 * \details The minimum compiler versions the library should support
 * ( These compilers have solid C++11 support):
 * - Visual Studio 2013 and up
 * - GCC 4.8 and up
 * - Clang 3.4 and up
 */

#pragma once
#ifndef _CL_BLAS_OPENCL_MULTI_H_
#define _CL_BLAS_OPENCL_MULTI_H_

#include "clBLAS-opencl.h"

/*!
 * The older clBLAS.h header will be renamed clBLAS-deprecated.h and it's use will be
 * dictated by the user defining CLBLAS_USE_DEPRECATED_2_X_API before including clBLAS.h
 */
#if !defined( CLBLAS_USE_DEPRECATED_2_X_API )

/*!
 * \defgroup OPENCL-STATE-MULTI multi device
 * \brief Functions to create or modify multiple device opencl state
 * \ingroup OPENCL-STATE
 */
/**@{*/

/*! \brief clblasControlMulti keeps library state to control multi-device operation
 * in the blas API
 */
typedef struct clblas_control_opencl_multi*  clblasControlMulti;

/*!
 * \brief setup a multi-device clblas control object
 * \details This creates and returns to the user a structure that bundles multi-device state
 * needed to be passed into a multi-device API call.  The user constrains the multi-gpu
 * algorithms to utilize only the devices contained in the set.  This control structure
 * does not create additional queues or events, but bumps the reference counts in the existing
 * opencl objects previosly created and passed in with the clblasControl structures
 *
 * \param[in] controls   An array of clblasControl's created with clblasCreateControl
 * \param[in] control_size   The number of clblasControl's in the controls array
 * \param[out] status   clblas error return value from function
 *
 * \returns \b On successful completion, a valid clblasControlMulti object
 */
CLBLAS_EXPORT clblasControlMulti
  clblasCreateControlMulti( clblasControl* controls, size_t control_size, clblasStatus* status );
/**@}*/

#endif

#endif // _CL_BLAS_OPENCL_MULTI_H_
