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
#ifndef _CL_BLAS_OPENCL_H_
#define _CL_BLAS_OPENCL_H_

// Standard OpenCL includes
#if defined( __APPLE__ ) || defined( __MACOSX )
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/*!
* \brief The older clBLAS.h header will be renamed clBLAS-deprecated.h and it's use will be
  dictated by the user defining CLBLAS_USE_DEPRECATED_2_X_API before including clBLAS.h
*/
#if !defined( CLBLAS_USE_DEPRECATED_2_X_API )

  /*!
   *   \brief clblas error codes definition, incorporating OpenCL error definitions
   *   \details This enumeration is a subset of the OpenCL error codes extended with some
   *   additional extra codes.
   */
  typedef enum clblasStatus_
  {
    /** @name Inherited OpenCL codes */
    /**@{*/
    clblasSuccess                         = CL_SUCCESS,
    clblasInvalidValue                    = CL_INVALID_VALUE,
    clblasInvalidCommandQueue             = CL_INVALID_COMMAND_QUEUE,
    clblasInvalidContext                  = CL_INVALID_CONTEXT,
    clblasInvalidMemObject                = CL_INVALID_MEM_OBJECT,
    clblasInvalidDevice                   = CL_INVALID_DEVICE,
    clblasInvalidEventWaitList            = CL_INVALID_EVENT_WAIT_LIST,
    clblasOutOfResources                  = CL_OUT_OF_RESOURCES,
    clblasOutOfHostMemory                 = CL_OUT_OF_HOST_MEMORY,
    clblasInvalidOperation                = CL_INVALID_OPERATION,
    clblasCompilerNotAvailable            = CL_COMPILER_NOT_AVAILABLE,
    clblasBuildProgramFailure             = CL_BUILD_PROGRAM_FAILURE,
    /**@}*/

    /** @name clBLAS extended error codes */
    /**@{*/
    clblasNotImplemented         = -1024, /**< Functionality is not implemented */
    clblasNotInitialized,                 /**< clblas library is not initialized yet */
    clblasInvalidMatA,                    /**< Matrix A is not a valid memory object */
    clblasInvalidMatB,                    /**< Matrix B is not a valid memory object */
    clblasInvalidMatC,                    /**< Matrix C is not a valid memory object */
    clblasInvalidVecX,                    /**< Vector X is not a valid memory object */
    clblasInvalidVecY,                    /**< Vector Y is not a valid memory object */
    clblasInvalidDim,                     /**< An input dimension (M,N,K) is invalid */
    clblasInvalidLeadDimA,                /**< Leading dimension A must not be less than the size of the first dimension */
    clblasInvalidLeadDimB,                /**< Leading dimension B must not be less than the size of the second dimension */
    clblasInvalidLeadDimC,                /**< Leading dimension C must not be less than the size of the third dimension */
    clblasInvalidIncX,                    /**< The increment for a vector X must not be 0 */
    clblasInvalidIncY,                    /**< The increment for a vector Y must not be 0 */
    clblasInsufficientMemMatA,            /**< The memory object for Matrix A is too small */
    clblasInsufficientMemMatB,            /**< The memory object for Matrix B is too small */
    clblasInsufficientMemMatC,            /**< The memory object for Matrix C is too small */
    clblasInsufficientMemVecX,            /**< The memory object for Vector X is too small */
    clblasInsufficientMemVecY             /**< The memory object for Vector Y is too small */
    /**@}*/
  } clblasStatus;

  /*!
  * \defgroup OPENCL-STATE OpenCL state setup
  * \brief Functions to create or modify opencl state
  */

  /*!
  * \defgroup OPENCL-STATE-SINGLE single device
  * \brief Functions to create or modify single device opencl state
  * \ingroup OPENCL-STATE
  */
  /**@{*/

  /*! \brief clblasControl keeps library state that is passed into each API
   * It's an opaque pointer that user has no access to except through an explicit API
   */
  typedef struct clblas_control_opencl*  clblasControl;

  /*!
  * \brief setup the clblas control object from external OpenCL queues
  * \details This creates and returns to the user a structure that bundles clblas state
  * needed to be passed into each individual API call.  The array of command queues it
  * accepts allows the library to efficiently compute tail computation on problems that do
  * not evenly divide into the hardware workgroup size.
  * \remark It is recommended to create an array of at least 4 queues, such that up
  * to 4 sub-tiles can be computing in parallel on the same device
  *
  * \param[in] queue   An array of cl_command_queue's
  * \pre queue should only contain cl_command_queue's that point to the same device
  * \param[in] queue_size   The number of queue's in the queue array
  * \param[out] status   clblas error return value from function
  *
  * \ingroup OPENCL-STATE-SINGLE
  *
  * \returns \b On successful completion, a valid clblasControl object
  */
  CLBLAS_EXPORT clblasControl
    clblasCreateControl( cl_command_queue* queue, size_t queue_size, clblasStatus* status );

  /*!
  * \brief Return an array of events from the last kernel execution
  * \details Each cl_event in the array corresponds to a cl_command_queue passed into the clblasCreateControl
  * API.  However, the library determines how many of the queue's it is going to use for a given problem,
  * and therefore only a subset of the events maybe used.
  *
  * \param[out] events  An array of cl_event handles of size event_size.  Each cl_event corresponds to a
  * queue passed into clblasCreateControl
  * \param[out] events_size  The length of valid cl_event objects in the returned array
  * \param[in] control  A valid clblasControl created with clblasCreateControl
  * clsparseControl object
  *
  * \returns \b clblasSuccess
  *
  * \ingroup OPENCL-STATE-SINGLE
  */
  CLBLAS_EXPORT clblasStatus
    clblasGetEvents( cl_event* events, size_t* events_size, clblasControl control );

  /*!
  * \brief Set the events that the next API call should on when enqueueing the operation
  * \details If the user provides out-of-order queues into the library, it is necessary to specify
  * dependencies between the executions of the kernel.  This API takes a list of events that the next
  * enqueue'd operation should wait on before it's own execution.
  *
  * \param[in] eventWaitList  An array of cl_event handles of size event_size.  Each cl_event
  * represents an operation that is a dependency of the next enqueued operation.
  * \param[in] events_size  The length of valid cl_event objects in the returned array
  * \param[in] control  A valid clblasControl created with clblasCreateControl
  * clsparseControl object
  * \remarks The event waitlist state is sticky, such that it will persist in the control object until
  * the user explicitely clears it.
  *
  * \returns \b clblasSuccess
  *
  * \ingroup OPENCL-STATE-SINGLE
  */
  CLBLAS_EXPORT clblasStatus
    clblasSetWaitEvents( const cl_event* eventWaitList, size_t events_size, clblasControl control );
/**@}*/

#else
    #include "clBLAS-deprecated.h"
#endif

#endif // _CL_BLAS_OPENCL_H_
