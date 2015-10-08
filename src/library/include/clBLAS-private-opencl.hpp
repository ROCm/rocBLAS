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
 * \brief clBLAS-private-opencl.hpp is an internal to the library only header file.  It is free
 * to assume that c++ mechanisms are available to use.
 */

#pragma once
#ifndef _CL_BLAS_PRIVATE_H_
#define _CL_BLAS_PRIVATE_H_

#include <clBLAS.h>

/*! \brief Structure to encapsulate OpenCL state information that
 * \details The contents of this control object only pertian to one device, and it will be considered an error
 * if multple cl_command_queue's are passed in bound to more than one device.  A seperate control structure will handle the case * of multi-GPU operations.
 * \pre The array of cl_command_queue's must all point to the same device
 * \note clBLAS can accelerate performance by using up to 4 cl_command_queue's to enqueue work to.  This usually happens when the matrix size is not evenly divisible by workgroup size, which creates 4 independant sub-tiles that can be processed in parallel.
 */
struct clblas_control_opencl
{
    /*! \brief The constructicon
     */
    clblas_control_opencl( const cl_command_queue* pQueue, size_t queue_size )
        : queue( queue_size ), events( queue_size ), async( CL_FALSE ), numEventsInWaitList( 0 ), eventWaitList( NULL )
    {
      for( size_t i = 0; i < queue_size; ++i )
      {
        queue[ i ] = pQueue[ i ];
        // Increment reference count since the library is caching a copy of the queue
        ::clRetainCommandQueue( pQueue );

        events[ i ] = NULL;
      }
    }

    /*! /brief The library will default to synchronous behavior, using a temporary internal cl_event
     * If the user wants asynchronous behavior, they can set this to true and then query the library for
     * an event per queue
     */
    cl_bool async;

    /*! This is the number of scalar values stored in the value buffer
     */
    std::vector< cl_command_queue > queue;

    /*! This offset is added to the cl_mem locations on device to define beginning of the data in the cl_mem buffers
     */
    std::vector< cl_event > events;

    /*! \brief The following is only used for queue's that support out-of-order execution.  This specifies
     * dependencies that the next enqueued operations must wait to complete first
     */
    const std::vector< cl_event > eventWaitList;
};

#endif // _CL_BLAS_PRIVATE_H_
