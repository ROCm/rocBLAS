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
 * \brief An internal to the library only header file.  It is free
 * to assume that c++ mechanisms are available to use.
 */

#pragma once
#ifndef _CL_BLAS_PRIVATE_H_
#define _CL_BLAS_PRIVATE_H_

#include "clBLAS-private-opencl-multi.hpp"

/*! \brief Structure to encapsulate OpenCL state information that
 * \details This structure is initialized with an array of previously constructed clblasControl objects.  Each
 * individual object represents a single device that the user would like the library to use to complete
 * the given operation.
 */
struct clblas_control_opencl_multi
{
    /*! \brief The constructicon
     */
    clblas_control_opencl_multi( const clblasControl* controls, size_t control_size )
        : queue( queue_size ), events( queue_size ), async( CL_FALSE )
    {
      for( size_t i = 0; i < control_size; ++i )
      {
        this->controls[ i ] = controls[ i ];
      }
    }

    /*! /brief This boolean controls if the BLAS API will wait for ALL of the individual control objects
     * to finish.  This represents a boolean logical AND operation.
     */
    cl_bool async;

    /*! The vector of control objects passed into the contstructor of this object
     */
    std::vector< clblasControl > controls;
};

#endif // _CL_BLAS_PRIVATE_H_
