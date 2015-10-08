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
 * \brief clBLAS-private-opencl.cpp is an internal to the library only header file.  It is free
 * to assume that c++ mechanisms are available to use.
 */

#include <clBLAS-private-opencl.hpp>

clblasControl
clblasCreateControl( cl_command_queue* queue, size_t queue_size, clblasStatus* status );
{
    clblasControl control = new clblas_control_opencl( queue, queue_size );

    clblasStatus err = clblasSuccess;
    if( !control )
    {
        control = nullptr;
        err = clblasOutOfHostMemory;
    }

    if( status != nullptr )
    {
        *status = err;
    }

    return control;
}

clblasStatus
clblasEnableAsync( clblasControl control, bool async )
{
    if( control == nullptr )
    {
        return clblasInvalidControlObject;
    }

    control->async = async;
    return clblasSuccess;
}

clblasStatus
clblasGetEvents( cl_event* events, size_t* events_size, clblasControl control )
{
    if( control == nullptr )
      return clblasInvalidControlObject;

    // keep the events valid on user side
    for( auto event: control->events )
      ::clRetainEvent( event );

    *event = control->event.data( );
    events_size = control->event.size( );

    return clblasSuccess;
}

clblasStatus
clblasSetWaitEvents( const cl_event* waitList,  size_t events_size, clblasControl control )
{
  if( control == nullptr )
      return clblasInvalidControlObject;

  for( auto event: control->eventWaitList )
    ::clReleaseEvent( event );

  control->eventWaitList.resize( events_size );

  // keep the events valid on user side
  for( size_t i = 0; i < events_size; ++i )
  {
    control->eventWaitList[ i ] = waitList[ i ];
    ::clRetainEvent( waitList[ i ] );
  }

  return clblasSuccess;
}
