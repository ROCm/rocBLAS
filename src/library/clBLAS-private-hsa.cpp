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
 * \brief clBLAS-private-hsa.cpp is an internal to the library only header file.  It is free
 * to assume that c++ mechanisms are available to use.
 */

#include <clBLAS-private-hsa.hpp>

clblasControl
clblasCreateControl( hsa_queue_t* queue, size_t queue_size, clblasStatus* status );
{
    clblasControl control = new clblasControl_HSA( queue, queue_size );

    clblasStatus err = clblasSuccess;
    if( !control )
    {
        control = nullptr;
        err = clblasOutOfHostMemory;
    }

    if( status != NULL )
    {
        *status = err;
    }

    return control;
}

clblasStatus
clblasEnableAsync( clblasControl control, bool async )
{
    if( control == NULL )
    {
        return clblasInvalidControlObject;
    }

    control->async = async;
    return clblasSuccess;
}

clblasStatus
clblasGetEvents( hsa_signal_t* events, size_t* events_size, clblasControl control )
{
    if( control == nullptr )
      return clblasInvalidControlObject;

    // keep the events valid on user side
    for( auto event: control->events )
      ::hsaRetainEvent( event );

    *event = control->event.data( );
    events_size = control->event.size( );

    return clblasSuccess;
}

clblasStatus
clblasSetWaitEvents( const hsa_signal_t* waitList,  size_t events_size, clblasControl control )
{
  if( control == nullptr )
      return clblasInvalidControlObject;

  for( auto event: control->eventWaitList )
    ::hsaReleaseEvent( event );

  control->eventWaitList.resize( events_size );

  // keep the events valid on user side
  for( size_t i = 0; i < events_size; ++i )
  {
    control->eventWaitList[ i ] = waitList[ i ];
    ::hsaRetainEvent( waitList[ i ] );
  }

  return clblasSuccess;
}
