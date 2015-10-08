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

#include "include/clBLAS-private-opencl-multi.hpp"

clblasControlMulti
clblasCreateControlMulti( clblasControl* controls, size_t control_size, clblasStatus* status );
{
    clblasControlMulti control = new clblas_control_opencl_multi( controls, control_size );

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
clblasEnableAsyncMulti( clblasControlMulti control, bool async )
{
    if( control == nullptr )
    {
        return clblasInvalidControlObject;
    }

    control->async = async;
    return clblasSuccess;
}
