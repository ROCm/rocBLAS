/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HANDLE_H
#define HANDLE_H
#include <hip/hip_runtime_api.h>
#include <vector>
#include <stdio.h>

#include "rocblas.h"

/*******************************************************************************
 * \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle() and the returned handle mus
 * It should be destroyed at the end using rocblas_destroy_handle().
 * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
******************************************************************************/
struct _rocblas_handle
{

    _rocblas_handle();
    ~_rocblas_handle();

    rocblas_status set_stream(hipStream_t stream);
    rocblas_status get_stream(hipStream_t* stream) const;

    rocblas_int device;
    hipDeviceProp_t device_properties;

    // rocblas by default take the system default stream 0 users cannot create
    hipStream_t rocblas_stream = 0;

    // default pointer_mode is on host
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;

    // default logging_mode is no logging
    rocblas_layer_mode layer_mode;
    FILE *rocblas_logfile;
    // need to read environment variable that contains layer_mode
};

#endif
