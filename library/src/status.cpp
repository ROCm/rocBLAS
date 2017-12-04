/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <hip/hip_runtime_api.h>
#include "rocblas.h"
#include "status.h"

/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 * TODO - enumerate library calls to hip runtime, enumerate possible errors from those calls
 ******************************************************************************/
rocblas_status get_rocblas_status_for_hip_status(hipError_t status)
{
    switch(status)
    {
    // success
    case hipSuccess:
        return rocblas_status_success;

    // internal hip memory allocation
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources:
        return rocblas_status_memory_error;

    // user-allocated hip memory
    case hipErrorInvalidDevicePointer: // hip memory
        return rocblas_status_invalid_pointer;

    // user-allocated device, stream, event
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return rocblas_status_invalid_handle;

    // library using hip incorrectly
    case hipErrorInvalidValue:
        return rocblas_status_internal_error;

    // hip runtime failing
    case hipErrorNoDevice: // no hip devices
    case hipErrorUnknown:
    default: return rocblas_status_internal_error;
    }
}
