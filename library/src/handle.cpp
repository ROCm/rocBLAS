
#include "definitions.h"
#include "status.h"
#include "handle.h"
#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle()
{

    // default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&device_properties, device));

    // rocblas by default take the system default stream 0 users cannot create
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    // rocblas by default take the system default stream which user cannot destory
}

/*******************************************************************************
 * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
 ******************************************************************************/

/*******************************************************************************
 * set stream:
   This API assumes user has already created a valid stream
   Associate the following rocblas API call with this user provided stream
 ******************************************************************************/
rocblas_status _rocblas_handle::set_stream(hipStream_t user_stream)
{

    // TODO: check the user_stream valid or not
    rocblas_stream = user_stream;
    return rocblas_status_success;
}

/*******************************************************************************
 * get stream
 ******************************************************************************/
rocblas_status _rocblas_handle::get_stream(hipStream_t* stream) const
{
    *stream = rocblas_stream;
    return rocblas_status_success;
}
