/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "definitions.h"
#include "handle.h"
#include <hip/hip_runtime_api.h>
#include <unistd.h>
#include <sys/param.h>
#include "logging.h"

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle() : layer_mode(rocblas_layer_mode_log_trace)
{
    // default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&device_properties, device));

    // rocblas by default take the system default stream 0 users cannot create

    // set layer_mode from vaule of environment variable ROCBLAS_LAYER
    char* str_layer_mode;
    if((str_layer_mode = getenv("ROCBLAS_LAYER")) == NULL)
    {
        layer_mode = rocblas_layer_mode_none;
    }
    else
    {
        layer_mode = (rocblas_layer_mode)(atoi(str_layer_mode));
    }

    // open log file
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        open_log_stream(
            &log_trace_os, &log_trace_ofs, "ROCBLAS_LOG_TRACE_PATH", "rocblas_log_trace.csv");

        *log_trace_os << "rocblas_create_handle";
    }

    // open log_bench file
    if(layer_mode & rocblas_layer_mode_log_bench)
    {
        open_log_stream(
            &log_bench_os, &log_bench_ofs, "ROCBLAS_LOG_BENCH_PATH", "rocblas_log_bench.csv");
    }
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    // rocblas by default take the system default stream which user cannot destroy

    // Close log files
    if(log_trace_ofs.is_open())
    {
        log_trace_ofs.close();
    }
    if(log_bench_ofs.is_open())
    {
        log_bench_ofs.close();
    }
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
