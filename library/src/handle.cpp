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
_rocblas_handle::_rocblas_handle()
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

    // allocate trsm temp buffers
    THROW_IF_HIP_ERROR(hipMalloc(&trsm_Y, WORKBUF_TRSM_Y_SZ));
    THROW_IF_HIP_ERROR(hipMalloc(&trsm_invA, WORKBUF_TRSM_INVA_SZ));
    THROW_IF_HIP_ERROR(hipMalloc(&trsm_invA_C, WORKBUF_TRSM_INVA_C_SZ));

    // open log file
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        open_log_stream(&log_trace_os, &log_trace_ofs, "ROCBLAS_LOG_TRACE_PATH");

        *log_trace_os << "rocblas_create_handle";
    }

    // open log_bench file
    if(layer_mode & rocblas_layer_mode_log_bench)
    {
        open_log_stream(&log_bench_os, &log_bench_ofs, "ROCBLAS_LOG_BENCH_PATH");
    }
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    // rocblas by default take the system default stream which user cannot destroy

    if(trsm_Y)
        hipFree(trsm_Y);
    
    if(trsm_invA)
        hipFree(trsm_invA);

    if(trsm_invA_C)
        hipFree(trsm_invA_C);

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

// trsm get pointers
void * _rocblas_handle::get_trsm_Y()
{
    return trsm_Y;
}

void * _rocblas_handle::get_trsm_invA()
{
    return trsm_invA;
}

void * _rocblas_handle::get_trsm_invA_C()
{
    return trsm_invA_C;
}

