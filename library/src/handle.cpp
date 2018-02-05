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
        log_trace_ofs = open_logfile("ROCBLAS_LOG_TRACE_PATH", "rocblas_log_trace.csv");

        if(log_trace_ofs.is_open() == true)
        {
            // logfile is open, start logging
            log_trace_ofs << "rocblas_create_handle";
        }
        else
        {
            // failed to open log file
            std::cerr << "rocBLAS ERROR: cannot open log trace file: " << std::endl;
            std::cerr << "rocBLAS ERROR: set environment variable" << std::endl;
            std::cerr << "rocBLAS ERROR: ROCBLAS_LOG_TRACE_PATH to the full" << std::endl;
            std::cerr << "rocBLAS ERROR: path for a rocblas logging file" << std::endl;
            std::cerr << "rocBLAS ERROR: turning off logging" << std::endl;

            // turn off logging by clearing bit for rocblas_layer_mode_log_trace;
            layer_mode = (rocblas_layer_mode)(layer_mode & (~rocblas_layer_mode_log_trace));
        }
    }

    // open log_bench file
    if(layer_mode & rocblas_layer_mode_log_bench)
    {
        log_bench_ofs = open_logfile("ROCBLAS_LOG_BENCH_PATH", "rocblas_log_bench.txt");

        if(log_bench_ofs.is_open() != true)
        {
            // failed to open log file
            std::cerr << "rocBLAS ERROR: cannot open log bench file: " << std::endl;
            std::cerr << "rocBLAS ERROR: set environment variable" << std::endl;
            std::cerr << "rocBLAS ERROR: ROCBLAS_LOGFILE_PATH to the full" << std::endl;
            std::cerr << "rocBLAS ERROR: path for a rocblas logging file" << std::endl;
            std::cerr << "rocBLAS ERROR: turning off logging" << std::endl;

            // turn off logging by clearing bit for rocblas_layer_mode_log_trace;
            layer_mode = (rocblas_layer_mode)(layer_mode & (~rocblas_layer_mode_log_trace));
        }
    }
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    // rocblas by default take the system default stream which user cannot destroy
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        log_trace_ofs.close();
    }
    if(layer_mode & rocblas_layer_mode_log_bench)
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
