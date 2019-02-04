/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include <cstdlib>

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle()
{
    // default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&device_properties, device));

    // rocblas by default take the system default stream 0 users cannot create

    // allocate trsm temp buffers
    THROW_IF_HIP_ERROR(hipMalloc(&trsm_Y, WORKBUF_TRSM_Y_SZ));
    THROW_IF_HIP_ERROR(hipMalloc(&trsm_invA, WORKBUF_TRSM_INVA_SZ));
    THROW_IF_HIP_ERROR(hipMalloc(&trsm_invA_C, WORKBUF_TRSM_INVA_C_SZ));

    // allocate trsv temp buffers
    THROW_IF_HIP_ERROR(hipMalloc(&trsv_x, WORKBUF_TRSV_X_SZ));
    THROW_IF_HIP_ERROR(hipMalloc(&trsv_alpha, WORKBUF_TRSV_ALPHA_SZ));
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    if(trsm_Y)
        hipFree(trsm_Y);
    if(trsm_invA)
        hipFree(trsm_invA);
    if(trsm_invA_C)
        hipFree(trsm_invA_C);
    if(trsv_x)
        hipFree(trsv_x);
    if(trsv_alpha)
        hipFree(trsv_alpha);
}

/*******************************************************************************
 * Static handle data
 ******************************************************************************/
rocblas_layer_mode _rocblas_handle::layer_mode = rocblas_layer_mode_none;
std::ofstream _rocblas_handle::log_trace_ofs;
std::ostream* _rocblas_handle::log_trace_os;
std::ofstream _rocblas_handle::log_bench_ofs;
std::ostream* _rocblas_handle::log_bench_os;
std::ofstream _rocblas_handle::log_profile_ofs;
std::ostream* _rocblas_handle::log_profile_os;
_rocblas_handle::init _rocblas_handle::handle_init;

/**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Open stream log_os for logging.
 *                  If the environment variable with name environment_variable_name
 *                  is not set, then stream log_os to std::cerr.
 *                  Else open a file at the full logfile path contained in
 *                  the environment variable.
 *                  If opening the file suceeds, stream to the file
 *                  else stream to std::cerr.
 *
 *  @param[in]
 *  environment_variable_name   const char*
 *                              Name of environment variable that contains
 *                              the full logfile path.
 *
 *  @parm[out]
 *  log_os      std::ostream*&
 *              Output stream. Stream to std:cerr if environment_variable_name
 *              is not set, else set to stream to log_ofs
 *
 *  @parm[out]
 *  log_ofs     std::ofstream&
 *              Output file stream. If log_ofs->is_open()==true, then log_os
 *              will stream to log_ofs. Else it will stream to std::cerr.
 */

static void open_log_stream(const char* environment_variable_name,
                            std::ostream*& log_os,
                            std::ofstream& log_ofs)

{
    // By default, output to cerr
    log_os = &std::cerr;

    // if environment variable is set, open file at logfile_pathname contained in the
    // environment variable
    auto logfile_pathname = getenv(environment_variable_name);
    if(logfile_pathname)
    {
        log_ofs.open(logfile_pathname, std::ios_base::trunc);

        // if log_ofs is open, then stream to log_ofs, else log_os is already set to std::cerr
        if(log_ofs.is_open())
            log_os = &log_ofs;
    }
}

/*******************************************************************************
 * Static runtime initialization
 ******************************************************************************/
_rocblas_handle::init::init()
{
    // set layer_mode from value of environment variable ROCBLAS_LAYER
    auto str_layer_mode = getenv("ROCBLAS_LAYER");
    if(str_layer_mode)
    {
        layer_mode = static_cast<rocblas_layer_mode>(strtol(str_layer_mode, 0, 0));

        // open log_trace file
        if(layer_mode & rocblas_layer_mode_log_trace)
            open_log_stream("ROCBLAS_LOG_TRACE_PATH", log_trace_os, log_trace_ofs);

        // open log_bench file
        if(layer_mode & rocblas_layer_mode_log_bench)
            open_log_stream("ROCBLAS_LOG_BENCH_PATH", log_bench_os, log_bench_ofs);

        // open log_profile file
        if(layer_mode & rocblas_layer_mode_log_profile)
            open_log_stream("ROCBLAS_LOG_PROFILE_PATH", log_profile_os, log_profile_ofs);
    }
}

/*******************************************************************************
 * Static reinitialization (for testing only)
 ******************************************************************************/
namespace rocblas {
void reinit_logs()
{
    _rocblas_handle::log_trace_ofs.close();
    _rocblas_handle::log_bench_ofs.close();
    _rocblas_handle::log_profile_ofs.close();
    new(&_rocblas_handle::handle_init) _rocblas_handle::init;
}
} // namespace rocblas
