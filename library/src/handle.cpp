/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "definitions.h"
#include "status.h"
#include "handle.h"
#include <hip/hip_runtime_api.h>
#include <unistd.h>
#include <pwd.h>
#include <sys/param.h>

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle() : layer_mode(rocblas_layer_mode_logging)
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
    if(layer_mode & rocblas_layer_mode_logging)
    {
        bool logfile_open = false;      // track if file is open
        std::string logfile_pathname_2; // logfile_pathname based on env variable

        // get current working directory
        char temp[MAXPATHLEN];
        std::string logfile_path = (getcwd(temp, MAXPATHLEN) ? std::string(temp) : std::string(""));
        std::string logfile_pathname_1 = logfile_path + "/rocblas_logfile.csv";

        log_ofs.open(logfile_pathname_1);

        if(log_ofs.is_open())
        {
            logfile_open = true;
        }
        else
        {
            // if cannot open logfile in cwd, try open file ROCBLAS_LOGFILE_PATH
            char const* tmp = getenv("ROCBLAS_LOGFILE_PATH");
            if(tmp != NULL)
            {
                logfile_pathname_2 = (std::string)tmp;
                log_ofs.open(logfile_pathname_2);
                if(log_ofs.is_open())
                {
                    logfile_open = true;
                }
            }
        }

        if(logfile_open == true)
        {
            // logfile is open, start logging
            log_ofs << "rocblas_create_handle";
        }
        else
        {
            // failed to open log file
            std::cerr << "rocBLAS ERROR: cannot open log file: " << logfile_pathname_1 << std::endl;
            std::cerr << "rocBLAS ERROR: cannot open log file: " << logfile_pathname_2 << std::endl;
            std::cerr << "rocBLAS ERROR: set environment variable" << std::endl;
            std::cerr << "rocBLAS ERROR: ROCBLAS_LOGFILE_PATH to the full" << std::endl;
            std::cerr << "rocBLAS ERROR: path for a rocblas logging file" << std::endl;
            std::cerr << "rocBLAS ERROR: turning off logging" << std::endl;

            // turn off logging by clearing bit for rocblas_layer_mode_logging;
            layer_mode = (rocblas_layer_mode)(layer_mode & (~rocblas_layer_mode_logging));
        }
    }
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    // rocblas by default take the system default stream which user cannot destroy
    if(layer_mode & rocblas_layer_mode_logging)
    {
        log_ofs.close();
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
