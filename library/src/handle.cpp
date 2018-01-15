/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "definitions.h"
#include "status.h"
#include "handle.h"
#include <hip/hip_runtime_api.h>
#include <unistd.h>
#include <pwd.h>

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
        int int_layer_mode = atoi(str_layer_mode);

        layer_mode = (rocblas_layer_mode)int_layer_mode;
    }

    // open log file
    if(layer_mode & rocblas_layer_mode_logging)
    {
        // construct filepath for log file in home directory
        const char* homedir_char;
        std::string homedir_str;

        if((homedir_char = getenv("HOME")) == NULL)
        {
            homedir_char = getpwuid(getuid())->pw_dir;
        }
        if(homedir_char == NULL)
        {
            std::cerr << "rocBLAS ERROR: cannot determine home directory for rocBLAS log file"
                      << std::endl;
            std::cerr << "rocBLAS ERROR: turn off logging or create a home directory" << std::endl;
            exit(-1);
        }
        else
        {
            homedir_str = std::string(homedir_char);
        }

        std::string filename  = "/rocblas_logfile.csv";
        std::string file_path = homedir_str + filename;

        // open log file
        log_ofs.open(file_path);
        log_ofs << "rocblas_create_handle";
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
