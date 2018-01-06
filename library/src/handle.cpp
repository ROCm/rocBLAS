/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "definitions.h"
#include "status.h"
#include "handle.h"
#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle() : layer_mode(rocblas_layer_mode_logging)
{
    // default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&device_properties, device));

    // rocblas by default take the system default stream 0 users cannot create
    char* str_layer_mode = getenv("ROCBLAS_LAYER");
    int   int_layer_mode = atoi(str_layer_mode);

    layer_mode = (rocblas_layer_mode) int_layer_mode;
    
    if(layer_mode & rocblas_layer_mode_logging)
    {
        // open log file in home directory
        const char *file_name = "/rocblas_logfile.csv";
        char *home_dir = getenv("HOME");
        char *file_path = (char *) malloc(strlen(home_dir) + strlen(file_name) + 1);
        strncpy(file_path, home_dir, strlen(home_dir) + 1);
        strncat(file_path, file_name, strlen(file_name) + 1);
        rocblas_logfile = fopen(file_path, "w");
        free(file_path);

        if (rocblas_logfile == NULL)
        {
            printf("ERROR: rocBLAS: could not open logging file %s\n",file_path);
        }
        else
        {
            if (layer_mode & rocblas_layer_mode_logging_synch)
            {
                fprintf(rocblas_logfile, "rocblas_handle,constructor,rocblas_layer_mode_logging_synch\n");
            }
            else
            {
                fprintf(rocblas_logfile, "rocblas_handle,constructor,rocblas_layer_mode_logging\n");
            }
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
        fprintf(rocblas_logfile, "rocblas_handle,destructor\n");
        fflush(rocblas_logfile);
        fclose(rocblas_logfile);
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
