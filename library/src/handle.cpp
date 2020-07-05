/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"

#if BUILD_WITH_TENSILE
#ifndef USE_TENSILE_HOST
#include "Tensile.h"
#endif
#else
// see TensileHost.cpp for normal rocblas_initialize definition
// it isn't compiled if not BUILD_WITH_TENSILE so defining here
extern "C" void rocblas_initialize() {}
#endif

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle()
{
#if BUILD_WITH_TENSILE
#ifndef USE_TENSILE_HOST
    static int dummy = (tensileInitialize(), 0);
#endif
#endif

    // default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&device_properties, device));

    // rocblas by default take the system default stream 0 users cannot create

    // Device memory size
    //
    // If ROCBLAS_DEVICE_MEMORY_SIZE is set to > 0, then rocBLAS will allocate
    // the size specified, and will not manage device memory itself.
    //
    // If ROCBLAS_DEVICE_MEMORY_SIZE is unset, invalid or 0, then rocBLAS will
    // allocate a default initial size and manage device memory itself,
    // growing it as necessary.
    const char* env = getenv("ROCBLAS_DEVICE_MEMORY_SIZE");
    if(env)
        device_memory_size = strtoul(env, nullptr, 0);
    else
    {
        env = getenv("WORKBUF_TRSM_B_CHNK");
        if(env)
        {
            static int once
                = (rocblas_cerr << "Warning: Environment variable WORKBUF_TRSM_B_CHNK is "
                                   "obsolete.\nUse ROCBLAS_DEVICE_MEMORY_SIZE instead."
                                << std::endl,
                   0);
            device_memory_size = strtoul(env, nullptr, 0);
            if(device_memory_size)
                device_memory_size = device_memory_size * 1024 + 1024 * 1024 * 2;
        }
    }

    device_memory_is_rocblas_managed = !env || !device_memory_size;
    if(device_memory_is_rocblas_managed)
        device_memory_size = DEFAULT_DEVICE_MEMORY_SIZE;

    // Allocate device memory
    THROW_IF_HIP_ERROR((hipMalloc)(&device_memory, device_memory_size));

    // Initialize logging
    init_logging();
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    if(device_memory_in_use)
    {
        rocblas_cerr
            << "rocBLAS internal error: Handle object destroyed while device memory still in use."
            << std::endl;
        rocblas_abort();
    }
    if(device_memory)
        (hipFree)(device_memory);
}

/*******************************************************************************
 * helper for allocating device memory
 ******************************************************************************/
void* _rocblas_handle::device_allocator(size_t size)
{
    if(device_memory_in_use)
    {
        rocblas_cerr << "rocBLAS internal error: Cannot allocate device memory while it is already "
                        "allocated."
                     << std::endl;
        rocblas_abort();
    }
    if(size > device_memory_size)
    {
        if(!device_memory_is_rocblas_managed)
            return nullptr;
        if(device_memory)
        {
            (hipFree)(device_memory);
            device_memory = nullptr;
        }
        device_memory_size = 0;
        if((hipMalloc)(&device_memory, size) == hipSuccess)
            device_memory_size = size;
        else
            return nullptr;
    }
    device_memory_in_use = true;
    return device_memory;
}

/*******************************************************************************
 * start device memory size queries
 ******************************************************************************/
extern "C" rocblas_status rocblas_start_device_memory_size_query(rocblas_handle handle)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(handle->device_memory_size_query)
        return rocblas_status_size_query_mismatch;
    handle->device_memory_size_query = true;
    handle->device_memory_query_size = 0;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * stop device memory size queries
 ******************************************************************************/
extern "C" rocblas_status rocblas_stop_device_memory_size_query(rocblas_handle handle, size_t* size)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!handle->device_memory_size_query)
        return rocblas_status_size_query_mismatch;
    if(!size)
        return rocblas_status_invalid_pointer;
    *size                            = handle->device_memory_query_size;
    handle->device_memory_size_query = false;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * get the device memory size
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_device_memory_size(rocblas_handle handle, size_t* size)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!size)
        return rocblas_status_invalid_pointer;
    *size = handle->device_memory_size;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * set the device memory size
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_device_memory_size(rocblas_handle handle, size_t size)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // Cannot change memory allocation when a device_malloc
    // object is alive and using device memory.
    if(handle->device_memory_in_use)
        return rocblas_status_internal_error;

    // Free existing device memory, if any
    if(handle->device_memory)
    {
        (hipFree)(handle->device_memory);
        handle->device_memory = nullptr;
    }
    handle->device_memory_size = 0;

    // A zero size requests rocBLAS to take over management of device memory.
    // A nonzero size forces rocBLAS to use that as a fixed size, and not change it.
    handle->device_memory_is_rocblas_managed = !size;
    if(size)
    {
        size           = handle->roundup_device_memory_size(size);
        auto hipStatus = (hipMalloc)(&handle->device_memory, size);
        if(hipStatus != hipSuccess)
            return get_rocblas_status_for_hip_status(hipStatus);
        handle->device_memory_size = size;
    }
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Returns whether device memory is rocblas-managed
 ******************************************************************************/
extern "C" bool rocblas_is_managing_device_memory(rocblas_handle handle)
{
    return handle && handle->device_memory_is_rocblas_managed;
}

/**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Return a stream opened for logging.
 *                  If the environment variable with name environment_variable_name
 *                  is set, then it indicates the name of the file to be opened.
 *                  If the environment variable with name environment_variable_name
 *                  is not set, and the environment variable ROCBLAS_LOG_PATH is set,
 *                  then ROCBLAS_LOG_PATH indicates the name of the file to open.
 *                  Otherwise open the stream to stderr.
 *
 *  @param[in]
 *  environment_variable_name   const char*
 *                              Name of environment variable that contains
 *                              the full logfile path.
 */

static auto open_log_stream(const char* environment_variable_name)
{
    const char* logfile;
    return (logfile = getenv(environment_variable_name)) != nullptr
                   || (logfile = getenv("ROCBLAS_LOG_PATH")) != nullptr
               ? std::make_unique<rocblas_ostream>(logfile)
               : std::make_unique<rocblas_ostream>(STDERR_FILENO);
}

/*******************************************************************************
 * Logging initialization
 ******************************************************************************/
void _rocblas_handle::init_logging()
{
    // set layer_mode from value of environment variable ROCBLAS_LAYER
    const char* str_layer_mode = getenv("ROCBLAS_LAYER");
    if(str_layer_mode)
    {
        layer_mode = static_cast<rocblas_layer_mode>(strtol(str_layer_mode, 0, 0));

        // open log_trace file
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace_os = open_log_stream("ROCBLAS_LOG_TRACE_PATH");

        // open log_bench file
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench_os = open_log_stream("ROCBLAS_LOG_BENCH_PATH");

        // open log_profile file
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile_os = open_log_stream("ROCBLAS_LOG_PROFILE_PATH");
    }
}
