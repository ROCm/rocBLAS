/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <mutex>
#include <unistd.h>

#if BUILD_WITH_TENSILE
#include "Tensile.h"
#ifdef USE_TENSILE_HOST
#include "tensile_host.hpp"
#endif
#endif

/*******************************************************************************
 * Static handle data                                                          *
 *******************************************************************************/
constexpr size_t _rocblas_handle::DEFAULT_DEVICE_MEMORY_SIZE;
constexpr size_t _rocblas_handle::MIN_CHUNK_SIZE;

/*******************************************************************************
 * Handle Constructor                                                          *
 *******************************************************************************/
_rocblas_handle::_rocblas_handle()
{
#if BUILD_WITH_TENSILE
    static int dummy = (tensileInitialize(), 0);
#ifdef USE_TENSILE_HOST
    // Cache the Tensile host on the first handle, since it takes
    // up to 10 seconds to load; later handles reuse the same host
    static TensileHost* hostImpl = createTensileHost();
    host                         = hostImpl;
#endif
#endif

    // default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&device_properties, device));

    // Initialize logging
    init_logging();

    // rocblas by default takes the system default stream 0 users cannot create

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
                = fputs("Warning: Environment variable WORKBUF_TRSM_B_CHNK is obsolete.\n"
                        "Use ROCBLAS_DEVICE_MEMORY_SIZE instead.\n",
                        stderr);
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
}

/*******************************************************************************
 * Handle Destructor                                                           *
 *******************************************************************************/
_rocblas_handle::~_rocblas_handle()
{
    if(device_memory_in_use)
    {
        fputs("rocBLAS internal error: Handle object destroyed while device memory still in use.\n",
              stderr);
        abort();
    }
    if(device_memory)
        (hipFree)(device_memory);

    delete log_trace;
    delete log_bench;
    delete log_profile;
}

/*******************************************************************************
 * Helper for allocating device memory                                         *
 *******************************************************************************/
void* _rocblas_handle::device_allocator(size_t size)
{
    if(device_memory_in_use)
    {
        fputs("rocBLAS internal error: Cannot allocate device memory while it is already "
              "allocated.\n",
              stderr);
        abort();
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
 * Start device memory size queries                                            *
 *******************************************************************************/
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
 * Stop device memory size queries                                             *
 *******************************************************************************/
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
 * Get the device memory size                                                  *
 *******************************************************************************/
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
 * Set the device memory size                                                  *
 *******************************************************************************/
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
 * Returns whether device memory is rocblas-managed                            *
 *******************************************************************************/
extern "C" bool rocblas_is_managing_device_memory(rocblas_handle handle)
{
    return handle && handle->device_memory_is_rocblas_managed;
}

/*******************************************************************************
 * Initialization of logging                                                   *
 *******************************************************************************/
void _rocblas_handle::init_logging()
{
    // set layer_mode from value of environment variable ROCBLAS_LAYER
    auto str_layer_mode = getenv("ROCBLAS_LAYER");
    if(str_layer_mode)
    {
        layer_mode = static_cast<rocblas_layer_mode>(strtol(str_layer_mode, 0, 0));

        // open log_trace file
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace = new rocblas_logging_stream(getenv("ROCBLAS_LOG_TRACE_PATH"));

        // open log_bench file
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench = new rocblas_logging_stream(getenv("ROCBLAS_LOG_BENCH_PATH"));

        // open log_profile file
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile = new rocblas_logging_stream(getenv("ROCBLAS_LOG_PROFILE_PATH"));
    }
}

/*****************************************************************************************
 * Construct a rocBLAS logging stream, logging output to a given filename                *
 * or to stderr if the filename is nullptr or cannot be created                          *
 *****************************************************************************************/
rocblas_logging_stream::rocblas_logging_stream(const char* filename)
    : filehandle(filename ? open(filename, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644)
                          : STDERR_FILENO)
{
    fprintf(stderr, "open(%s) returned %d\n", filename, filehandle);
    if(filehandle == -1)
    {
        fprintf(stderr, "Error opening %s: %m\nLogging to stderr instead.\n", filename);
        filehandle = STDERR_FILENO;
    }
}

/*****************************************************************************************
 * Flush a logging stream to a file or stderr                                            *
 *****************************************************************************************/
void rocblas_logging_stream::flush()
{
    if(filehandle >= 0)
    {
        std::string s = str();
        if(s.size())
        {
            static std::mutex           mutex;
            std::lock_guard<std::mutex> lock(mutex);

            do
            {
                ssize_t written;
                do
                    written = ::write(filehandle, s.data(), s.size());
                while(written == 0 || (written < 0 && errno == EINTR));

                if(written < 0)
                {
                    // Write error message to stdout if error occurred on stderr
                    fprintf(filehandle == STDERR_FILENO ? stdout : stderr,
                            "Error when writing to log file: %m\n");
                    filehandle = -1; // Stop further output to this filehandle.
                    break;
                }

                // Remove the bytes written so far
                s = s.substr(written);
            } while(s.size());
        }
    }

    // Erase the logging stream string buffer
    clear();
    str({});
}

/*****************************************************************************************
 * Destroy a rocBLAS logging stream, by flushing any data and closing its filehandle     *
 * if it's not stderr.                                                                   *
 *****************************************************************************************/
rocblas_logging_stream::~rocblas_logging_stream()
{
    if(filehandle >= 0)
    {
        flush();
        if(filehandle != STDERR_FILENO)
            ::close(filehandle);
    }
}
