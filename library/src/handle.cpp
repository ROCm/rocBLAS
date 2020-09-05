/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.hpp"
#include <cstdarg>

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
static inline int getDevice()
{
    int device;
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    return device;
}

_rocblas_handle::_rocblas_handle()
    : device(getDevice()) // active device is handle device
{
#if BUILD_WITH_TENSILE
#ifndef USE_TENSILE_HOST
    static int once = (tensileInitialize(), 0);
#endif
#endif

    // Device memory size
    const char* env = getenv("ROCBLAS_DEVICE_MEMORY_SIZE");
    if(env)
        device_memory_size = strtoul(env, nullptr, 0);
    else if(getenv("WORKBUF_TRSM_B_CHNK"))
    {
        static auto& once = rocblas_cerr << "Warning: Environment variable WORKBUF_TRSM_B_CHNK is "
                                            "obsolete.\nUse ROCBLAS_DEVICE_MEMORY_SIZE instead."
                                         << std::endl;
    }

    if(!env || !device_memory_size)
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
    auto hipStatus = (hipFree)(device_memory);
    if(hipStatus != hipSuccess)
    {
        rocblas_cerr << "rocBLAS error during hipFree in handle destructor: "
                     << rocblas_status_to_string(get_rocblas_status_for_hip_status(hipStatus))
                     << std::endl;
        rocblas_abort();
    };
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
 * Get the device memory size
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
 * Set the device memory size
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
    RETURN_IF_HIP_ERROR((hipFree)(handle->device_memory));
    handle->device_memory      = nullptr;
    handle->device_memory_size = 0;

    // Allocate size rounded up to MIN_CHUNK_SIZE
    size           = roundup_device_memory_size(size);
    auto hipStatus = (hipMalloc)(&handle->device_memory, size);
    if(hipStatus != hipSuccess)
    {
        handle->device_memory = nullptr;
        return get_rocblas_status_for_hip_status(hipStatus);
    }
    else
    {
        handle->device_memory_size = size;
        return rocblas_status_success;
    }
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Returns whether device memory is rocblas-managed
 * (Returns always false now, since it is impractical to manage in rocBLAS.)
 ******************************************************************************/
extern "C" bool rocblas_is_managing_device_memory(rocblas_handle)
{
    return false;
}

/* \brief
   \details
   Returns true if the handle is in device memory size query mode.
   @param[in]
   handle           rocblas handle
 ******************************************************************************/
extern "C" bool rocblas_is_device_memory_size_query(rocblas_handle handle)
{
    return handle && handle->is_device_memory_size_query();
}

// Helper function to round up sizes and compute total size
static inline size_t va_total_device_memory_size(size_t count, va_list ap)
{
    size_t total = 0;
    while(count--)
        total += roundup_device_memory_size(va_arg(ap, size_t));
    return total;
}

/* \brief
   \details
   Sets the optimal device memory size during a query
   Returns rocblas_status_size_increased if the maximum size was increased,
   rocblas_status_size_unchanged if the maximum size was unchanged, or
   rocblas_status_size_query_mismatch if the handle is not in query mode.
   @param[in]
   handle           rocblas handle
   count            number of sizes
   ...              sizes needed for optimal execution of the current kernel
 ******************************************************************************/
extern "C" rocblas_status
    rocblas_set_optimal_device_memory_size_impl(rocblas_handle handle, size_t count, ...)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    va_list ap;
    va_start(ap, count);
    size_t total = va_total_device_memory_size(count, ap);
    va_end(ap);
    return handle->set_optimal_device_memory_size(total);
}

/*! \brief
    \details
    Borrows size bytes from the device memory allocated in handle.
    Returns rocblas_status_invalid_handle if handle is nullptr; rocblas_status_invalid_pointer if res is nullptr; otherwise rocblas_status_success
    @param[in]
    handle          rocblas handle
    count           number of sizes
    ...             sizes to allocate
    @param[out]
    res             pointer to pointer to struct rocblas_device_malloc_base
 ******************************************************************************/
extern "C" rocblas_status rocblas_device_malloc_alloc(rocblas_handle               handle,
                                                      rocblas_device_malloc_base** res,
                                                      size_t                       count,
                                                      ...)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!res)
        return rocblas_status_invalid_pointer;
    if(!count)
        return rocblas_status_invalid_size;

    *res = nullptr; // in case of exception

    // Compute the total of the rounded up sizes
    va_list ap;
    va_start(ap, count);
    size_t total = va_total_device_memory_size(count, ap);
    va_end(ap);

    // Borrow allocated memory from the handle
    auto mem = handle->device_malloc_count(count, total);

    // If unsuccessful
    if(!mem)
        return rocblas_status_memory_error;

    // Get the base of the allocated pointers
    char* addr = static_cast<char*>(mem[0]);

    // Compute each pointer based on offsets
    va_start(ap, count);
    for(size_t i = 0; i < count; ++i)
    {
        size_t size = roundup_device_memory_size(va_arg(ap, size_t));
        mem[i]      = size ? addr : nullptr;
        addr += size;
    }
    va_end(ap);

    // Move it to the heap
    *res = new auto(std::move(mem));

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*! \brief
    \details
    Tells whether an allocation succeeded
    @param[in]
    ptr             pointer to struct rocblas_device_malloc_base
 ******************************************************************************/
extern "C" bool rocblas_device_malloc_success(rocblas_device_malloc_base* ptr)
{
    using _device_malloc = decltype(rocblas_handle {}->device_malloc(0));
    return ptr && *static_cast<_device_malloc*>(ptr);
}

/*! \brief
    \details
    Converts rocblas_device_malloc() to a pointer if it only has one pointer.
    Retuns rocblas_status_invalid_pointer if ptr or res is nullptr, there is more than one pointer, or the underyling object is not from rocblas_device_malloc(); rocblas_status_success otherwise
    @param[in]
    ptr             pointer to struct rocblas_device_malloc_base
    @param[out]
    res             pointer to pointer to void
*/
extern "C" rocblas_status rocblas_device_malloc_ptr(rocblas_device_malloc_base* ptr, void** res)
try
{
    using _device_malloc = decltype(rocblas_handle {}->device_malloc(0));
    if(!ptr || !res)
        return rocblas_status_invalid_pointer;
    *res = static_cast<void*>(*static_cast<_device_malloc*>(ptr));
    return rocblas_status_success;
}
catch(...)
{
    return rocblas_status_invalid_pointer;
}

/*! \brief
    \details
    Gets a pointer to device memory allocated by rocblas_device_malloc().
    Returns rocblas_status_invalid_pointer if ptr or res is nullptr or the underyling object is not from rocblas_device_malloc(); rocblas_status_success otherwise
    @param[in]
    ptr             pointer to struct rocblas_device_malloc_base
    index           index of the pointer to get
    @param[out]
    res             pointer to pointer to void
*/
extern "C" rocblas_status
    rocblas_device_malloc_get(rocblas_device_malloc_base* ptr, size_t index, void** res)
try
{
    using _device_malloc = decltype(rocblas_handle {}->device_malloc(0));
    if(!ptr || !res)
        return rocblas_status_invalid_pointer;
    *res = (*static_cast<_device_malloc*>(ptr))[index];
    return rocblas_status_success;
}
catch(...)
{
    return rocblas_status_invalid_pointer;
}

/*! \brief
    \details
    Frees memory borrowed from the device memory allocated in handle.
    @param[in]
    ptr             pointer to struct rocblas_device_malloc_base
*/
extern "C" rocblas_status rocblas_device_malloc_free(rocblas_device_malloc_base* ptr)
{
    using _device_malloc = decltype(rocblas_handle {}->device_malloc(0));
    delete static_cast<_device_malloc*>(ptr);
    return rocblas_status_success;
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
