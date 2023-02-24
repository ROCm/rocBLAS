/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#include "handle.hpp"
#include <cstdarg>
#include <limits>
#ifdef WIN32
#include <windows.h>
#endif

#if BUILD_WITH_TENSILE
#else
// see TensileHost.cpp for normal rocblas_initialize definition
// it isn't compiled if not BUILD_WITH_TENSILE so defining here
extern "C" void rocblas_initialize() {}
#endif

// forcing early cleanup
extern "C" void rocblas_shutdown()
{
    rocblas_internal_ostream::clear_workers();
}

/* read environment variable */
/* On windows, getenv take a copy of the environment at the beginning of the process */
/* This behavior is not suited for the purpose of the tests */
const char* read_env(const char* env_var)
{
#ifdef WIN32
    const DWORD              nSize = _MAX_PATH;
    static thread_local char lpBuffer[nSize];
    lpBuffer[0] = 0; // terminate for reuse
    if(GetEnvironmentVariableA(env_var, lpBuffer, nSize) == 0)
        return nullptr;
    else
        return lpBuffer;
#else
    return getenv(env_var);
#endif
}

// This variable can be set in hipBLAS or other libraries to change the default
// device memory size
static thread_local size_t t_rocblas_device_malloc_default_memory_size;

extern "C" void rocblas_device_malloc_set_default_memory_size(size_t size)
{
    t_rocblas_device_malloc_default_memory_size = size;
}

static inline int getActiveDevice()
{
    int device;
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    return device;
}

static inline int getActiveArch(int deviceId)
{
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties(&deviceProperties, deviceId);
    return deviceProperties.gcnArch;
}

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle()
    : device(getActiveDevice()) // active device is handle device
    , arch(getActiveArch(device))
{
    archMajor = arch / 100; // this may need to switch to string handling in the future

    //ROCBLAS_STREAM_ORDER_ALLOC
    const char* stream_order_alloc_env = read_env("ROCBLAS_STREAM_ORDER_ALLOC");

    if(stream_order_alloc_env)
    {
        int stream_order_alloc_env_val = strtoul(stream_order_alloc_env, nullptr, 0);
        stream_order_alloc             = stream_order_alloc_env_val ? true : false;
    }

    // Device memory size
    const char* env = read_env("ROCBLAS_DEVICE_MEMORY_SIZE");
    if(env)
        device_memory_size = strtoul(env, nullptr, 0);

    if(env && device_memory_size)
    {
        device_memory_owner = rocblas_device_memory_ownership::user_managed;
    }
    else
    {
        device_memory_owner = rocblas_device_memory_ownership::rocblas_managed;

        if(!env)
        {
            if(t_rocblas_device_malloc_default_memory_size)
            {
                device_memory_size = t_rocblas_device_malloc_default_memory_size;
                t_rocblas_device_malloc_default_memory_size = 0;
            }
            else
            {
                device_memory_size = DEFAULT_DEVICE_MEMORY_SIZE;
            }
        }
    }

    if(!stream_order_alloc)
    { // Allocate device memory
        if(device_memory_size)
            THROW_IF_HIP_ERROR((hipMalloc)(&device_memory, device_memory_size));
    }
    else
    {
// hipMallocAsync and hipFreeAsync are defined in hip version 5.2.0
// Support for default stream added in hip version 5.3.0
#if HIP_VERSION >= 50300000
        // The following allocation & free of device memory using hipMallocAsync/hipFreeAsync will allocate memory from
        // the OS and release it to default memory pool. Further allocation of memory using hipMallocAsync
        // will be from the memory pool and it will be faster.
        THROW_IF_HIP_ERROR((hipMallocAsync)(&device_memory, device_memory_size, stream));

        THROW_IF_HIP_ERROR((hipFreeAsync)(device_memory, stream));

        device_memory = nullptr;
#else
        rocblas_cerr
            << "rocBLAS internal error: Stream order allocation is supported on ROCm 5.3 and above."
            << std::endl;
        rocblas_abort();
#endif
    }

    // Initialize logging
    init_logging();

    // Initialize numerical checking
    init_check_numerics();
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
    // Free device memory unless it's user-owned
    if(device_memory_owner != rocblas_device_memory_ownership::user_owned)
    {
        hipError_t hipStatus;
        if(!stream_order_alloc)
        {
            hipStatus = (hipFree)(device_memory);
            if(hipStatus != hipSuccess)
            {
                rocblas_cerr
                    << "rocBLAS error during freeing of allocated memory in handle destructor: "
                    << rocblas_status_to_string(get_rocblas_status_for_hip_status(hipStatus))
                    << std::endl;
                rocblas_abort();
            };
        }
        else
        {
// hipMallocAsync and hipFreeAsync are defined in hip version 5.2.0
// Support for default stream added in hip version 5.3.0
#if HIP_VERSION >= 50300000
            hipStatus = (device_memory) ? (hipFreeAsync)(device_memory, stream) : hipSuccess;
            if(hipStatus != hipSuccess)
            {
                rocblas_cerr << "rocBLAS error during freeing of allocated memory in handle "
                                "destructor (stream order allocation): "
                             << rocblas_status_to_string(
                                    get_rocblas_status_for_hip_status(hipStatus))
                             << std::endl;
                rocblas_abort();
            };

            hipMemPool_t mem_pool;
            int          device;
            hipGetDevice(&device);
            hipDeviceGetDefaultMemPool(&mem_pool, device);

            //Releases device memory back to OS
            hipMemPoolTrimTo(mem_pool, 0);
#endif
        }
        // Free memory allocated on the host
        for(auto host_mem : host_mem_pointers)
        {
            free(host_mem);
        }
    }
}

/*******************************************************************************
 * helper for allocating device memory
 ******************************************************************************/
#if ROCBLAS_REALLOC_ON_DEMAND
bool _rocblas_handle::device_allocator(size_t size)
{
    bool success = size <= device_memory_size - device_memory_in_use;
    if(!success && device_memory_owner == rocblas_device_memory_ownership::rocblas_managed)
    {
        if(device_memory_in_use)
        {
            rocblas_cerr << "rocBLAS internal error: Cannot reallocate device memory while it is "
                            "already in use.";
            rocblas_abort();
        }

        // Temporarily change the thread's default device ID to the handle's device ID
        // cppcheck-suppress unreadVariable
        auto saved_device_id = push_device_id();

        device_memory_size = 0;
        if(!device_memory || (hipFree)(device_memory) == hipSuccess)
        {
            success = (hipMalloc)(&device_memory, size) == hipSuccess;
            if(success)
                device_memory_size = size;
            else
                device_memory = nullptr;
        }
    }
    return success;
}
#endif

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
 * Free any allocated memory unless owned by user, and reset the handle to being
 * rocBLAS-managed
 ******************************************************************************/
static rocblas_status free_existing_device_memory(rocblas_handle handle)
{
    // Cannot change memory allocation when a device_malloc object is alive and
    // using device memory. This should never happen unless this function is
    // called from inside library code which borrows allocated device memory.
    if(handle->device_memory_in_use)
        return rocblas_status_internal_error;

    // Free existing device memory in handle, unless owned by user
    if(handle->device_memory
       && handle->device_memory_owner != rocblas_device_memory_ownership::user_owned)
    {
        if(!handle->stream_order_alloc)
            RETURN_IF_HIP_ERROR((hipFree)(handle->device_memory));
// hipMallocAsync and hipFreeAsync are defined in hip version 5.2.0
// Support for default stream added in hip version 5.3.0
#if HIP_VERSION >= 50300000
        else
            RETURN_IF_HIP_ERROR((hipFreeAsync)(handle->device_memory, handle->stream));
#endif
    }

    // Clear the memory size and address, and set the memory to be rocBLAS-managed
    handle->device_memory_size  = 0;
    handle->device_memory       = nullptr;
    handle->device_memory_owner = rocblas_device_memory_ownership::rocblas_managed;

    return rocblas_status_success;
}

/*******************************************************************************
 * Set the device memory size
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_device_memory_size(rocblas_handle handle, size_t size)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // Temporarily change the thread's default device ID to the handle's device ID
    auto saved_device_id = handle->push_device_id();

    // Free any allocated memory unless owned by user, and set device memory to
    // the default of being rocBLAS-managed
    rocblas_status status = free_existing_device_memory(handle);
    if(status != rocblas_status_success)
        return status;

    // A zero specified size makes it rocBLAS-managed, and defers allocation
    if(!size)
        return rocblas_status_success;

    // Allocate size rounded up to MIN_CHUNK_SIZE
    size = roundup_device_memory_size(size);

    hipError_t hipStatus;
    if(!handle->stream_order_alloc)
        hipStatus = (hipMalloc)(&handle->device_memory, size);
// hipMallocAsync and hipFreeAsync are defined in hip version 5.2.0
// Support for default stream added in hip version 5.3.0
#if HIP_VERSION >= 50300000
    else
        hipStatus = (hipMallocAsync)(&handle->device_memory, size, handle->stream);
#endif

    if(hipStatus != hipSuccess)
    {
        // If allocation fails, nullify device memory address and return error
        // Leave the memory under rocBLAS management for future calls
        handle->device_memory = nullptr;
        return get_rocblas_status_for_hip_status(hipStatus);
    }
    else
    {
        // If allocation succeeds, set size, mark it under user-management, and return success
        handle->device_memory_size  = size;
        handle->device_memory_owner = rocblas_device_memory_ownership::user_managed;
        return rocblas_status_success;
    }
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * Set the device memory workspace
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_workspace(rocblas_handle handle, void* addr, size_t size)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // Temporarily change the thread's default device ID to the handle's device ID
    auto saved_device_id = handle->push_device_id();

    // Free any allocated memory unless owned by user, and set device memory to
    // the default of being rocBLAS-managed
    rocblas_status status = free_existing_device_memory(handle);
    if(status != rocblas_status_success)
        return status;

    // For nonzero size and non-nullptr address, mark device memory as user-owned,
    // with a specific size and address; otherwise leave it as rocBLAS-managed
    if(size && addr)
    {
        handle->device_memory_owner = rocblas_device_memory_ownership::user_owned;
        handle->device_memory_size  = size;
        handle->device_memory       = addr;
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
#if ROCBLAS_REALLOC_ON_DEMAND
    return handle
           && handle->device_memory_owner == rocblas_device_memory_ownership::rocblas_managed;
#else
    return false;
#endif
}

/*******************************************************************************
 * Returns whether device memory is user-managed
 ******************************************************************************/
extern "C" bool rocblas_is_user_managing_device_memory(rocblas_handle handle)
{
    return handle && handle->device_memory_owner == rocblas_device_memory_ownership::user_managed;
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
    using _device_malloc = decltype(rocblas_handle {} -> device_malloc(0));
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
    using _device_malloc = decltype(rocblas_handle {} -> device_malloc(0));
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
    using _device_malloc = decltype(rocblas_handle {} -> device_malloc(0));
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
    using _device_malloc = decltype(rocblas_handle {} -> device_malloc(0));
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
    logfile = read_env(environment_variable_name);
    if(!logfile)
        logfile = read_env("ROCBLAS_LOG_PATH");
    return logfile ? std::make_unique<rocblas_internal_ostream>(logfile)
                   : std::make_unique<rocblas_internal_ostream>(STDERR_FILENO);
}

/*******************************************************************************
 * Logging initialization
 ******************************************************************************/
void _rocblas_handle::init_logging()
{
    // set layer_mode from value of environment variable ROCBLAS_LAYER
    const char* str_layer_mode = read_env("ROCBLAS_LAYER");
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

/*******************************************************************************
 * Solution fitness query, for internal testing only
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_solution_fitness_query(rocblas_handle handle, double* fitness)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    handle->solution_fitness_query = fitness;
    if(fitness)
        *fitness = std::numeric_limits<double>::lowest();
    return rocblas_status_success;
}

/*******************************************************************************
 * Choose performance metric used to select solution
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_performance_metric(rocblas_handle             handle,
                                                         rocblas_performance_metric metric)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    handle->performance_metric = metric;
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_get_performance_metric(rocblas_handle              handle,
                                                         rocblas_performance_metric* metric)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(metric)
    {
        *metric = handle->performance_metric;
        return rocblas_status_success;
    }
    else
        return rocblas_status_invalid_pointer;
}

/*******************************************************************************
 * Numeric_check initialization
 ******************************************************************************/
void _rocblas_handle::init_check_numerics()
{
    // set check_numerics from value of environment variable ROCBLAS_CHECK_NUMERICS
    const char* str_check_numerics_mode = read_env("ROCBLAS_CHECK_NUMERICS");
    if(str_check_numerics_mode)
    {
        check_numerics
            = static_cast<rocblas_check_numerics_mode>(strtol(str_check_numerics_mode, 0, 0));
    }
}
