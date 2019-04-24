/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HANDLE_H
#define HANDLE_H

#include <fstream>
#include <iostream>
#include <utility>
#include <tuple>
#include <initializer_list>
#include <array>
#include "rocblas.h"
#include "definitions.h"
#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle() and the returned handle mus
 * It should be destroyed at the end using rocblas_destroy_handle().
 * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
******************************************************************************/
struct _rocblas_handle
{
    _rocblas_handle();
    ~_rocblas_handle();

    /*******************************************************************************
     * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
     ******************************************************************************/

    /*******************************************************************************
     * set stream:
        This API assumes user has already created a valid stream
        Associate the following rocblas API call with this user provided stream
     ******************************************************************************/
    rocblas_status set_stream(hipStream_t user_stream)
    {
        // TODO: check the user_stream valid or not
        rocblas_stream = user_stream;
        return rocblas_status_success;
    }

    /*******************************************************************************
     * get stream
     ******************************************************************************/
    rocblas_status get_stream(hipStream_t* stream) const
    {
        *stream = rocblas_stream;
        return rocblas_status_success;
    }

    rocblas_int device;
    hipDeviceProp_t device_properties;

    // rocblas by default take the system default stream 0 users cannot create
    hipStream_t rocblas_stream = 0;

    // default pointer_mode is on host
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;

    // default logging_mode is no logging
    static rocblas_layer_mode layer_mode;

    // logging streams
    static std::ofstream log_trace_ofs;
    static std::ostream* log_trace_os;
    static std::ofstream log_bench_ofs;
    static std::ostream* log_bench_os;
    static std::ofstream log_profile_ofs;
    static std::ostream* log_profile_os;

    // static data for startup initialization
    static struct init
    {
        init();
    } handle_init;

    static int device_arch_id()
    {
        static int id = get_device_arch_id();
        return id;
    }

    // C interfaces for manipulating device memory
    friend _rocblas_handle*(::rocblas_query_device_memory_size)(_rocblas_handle*, size_t*);
    friend size_t(::rocblas_get_device_memory_size)(_rocblas_handle*);
    friend rocblas_status(::rocblas_set_device_memory_size)(_rocblas_handle*, size_t);

    // C++ interfaces to the above (i.e. handle->method() instead of method(handle))
    auto set_device_memory_size(size_t size) { return rocblas_set_device_memory_size(this, size); }
    auto get_device_memory_size() const { return device_memory_size; }
    auto query_device_memory_size(size_t* size)
    {
        return rocblas_query_device_memory_size(this, size);
    }

    // Returns whether the current kernel call is a device memory size query
    bool is_device_memory_size_query() const { return device_memory_size_query != nullptr; }

    // Sets the optimum size of device memory for a kernel call
    rocblas_status set_queried_device_memory_size(size_t size)
    {
        if(!is_device_memory_size_query())
            return rocblas_status_internal_error;
        *device_memory_size_query = size;
        device_memory_size_query  = nullptr;
        return rocblas_status_success;
    }

    // Allocate one or more sizes
    template <typename... Ss>
    auto device_memory_alloc(Ss... sizes)
    {
        static constexpr size_t MIN_CHUNK_SIZE = 64;
        static_assert(sizeof...(sizes) > 0, "There must be at least one argument");
        static_assert(MIN_CHUNK_SIZE > 0 && !(MIN_CHUNK_SIZE & (MIN_CHUNK_SIZE - 1)),
                      "MIN_CHUNK_SIZE must be a power of two");
        size_t total     = 0, oldtotal;
        size_t offsets[] = {(oldtotal = total,
                             total += ((static_cast<size_t>(sizes) - 1) | (MIN_CHUNK_SIZE - 1)) + 1,
                             oldtotal)...};
        void* ptr = device_memory_allocator(total);
        total     = 0;
        auto pointers[] = {((void)sizes, ptr ? (void*)((char*)ptr + offsets[total++]) : nullptr)...};
        total = 0;
        return _device_memory_alloc(this, ((void)sizes, pointers[total++])...);
    }

#if 0
    // Allocate one or more sizes
    auto device_memory_alloc(std::initializer_list<void*&> ptr, std::initializer_list<size_t> size)
    {
        static constexpr size_t MIN_CHUNK_SIZE = 64;
        static_assert(MIN_CHUNK_SIZE > 0 && !(MIN_CHUNK_SIZE & (MIN_CHUNK_SIZE - 1)),
                      "MIN_CHUNK_SIZE must be a power of two");
        static_assert(ptr.size() > 0, "Lists must contain at least one element");
        static_assert(ptr.size() == size.size(), "Sizes of both lists must be the same");

        return _device_memory_alloc<sizeof...(sizes)>(
            this, {(((static_cast<size_t>(sizes) - 1) | (MIN_CHUNK_SIZE - 1)) + 1)...});
    }
#endif

    private:
    // device memory work buffer
    static constexpr size_t DEFAULT_DEVICE_MEMORY_SIZE = 1048576;
    size_t device_memory_size                          = 0;
    bool device_memory_rocblas_managed                 = true;
    bool device_memory_inuse                           = false;
    void* device_memory                                = nullptr;
    size_t* device_memory_size_query                   = nullptr;

    // Helper for device memory allocator
    void* device_memory_allocator(size_t);

    // Opaque smart allocator class to perform device memory allocations
    class _device_memory_alloc
    {
        friend class _rocblas_handle;
        rocblas_handle handle;
        bool nonnull;

        _device_memory_alloc(rocblas_handle handle, bool nonnull) : handle(handle), nonnull(nonnull)
        {
        }

        public:
        ~_device_memory_alloc() { handle->device_memory_inuse = false; }
        explicit operator bool() const { return nonnull; }
    };

    static int get_device_arch_id()
    {
        int deviceId;
        hipGetDevice(&deviceId);
        hipDeviceProp_t deviceProperties;
        hipGetDeviceProperties(&deviceProperties, deviceId);
        return deviceProperties.gcnArch;
    }
};

// For functions which don't use temporary device memory, and won't be likely
// to use them in the future, the RETURN_ZERO_DEVICE_MEMORY_IF_QUERIED(handle)
// macro can be used to return from a rocblas function with a requested size of 0.
//
// rocblas_status func(rocblas_handle handle, ...)
// {
//     RETURN_ZERO_DEVICE_MEMORY_IF_QUERIED(handle);
//     ...
// }
#define RETURN_ZERO_DEVICE_MEMORY_IF_QUERIED(h)               \
    do                                                        \
    {                                                         \
        rocblas_handle handle = (h);                          \
        if(!handle)                                           \
            return rocblas_status_invalid_handle;             \
        if(handle->is_device_memory_size_query())             \
            return handle->set_queried_device_memory_size(0); \
    } while(0)

namespace rocblas {
void reinit_logs(); // Reinitialize static data (for testing only)
}

#endif
