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
    private:
    // Emulate C++17 std::conjunction
    template <typename...>
    struct conjunction : std::true_type
    {
    };
    template <typename B>
    struct conjunction<B> : B
    {
    };
    template <typename B, typename... Bn>
    struct conjunction<B, Bn...> : std::conditional<B{}, conjunction<Bn...>, B>::type
    {
    };

    public:
    _rocblas_handle();
    ~_rocblas_handle();

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

    // C++ interfaces to the above (i.e. handle->method(...) instead of method(handle, ...))
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
    template <typename... Ss,
              typename = typename std::enable_if<
                  sizeof...(Ss) != 0 && conjunction<std::is_convertible<Ss, size_t>...>{}>::type>
    auto device_memory_alloc(Ss... sizes)
    {
        return _device_memory_alloc<sizeof...(Ss)>(this, static_cast<size_t>(sizes)...);
    }

    private:
    // device memory work buffer
    static constexpr size_t DEFAULT_DEVICE_MEMORY_SIZE = 1048576;
    static constexpr size_t MIN_CHUNK_SIZE             = 64;

    size_t device_memory_size          = 0;
    bool device_memory_rocblas_managed = true;
    bool device_memory_inuse           = false;
    void* device_memory                = nullptr;
    size_t* device_memory_size_query   = nullptr;

    // Helper for device memory allocator
    void* device_memory_allocator(size_t);

    // Opaque smart allocator class to perform device memory allocations
    template <size_t N>
    class _device_memory_alloc
    {
        friend struct _rocblas_handle;
        rocblas_handle handle;
        std::array<void*, N> pointers;

        // Allocate one or more pointers to buffers of different sizes
        template <typename... Ss>
        auto allocate_pointers(Ss... sizes) -> std::array<void*, sizeof...(Ss)>
        {
            static_assert(MIN_CHUNK_SIZE > 0 && !(MIN_CHUNK_SIZE & (MIN_CHUNK_SIZE - 1)),
                          "MIN_CHUNK_SIZE must be a power of two");

            // This creates a sequential list of partial sums which are the offsets of each of the
            // allocated arrays. The sizes are rounded up to the next multiple of MIN_CHUNK_SIZE.
            // The entire expression to the left of ... is evaluated once for each value in sizes.
            // The comma expression in ( ) returns its last value. total contains the total of all
            // sizes at the end of the calculation of offsets.
            size_t oldtotal;
            size_t total     = 0;
            size_t offsets[] = {
                (oldtotal = total, total += ((sizes - 1) | (MIN_CHUNK_SIZE - 1)) + 1, oldtotal)...};

            // We allocate the total amount needed. This is a constant-time operation if the space
            // is already available.
            void* ptr = handle->device_memory_allocator(total);

            // An array of pointers to all of the allocated arrays is formed.
            // sizes is only used to expand the parameter pack.
            // Note: Compilers are able to scalarize these arrays, and do everything in registers.
            total = 0;
            return {{((void)sizes, (void*)((char*)ptr + offsets[total++]))...}};
        }

        // Constructor
        template <typename... Ss, typename = typename std::enable_if<sizeof...(Ss) == N>::type>
        _device_memory_alloc(rocblas_handle handle, Ss... sizes)
            : handle(handle), pointers(allocate_pointers(sizes...))
        {
        }

        // Create a tuple of references to the pointers, to be assigned to std::tie(...)
        template <size_t... I>
        auto make_pointer_tuple(std::index_sequence<I...>)
        {
            return std::tie(pointers[I]...);
        }

        public:
        // The destructor marks the device memory as no longer in use
        ~_device_memory_alloc() { handle->device_memory_inuse = false; }

        // Conversion to any pointer type, but only if N==1
        // (is_void is only used to make the enable_if a dependent expression)
        template <typename T, typename = typename std::enable_if<(std::is_void<T>{}, N==1)>::type>
        explicit operator T*() const
        {
            return static_cast<T*>(pointers[0]);
        }

        // Conversion to bool to tell if the allocation succeeded
        explicit operator bool() const { return pointers[0] != nullptr; }

        // Conversion to std::tuple<void*&...> to be assigned to std::tie()
        operator auto() { return make_pointer_tuple(std::make_index_sequence<N>{}); }
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
