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

#pragma once

#include "macros.hpp"
#include "rocblas.h"
#include "rocblas_ostream.hpp"
#include "utility.hpp"
#include <array>
#include <cstddef>
#include <hip/hip_runtime.h>
#include <memory>
#include <tuple>
#include <type_traits>
#ifdef WIN32
#include <stdio.h>
#define STDOUT_FILENO _fileno(stdout)
#define STDERR_FILENO _fileno(stderr)
#else
#include <unistd.h>
#endif
#include <utility>

// forcing early cleanup
extern "C" ROCBLAS_EXPORT void rocblas_shutdown();

// Whether rocBLAS can reallocate device memory on demand, at the cost of only
// allowing one allocation at a time, and at the cost of potential synchronization.
// If this is 0, then stack-like allocation is allowed, but reallocation on demand
// does not occur.
#define ROCBLAS_REALLOC_ON_DEMAND 1

// Round up size to the nearest MIN_CHUNK_SIZE
constexpr size_t roundup_device_memory_size(size_t size)
{
    size_t MIN_CHUNK_SIZE = 64;
    return ((size + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
}

// Empty base class for device memory allocation
struct rocblas_device_malloc_base
{
};

// enum representing state of rocBLAS device memory ownership
enum class rocblas_device_memory_ownership
{
    rocblas_managed,
    user_managed,
    user_owned,
};

// helper function in handle.cpp
static rocblas_status free_existing_device_memory(rocblas_handle);

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
    template <class...>
    struct conjunction : std::true_type
    {
    };
    template <class T, class... Ts>
    struct conjunction<T, Ts...> : std::integral_constant<bool, T{} && conjunction<Ts...>{}>
    {
    };

    // Class for saving and restoring default device ID
    // clang-format off
    class [[nodiscard]] _rocblas_saved_device_id
    {
        int device_id;
        int old_device_id;

    public:
        // Constructor
        explicit _rocblas_saved_device_id(int device_id)
            : device_id(device_id)
            , old_device_id(-1)
        {
            hipGetDevice(&old_device_id);
            if(device_id != old_device_id)
                hipSetDevice(device_id);
        }

        // Old device ID is restored on destruction
        ~_rocblas_saved_device_id()
        {
            if(device_id != old_device_id)
                hipSetDevice(old_device_id);
        }

        // Move constructor
        _rocblas_saved_device_id(_rocblas_saved_device_id&& other)
            : device_id(other.device_id)
            , old_device_id(other.old_device_id)
        {
            other.device_id = other.old_device_id;
        }

        _rocblas_saved_device_id(const _rocblas_saved_device_id&) = delete;
        _rocblas_saved_device_id& operator=(const _rocblas_saved_device_id&) = delete;
        _rocblas_saved_device_id& operator=(_rocblas_saved_device_id&&) = delete;
    };
    // clang-format on

    // Class for temporarily modifying a state, restoring it on destruction
    // clang-format off
    template <typename STATE>
    class [[nodiscard]] _pushed_state
    {
        STATE* statep;
        STATE  old_state;

    public:
        // Constructor
        _pushed_state(STATE& state, STATE new_state)
            : statep(&state)
            , old_state(std::move(state))
        {
            state = std::move(new_state);
        }

        // Temporary object implicitly converts to old state
        operator const STATE&() const&
        {
            return old_state;
        }

        // Old state is restored on destruction
        ~_pushed_state()
        {
            if(statep)
                *statep = std::move(old_state);
        }

        // Move constructor
        _pushed_state(_pushed_state&& other)
            : statep(other.statep)
            , old_state(std::move(other.old_state))
        {
            other.statep = nullptr;
        }

        _pushed_state(const _pushed_state&) = delete;
        _pushed_state& operator=(const _pushed_state&) = delete;
        _pushed_state& operator=(_pushed_state&&) = delete;
    };
    // clang-format on

public:
    _rocblas_handle();
    ~_rocblas_handle();

    _rocblas_handle(const _rocblas_handle&) = delete;
    _rocblas_handle& operator=(const _rocblas_handle&) = delete;

    // Set the HIP default device ID to the handle's device ID, and restore on exit
    auto push_device_id()
    {
        return _rocblas_saved_device_id(device);
    }

    int getDevice()
    {
        return device;
    }

    int getArch()
    {
        return arch;
    }

    int getArchMajor()
    {
        return archMajor;
    }

    // hipEvent_t pointers (for internal use only)
    hipEvent_t startEvent = nullptr;
    hipEvent_t stopEvent  = nullptr;

    // default pointer_mode is on host
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;

    // default logging_mode is no logging
    rocblas_layer_mode layer_mode = rocblas_layer_mode_none;

    // default atomics mode allows atomic operations
    rocblas_atomics_mode atomics_mode = rocblas_atomics_allowed;

    // Selects the benchmark library to be used for solution selection
    rocblas_performance_metric performance_metric = rocblas_default_performance_metric;

    // default check_numerics_mode is no numeric_check
    rocblas_check_numerics_mode check_numerics = rocblas_check_numerics_mode_no_check;

    // used by hipBLAS to set int8 datatype to int8_t or rocblas_int8x4
    rocblas_int8_type_for_hipblas rocblas_int8_type = rocblas_int8_type_for_hipblas_default;

    // logging streams
    std::unique_ptr<rocblas_internal_ostream> log_trace_os;
    std::unique_ptr<rocblas_internal_ostream> log_bench_os;
    std::unique_ptr<rocblas_internal_ostream> log_profile_os;
    void                                      init_logging();
    void                                      init_check_numerics();

    // C interfaces for manipulating device memory
    friend rocblas_status(::rocblas_start_device_memory_size_query)(_rocblas_handle*);
    friend rocblas_status(::rocblas_stop_device_memory_size_query)(_rocblas_handle*, size_t*);
    friend rocblas_status(::rocblas_get_device_memory_size)(_rocblas_handle*, size_t*);
    friend rocblas_status(::rocblas_set_device_memory_size)(_rocblas_handle*, size_t);
    friend rocblas_status(::free_existing_device_memory)(rocblas_handle);
    friend rocblas_status(::rocblas_set_workspace)(_rocblas_handle*, void*, size_t);
    friend bool(::rocblas_is_managing_device_memory)(_rocblas_handle*);
    friend bool(::rocblas_is_user_managing_device_memory)(_rocblas_handle*);
    friend rocblas_status(::rocblas_set_stream)(_rocblas_handle*, hipStream_t);

    // C interfaces that interact with the solution selection process
    friend rocblas_status(::rocblas_set_solution_fitness_query)(_rocblas_handle*, double*);
    friend rocblas_status(::rocblas_set_performance_metric)(_rocblas_handle*,
                                                            rocblas_performance_metric);
    friend rocblas_status(::rocblas_get_performance_metric)(_rocblas_handle*,
                                                            rocblas_performance_metric*);

    // Returns whether the current kernel call is a device memory size query
    bool is_device_memory_size_query() const
    {
        return device_memory_size_query;
    }

    size_t get_available_workspace()
    {
        return (device_memory_size - device_memory_in_use);
    }

    // Get the solution fitness query
    auto* get_solution_fitness_query() const
    {
        return solution_fitness_query;
    }

    // Sets the optimal size(s) of device memory for a kernel call
    // Maximum size is accumulated in device_memory_query_size
    // Returns rocblas_status_size_increased or rocblas_status_size_unchanged
    template <typename... Ss,
              std::enable_if_t<sizeof...(Ss) && conjunction<std::is_convertible<Ss, size_t>...>{},
                               int> = 0>
    rocblas_status set_optimal_device_memory_size(Ss... sizes)
    {
        if(!device_memory_size_query)
            return rocblas_status_size_query_mismatch;

#if __cplusplus >= 201703L
        // Compute the total size, rounding up each size to multiples of MIN_CHUNK_SIZE
        size_t total = (roundup_device_memory_size(sizes) + ...);
#else
        size_t total = 0;
        auto   dummy = {total += roundup_device_memory_size(sizes)...};
#endif

        return total > device_memory_query_size ? device_memory_query_size = total,
                                                  rocblas_status_size_increased
                                                : rocblas_status_size_unchanged;
    }

    // Temporarily change pointer mode, returning object which restores old mode when destroyed
    auto push_pointer_mode(rocblas_pointer_mode mode)
    {
        return _pushed_state<rocblas_pointer_mode>(pointer_mode, mode);
    }

    // Whether to use any_order scheduling in Tensile calls
    bool any_order = false;

    // Temporarily change any_order flag
    auto push_any_order(bool new_any_order)
    {
        return _pushed_state<bool>(any_order, new_any_order);
    }

    // Return the current stream
    hipStream_t get_stream() const
    {
        return stream;
    }

    bool is_stream_in_capture_mode()
    {
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        bool                   status = hipStreamIsCapturing(stream, &capture_status) == hipSuccess;
        if(!status)
            rocblas_cerr << "Stream capture check failed" << std::endl;
        if(capture_status == hipStreamCaptureStatusActive)
            return true;
        else
            return false;
    }

    void* host_malloc(size_t size)
    {
        void* ptr = malloc(size);
        if(ptr)
        {
            host_mem_pointers.push_back(ptr);
            return ptr;
        }
        else
        {
            rocblas_cerr << " host_malloc FAILED " << std::endl;
            rocblas_abort();
        }
    }

    bool skip_alpha_beta_memcpy()
    {
        return alpha_beta_memcpy_complete;
    }

    void alpha_beta_memcpy_completed()
    {
        alpha_beta_memcpy_complete = true;
    }

private:
    // device memory work buffer
    static constexpr size_t DEFAULT_DEVICE_MEMORY_SIZE = 32 * 1024 * 1024;

    // Variables holding state of device memory allocation
    void*                           device_memory              = nullptr;
    size_t                          device_memory_size         = 0;
    size_t                          device_memory_in_use       = 0;
    bool                            device_memory_size_query   = false;
    bool                            alpha_beta_memcpy_complete = false;
    rocblas_device_memory_ownership device_memory_owner;
    size_t                          device_memory_query_size;
    std::vector<void*>              host_mem_pointers;

    bool stream_order_alloc = false;

    // Solution fitness query (used for internal testing)
    double* solution_fitness_query = nullptr;

    // rocblas by default take the system default stream 0 users cannot create
    hipStream_t stream = 0;

#if ROCBLAS_REALLOC_ON_DEMAND
    // Helper for device memory allocator
    bool device_allocator(size_t size);
#endif

    // Device ID is created at handle creation time and remains in effect for the life of the handle.
    const int device;

    // Arch ID is created at handle creation time and remains in effect for the life of the handle.
    const int arch;
    int       archMajor;

    // Opaque smart allocator class to perform device memory allocations
    // clang-format off
    class [[nodiscard]] _device_malloc : public rocblas_device_malloc_base
    {
    protected:
        // Order is important (pointers member declared last):
        rocblas_handle handle;
        size_t         prev_device_memory_in_use;
        size_t         size;
        void*          dev_mem = nullptr;
        hipStream_t    stream_in_use;
        bool           success;

    private:
        std::vector<void*> pointers; // Important: must come last

        // Allocate one or more pointers to buffers of different sizes
        template <typename... Ss>
        decltype(pointers) allocate_pointers(Ss... sizes)
        {
            // This creates a list of partial sums which are the offsets of each of the allocated
            // arrays. The sizes are rounded up to the next multiple of MIN_CHUNK_SIZE.
            // size contains the total of all sizes at the end of the calculation of offsets.
            size = 0;
            size_t old;
            const size_t offsets[] = {(old = size, size += roundup_device_memory_size(sizes), old)...};
            char* addr = nullptr;

            if(handle->stream_order_alloc &&
                handle->device_memory_owner == rocblas_device_memory_ownership::rocblas_managed)
            {
// hipMallocAsync and hipFreeAsync are defined in hip version 5.2.0
// Support for default stream added in hip version 5.3.0
#if HIP_VERSION >= 50300000
                if(!size)
                    return decltype(pointers)(sizeof...(sizes));

                hipError_t hipStatus = hipMallocAsync(&dev_mem, size, stream_in_use);
                if(hipStatus != hipSuccess)
                {
                    success = false;
                    rocblas_cerr << " rocBLAS internal error: hipMallocAsync() failed to allocate memory of size : " << size << std::endl;
                    return decltype(pointers)(sizeof...(sizes));
                }
                addr = static_cast<char*>(dev_mem);
#endif
            }
            else
            {
    #if ROCBLAS_REALLOC_ON_DEMAND
                success = handle->device_allocator(size);
    #else
                success = size <= handle->device_memory_size - handle->device_memory_in_use;
    #endif
                // If allocation failed, return an array of nullptr's
                // If total size is 0, return an array of nullptr's, but leave it marked as successful
                if(!success || !size)
                    return decltype(pointers)(sizeof...(sizes));

                // We allocate the total amount needed, taking it from the available device memory.
                addr = static_cast<char*>(handle->device_memory) + handle->device_memory_in_use;
                handle->device_memory_in_use += size;
            }
            // An array of pointers to all of the allocated arrays is formed.
            // If a size is 0, the corresponding pointer is nullptr
            size_t i = 0;
            // cppcheck-suppress arrayIndexOutOfBounds
            return {!sizes ? i++, nullptr : addr + offsets[i++]...};
        }

    public:
        // Constructor
        template <typename... Ss>
        explicit _device_malloc(rocblas_handle handle, Ss... sizes)
            : handle(handle)
            , prev_device_memory_in_use(handle->device_memory_in_use)
            , size(0)
            , stream_in_use(handle->stream)
            , success(true)
            , pointers(allocate_pointers(size_t(sizes)...))
        {

        }

        // Constructor for allocating count pointers of a certain total size
        explicit _device_malloc(rocblas_handle handle, std::nullptr_t, size_t count, size_t total)
            : handle(handle)
            , prev_device_memory_in_use(handle->device_memory_in_use)
            , size(roundup_device_memory_size(total))
            , stream_in_use(handle->stream)
            , success(true)
        {
            if(handle->stream_order_alloc &&
                handle->device_memory_owner == rocblas_device_memory_ownership::rocblas_managed)
            {
// hipMallocAsync and hipFreeAsync are defined in hip version 5.2.0
// Support for default stream added in hip version 5.3.0
#if HIP_VERSION >= 50300000
                bool status = hipMallocAsync(&dev_mem, size, stream_in_use) == hipSuccess ;

                for(auto i= 0 ; i < count ; i++)
                    pointers.push_back(status ? dev_mem : nullptr);
#endif
            }
            else
            {
#if ROCBLAS_REALLOC_ON_DEMAND
            success = handle->device_allocator(size);
#else
            success = size <= handle->device_memory_size - handle->device_memory_in_use;
#endif
            for(auto i= 0 ; i < count ; i++)
            {    pointers.push_back(success ? static_cast<char*>(handle->device_memory)
                                         + handle->device_memory_in_use : nullptr);
            }

            if(success)
                handle->device_memory_in_use += size;
            }
        }

        // Move constructor
        // Warning: This should only be used to move temporary expressions,
        // such as the return values of functions and initialization with
        // rvalues. If std::move() is used to move a _device_malloc object
        // from a variable, then there must not be any alive allocations made
        // between the initialization of the variable and the object that it
        // moves to, or the LIFO ordering will be violated and flagged.
        _device_malloc(_device_malloc&& other) noexcept
            : handle(other.handle)
            , prev_device_memory_in_use(other.prev_device_memory_in_use)
            , size(other.size)
            , dev_mem(other.dev_mem)
            , stream_in_use(other.stream_in_use)
            , success(other.success)
            , pointers(std::move(other.pointers))
        {
            other.success = false;
        }

        // Move assignment is allowed as long as the object being assigned to
        // is 0-sized or an unsuccessful previous allocation.
        _device_malloc& operator=(_device_malloc&& other) & noexcept
        {
            this->~_device_malloc();
            return *new(this) _device_malloc(std::move(other));
        }

        // Copying and copy-assignment are deleted
        _device_malloc(const _device_malloc&) = delete;
        _device_malloc& operator=(const _device_malloc&) = delete;

        // The destructor marks the device memory as no longer in use
        ~_device_malloc()
        {
            // If success == false or size == 0, the destructor is a no-op
            if(success && size)
            {
                if(handle->stream_order_alloc &&
                    handle->device_memory_owner == rocblas_device_memory_ownership::rocblas_managed)
                {
// hipMallocAsync and hipFreeAsync are defined in hip version 5.2.0
// Support for default stream added in hip version 5.3.0
#if HIP_VERSION >= 50300000
                        if(dev_mem)
                        {

                            bool status = hipFreeAsync(dev_mem, stream_in_use) == hipSuccess ;
                            if(!status)
                            {
                                rocblas_cerr << " rocBLAS internal error: hipFreeAsync() Failed, "
                                "device memory could not be released to default memory pool" << std::endl;
                                rocblas_abort();
                            }
                            dev_mem = nullptr;
                        }
#endif
                }
                else
                {
                    // Subtract size from the handle's device_memory_in_use, making sure
                    // it matches the device_memory_in_use when this object was created.
                    if((handle->device_memory_in_use -= size) != prev_device_memory_in_use)
                    {
                        rocblas_cerr
                            << "rocBLAS internal error: device_malloc() RAII object not "
                            "destroyed in LIFO order.\n"
                            "Objects returned by device_malloc() must be 0-sized, "
                            "unsuccessfully allocated,\n"
                            "or destroyed in the reverse order that they are created.\n"
                            "device_malloc() objects cannot be assigned to unless they are 0-sized\n"
                            "or they were unsuccessfully allocated previously."
                            << std::endl;
                        rocblas_abort();
                    }
                }
            }
        }

        // In the following functions, the trailing & prevents the functions from
        // applying to rvalue temporaries, to catch common mistakes such as:
        // void *p = (void*) handle->device_malloc(), which is a dangling pointer.

        // Conversion to bool to tell if the allocation succeeded
        explicit operator bool() &
        {
            return success;
        }

        // Return the ith pointer
        void*& operator[](size_t i) &
        {
            return pointers.at(i);
        }

        // Conversion to any pointer type (if pointers.size() == 1)
        template <typename T>
        explicit operator T*() &
        {
            // Index 1 - pointers.size() is used to make at() throw if size() != 1
            // but to otherwise return the first element.
            return static_cast<T*>(pointers.at(1 - pointers.size()));
        }
    };
    // clang-format on

    // Allocate workspace for GSU based on the needs.
    // clang-format off
    class [[nodiscard]] _gsu_malloc_by_size final : _device_malloc
    {
    public:
        explicit _gsu_malloc_by_size(rocblas_handle handle, size_t requested_Workspace_Size)
        : _device_malloc(handle, requested_Workspace_Size)
        {
            handle->gsu_workspace_size = success ? size : 0;
            handle->gsu_workspace = static_cast<void*>(*this);
        }

        ~_gsu_malloc_by_size()
        {
            if(success)
            {
                handle->gsu_workspace_size = 0;
                handle->gsu_workspace      = nullptr;
            }
        }

        // Move constructor allows initialization by rvalues and returns from functions
        _gsu_malloc_by_size(_gsu_malloc_by_size&&) = default;
    };
    // clang-format on

public:
    // Allocate one or more sizes
    template <typename... Ss,
              std::enable_if_t<sizeof...(Ss) && conjunction<std::is_convertible<Ss, size_t>...>{},
                               int> = 0>
    auto device_malloc(Ss... sizes)
    {
        return _device_malloc(this, size_t(sizes)...);
    }

    // Allocate count pointers, reserving "size" total bytes
    auto device_malloc_count(size_t count, size_t size)
    {
        return _device_malloc(this, nullptr, count, size);
    }

    // Variables holding state of GSU device memory allocation
    size_t gsu_workspace_size = 0;
    void*  gsu_workspace      = nullptr;

    auto gsu_malloc_by_size(size_t requested_Workspace_Size)
    {
        return _gsu_malloc_by_size(this, requested_Workspace_Size);
    };
};

// For functions which don't use temporary device memory, and won't be likely
// to use them in the future, the RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle)
// macro can be used to return from a rocblas function with a requested size of 0.
#define RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(h) \
    do                                               \
    {                                                \
        if((h)->is_device_memory_size_query())       \
            return rocblas_status_size_unchanged;    \
    } while(0)

// Warn about potentially unsafe and synchronizing uses of hipMalloc and hipFree
#define hipMalloc(ptr, size)                                                                     \
    _Pragma(                                                                                     \
        "GCC warning \"Direct use of hipMalloc in rocBLAS is deprecated; see CONTRIBUTING.md\"") \
        hipMalloc(ptr, size)
#define hipFree(ptr)                                                                               \
    _Pragma("GCC warning \"Direct use of hipFree in rocBLAS is deprecated; see CONTRIBUTING.md\"") \
        hipFree(ptr)
