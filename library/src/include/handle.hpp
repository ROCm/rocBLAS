/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCBLAS_HANDLE_H_
#define _ROCBLAS_HANDLE_H_

#include "rocblas.h"
#include "rocblas_ostream.hpp"
#include "utility.hpp"
#include <array>
#include <cstddef>
#include <hip/hip_runtime.h>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <utility>

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
        _rocblas_saved_device_id(_rocblas_saved_device_id && other)
            : device_id(other.device_id)
            , old_device_id(other.old_device_id)
        {
            other.device_id = other.old_device_id;
        }

        _rocblas_saved_device_id(const _rocblas_saved_device_id&) = delete;
        _rocblas_saved_device_id& operator=(const _rocblas_saved_device_id&) = delete;
        _rocblas_saved_device_id& operator=(_rocblas_saved_device_id&&) = delete;
    };

    // Class for temporarily modifying a state, restoring it on destruction
    template <typename STATE>
    class [[nodiscard]] _pushed_state
    {
        STATE* statep;
        STATE  old_state;

    public:
        // Constructor
        _pushed_state(STATE & state, STATE new_state)
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
        _pushed_state(_pushed_state && other)
            : statep(other.statep)
            , old_state(std::move(other.old_state))
        {
            other.statep = nullptr;
        }

        _pushed_state(const _pushed_state&) = delete;
        _pushed_state& operator=(const _pushed_state&) = delete;
        _pushed_state& operator=(_pushed_state&&) = delete;
    };

public:
    _rocblas_handle();
    ~_rocblas_handle();

    // Set the HIP default device ID to the handle's device ID, and restore on exit
    auto push_device_id()
    {
        return _rocblas_saved_device_id(device);
    }

    // rocblas by default take the system default stream 0 users cannot create
    hipStream_t rocblas_stream = 0;

    // hipEvent_t pointers (for internal use only)
    hipEvent_t startEvent = nullptr;
    hipEvent_t stopEvent  = nullptr;

    // default pointer_mode is on host
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;

    // default logging_mode is no logging
    rocblas_layer_mode layer_mode = rocblas_layer_mode_none;

    // default atomics mode allows atomic operations
    rocblas_atomics_mode atomics_mode = rocblas_atomics_allowed;

    // logging streams
    std::unique_ptr<rocblas_ostream> log_trace_os;
    std::unique_ptr<rocblas_ostream> log_bench_os;
    std::unique_ptr<rocblas_ostream> log_profile_os;
    void                             init_logging();

    // C interfaces for manipulating device memory
    friend rocblas_status(::rocblas_start_device_memory_size_query)(_rocblas_handle*);
    friend rocblas_status(::rocblas_stop_device_memory_size_query)(_rocblas_handle*, size_t*);
    friend rocblas_status(::rocblas_get_device_memory_size)(_rocblas_handle*, size_t*);
    friend rocblas_status(::rocblas_set_device_memory_size)(_rocblas_handle*, size_t);
    friend bool(::rocblas_is_managing_device_memory)(_rocblas_handle*);

    // Returns whether the current kernel call is a device memory size query
    bool is_device_memory_size_query() const
    {
        return device_memory_size_query;
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

private:
    // device memory work buffer
    static constexpr size_t DEFAULT_DEVICE_MEMORY_SIZE = 4 * 1048576;

    // Variables holding state of device memory allocation
    void*  device_memory            = nullptr;
    size_t device_memory_size       = 0;
    size_t device_memory_in_use     = 0;
    bool   device_memory_size_query = false;
    size_t device_memory_query_size;

    // Device ID is created at handle creation time and remains in effect for the life of the handle.
    const int device;

    // Opaque smart allocator class to perform device memory allocations
    class [[nodiscard]] _device_malloc : public rocblas_device_malloc_base
    {
    public:
        // Order is important:
        rocblas_handle handle;
        size_t         prev_device_memory_in_use;
        size_t         size;
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
            size_t offsets[] = {(old = size, size += roundup_device_memory_size(sizes), old)...};

            // If allocation failed, return an array of nullptr's
            // If total size is 0, return an array of nullptr's, but leave it marked as successful
            success = size <= handle->device_memory_size - handle->device_memory_in_use;
            if(!success || !size)
                return decltype(pointers)(sizeof...(sizes));

            // We allocate the total amount needed, taking it from the available device memory.
            char* addr = static_cast<char*>(handle->device_memory) + handle->device_memory_in_use;
            handle->device_memory_in_use += size;

            // An array of pointers to all of the allocated arrays is formed.
            // If a size is 0, the corresponding pointer is nullptr
            size_t i = 0;
            return {!sizes ? i++, nullptr : addr + offsets[i++]...};
        }

    public:
        // Constructor
        template <typename... Ss>
        explicit _device_malloc(rocblas_handle handle, Ss... sizes)
            : handle(handle)
            , prev_device_memory_in_use(handle->device_memory_in_use)
            , size(0)
            , success(false)
            , pointers(allocate_pointers(size_t(sizes)...))
        {
        }

        // Constructor for allocating count pointers of a certain total size
        explicit _device_malloc(rocblas_handle handle, std::nullptr_t, size_t count, size_t total)
            : handle(handle)
            , prev_device_memory_in_use(handle->device_memory_in_use)
            , size(roundup_device_memory_size(total))
            , success(size <= handle->device_memory_size - handle->device_memory_in_use)
            , pointers(count,
                       success ? static_cast<char*>(handle->device_memory)
                                     + handle->device_memory_in_use
                               : nullptr)
        {
            if(success)
                handle->device_memory_in_use += size;
        }

        // Move constructor
        // Warning: This should only be used to move temporary expressions,
        // such as the return values of functions and initialization with
        // rvalues. If std::move() is used to move a _device_malloc object
        // from a variable, then there must not be any alive allocations made
        // between the initialization of the variable and the object that it
        // moves to, or the LIFO ordering will be violated and flagged.
        _device_malloc(_device_malloc && other) noexcept
            : handle(other.handle)
            , prev_device_memory_in_use(other.prev_device_memory_in_use)
            , size(other.size)
            , success(other.success)
            , pointers(std::move(other.pointers))
        {
            other.success = false;
        }

        // Move assignment is allowed as long as the object being assigned to
        // is 0-sized or an unsuccessful previous allocation.
        _device_malloc& operator=(_device_malloc&& other)& noexcept
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

        // In the following functions, the trailing & prevents the functions from
        // applying to rvalue temporaries, to catch common mistakes such as:
        // void *p = (void*) handle->device_malloc(), which is a dangling pointer.

        // Conversion to bool to tell if the allocation succeeded
        explicit operator bool()&
        {
            return success;
        }

        // Return the ith pointer
        void*& operator[](size_t i)&
        {
            return pointers.at(i);
        }

        // Conversion to any pointer type (if pointers.size() == 1)
        template <typename T>
        explicit operator T*()&
        {
            // Index 1 - pointers.size() is used to make at() throw if size() != 1
            // but to otherwise return the first element.
            return static_cast<T*>(pointers.at(1 - pointers.size()));
        }
    };

    // For HPA kernel calls, all available device memory is allocated and passed to Tensile
    class [[nodiscard]] _gsu_malloc final : _device_malloc
    {
    public:
        explicit _gsu_malloc(rocblas_handle handle)
            : _device_malloc(handle, handle->device_memory_size - handle->device_memory_in_use)
        {
            handle->gsu_workspace_size = success ? size : 0;
            handle->gsu_workspace      = static_cast<void*>(*this);
        }

        ~_gsu_malloc()
        {
            if(success)
            {
                handle->gsu_workspace_size = 0;
                handle->gsu_workspace      = nullptr;
            }
        }

        // Move constructor allows initialization by rvalues and returns from functions
        _gsu_malloc(_gsu_malloc &&) = default;
    };

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

    // gsu_malloc() returns a proxy object which manages GSU memory for the handle.
    // The returned object needs to be kept alive for as long as the GSU memory is needed.
    auto gsu_malloc()
    {
        return _gsu_malloc(this);
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

#endif
