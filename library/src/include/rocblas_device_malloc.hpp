/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

// RAII helper class for device memory allocation outside of rocBLAS
//
// rocblas_device_malloc mem(handle, size);
//
// if(!mem) return rocblas_status_memory_error;
//
// void* ptr = static_cast<void*>(mem);
//
// rocblas_trsm_template_mem(..., mem, ...)
//
// This header should be included in other projects to use the C++
// device memory allocation API. It is unlikely to change very often,
// so it is safe to copy to other projects.

#include "rocblas.h"
#include <new>
#include <type_traits>

class [[nodiscard]] rocblas_device_malloc
{
    rocblas_handle              handle;
    rocblas_device_malloc_base* dm_ptr;

    // Emulate C++17 std::conjunction
    template <class...>
    struct conjunction : std::true_type
    {
    };
    template <class T, class... Ts>
    struct conjunction<T, Ts...> : std::integral_constant<bool, T{} && conjunction<Ts...>{}>
    {
    };

public:
    // Allocate memory in a RAII class
    template <typename... Ss,
              std::enable_if_t<sizeof...(Ss) && conjunction<std::is_convertible<Ss, size_t>...>{},
                               int> = 0>
    rocblas_device_malloc(rocblas_handle handle, Ss && ... sizes)
        : handle(handle)
        , dm_ptr(nullptr)
    {
        if(rocblas_device_malloc_alloc(handle, &dm_ptr, sizeof...(sizes), size_t(sizes)...)
           != rocblas_status_success)
            throw std::bad_alloc();
    }

    // Move constructor
    rocblas_device_malloc(rocblas_device_malloc && other) noexcept
        : handle(other.handle)
        , dm_ptr(other.dm_ptr)
    {
        other.dm_ptr = nullptr;
    }

    // Access a particular element
    void* operator[](size_t index)&
    {
        void* res = nullptr;
        if(dm_ptr
           && rocblas_device_malloc_get(handle, dm_ptr, index, &res) != rocblas_status_success)
            res = nullptr;
        return res;
    }

    // Conversion to a pointer type, to get the address of the device memory
    // It is lvalue-qualified to avoid the common mistake of writing:
    // void* ptr = (void*) rocblas_device_malloc(handle, size);
    // ... which is incorrect, since the RAII temporary expression will be
    // destroyed, and the pointer will be left dangling.
    template <typename T>
    explicit operator T*()&
    {
        return static_cast<T*>((*this)[0]);
    }

    // Conversion to bool indicates whether allocation succeeded
    // It is lvalue-qualified so that it cannot bind to temporaries
    explicit operator bool()&
    {
        return rocblas_device_malloc_success(handle, dm_ptr);
    }

    // Conversion to rocblas_device_malloc_base reference, to pass to rocBLAS
    // It is lvalue-qualified so that it cannot bind to temporaries
    operator rocblas_device_malloc_base&()&
    {
        return *dm_ptr;
    }

    // Destructor calls the C API to mark the data as freed
    ~rocblas_device_malloc()
    {
        if(dm_ptr)
            rocblas_device_free(handle, dm_ptr);
    }

    // Copying and assigning to rocblas_device_malloc are deleted
    rocblas_device_malloc(const rocblas_device_malloc&) = delete;
    rocblas_device_malloc& operator=(const rocblas_device_malloc&) = delete;
    rocblas_device_malloc& operator=(rocblas_device_malloc&&) = delete;
};

// Set optimal device memory size in handle
template <
    typename... Ss,
    std::enable_if_t<sizeof...(Ss) && conjunction<std::is_convertible<Ss, size_t>...>{}, int> = 0>
rocblas_status rocblas_set_optimal_device_memory_size(rocblas_handle handle, Ss&&... sizes)
{
    return rocblas_set_optimal_device_memory_size_impl(handle, sizeof...(sizes), size_t(sizes)...);
}
