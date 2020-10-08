/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

// RAII helper class for device memory allocation outside of rocBLAS
//
// rocblas_device_malloc mem(handle, size1, size2, size3, ...);
//
// roblcas_set_optimal_device_memory_size(handle, size1, size2, size3, ...);
//
// if(!mem) return rocblas_status_memory_error;
//
// void* ptr = static_cast<void*>(mem); // Only works if there is one size
//
// rocblas_trsm_template_mem(..., mem, ...)
//
// This header should be included in other projects to use the rocblas_handle
// C++ device memory allocation API. It is unlikely to change very often.

#include "rocblas.h"
#include <new>
#include <type_traits>

// Emulate C++17 std::conjunction
template <class...>
struct rocblas_conjunction : std::true_type
{
};
template <class T, class... Ts>
struct rocblas_conjunction<T, Ts...>
    : std::integral_constant<bool, T{} && rocblas_conjunction<Ts...>{}>
{
};

class [[nodiscard]] rocblas_device_malloc
{
    rocblas_handle              handle;
    rocblas_device_malloc_base* dm_ptr;

public:
    // Allocate memory in a RAII class
    template <
        typename... Ss,
        std::enable_if_t<sizeof...(Ss) && rocblas_conjunction<std::is_convertible<Ss, size_t>...>{},
                         int> = 0>
    explicit rocblas_device_malloc(rocblas_handle handle, Ss... sizes)
        : handle(handle)
        , dm_ptr(nullptr)
    {
        if(rocblas_device_malloc_alloc(handle, &dm_ptr, sizeof...(sizes), size_t(sizes)...)
           != rocblas_status_success)
            throw std::bad_alloc();
    }

    // Move constructor
    // clang-format off
    rocblas_device_malloc(rocblas_device_malloc&& other) noexcept
        // clang-format on
        : handle(other.handle)
        , dm_ptr(other.dm_ptr)
    {
        other.dm_ptr = nullptr;
    }

    // Conversion to a pointer type, to get the address of the device memory
    // It is lvalue-qualified to avoid the common mistake of writing:
    // void* ptr = (void*) rocblas_device_malloc(handle, size);
    // ... which is incorrect, since the RAII temporary expression will be
    // destroyed, and the pointer will be left dangling.
    template <typename T>
    // clang-format off
    explicit operator T*() &
    // clang-format on
    {
        void* res;
        if(!dm_ptr || rocblas_device_malloc_ptr(dm_ptr, &res) != rocblas_status_success)
            res = nullptr;
        return static_cast<T*>(res);
    }

    // Access a particular element
    // It is lvalue-qualified so that it cannot bind to temporaries
    // clang-format off
    void* operator[](size_t index) &
    // clang-format on
    {
        void* res;
        if(!dm_ptr || rocblas_device_malloc_get(dm_ptr, index, &res) != rocblas_status_success)
            res = nullptr;
        return res;
    }

    // Conversion to bool indicates whether allocation succeeded
    // It is lvalue-qualified so that it cannot bind to temporaries
    // clang-format off
    explicit operator bool() &
    // clang-format on
    {
        return rocblas_device_malloc_success(dm_ptr);
    }

    // Conversion to rocblas_device_malloc_base reference, to pass to rocBLAS
    // It is lvalue-qualified so that it cannot bind to temporaries
    // clang-format off
    operator rocblas_device_malloc_base&() &
    // clang-format on
    {
        return *dm_ptr;
    }

    // Destructor calls the C API to mark the data as freed
    ~rocblas_device_malloc()
    {
        if(dm_ptr)
            rocblas_device_malloc_free(dm_ptr);
    }

    // Copying and assigning to rocblas_device_malloc are deleted
    rocblas_device_malloc(const rocblas_device_malloc&) = delete;
    rocblas_device_malloc& operator=(const rocblas_device_malloc&) = delete;
    rocblas_device_malloc& operator=(rocblas_device_malloc&&) = delete;
};

// Set optimal device memory size in handle
template <
    typename... Ss,
    std::enable_if_t<sizeof...(Ss) && rocblas_conjunction<std::is_convertible<Ss, size_t>...>{},
                     int> = 0>
inline rocblas_status rocblas_set_optimal_device_memory_size(rocblas_handle handle, Ss... sizes)
{
    return rocblas_set_optimal_device_memory_size_impl(handle, sizeof...(sizes), size_t(sizes)...);
}
