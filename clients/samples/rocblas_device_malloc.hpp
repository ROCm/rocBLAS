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
// rocblas_device_malloc mem4 = rocblas_device_malloc::nullptrs(4);
// rocblas_trsm_template_mem(..., mem4, ...)
//
// This header should be included in other projects to use the C++
// device memory allocation API. It is unlikely to change very often,
// so it is safe to copy to other projects.

#include "rocblas.h"

class [[nodiscard]] rocblas_device_malloc
{
    rocblas_handle              handle;
    rocblas_device_malloc_base* dm_ptr;

    // Private constructor for empty object
    explicit rocblas_device_malloc(rocblas_handle handle)
        : handle(handle)
        , dm_ptr(nullptr)
    {
    }

public:
    // Allocate memory in a RAII class
    rocblas_device_malloc(rocblas_handle handle, size_t size)
        : rocblas_device_malloc(handle)
    {
        rocblas_device_malloc_alloc(handle, size, &dm_ptr);
    }

    // Allocate an object with count nullptrs, to be passed to rocBLAS
    // This is used to construct objects to pass to rocblas_trsm_template_mem
    // and other rocBLAS functions which expect multiple sizes to fill.
    static rocblas_device_malloc nullptrs(rocblas_handle handle, size_t count)
    {
        rocblas_device_malloc nptr(handle);
        rocblas_device_malloc_nullptrs(handle, count, &nptr.dm_ptr);
        return nptr;
    }

    // Move constructor
    rocblas_device_malloc(rocblas_device_malloc && other) noexcept
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
    explicit operator T*()&
    {
        void* res;
        return dm_ptr && rocblas_device_malloc_get(handle, dm_ptr, &res) == rocblas_status_success
                   ? static_cast<T*>(res)
                   : nullptr;
    }

    // Conversion to bool indicates whether allocation succeeded
    // It is lvalue-qualified so that it cannot bind to temporaries
    explicit operator bool()&
    {
        return static_cast<void*>(*this) != nullptr;
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
