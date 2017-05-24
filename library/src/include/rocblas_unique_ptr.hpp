#ifndef GUARD_ROCBLAS_MANAGE_PTR_HPP
#define GUARD_ROCBLAS_MANAGE_PTR_HPP

#include <memory>

namespace rocblas
{
    void* device_malloc(size_t byte_size)
    {
        void *pointer;
        PRINT_IF_HIP_ERROR(hipMalloc(&pointer, byte_size));
        return pointer;
    }

    void device_free(void *ptr)
    {   
        PRINT_IF_HIP_ERROR(hipFree(ptr));
    }
} // namespace rocblas

using rocblas_unique_ptr = std::unique_ptr<void, void(*)(void*)>;

#endif
