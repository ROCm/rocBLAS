#ifndef GUARD_ROCBLAS_MANAGE_PTR
#define GUARD_ROCBLAS_MANAGE_PTR

#include <memory>

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        \
}                                                      \
    }

namespace rocblas_test {
// device_malloc wraps hipMalloc and provides same API as malloc
static void* device_malloc(size_t byte_size)
{
    void* pointer;
    PRINT_IF_HIP_ERROR(hipMalloc(&pointer, byte_size));
    return pointer;
}

// device_free wraps hipFree and provides same API as free
static void device_free(void* ptr) { PRINT_IF_HIP_ERROR(hipFree(ptr)); }

struct handle_struct
{
    rocblas_handle handle;
    handle_struct()
    {
        rocblas_status status = rocblas_create_handle(&handle);
        verify_rocblas_status_success(status, "ERROR: handle_struct constructor");
    }

    ~handle_struct()
    {
        rocblas_status status = rocblas_destroy_handle(handle);
        verify_rocblas_status_success(status, "ERROR: handle_struct destructor");
    }
};

} // namespace rocblas_test

using rocblas_unique_ptr = std::unique_ptr<void, void (*)(void*)>;

#endif
