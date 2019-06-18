#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime_api.h>
#include <rocblas_constant.h>

// Constants to be used in rocBLAS kernels
// Note: Order is important, and must match rocblas_constant_t.
// Cannot use designated array index initialiers in C++.
template <typename T>
static constexpr T constant_values[rocblas_constant_count] = {-1, 0, 1};

// Constants stored in device constant memory
template <typename T>
static __device__ __constant__ T device_constant_values_store[rocblas_constant_count];

// Initialization of device constants at program startup time
template <typename T>
static const T* initialize_constants()
{
    T* addr;
    if(hipSuccess
           != hipMemCpyToSymbol(
               device_constant_values_store<T>, constant_values<T>, sizeof(constant_values<T>))
       || hipSuccess != hipGetSymbolAddress((void**)&addr, device_constant_values_store<T>))
    {
        fputs("Error initializing rocBLAS device constants\n", stderr);
        abort();
    }
    return addr;
}

// This initializes a pointer to constants in device memory
template <typename T>
static const T* device_constant_values = initialize_constants<T>();

// This returns a pointer to a host or device constant depending on pointer mode
template <typename T>
const T* rocblas_get_constant(rocblas_handle handle, rocblas_constant_t c)
{
    return !handle || c < 0 || c >= rocblas_constant_count
               ? nullptr
               : handle->pointer_mode == rocblas_pointer_mode_device
                     ? rocblas::device_constant_values<T> + c
                     : rocblas::constant_values<T> + c;
}

// Explicitly instantiate rocblas_get_constant() for all supported types
const float*  rocblas_get_constant<float>(rocblas_handle, rocblas_constant_t);
const double* rocblas_get_constant<double>(rocblas_handle, rocblas_constant_t);
