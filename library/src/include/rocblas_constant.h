#ifndef ROCBLAS_CONSTANT_H_
#define ROCBLAS_CONSTANT_H_

#include "rocblas-types.h"

typedef enum
{
    rocblas_constant_neg1,
    rocblas_constant_zero,
    rocblas_constant_one,
    rocblas_constant_count
} rocblas_constant_t;

// This returns a pointer to a host or device constant depending on pointer mode
template <typename T>
const T* rocblas_get_constant(rocblas_handle, rocblas_constant_t);

#endif // ROCBLAS_CONSTANT_H_
