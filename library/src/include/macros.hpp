/* ************************************************************************
 * Copyright 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

/*******************************************************************************
 * Macros
 ******************************************************************************/

#ifdef WIN32
#define ROCBLAS_KERNEL __global__ static
#else
#define ROCBLAS_KERNEL __global__
#endif

// A storage-class-specifier other than thread_local shall not be specified in an explicit specialization.
// ROCBLAS_KERNEL_INSTANTIATE should be used where kernels are instantiated to avoid use of static.
#ifdef WIN32
#define ROCBLAS_KERNEL_INSTANTIATE __global__
#else
#define ROCBLAS_KERNEL_INSTANTIATE __global__
#endif

#ifdef WIN32
#define ROCBLAS_KERNEL_ILF __device__ __attribute__((always_inline))
#else
#define ROCBLAS_KERNEL_ILF __device__ __attribute__((always_inline))
#endif
