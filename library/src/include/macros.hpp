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

#ifdef WIN32
#define ROCBLAS_KERNEL_ILF __device__ __attribute__((always_inline))
#else
#define ROCBLAS_KERNEL_ILF __device__ __attribute__((always_inline))
#endif
