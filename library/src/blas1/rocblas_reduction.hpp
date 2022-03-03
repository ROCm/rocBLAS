/* ************************************************************************
 * Copyright 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "utility.hpp"
#include <hip/hip_runtime.h>

static constexpr int rocblas_log2ui(int x)
{
    unsigned int ax = (unsigned int)x;
    int          v  = 0;
    while(ax >>= 1)
    {
        v++;
    }
    return v;
}

template <int N, typename T>
__inline__ __device__ T wavefront_reduce(T val)
{
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        val += __shfl_down(val, offset);
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocblas_float_complex wavefront_reduce(rocblas_float_complex val)
{
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        val.real(val.real() + __shfl_down(val.real(), offset));
        val.imag(val.imag() + __shfl_down(val.imag(), offset));
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocblas_double_complex wavefront_reduce(rocblas_double_complex val)
{
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        val.real(val.real() + __shfl_down(val.real(), offset));
        val.imag(val.imag() + __shfl_down(val.imag(), offset));
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocblas_bfloat16 wavefront_reduce(rocblas_bfloat16 val)
{
    union
    {
        int              i;
        rocblas_bfloat16 h;
    } tmp;
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        tmp.h = val;
        tmp.i = __shfl_down(tmp.i, offset);
        val += tmp.h;
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocblas_half wavefront_reduce(rocblas_half val)
{
    union
    {
        int          i;
        rocblas_half h;
    } tmp;
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        tmp.h = val;
        tmp.i = __shfl_down(tmp.i, offset);
        val += tmp.h;
        offset >>= 1;
    }
    return val;
}

template <rocblas_int NB, typename T>
__inline__ __device__ T rocblas_dot_block_reduce(T val)
{
    __shared__ T psums[warpSize];

    rocblas_int wavefront = hipThreadIdx_x / warpSize;
    rocblas_int wavelet   = hipThreadIdx_x % warpSize;

    if(wavefront == 0)
        psums[wavelet] = T(0);
    __syncthreads();

    val = wavefront_reduce<warpSize>(val); // sum over wavefront
    if(wavelet == 0)
        psums[wavefront] = val; // store sum for wavefront

    __syncthreads(); // Wait for all wavefront reductions

    // ensure wavefront was run
    static constexpr rocblas_int num_wavefronts = NB / warpSize;
    val = (hipThreadIdx_x < num_wavefronts) ? psums[wavelet] : T(0);
    if(wavefront == 0)
        val = wavefront_reduce<num_wavefronts>(val); // sum wavefront sums

    return val;
}

inline size_t rocblas_reduction_kernel_block_count(rocblas_int n, rocblas_int NB)
{
    if(n <= 0)
        n = 1; // avoid sign loss issues
    return size_t(n - 1) / NB + 1;
}

/*! \brief rocblas_reduction_batched_kernel_workspace_size
    Work area for reduction must be at lease sizeof(To) * (blocks + 1) * batch_count

    @param[in]
    outputType To*
        Type of output values
    @param[in]
    batch_count rocblas_int
        Number of batches
    ********************************************************************/
template <rocblas_int NB, typename To>
size_t rocblas_reduction_kernel_workspace_size(rocblas_int n, rocblas_int batch_count = 1)
{
    if(n <= 0)
        n = 1; // allow for return value of empty set
    if(batch_count <= 0)
        batch_count = 1;
    auto blocks = rocblas_reduction_kernel_block_count(n, NB);
    return sizeof(To) * (blocks + 1) * batch_count;
}

/*! \brief rocblas_reduction_batched_kernel_workspace_size
    Work area for reduction must be at lease sizeof(To) * (blocks + 1) * batch_count

    @param[in]
    outputType To*
        Type of output values
    @param[in]
    batch_count rocblas_int
        Number of batches
    ********************************************************************/
template <rocblas_int NB, typename To>
size_t
    rocblas_reduction_kernel_workspace_size(rocblas_int n, rocblas_int batch_count, To* output_type)
{
    return rocblas_reduction_kernel_workspace_size<NB, To>(n, batch_count);
}

template <rocblas_int NB>
size_t rocblas_reduction_kernel_workspace_size(rocblas_int      n,
                                               rocblas_int      batch_count,
                                               rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:
        return rocblas_reduction_kernel_workspace_size<NB, rocblas_half>(n, batch_count);
    case rocblas_datatype_bf16_r:
        return rocblas_reduction_kernel_workspace_size<NB, rocblas_bfloat16>(n, batch_count);
    case rocblas_datatype_f32_r:
        return rocblas_reduction_kernel_workspace_size<NB, float>(n, batch_count);
    case rocblas_datatype_f64_r:
        return rocblas_reduction_kernel_workspace_size<NB, double>(n, batch_count);
    case rocblas_datatype_f32_c:
        return rocblas_reduction_kernel_workspace_size<NB, rocblas_float_complex>(n, batch_count);
    case rocblas_datatype_f64_c:
        return rocblas_reduction_kernel_workspace_size<NB, rocblas_double_complex>(n, batch_count);
    default:
        return 0;
    }
}
