#ifndef REDUCTION_BATCHED_H_
#define REDUCTION_BATCHED_H_

#include "handle.h"
#include "rocblas.h"
#include "utility.h"
#include "reduction.h"
#include <type_traits>
#include <utility>

/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 *    It reuses later stage reduction kernels from reduction.h
 *    See that header file for more documentation
 * ===========================================================================
 */

// BLAS Level 1 includes routines and functions performing vector-vector
// operations. Most BLAS 1 routines are about reduction: compute the norm,
// calculate the dot production of two vectors, find the maximum/minimum index
// of the element of the vector. As you may observed, although the computation
// type is different, the core algorithm is the same: scan all element of the
// vector(s) and reduce to one single result.
//
// The reduction algorithm on GPU is called [parallel
// reduction](https://raw.githubusercontent.com/mateuszbuda/GPUExample/master/reduce3.png)
// which is adopted in rocBLAS. At the beginning, all the threads in the thread
// block participate. After each step of reduction (like a tree), the number of
// participating threads decrease by half. At the end of the parallel reduction,
// only one thread (usually thread 0) owns the result in its thread block.
//
// Classically, the BLAS 1 reduction needs more than one GPU kernel to finish,
// because the lack of global synchronization of thread blocks without exiting
// the kernel. The first kernels gather partial results, write into a temporary
// working buffer. The second kernel finishes the final reduction.
//
// For example, BLAS 1 routine i*amax is to find index of the maximum absolute
// value element of a vector. In this routine:
//
// Kernel 1: launch many thread block as needed. Each thread block works on a
// subset of the vector. Each thread block use the parallel reduction to find a
// local index with the maximum absolute value of the subset. There are
// number-of-the-thread-blocks local results.The results are written into a
// temporary working buffer. The working buffer has number-of-the-thread-blocks
// elements.
//
// Kernel 2: launch only one thread block which reads the temporary work buffer and
// reduces to final result still with the parallel reduction.
//
// As you may see, if there is a mechanism to synchronize all the thread blocks
// after local index is obtained in kernel 1 (without ending the kernel), then
// Kernel 2's computation can be merged into Kernel 1. One such mechanism is called
// atomic operation. However, atomic operation is new and is not used in rocBLAS
// yet. rocBLAS still use the classic standard parallel reduction right now.


// kernel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <rocblas_int NB,
          typename FETCH,
          typename REDUCE = rocblas_reduce_sum,
          typename Ti,
          typename To>
__global__ void rocblas_reduction_batched_kernel_part1(
    rocblas_int n, rocblas_int nblocks, const Ti* const xvec[], rocblas_int incx, To* workspace)
{
    ptrdiff_t     tx  = hipThreadIdx_x;
    ptrdiff_t     tid = hipBlockIdx_x * hipBlockDim_x + tx;
    __shared__ To tmp[NB];

    const Ti* x = xvec[hipBlockIdx_y];

    // bound
    if(tid < n)
        tmp[tx] = FETCH{}(x[tid * incx], tid);
    else
        tmp[tx] = default_value<To>{}(); // pad with default value

    rocblas_reduction<NB, REDUCE>(tx, tmp);

    __syncthreads();

    if(tx == 0)
        workspace[hipBlockIdx_y * nblocks + hipBlockIdx_x] = tmp[0];
}

// kernel 2 gathers all the partial results in workspace and finishes the final reduction;
// number of threads (NB) loop blocks
template <rocblas_int NB,
          typename REDUCE   = rocblas_reduce_sum,
          typename FINALIZE = rocblas_finalize_identity,
          typename To,
          typename Tr>
__global__ void
    rocblas_reduction_batched_kernel_part2(rocblas_int nblocks, To* workspace, Tr* result)
{
    rocblas_int   tx = hipThreadIdx_x;
    __shared__ To tmp[NB];

    if(tx < nblocks)
    {
        To* work = workspace + hipBlockIdx_y * nblocks;
        tmp[tx]  = work[tx];

        // bound, loop
        for(rocblas_int i = tx + NB; i < nblocks; i += NB)
            REDUCE{}(tmp[tx], work[i]);
    }
    else
    { // pad with default value
        tmp[tx] = default_value<To>{}();
    }

    if(nblocks < 32)
    {
        // no need parallel reduction
        __syncthreads();

        if(tx == 0)
            for(rocblas_int i = 1; i < nblocks; i++)
                REDUCE{}(tmp[0], tmp[i]);
    }
    else
    {
        // parallel reduction
        rocblas_reduction<NB, REDUCE>(tx, tmp);
    }

    // Store result on device or in workspace
    if(tx == 0)
        result[hipBlockIdx_y] = FINALIZE{}(tmp[0]);
}

// At least two kernels are needed to finish the reduction
// kennel 1 write partial result per thread block in workspace, blocks partial results
// kernel 2 gathers all the partial result in workspace and finishes the final reduction.
template <rocblas_int NB,
          typename FETCH,
          typename REDUCE   = rocblas_reduce_sum,
          typename FINALIZE = rocblas_finalize_identity,
          typename Ti,
          typename To,
          typename Tr>
rocblas_status rocblas_reduction_batched_kernel(rocblas_handle __restrict__ handle,
                                                rocblas_int     n,
                                                const Ti* const x[],
                                                rocblas_int     incx,
                                                Tr*             result,
                                                To*             workspace,
                                                rocblas_int     blocks,
                                                rocblas_int     batch_count)
{
    hipLaunchKernelGGL((rocblas_reduction_batched_kernel_part1<NB, FETCH, REDUCE>),
                       dim3(blocks, batch_count),
                       NB,
                       0,
                       handle->rocblas_stream,
                       n,
                       blocks,
                       x,
                       incx,
                       workspace);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((rocblas_reduction_batched_kernel_part2<NB, REDUCE, FINALIZE>),
                           dim3(1, batch_count),
                           NB,
                           0,
                           handle->rocblas_stream,
                           blocks,
                           workspace,
                           result);
    }
    else
    {
        // If in host pointer mode, workspace is converted to Tr* and the result is
        // placed there, and then copied from device to host. If To is a class type,
        // it must be a standard layout type and its first member must be of type Tr.
        static_assert(std::is_standard_layout<To>{}, "To must be a standard layout type");

        if(blocks > 1 || batch_count > 1)
        {
            hipLaunchKernelGGL((rocblas_reduction_batched_kernel_part2<NB, REDUCE, FINALIZE>),
                               dim3(1, batch_count),
                               NB,
                               0,
                               handle->rocblas_stream,
                               blocks,
                               workspace,
                               (Tr*)(workspace + batch_count * blocks));
        }

        if(std::is_same<FINALIZE, rocblas_finalize_identity>{} || blocks > 1
           || batch_count > 1)
        {
            // If FINALIZE is trivial or kernel part2 was called, result is in the
            // beginning of workspace[0]+offset, and can be copied directly.
            RETURN_IF_HIP_ERROR(hipMemcpy(result,
                                          workspace + batch_count * blocks,
                                          batch_count * sizeof(Tr),
                                          hipMemcpyDeviceToHost));
        }
        else
        {
            // If FINALIZE is not trivial and kernel part2 was not called, then
            // workspace[0] needs to be finalized on host.
            To res[batch_count];
            RETURN_IF_HIP_ERROR(
                hipMemcpy(res, workspace, batch_count * sizeof(To), hipMemcpyDeviceToHost));
            for(int i = 0; i < batch_count; i++)
                result[i] = FINALIZE{}(res[i]);
        }
    }

    return rocblas_status_success;
}

#endif
