#ifndef REDUCTION_STRIDED_BATCHED_H_
#define REDUCTION_STRIDED_BATCHED_H_

#include "handle.h"
#include "reduction_batched.h"
#include "rocblas.h"
#include "utility.h"
#include <type_traits>
#include <utility>

/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 *    It reuses later stage reduction kernels from reduction_batched.h
 *    See that header file for more documentation
 * ===========================================================================
 */

// kernel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <rocblas_int NB,
          typename FETCH,
          typename REDUCE = rocblas_reduce_sum,
          typename Ti,
          typename To>
__global__ void rocblas_reduction_strided_batched_kernel_part1(rocblas_int n,
                                                               rocblas_int nblocks,
                                                               const Ti*   xvec,
                                                               rocblas_int incx,
                                                               rocblas_int stridex,
                                                               To*         workspace)
{
    ptrdiff_t     tx  = hipThreadIdx_x;
    ptrdiff_t     tid = hipBlockIdx_x * hipBlockDim_x + tx;
    __shared__ To tmp[NB];

    const Ti* x = xvec + hipBlockIdx_y * stridex;

    // bound
    if(tid < n)
        tmp[tx] = FETCH{}(x[tid * incx], tid);
    else
        tmp[tx] = default_value<To>{}(); // pad with default value

    rocblas_reduction<NB, REDUCE>(tx, tmp);

    if(tx == 0)
        workspace[hipBlockIdx_y * nblocks + hipBlockIdx_x] = tmp[0];
}

// kernel 2 is used from non-strided reduction_batched see include file

/*! \brief

    \details
    rocblas_reduction_strided_batched_kernel computes a reduction over multiple vectors x_i
              Template parameters allow threads per block, data, and specific phase kernel overrides
              At least two kernels are needed to finish the reduction
              kernel 1 write partial result per thread block in workspace, blocks partial results
              kernel 2 gathers all the partial result in workspace and finishes the final reduction.
    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int
              number of elements in each vector x_i
    @param[in]
    x         pointer to the first vector x_i on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of each x_i.
    @param[in]
    stridex   specifies the pointer increment between batches for x. stridex must be >= n*incx

    @param[out]
    result
              pointer to array of batch_count size for results. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.

    @param[out]
    workspace *To  
              temporary GPU buffer for inidividual block results per batch
              and results buffer in case result pointer is to host memory
              Size must be (blocks+1)*batch_count*sizeof(To)
    @param[in]
    blocks    rocblas_int
              number of thread blocks 
    @param[in]
    batch_count rocblas_int
              number of instances in the batch
    ********************************************************************/
template <rocblas_int NB,
          typename FETCH,
          typename REDUCE   = rocblas_reduce_sum,
          typename FINALIZE = rocblas_finalize_identity,
          typename Ti,
          typename To,
          typename Tr>
rocblas_status rocblas_reduction_strided_batched_kernel(rocblas_handle __restrict__ handle,
                                                        rocblas_int n,
                                                        const Ti*   x,
                                                        rocblas_int incx,
                                                        rocblas_int stridex,
                                                        Tr*         result,
                                                        To*         workspace,
                                                        rocblas_int blocks,
                                                        rocblas_int batch_count)
{
    hipLaunchKernelGGL((rocblas_reduction_strided_batched_kernel_part1<NB, FETCH, REDUCE>),
                       dim3(blocks, batch_count),
                       NB,
                       0,
                       handle->rocblas_stream,
                       n,
                       blocks,
                       x,
                       incx,
                       stridex,
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

        bool reduceKernel = blocks > 1 || batch_count > 1;
        if(reduceKernel)
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

        if(std::is_same<FINALIZE, rocblas_finalize_identity>{} || reduceKernel)
        {
            // If FINALIZE is trivial or kernel part2 was called, result is in the
            // beginning of workspace[0]+offset, and can be copied directly.
            size_t offset = reduceKernel ? batch_count * blocks : 0;
            RETURN_IF_HIP_ERROR(hipMemcpy(
                result, workspace + offset, batch_count * sizeof(Tr), hipMemcpyDeviceToHost));
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
