/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "reduction_strided_batched.h"
#include "rocblas.h"
#include "utility.h"

template <rocblas_int NB, bool CONJ, typename T, typename U, typename V = T>
__global__ void dot_kernel_part1(rocblas_int    n,
                                 const U        xa,
                                 ptrdiff_t      shiftx,
                                 rocblas_int    incx,
                                 rocblas_stride stridex,
                                 const U        ya,
                                 ptrdiff_t      shifty,
                                 rocblas_int    incy,
                                 rocblas_int    stridey,
                                 V*             workspace)
{
    ptrdiff_t tx  = hipThreadIdx_x;
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + tx;

    __shared__ V tmp[NB];
    const T*     x;
    const T*     y;

    // bound
    if(tid < n)
    {
        x       = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
        y       = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);
        tmp[tx] = V(y[tid * incy]) * V(CONJ ? conj(x[tid * incx]) : x[tid * incx]);
    }
    else
        tmp[tx] = V(0); // pad with zero

    rocblas_sum_reduce<NB>(tx, tmp);

    if(tx == 0)
        workspace[hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x] = tmp[0];
}

// assume workspace has already been allocated, recommened for repeated calling of dot_strided_batched product
// routine
template <rocblas_int NB, bool CONJ, typename T, typename U, typename V = T>
rocblas_status rocblas_dot_template(rocblas_handle __restrict__ handle,
                                    rocblas_int    n,
                                    const U        x,
                                    rocblas_int    offsetx,
                                    rocblas_int    incx,
                                    rocblas_stride stridex,
                                    const U        y,
                                    rocblas_int    offsety,
                                    rocblas_int    incy,
                                    rocblas_stride stridey,
                                    rocblas_int    batch_count,
                                    T*             results,
                                    V*             workspace)
{
    // At least two kernels are needed to finish the reduction
    // kennel 1 write partial results per thread block in workspace, number of partial results is
    // blocks
    // kernel 2 gather all the partial results in workspace and finish the final reduction. number of
    // threads (NB) loop blocks

    // Quick return if possible.
    if(n <= 0 || batch_count == 0)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        else if(rocblas_pointer_mode_device == handle->pointer_mode && batch_count > 0)
        {
            RETURN_IF_HIP_ERROR(hipMemset(results, 0, batch_count * sizeof(T)));
        }
        else
            for(int i = 0; i < batch_count; i++)
            {
                results[i] = T(0);
            }

        return rocblas_status_success;
    }

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    rocblas_int blocks = rocblas_reduction_kernel_block_count(n, NB);
    dim3        grid(blocks, batch_count);
    dim3        threads(NB);

    hipLaunchKernelGGL((dot_kernel_part1<NB, CONJ, T>),
                       grid,
                       threads,
                       0,
                       handle->rocblas_stream,
                       n,
                       x,
                       shiftx,
                       incx,
                       stridex,
                       y,
                       shifty,
                       incy,
                       stridey,
                       workspace);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL(rocblas_reduction_strided_batched_kernel_part2<NB>,
                           dim3(1, batch_count),
                           threads,
                           0,
                           handle->rocblas_stream,
                           blocks,
                           workspace,
                           results);
    }
    else
    {
        hipLaunchKernelGGL(rocblas_reduction_strided_batched_kernel_part2<NB>,
                           dim3(1, batch_count),
                           threads,
                           0,
                           handle->rocblas_stream,
                           blocks,
                           workspace,
                           (V*)(workspace + size_t(batch_count) * blocks));

        // result is in the beginning of workspace[0]+offset
        size_t offset = size_t(batch_count) * blocks;
        V      res_V[batch_count];
        RETURN_IF_HIP_ERROR(
            hipMemcpy(res_V, workspace + offset, sizeof(V) * batch_count, hipMemcpyDeviceToHost));
        for(rocblas_int i = 0; i < batch_count; i++)
            results[i] = T(res_V[i]);
    }

    return rocblas_status_success;
}
