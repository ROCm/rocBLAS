/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "reduction.h"
#include "reduction_batched.h"
#include "rocblas.h"
#include "utility.h"

template <rocblas_int NB, bool CONJ, typename T, typename U, typename V>
__global__ void dot_kernel_part1(
    rocblas_int n, const U xa, rocblas_int    offsetx, rocblas_int incx, rocblas_int stridex, const U  ya, rocblas_int    offsety, rocblas_int incy, rocblas_int stridey, V workspace)
{
    ptrdiff_t tx  = hipThreadIdx_x;
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + tx;

    __shared__ V tmp[NB];
    const T* x;
    const T* y;

    // bound
    if(tid < n)
    {
        x = load_ptr_batch(xa, hipBlockIdx_y, offsetx, stridex);
        y = load_ptr_batch(ya, hipBlockIdx_y, offsety, stridey);
        if(incx < 0)
            x -= ptrdiff_t(incx) * (n - 1);
        if(incy < 0)
            y -= ptrdiff_t(incy) * (n - 1);
        tmp[tx] = V(y[tid * incy]) * V(CONJ ? conj(x[tid * incx]) : x[tid * incx]);
    }
    else
        tmp[tx] = V(0); // pad with zero

    rocblas_sum_reduce<NB>(tx, tmp);

    if(tx == 0)
        workspace[hipBlockIdx_x + hipBlockIdx_y*hipGridDim_x] = tmp[0];
}

// assume workspace has already been allocated, recommened for repeated calling of dot_strided_batched product
// routine
template <rocblas_int NB, bool CONJ, typename T, typename U , typename V, typename V2 = V>
rocblas_status rocblas_dot_template(rocblas_handle __restrict__ handle,
                                        rocblas_int n,
                                        const U    x,
                                        rocblas_int    offsetx,
                                        rocblas_int incx,
                                        rocblas_int stridex,
                                        const U    y,
                                        rocblas_int    offsety,
                                        rocblas_int incy,
                                        rocblas_int stridey,
                                        rocblas_int    batch_count,
                                        V          result,
                                        V2          workspace,
                                        rocblas_int blocks)
{
    // At least two kernels are needed to finish the reduction
    // kennel 1 write partial result per thread block in workspace, number of partial result is
    // blocks
    // kernel 2 gather all the partial result in workspace and finish the final reduction. number of
    // threads (NB) loop blocks

    // Quick return if possible.
    if(n <= 0 || batch_count == 0)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        else if(rocblas_pointer_mode_device == handle->pointer_mode && batch_count > 0)
        {
            RETURN_IF_HIP_ERROR(hipMemset(result, 0, batch_count * sizeof(T)));
        }
        else
            for(int i = 0; i < batch_count; i++)
            {
                result[i] = T(0);
            }

        return rocblas_status_success;
    }

    dim3 grid(blocks);
    dim3 threads(NB);

    if(incx < 0)
        x -= ptrdiff_t(incx) * (n - 1);
    if(incy < 0)
        y -= ptrdiff_t(incy) * (n - 1);

    hipLaunchKernelGGL(dot_kernel_part1<NB, CONJ, T>,
                        grid,
                        threads,
                        0,
                        handle->rocblas_stream,
                        n,
                        x,
                        incx,
                        y,
                        incy,
                        workspace);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL(rocblas_reduction_batched_kernel_part2<NB>,
                            1,
                            threads,
                            0,
                            handle->rocblas_stream,
                            blocks,
                            workspace,
                            result);
    }
    else
    {
        hipLaunchKernelGGL(rocblas_reduction_batched_kernel_part2<NB>,
                            1,
                            threads,
                            0,
                            handle->rocblas_stream,
                            blocks,
                            workspace,
                            workspace);

        V2 res_V2;
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&res_V2, workspace, sizeof(res_V2), hipMemcpyDeviceToHost));
        *result = T(res_V2);
    }

    return rocblas_status_success;
}
