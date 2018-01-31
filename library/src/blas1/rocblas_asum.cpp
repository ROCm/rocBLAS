/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"

#include "status.h"
#include "definitions.h"
#include "device_template.h"
#include "fetch_template.h"
#include "rocblas_unique_ptr.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"

template <typename T1, typename T2, rocblas_int NB>
__global__ void asum_kernel_part1(rocblas_int n, const T1* x, rocblas_int incx, T2* workspace)
{
    rocblas_int tx  = hipThreadIdx_x;
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    __shared__ T2 shared_tep[NB];
    // bound
    if(tid < n)
    {
        T2 real        = fetch_real<T1, T2>(x[tid * incx]);
        T2 imag        = fetch_imag<T1, T2>(x[tid * incx]);
        shared_tep[tx] = fabs(real) + fabs(imag);
    }
    else
    { // pad with zero
        shared_tep[tx] = 0.0;
    }

    rocblas_sum_reduce<NB, T2>(tx, shared_tep);

    if(tx == 0)
        workspace[hipBlockIdx_x] = shared_tep[0];
}

template <typename T, rocblas_int NB, rocblas_int flag>
__global__ void asum_kernel_part2(rocblas_int n, T* workspace, T* result)
{
    rocblas_int tx = hipThreadIdx_x;

    __shared__ T shared_tep[NB];

    shared_tep[tx] = 0.0;

    // bound, loop
    for(rocblas_int i = tx; i < n; i += NB)
    {
        shared_tep[i] += workspace[i];
    }

    __syncthreads();

    if(n < 32)
    {
        // no need parallel reduction
        if(tx == 0)
        {
            for(rocblas_int i = 1; i < n; i++)
            {
                shared_tep[0] += shared_tep[i];
            }
        }
    }
    else
    {
        // parallel reduction, TODO bug
        rocblas_sum_reduce<NB, T>(tx, shared_tep);
    }

    if(tx == 0)
    {
        if(flag)
        {
            // flag == 1, write to result of device memory
            *result = shared_tep[0]; // result[0] works, too
        }
        else
        {
            workspace[0] = shared_tep[0];
        }
    }
}

// HIP support up to 1024 threads/work itmes per thread block/work group
#define NB_X 1024

// assume workspace has already been allocated, recommened for repeated calling of asum product
// routine
template <typename T1, typename T2>
rocblas_status rocblas_asum_template_workspace(rocblas_handle handle,
                                               rocblas_int n,
                                               const T1* x,
                                               rocblas_int incx,
                                               T2* result,
                                               T2* workspace,
                                               rocblas_int lworkspace)
{
    rocblas_int blocks = (n - 1) / NB_X + 1;

    // At least two kernels are needed to finish the reduction
    // kennel 1 write partial result per thread block in workspace, number of partial result is
    // blocks
    // kernel 2 gather all the partial result in workspace and finish the final reduction. number of
    // threads (NB_X) loop blocks

    if(lworkspace < blocks)
    {
        printf("size workspace = %d is too small, allocate at least %d", lworkspace, blocks);
        return rocblas_status_not_implemented;
    }

    dim3 grid(blocks, 1, 1);
    dim3 threads(NB_X, 1, 1);

    hipStream_t rocblas_stream = handle->rocblas_stream;

    hipLaunchKernelGGL((asum_kernel_part1<T1, T2, NB_X>),
                       dim3(grid),
                       dim3(threads),
                       0,
                       rocblas_stream,
                       n,
                       x,
                       incx,
                       workspace);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        // the last argument 1 indicate the result is on device, not memcpy is required
        hipLaunchKernelGGL((asum_kernel_part2<T2, NB_X, 1>),
                           dim3(1, 1, 1),
                           dim3(threads),
                           0,
                           rocblas_stream,
                           blocks,
                           workspace,
                           result);
    }
    else
    {
        // the last argument 0 indicate the result is on host
        // workspace[0] has a copy of the final result, if the result pointer is on host, a memory
        // copy is required
        // printf("it is a host pointer\n");
        // only for blocks > 1, otherwise the final result is already reduced in workspace[0]
        if(blocks > 1)
            hipLaunchKernelGGL((asum_kernel_part2<T2, NB_X, 0>),
                               dim3(1, 1, 1),
                               dim3(threads),
                               0,
                               rocblas_stream,
                               blocks,
                               workspace,
                               result);
        RETURN_IF_HIP_ERROR(hipMemcpy(result, workspace, sizeof(T2), hipMemcpyDeviceToHost));
    }

    return rocblas_status_success;
}

/* ============================================================================================ */

/*! \brief BLAS Level 1 API

    \details
    asum computes the sum of the absolute values of elements of a real vector x,
         or the sum of absolute values of the real and imaginary parts of elements if x is a complex
   vector

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of y.
    @param[inout]
    result
              store the asum product. either on the host CPU or device GPU.
              result is 0.0 if n <= 0 or incx <= 0.
    ********************************************************************/

// allocate workspace inside this API
template <typename T1, typename T2>
rocblas_status rocblas_asum_template(
    rocblas_handle handle, rocblas_int n, const T1* x, rocblas_int incx, T2* result)
{
    if(nullptr == handle)
        return rocblas_status_invalid_handle;

    log_function(handle, replaceX<T1>("rocblas_Xasum"), n, (const void*&)x, incx);

    log_bench(handle, "./rocblas-bench -f asum -r", replaceX<T1>("X"), "-n", n, "--incx", incx);

    if(nullptr == x)
    {
        return rocblas_status_invalid_pointer;
    }
    else if(nullptr == result)
    {
        return rocblas_status_invalid_pointer;
    }

    /*
     * Quick return if possible.
     */
    if(n <= 0 || incx <= 0)
    {
        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(T2)));
        }
        else
        {
            *result = 0.0;
        }
        return rocblas_status_success;
    }

    rocblas_int blocks = (n - 1) / NB_X + 1;

    rocblas_status status;

    auto workspace =
        rocblas_unique_ptr{rocblas::device_malloc(sizeof(T2) * blocks), rocblas::device_free};
    if(!workspace)
    {
        return rocblas_status_memory_error;
    }

    status = rocblas_asum_template_workspace<T1, T2>(
        handle, n, x, incx, result, (T2*)workspace.get(), blocks);

    return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status
rocblas_sasum(rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result)
{
    return rocblas_asum_template<float, float>(handle, n, x, incx, result);
}

extern "C" rocblas_status rocblas_dasum(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result)
{
    return rocblas_asum_template<double, double>(handle, n, x, incx, result);
}

extern "C" rocblas_status rocblas_scasum(rocblas_handle handle,
                                         rocblas_int n,
                                         const rocblas_float_complex* x,
                                         rocblas_int incx,
                                         float* result)
{
    return rocblas_asum_template<rocblas_float_complex, float>(handle, n, x, incx, result);
}

extern "C" rocblas_status rocblas_dzasum(rocblas_handle handle,
                                         rocblas_int n,
                                         const rocblas_double_complex* x,
                                         rocblas_int incx,
                                         double* result)
{
    return rocblas_asum_template<rocblas_double_complex, double>(handle, n, x, incx, result);
}
