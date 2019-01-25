/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"

#include "status.h"
#include "definitions.h"
#include "reduction.h"
#include "rocblas_unique_ptr.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {
// HIP support up to 1024 threads/work itemes per thread block/work group
// setting to 512 for gfx803.
constexpr int NB = 512;

template <typename T>
__global__ void dot_kernel_part1(
    rocblas_int n, const T* x, rocblas_int incx, const T* y, rocblas_int incy, T* workspace)
{
    ssize_t tx  = hipThreadIdx_x;
    ssize_t tid = hipBlockIdx_x * hipBlockDim_x + tx;

    __shared__ T tmp[NB];

    // bound
    if(tid < n)
        tmp[tx] = y[tid * incy] * x[tid * incx];
    else
        tmp[tx] = 0; // pad with zero

    rocblas_sum_reduce<NB>(tx, tmp);

    if(tx == 0)
        workspace[hipBlockIdx_x] = tmp[0];
}

// assume workspace has already been allocated, recommened for repeated calling of dot product
// routine
template <typename T>
rocblas_status rocblas_dot_workspace(rocblas_handle __restrict__ handle,
                                     rocblas_int n,
                                     const T* x,
                                     rocblas_int incx,
                                     const T* y,
                                     rocblas_int incy,
                                     T* result,
                                     T* workspace,
                                     rocblas_int blocks)
{
    // At least two kernels are needed to finish the reduction
    // kennel 1 write partial result per thread block in workspace, number of partial result is
    // blocks
    // kernel 2 gather all the partial result in workspace and finish the final reduction. number of
    // threads (NB) loop blocks

    dim3 grid(blocks);
    dim3 threads(NB);

    if(incx < 0)
        x -= ssize_t(incx) * (n - 1);
    if(incy < 0)
        y -= ssize_t(incy) * (n - 1);

    hipLaunchKernelGGL(
        dot_kernel_part1, grid, threads, 0, handle->rocblas_stream, n, x, incx, y, incy, workspace);

    hipLaunchKernelGGL((rocblas_reduction_kernel_part2<NB>),
                       1,
                       threads,
                       0,
                       handle->rocblas_stream,
                       blocks,
                       workspace,
                       handle->pointer_mode != rocblas_pointer_mode_device ? workspace : result);
    if(handle->pointer_mode != rocblas_pointer_mode_device)
        RETURN_IF_HIP_ERROR(hipMemcpy(result, workspace, sizeof(*result), hipMemcpyDeviceToHost));

    return rocblas_status_success;
}

template <typename>
constexpr char rocblas_dot_name[] = "unknown";
template <>
constexpr char rocblas_dot_name<float>[] = "rocblas_sdot";
template <>
constexpr char rocblas_dot_name<double>[] = "rocblas_ddot";
template <>
constexpr char rocblas_dot_name<rocblas_float_complex>[] = "rocblas_cdot";
template <>
constexpr char rocblas_dot_name<rocblas_double_complex>[] = "rocblas_zdot";

/* ============================================================================================ */

/*! \brief BLAS Level 1 API

    \details
    dot(u)  perform dot product of vector x and y

        result = x * y;

    dotc  perform dot product of complex vector x and complex y

        result = conjugate (x) * y;

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
              store the dot product. either on the host CPU or device GPU.
              return is 0 if n <= 0.

    ********************************************************************/

// allocate workspace inside this API
template <typename T>
rocblas_status rocblas_dot(rocblas_handle handle,
                           rocblas_int n,
                           const T* x,
                           rocblas_int incx,
                           const T* y,
                           rocblas_int incy,
                           T* result)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    auto layer_mode = handle->layer_mode;

    if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, rocblas_dot_name<T>, n, x, incx, y, incy);

    if(layer_mode & rocblas_layer_mode_log_bench)
        log_bench(handle,
                  "./rocblas-bench -f dot -r",
                  rocblas_precision_string<T>,
                  "-n",
                  n,
                  "--incx",
                  incx,
                  "--incy",
                  incy);

    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle, rocblas_dot_name<T>, "N", n, "incx", incx, "incy", incy);

    if(!x || !y || !result)
        return rocblas_status_invalid_pointer;

    /*
     * Quick return if possible.
     */
    if(n <= 0)
    {
        if(rocblas_pointer_mode_device == handle->pointer_mode)
            RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(*result)));
        else
            *result = 0;
        return rocblas_status_success;
    }

    rocblas_int blocks = (n - 1) / NB + 1;

    auto workspace =
        rocblas_unique_ptr{rocblas::device_malloc(sizeof(T) * blocks), rocblas::device_free};
    if(!workspace)
        return rocblas_status_memory_error;

    auto status =
        rocblas_dot_workspace<T>(handle, n, x, incx, y, incy, result, (T*)workspace.get(), blocks);

    return status;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sdot(rocblas_handle handle,
                            rocblas_int n,
                            const float* x,
                            rocblas_int incx,
                            const float* y,
                            rocblas_int incy,
                            float* result)
{
    return rocblas_dot(handle, n, x, incx, y, incy, result);
}

rocblas_status rocblas_ddot(rocblas_handle handle,
                            rocblas_int n,
                            const double* x,
                            rocblas_int incx,
                            const double* y,
                            rocblas_int incy,
                            double* result)
{
    return rocblas_dot(handle, n, x, incx, y, incy, result);
}

#if 0 //  complex not supported

rocblas_status rocblas_cdotu(rocblas_handle handle,
                             rocblas_int n,
                             const rocblas_float_complex* x,
                             rocblas_int incx,
                             const rocblas_float_complex* y,
                             rocblas_int incy,
                             rocblas_float_complex* result)
{
    return rocblas_dot(handle, n, x, incx, y, incy, result);
}

rocblas_status rocblas_zdotu(rocblas_handle handle,
                             rocblas_int n,
                             const rocblas_double_complex* x,
                             rocblas_int incx,
                             const rocblas_double_complex* y,
                             rocblas_int incy,
                             rocblas_double_complex* result)
{
    return rocblas_dot(handle, n, x, incx, y, incy, result);
}

#endif

} // extern "C"
