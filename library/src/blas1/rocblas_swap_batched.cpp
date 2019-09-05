/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    constexpr int NB = 256;

    template <typename T>
    __global__ void
        swap_kernel_batched(rocblas_int n, T* x[], rocblas_int incx, T* y[], rocblas_int incy)
    {
        ssize_t tid = blockIdx.x * blockDim.x + threadIdx.x; // only dim1

        if(tid < n)
        {
            T* xb = x[blockIdx.y];
            T* yb = y[blockIdx.y];
            // in case of negative inc shift pointer to end of data for negative indexing tid*inc
            xb -= (incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0;
            yb -= (incy < 0) ? ptrdiff_t(incy) * (n - 1) : 0;

            auto tmp       = yb[tid * incy];
            yb[tid * incy] = xb[tid * incx];
            xb[tid * incx] = tmp;
        }
    }

    template <typename>
    constexpr char rocblas_swap_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_swap_batched_name<float>[] = "rocblas_sswap_batched";
    template <>
    constexpr char rocblas_swap_batched_name<double>[] = "rocblas_dswap_batched";
    template <>
    constexpr char rocblas_swap_batched_name<rocblas_float_complex>[] = "rocblas_cswap_batched";
    template <>
    constexpr char rocblas_swap_batched_name<rocblas_double_complex>[] = "rocblas_zswap_batched";

    template <class T>
    rocblas_status rocblas_swap_batched(rocblas_handle handle,
                                        rocblas_int    n,
                                        T*             x[],
                                        rocblas_int    incx,
                                        T*             y[],
                                        rocblas_int    incy,
                                        rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_swap_batched_name<T>, n, x, incx, y, incy, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f swap_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy,
                      "--batch",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_swap_batched_name<T>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "batch",
                        batch_count);

        if(!x || !y)
            return rocblas_status_invalid_pointer;

        if(batch_count < 0)
            return rocblas_status_invalid_size;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Quick return if possible.
        if(n <= 0 || batch_count == 0)
            return rocblas_status_success;

        hipStream_t rocblas_stream = handle->rocblas_stream;
        rocblas_int blocks         = (n - 1) / NB + 1;
        dim3        grid(blocks, batch_count);
        dim3        threads(NB);

        hipLaunchKernelGGL(
            swap_kernel_batched, grid, threads, 0, rocblas_stream, n, x, incx, y, incy);

        return rocblas_status_success;
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sswap_batched(rocblas_handle handle,
                                     rocblas_int    n,
                                     float*         x[],
                                     rocblas_int    incx,
                                     float*         y[],
                                     rocblas_int    incy,
                                     rocblas_int    batch_count)
{
    return rocblas_swap_batched(handle, n, x, incx, y, incy, batch_count);
}

rocblas_status rocblas_dswap_batched(rocblas_handle handle,
                                     rocblas_int    n,
                                     double*        x[],
                                     rocblas_int    incx,
                                     double*        y[],
                                     rocblas_int    incy,
                                     rocblas_int    batch_count)
{
    return rocblas_swap_batched(handle, n, x, incx, y, incy, batch_count);
}

rocblas_status rocblas_cswap_batched(rocblas_handle         handle,
                                     rocblas_int            n,
                                     rocblas_float_complex* x[],
                                     rocblas_int            incx,
                                     rocblas_float_complex* y[],
                                     rocblas_int            incy,
                                     rocblas_int            batch_count)
{
    return rocblas_swap_batched(handle, n, x, incx, y, incy, batch_count);
}

rocblas_status rocblas_zswap_batched(rocblas_handle          handle,
                                     rocblas_int             n,
                                     rocblas_double_complex* x[],
                                     rocblas_int             incx,
                                     rocblas_double_complex* y[],
                                     rocblas_int             incy,
                                     rocblas_int             batch_count)
{
    return rocblas_swap_batched(handle, n, x, incx, y, incy, batch_count);
}

} // extern "C"
