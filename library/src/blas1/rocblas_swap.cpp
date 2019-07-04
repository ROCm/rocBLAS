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
    __global__ void swap_kernel(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
    {
        ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        if(tid < n)
        {
            auto tmp      = y[tid * incy];
            y[tid * incy] = x[tid * incx];
            x[tid * incx] = tmp;
        }
    }

    template <typename>
    constexpr char rocblas_swap_name[] = "unknown";
    template <>
    constexpr char rocblas_swap_name<float>[] = "rocblas_sswap";
    template <>
    constexpr char rocblas_swap_name<double>[] = "rocblas_dswap";
    template <>
    constexpr char rocblas_swap_name<rocblas_half>[] = "rocblas_hswap";
    template <>
    constexpr char rocblas_swap_name<rocblas_float_complex>[] = "rocblas_cswap";
    template <>
    constexpr char rocblas_swap_name<rocblas_double_complex>[] = "rocblas_zswap";

    template <class T>
    rocblas_status rocblas_swap(
        rocblas_handle handle, rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_swap_name<T>, n, x, incx, y, incy);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f swap -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_swap_name<T>, "N", n, "incx", incx, "incy", incy);

        if(!x || !y)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Quick return if possible.
        if(n <= 0)
            return rocblas_status_success;

        hipStream_t rocblas_stream = handle->rocblas_stream;
        int         blocks         = (n - 1) / NB + 1;
        dim3        grid(blocks);
        dim3        threads(NB);

        if(incx < 0)
            x -= ptrdiff_t(incx) * (n - 1);
        if(incy < 0)
            y -= ptrdiff_t(incy) * (n - 1);

        hipLaunchKernelGGL(swap_kernel, grid, threads, 0, rocblas_stream, n, x, incx, y, incy);

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

rocblas_status rocblas_sswap(
    rocblas_handle handle, rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    return rocblas_swap(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_dswap(
    rocblas_handle handle, rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    return rocblas_swap(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_cswap(rocblas_handle         handle,
                             rocblas_int            n,
                             rocblas_float_complex* x,
                             rocblas_int            incx,
                             rocblas_float_complex* y,
                             rocblas_int            incy)
{
    return rocblas_swap(handle, n, x, incx, y, incy);
}

rocblas_status rocblas_zswap(rocblas_handle          handle,
                             rocblas_int             n,
                             rocblas_double_complex* x,
                             rocblas_int             incx,
                             rocblas_double_complex* y,
                             rocblas_int             incy)
{
    return rocblas_swap(handle, n, x, incx, y, incy);
}

} // extern "C"
