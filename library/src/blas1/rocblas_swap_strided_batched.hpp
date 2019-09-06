/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"


template <typename T>
__global__ void rocblas_swap_kernel_strided_batched(rocblas_int n,
                                            T*          x,
                                            rocblas_int shiftx,
                                            rocblas_int incx,
                                            rocblas_int stridex,
                                            T*          y,
                                            rocblas_int shifty,
                                            rocblas_int incy,
                                            rocblas_int stridey)
{
    ssize_t tid = blockIdx.x * blockDim.x + threadIdx.x; // only dim1

    if(tid < n)
    {
        T* xb = x + blockIdx.y * stridex + shiftx;
        T* yb = y + blockIdx.y * stridey + shifty; 
        // in case of negative inc shift pointer to end of data for negative indexing tid*inc
        xb -= (incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0;
        yb -= (incy < 0) ? ptrdiff_t(incy) * (n - 1) : 0;

        auto tmp       = yb[tid * incy];
        yb[tid * incy] = xb[tid * incx];
        xb[tid * incx] = tmp;
    }
}

template <typename>
constexpr char rocblas_swap_strided_batched_name[] = "unknown";
template <>
constexpr char rocblas_swap_strided_batched_name<float>[] = "rocblas_sswap_strided_batched";
template <>
constexpr char rocblas_swap_strided_batched_name<double>[] = "rocblas_dswap_strided_batched";
template <>
constexpr char rocblas_swap_strided_batched_name<rocblas_float_complex>[]
    = "rocblas_cswap_strided_batched";
template <>
constexpr char rocblas_swap_strided_batched_name<rocblas_double_complex>[]
    = "rocblas_zswap_strided_batched";

template <class T>
rocblas_status rocblas_swap_strided_batched(rocblas_handle handle,
                                            rocblas_int    n,
                                            T*             x,
                                            rocblas_int    shiftx,
                                            rocblas_int    incx,
                                            rocblas_int    stridex,
                                            T*             y,
                                            rocblas_int    shifty,
                                            rocblas_int    incy,
                                            rocblas_int    stridey,
                                            rocblas_int    batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle,
                    rocblas_swap_strided_batched_name<T>,
                    n,
                    x,
                    incx,
                    stridex,
                    y,
                    incy,
                    stridey,
                    batch_count);
    if(layer_mode & rocblas_layer_mode_log_bench)
        log_bench(handle,
                    "./rocblas-bench -f swap_strided_batched -r",
                    rocblas_precision_string<T>,
                    "-n",
                    n,
                    "--incx",
                    incx,
                    "--incy",
                    incy,
                    "--stride_x",
                    stridex,
                    "--stride_y",
                    stridey,
                    "--batch",
                    batch_count);
    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle,
                    rocblas_swap_strided_batched_name<T>,
                    "N",
                    n,
                    "incx",
                    incx,
                    "stride_x",
                    stridex,
                    "incy",
                    incy,
                    "stride_y",
                    stridey,
                    "batch",
                    batch_count);

    if(!x || !y)
        return rocblas_status_invalid_pointer;

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    if(!incx || !incy || (stridex < n * abs_incx) || (stridey < n * abs_incy) || (batch_count < 0))
        return rocblas_status_invalid_size;

    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    // Quick return if possible.
    if(n <= 0 || batch_count == 0)
        return rocblas_status_success;

    constexpr int NB = 256;
    hipStream_t rocblas_stream = handle->rocblas_stream;
    rocblas_int blocks         = (n - 1) / NB + 1;
    dim3        grid(blocks, batch_count);
    dim3        threads(NB);

    hipLaunchKernelGGL(rocblas_swap_kernel_strided_batched,
                        grid,
                        threads,
                        0,
                        rocblas_stream,
                        n,
                        x,
                        shiftx,
                        incx,
                        stridex,
                        y,
                        shifty,
                        incy,
                        stridey);

    return rocblas_status_success;
}


/* ============================================================================================ */


