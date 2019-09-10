/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"


template <typename T, typename U>
__global__ void rocblas_syr_batched_kernel(rocblas_fill uplo,
                            rocblas_int  n,
                            U            alpha_device_host,
                            const T* const __restrict__ xvec[],
                            rocblas_int shiftx,
                            rocblas_int incx,
                            T*          Avec[],
                            rocblas_int shiftA,
                            rocblas_int lda)
{
    auto        alpha = load_scalar(alpha_device_host);
    rocblas_int tx    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty    = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    const T* x = xvec[hipBlockIdx_z] + shiftx;
    T* A = Avec[hipBlockIdx_z] + shiftA;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    x -= (incx < 0) ? ptrdiff_t(incx) * (n - 1) : 0;

    if(uplo == rocblas_fill_lower ? tx < n && ty <= tx : ty < n && tx <= ty)
        A[tx + lda * ty] += alpha * x[tx * incx] * x[ty * incx];
}

template <typename>
constexpr char rocblas_syr_batched_name[] = "unknown";
template <>
constexpr char rocblas_syr_batched_name<float>[] = "rocblas_ssyr_batched";
template <>
constexpr char rocblas_syr_batched_name<double>[] = "rocblas_dsyr_batched";

template <typename T>
rocblas_status rocblas_syr_batched(rocblas_handle handle,
                            rocblas_fill   uplo,
                            rocblas_int    n,
                            const T*       alpha,
                            const T* const x[],
                            rocblas_int    shiftx,
                            rocblas_int    incx,
                            T*             A[],
                            rocblas_int    shiftA,
                            rocblas_int    lda,
                            rocblas_int    batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
    if(!alpha)
        return rocblas_status_invalid_pointer;
    auto layer_mode = handle->layer_mode;
    if(layer_mode
        & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
            | rocblas_layer_mode_log_profile))
    {
        auto uplo_letter = rocblas_fill_letter(uplo);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_syr_batched_name<T>, uplo, n, *alpha, x, incx, A, lda);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                            "./rocblas-bench -f syr_batched -r",
                            rocblas_precision_string<T>,
                            "--uplo",
                            uplo_letter,
                            "-n",
                            n,
                            "--alpha",
                            *alpha,
                            "--incx",
                            incx,
                            "--lda",
                            lda,
                            "--batch",
                            batch_count);
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_syr_batched_name<T>, uplo, n, alpha, x, incx, A, lda, batch_count);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_syr_batched_name<T>,
                        "uplo",
                        uplo_letter,
                        "N",
                        n,
                        "incx",
                        incx,
                        "lda",
                        lda,
                        "batch",
                        batch_count);
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    if(!x || !A)
        return rocblas_status_invalid_pointer;
    if(n < 0 || !incx || lda < n || lda < 1 || batch_count < 0)
        return rocblas_status_invalid_size;

    // Quick return if possible. Not Argument error
    if(!n || batch_count == 0)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    static constexpr int GEMV_DIM_X = 128;
    static constexpr int GEMV_DIM_Y = 8;
    rocblas_int          blocksX    = (n - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / GEMV_DIM_Y + 1;

    dim3 syr_batched_grid(blocksX, blocksY, batch_count);
    dim3 syr_batched_threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rocblas_syr_batched_kernel,
                            syr_batched_grid,
                            syr_batched_threads,
                            0,
                            rocblas_stream,
                            uplo,
                            n,
                            alpha,
                            x,
                            shiftx,
                            incx,
                            A,
                            shiftA,
                            lda);
    else
        hipLaunchKernelGGL(rocblas_syr_batched_kernel,
                            syr_batched_grid,
                            syr_batched_threads,
                            0,
                            rocblas_stream,
                            uplo,
                            n,
                            *alpha,
                            x,
                            shiftx,
                            incx,
                            A,
                            shiftA,
                            lda);

    return rocblas_status_success;
}


