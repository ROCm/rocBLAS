#pragma once

#include "rocblas_batched_arrays.h"

template <typename U>
struct rocblas_utils
{
    static void logging_trace(rocblas_handle handle,
                          rocblas_int    n,
                          U              x,
                          rocblas_int    incx,
                          rocblas_stride stridex,
                          rocblas_int    batch_count,
                          const char*    name);

    static void logging_bench(rocblas_handle handle,
                          rocblas_int    n,
                          U              x,
                          rocblas_int    incx,
                          rocblas_stride stridex,
                          rocblas_int    batch_count,
                          const char*    name);

    static void logging_profile(rocblas_handle handle,
                            rocblas_int    n,
                            U              x,
                            rocblas_int    incx,
                            rocblas_stride stridex,
                            rocblas_int    batch_count,
                            const char*    name);
};

template <typename T>
struct rocblas_utils<const_batched_arrays<T>>
{

    static void logging_profile(rocblas_handle          handle,
                            rocblas_int             n,
                            const_batched_arrays<T> x,
                            rocblas_int             incx,
                            rocblas_stride          stridex,
                            rocblas_int             batch_count,
                            const char*             name)
    {
        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            log_profile(
                handle, name, "N", n, "incx", incx, "stride_x", stridex, "batch", batch_count);
        }
    };

    static void logging_trace(rocblas_handle          handle,
                          rocblas_int             n,
                          const_batched_arrays<T> x,
                          rocblas_int             incx,
                          rocblas_stride          stridex,
                          rocblas_int             batch_count,
                          const char*             name)
    {
        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
        {
            log_trace(handle, name, n, x, incx, batch_count);
        }
    };

    static void logging_bench(rocblas_handle          handle,
                          rocblas_int             n,
                          const_batched_arrays<T> x,
                          rocblas_int             incx,
                          rocblas_stride          stridex,
                          rocblas_int             batch_count,
                          const char*             name)
    {
        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_bench)
        {
            log_bench(handle,
                      "./rocblas-bench",
                      "-f",
                      name,
                      "-r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--stride_x",
                      stridex,
                      "--batch",
                      batch_count);
        }
    };
};

template <typename T>
struct rocblas_utils<const_strided_batched_arrays<T>>
{

    static void logging_profile(rocblas_handle                  handle,
                            rocblas_int                     n,
                            const_strided_batched_arrays<T> x,
                            rocblas_int                     incx,
                            rocblas_stride                  stridex,
                            rocblas_int                     batch_count,
                            const char*                     name)
    {
        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            if(1 == batch_count && 0 == stridex)
            {
                log_profile(handle, name, "N", n, "incx", incx);
            }
            else
            {
                log_profile(
                    handle, name, "N", n, "incx", incx, "stride_x", stridex, "batch", batch_count);
            }
        }
    };

    static void logging_bench(rocblas_handle                  handle,
                          rocblas_int                     n,
                          const_strided_batched_arrays<T> x,
                          rocblas_int                     incx,
                          rocblas_stride                  stridex,
                          rocblas_int                     batch_count,
                          const char*                     name)
    {
        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_bench)
        {
            if(1 == batch_count && 0 == stridex)
            {
                log_bench(handle,
                          "./rocblas-bench",
                          "-f",
                          name,
                          "-r",
                          rocblas_precision_string<T>,
                          "-n",
                          n,
                          "--incx",
                          incx);
            }
            else
            {
                log_bench(handle,
                          "./rocblas-bench",
                          "-f",
                          name,
                          "-r",
                          rocblas_precision_string<T>,
                          "-n",
                          n,
                          "--incx",
                          incx,
                          "--stride_x",
                          stridex,
                          "--batch",
                          batch_count);
            }
        }
    };

    static void logging_trace(rocblas_handle                  handle,
                          rocblas_int                     n,
                          const_strided_batched_arrays<T> x,
                          rocblas_int                     incx,
                          rocblas_stride                  stridex,
                          rocblas_int                     batch_count,
                          const char*                     name)
    {
        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
        {
            if(1 == batch_count && 0 == stridex)
            {
                log_trace(handle, name, n, x, incx);
            }
            else
            {
                log_trace(handle, name, n, x, incx, stridex, batch_count);
            }
        }
    };
};
