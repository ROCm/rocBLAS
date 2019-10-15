#pragma once

#include "rocblas_batched_arrays.h"

struct rocblas_reduction_utils
{
    template <typename U>
    static void logging_profile(rocblas_handle handle,
                                rocblas_int    n,
                                U              x,
                                rocblas_int    incx,
                                rocblas_stride stridex,
                                rocblas_int    batch_count,
                                const char*    name)
    {
        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            if(is_strided_batched_arrays<U>::value)
            {
                if(1 == batch_count && 0 == stridex)
                {
                    log_profile(handle, name, "N", n, "incx", incx);
                }
                else
                {
                    log_profile(handle,
                                name,
                                "N",
                                n,
                                "incx",
                                incx,
                                "stride_x",
                                stridex,
                                "batch",
                                batch_count);
                }
            }
            else
            {
                log_profile(handle, name, "N", n, "incx", incx, "batch", batch_count);
            }
        }
    };
    template <typename U>
    static void logging_trace(rocblas_handle handle,
                              rocblas_int    n,
                              U              x,
                              rocblas_int    incx,
                              rocblas_stride stridex,
                              rocblas_int    batch_count,
                              const char*    name)
    {
        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
        {
            if(is_strided_batched_arrays<U>::value)
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
            else
            {
                log_trace(handle, name, n, x, incx, batch_count);
            }
        }
    };

    template <typename U>
    static void logging_bench(rocblas_handle handle,
                              rocblas_int    n,
                              U              x,
                              rocblas_int    incx,
                              rocblas_stride stridex,
                              rocblas_int    batch_count,
                              const char*    name)
    {
        auto precision_string = rocblas_precision_string<batched_data_t<U>>;
        auto layer_mode       = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_bench)
        {
            if(is_strided_batched_arrays<U>::value)
            {
                if(1 == batch_count && 0 == stridex)
                {
                    log_bench(handle,
                              "./rocblas-bench",
                              "-f",
                              name,
                              "-r",
                              precision_string,
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
                              precision_string,
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
            else
            {
                log_bench(handle,
                          "./rocblas-bench",
                          "-f",
                          name,
                          "-r",
                          precision_string,
                          "-n",
                          n,
                          "--incx",
                          incx,
                          "--batch",
                          batch_count);
            }
        }
    };
};
