/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.h"
#include "rocblas_sbmv.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_sbmv_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_sbmv_batched_name<float>[] = "rocblas_ssbmv_batched";
    template <>
    constexpr char rocblas_sbmv_batched_name<double>[] = "rocblas_dsbmv_batched";

    template <typename T, typename U, typename V, typename W>
    rocblas_status rocblas_sbmv_batched_impl(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             rocblas_int    k,
                                             const V*       alpha,
                                             const U*       A,
                                             rocblas_int    lda,
                                             const U*       x,
                                             rocblas_int    incx,
                                             const V*       beta,
                                             W*             y,
                                             rocblas_int    incy,
                                             rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter = rocblas_fill_letter(uplo);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_sbmv_batched_name<T>,
                              uplo,
                              n,
                              k,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              x,
                              incx,
                              log_trace_scalar_value(beta),
                              y,
                              incy,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f sbmv_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              "-k",
                              k,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--incx",
                              incx,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--incy",
                              incy,
                              "--batch_count",
                              batch_count);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_sbmv_batched_name<T>,
                              uplo,
                              n,
                              k,
                              alpha,
                              A,
                              lda,
                              x,
                              incx,
                              beta,
                              y,
                              incy,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_sbmv_batched_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "K",
                            k,
                            "lda",
                            lda,
                            "incx",
                            incx,
                            "incy",
                            incy,
                            "batch_count",
                            batch_count);
        }

        rocblas_status arg_status = rocblas_sbmv_arg_check<T>(handle,
                                                              uplo,
                                                              n,
                                                              k,
                                                              alpha,
                                                              0,
                                                              A,
                                                              0,
                                                              lda,
                                                              0,
                                                              x,
                                                              0,
                                                              incx,
                                                              0,
                                                              beta,
                                                              0,
                                                              y,
                                                              0,
                                                              incy,
                                                              0,
                                                              batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        return rocblas_sbmv_template<T>(handle,
                                        uplo,
                                        n,
                                        k,
                                        alpha,
                                        0,
                                        A,
                                        0,
                                        lda,
                                        0,
                                        x,
                                        0,
                                        incx,
                                        0,
                                        beta,
                                        0,
                                        y,
                                        0,
                                        incy,
                                        0,
                                        batch_count);
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, T_)                                                      \
    rocblas_status routine_name_(rocblas_handle  handle,                             \
                                 rocblas_fill    uplo,                               \
                                 rocblas_int     n,                                  \
                                 rocblas_int     k,                                  \
                                 const T_* const alpha,                              \
                                 const T_* const A[],                                \
                                 rocblas_int     lda,                                \
                                 const T_* const x[],                                \
                                 rocblas_int     incx,                               \
                                 const T_*       beta,                               \
                                 T_*             y[],                                \
                                 rocblas_int     incy,                               \
                                 rocblas_int     batch_count)                        \
    try                                                                              \
    {                                                                                \
        return rocblas_sbmv_batched_impl<T_>(                                        \
            handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy, batch_count); \
    }                                                                                \
    catch(...)                                                                       \
    {                                                                                \
        return exception_to_rocblas_status();                                        \
    }

IMPL(rocblas_ssbmv_batched, float);
IMPL(rocblas_dsbmv_batched, double);

#undef IMPL

} // extern "C"
