/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.h"
#include "rocblas_symv.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_symv_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_symv_strided_batched_name<float>[] = "rocblas_ssymv_strided_batched";
    template <>
    constexpr char rocblas_symv_strided_batched_name<double>[] = "rocblas_dsymv_strided_batched";
    template <>
    constexpr char rocblas_symv_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_csymv_strided_batched";
    template <>
    constexpr char rocblas_symv_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_zsymv_strided_batched";

    template <typename T, typename U, typename V, typename W>
    rocblas_status rocblas_symv_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     rocblas_int    n,
                                                     const V*       alpha,
                                                     const U*       A,
                                                     rocblas_int    lda,
                                                     rocblas_stride strideA,
                                                     const U*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     const V*       beta,
                                                     W*             y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stridey,
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
                              rocblas_symv_strided_batched_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              strideA,
                              x,
                              incx,
                              stridex,
                              log_trace_scalar_value(beta),
                              y,
                              incy,
                              stridey,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f symv_strided_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--stride_a",
                              strideA,
                              "--incx",
                              incx,
                              "--stride_x",
                              stridex,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--incy",
                              incy,
                              "--stride_y",
                              stridey,
                              "--batch_count",
                              batch_count);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_symv_strided_batched_name<T>,
                              uplo,
                              n,
                              alpha,
                              A,
                              lda,
                              strideA,
                              x,
                              incx,
                              stridex,
                              beta,
                              y,
                              incy,
                              stridey,
                              batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_symv_strided_batched_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "lda",
                            lda,
                            "stride_a",
                            strideA,
                            "incx",
                            incx,
                            "stride_x",
                            stridex,
                            "incy",
                            incy,
                            "stride_y",
                            stridey,
                            "batch_count",
                            batch_count);
        }

        rocblas_status arg_status = rocblas_symv_arg_check<T>(handle,
                                                              uplo,
                                                              n,
                                                              alpha,
                                                              0,
                                                              A,
                                                              0,
                                                              lda,
                                                              strideA,
                                                              x,
                                                              0,
                                                              incx,
                                                              stridex,
                                                              beta,
                                                              0,
                                                              y,
                                                              0,
                                                              incy,
                                                              stridey,
                                                              batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        return rocblas_symv_template<T>(handle,
                                        uplo,
                                        n,
                                        alpha,
                                        0,
                                        A,
                                        0,
                                        lda,
                                        strideA,
                                        x,
                                        0,
                                        incx,
                                        stridex,
                                        beta,
                                        0,
                                        y,
                                        0,
                                        incy,
                                        stridey,
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

#define IMPL(routine_name_, T_)                                    \
    rocblas_status routine_name_(rocblas_handle  handle,           \
                                 rocblas_fill    uplo,             \
                                 rocblas_int     n,                \
                                 const T_* const alpha,            \
                                 const T_*       A,                \
                                 rocblas_int     lda,              \
                                 rocblas_stride  strideA,          \
                                 const T_*       x,                \
                                 rocblas_int     incx,             \
                                 rocblas_stride  stridex,          \
                                 const T_*       beta,             \
                                 T_*             y,                \
                                 rocblas_int     incy,             \
                                 rocblas_stride  stridey,          \
                                 rocblas_int     batch_count)      \
    try                                                            \
    {                                                              \
        return rocblas_symv_strided_batched_impl<T_>(handle,       \
                                                     uplo,         \
                                                     n,            \
                                                     alpha,        \
                                                     A,            \
                                                     lda,          \
                                                     strideA,      \
                                                     x,            \
                                                     incx,         \
                                                     stridex,      \
                                                     beta,         \
                                                     y,            \
                                                     incy,         \
                                                     stridey,      \
                                                     batch_count); \
    }                                                              \
    catch(...)                                                     \
    {                                                              \
        return exception_to_rocblas_status();                      \
    }

IMPL(rocblas_ssymv_strided_batched, float);
IMPL(rocblas_dsymv_strided_batched, double);
IMPL(rocblas_csymv_strided_batched, rocblas_float_complex);
IMPL(rocblas_zsymv_strided_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
