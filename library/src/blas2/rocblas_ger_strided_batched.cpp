/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_ger.hpp"
#include "utility.h"

namespace
{
    template <bool, typename>
    constexpr char rocblas_ger_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_ger_strided_batched_name<false, float>[]
        = "rocblas_sger_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_name<false, double>[]
        = "rocblas_dger_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_name<false, rocblas_float_complex>[]
        = "rocblas_cgeru_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_name<false, rocblas_double_complex>[]
        = "rocblas_zgeru_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_name<true, rocblas_float_complex>[]
        = "rocblas_cgerc_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_name<true, rocblas_double_complex>[]
        = "rocblas_zgerc_strided_batched";

    template <bool, typename>
    constexpr char rocblas_ger_strided_batched_fn_name[] = "unknown";
    template <>
    constexpr char rocblas_ger_strided_batched_fn_name<false, float>[] = "ger_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_fn_name<false, double>[] = "ger_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_fn_name<false, rocblas_float_complex>[]
        = "geru_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_fn_name<false, rocblas_double_complex>[]
        = "geru_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_fn_name<true, rocblas_float_complex>[]
        = "gerc_strided_batched";
    template <>
    constexpr char rocblas_ger_strided_batched_fn_name<true, rocblas_double_complex>[]
        = "gerc_strided_batched";

    template <bool CONJ, typename T>
    rocblas_status rocblas_ger_strided_batched_impl(rocblas_handle handle,
                                                    rocblas_int    m,
                                                    rocblas_int    n,
                                                    const T*       alpha,
                                                    const T*       x,
                                                    rocblas_int    incx,
                                                    rocblas_stride stridex,
                                                    const T*       y,
                                                    rocblas_int    incy,
                                                    rocblas_stride stridey,
                                                    T*             A,
                                                    rocblas_int    lda,
                                                    rocblas_stride strideA,
                                                    rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_ger_strided_batched_name<CONJ, T>,
                          m,
                          n,
                          log_trace_scalar_value(alpha),
                          x,
                          incx,
                          stridex,
                          y,
                          incy,
                          stridey,
                          A,
                          lda,
                          strideA,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f",
                          rocblas_ger_strided_batched_fn_name<CONJ, T>,
                          "-r",
                          rocblas_precision_string<T>,
                          "-m",
                          m,
                          "-n",
                          n,
                          LOG_BENCH_SCALAR_VALUE(alpha),
                          "--incx",
                          incx,
                          "--stride_x",
                          stridex,
                          "--incy",
                          incy,
                          "--stride_y",
                          stridey,
                          "--lda",
                          lda,
                          "--stride_a",
                          strideA,
                          "--batch_count",
                          batch_count);
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_ger_strided_batched_name<CONJ, T>,
                          m,
                          n,
                          alpha,
                          x,
                          incx,
                          stridex,
                          y,
                          incy,
                          stridey,
                          A,
                          lda,
                          strideA,
                          batch_count);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_ger_strided_batched_name<CONJ, T>,
                        "M",
                        m,
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
                        "lda",
                        lda,
                        "stride_a",
                        strideA,
                        "batch_count",
                        batch_count);

        rocblas_status arg_status = rocblas_ger_arg_check<CONJ, T>(m,
                                                                   n,
                                                                   alpha,
                                                                   0,
                                                                   x,
                                                                   0,
                                                                   incx,
                                                                   stridex,
                                                                   y,
                                                                   0,
                                                                   incy,
                                                                   stridey,
                                                                   A,
                                                                   0,
                                                                   lda,
                                                                   strideA,
                                                                   batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        rocblas_ger_template<CONJ, T>(handle,
                                      m,
                                      n,
                                      alpha,
                                      0,
                                      x,
                                      0,
                                      incx,
                                      stridex,
                                      y,
                                      0,
                                      incy,
                                      stridey,
                                      A,
                                      0,
                                      lda,
                                      strideA,
                                      batch_count);

        return rocblas_status_success;
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

#define IMPL(routine_name_, CONJ_, T_)                                   \
    rocblas_status routine_name_(rocblas_handle handle,                  \
                                 rocblas_int    m,                       \
                                 rocblas_int    n,                       \
                                 const T_*      alpha,                   \
                                 const T_*      x,                       \
                                 rocblas_int    incx,                    \
                                 rocblas_stride stridex,                 \
                                 const T_*      y,                       \
                                 rocblas_int    incy,                    \
                                 rocblas_stride stridey,                 \
                                 T_*            A,                       \
                                 rocblas_int    lda,                     \
                                 rocblas_stride strideA,                 \
                                 rocblas_int    batch_count)             \
    try                                                                  \
    {                                                                    \
        return rocblas_ger_strided_batched_impl<CONJ_, T_>(handle,       \
                                                           m,            \
                                                           n,            \
                                                           alpha,        \
                                                           x,            \
                                                           incx,         \
                                                           stridex,      \
                                                           y,            \
                                                           incy,         \
                                                           stridey,      \
                                                           A,            \
                                                           lda,          \
                                                           strideA,      \
                                                           batch_count); \
    }                                                                    \
    catch(...)                                                           \
    {                                                                    \
        return exception_to_rocblas_status();                            \
    }

IMPL(rocblas_sger_strided_batched, false, float);
IMPL(rocblas_dger_strided_batched, false, double);
IMPL(rocblas_cgeru_strided_batched, false, rocblas_float_complex);
IMPL(rocblas_zgeru_strided_batched, false, rocblas_double_complex);
IMPL(rocblas_cgerc_strided_batched, true, rocblas_float_complex);
IMPL(rocblas_zgerc_strided_batched, true, rocblas_double_complex);

#undef IMPL

} // extern "C"
