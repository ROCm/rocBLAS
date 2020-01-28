/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_symv.hpp"
#include "logging.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_symv_name[] = "unknown";
    template <>
    constexpr char rocblas_symv_name<float>[] = "rocblas_ssymv";
    template <>
    constexpr char rocblas_symv_name<double>[] = "rocblas_dsymv";
    template <>
    constexpr char rocblas_symv_name<rocblas_float_complex>[] = "rocblas_csymv";
    template <>
    constexpr char rocblas_symv_name<rocblas_double_complex>[] = "rocblas_zsymv";

    template <typename T, typename U, typename V, typename W>
    rocblas_status rocblas_symv_impl(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const V*       alpha,
                                     const U*       A,
                                     rocblas_int    lda,
                                     const U*       x,
                                     rocblas_int    incx,
                                     const V*       beta,
                                     W*             y,
                                     rocblas_int    incy)
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
                              rocblas_symv_name<T>,
                              uplo,
                              n,
                              log_trace_scalar_value(alpha),
                              A,
                              lda,
                              x,
                              incx,
                              log_trace_scalar_value(beta),
                              y,
                              incy);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f symv -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(alpha),
                              "--lda",
                              lda,
                              "--incx",
                              incx,
                              LOG_BENCH_SCALAR_VALUE(beta),
                              "--incy",
                              incy);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_symv_name<T>,
                              uplo,
                              n,
                              alpha,
                              A,
                              lda,
                              x,
                              incx,
                              beta,
                              y,
                              incy);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_symv_name<T>,
                            "uplo",
                            uplo_letter,
                            "N",
                            n,
                            "lda",
                            lda,
                            "incx",
                            incx,
                            "incy",
                            incy);
        }

        rocblas_status arg_status = rocblas_symv_arg_check<T>(
            handle, uplo, n, alpha, 0, A, 0, lda, 0, x, 0, incx, 0, beta, 0, y, 0, incy, 0, 1);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        return rocblas_symv_template<T>(
            handle, uplo, n, alpha, 0, A, 0, lda, 0, x, 0, incx, 0, beta, 0, y, 0, incy, 0, 1);
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

#define IMPL(routine_name_, T_)                                                               \
    rocblas_status routine_name_(rocblas_handle handle,                                       \
                                 rocblas_fill   uplo,                                         \
                                 rocblas_int    n,                                            \
                                 const T_*      alpha,                                        \
                                 const T_*      A,                                            \
                                 rocblas_int    lda,                                          \
                                 const T_*      x,                                            \
                                 rocblas_int    incx,                                         \
                                 const T_*      beta,                                         \
                                 T_*            y,                                            \
                                 rocblas_int    incy)                                         \
    try                                                                                       \
    {                                                                                         \
        return rocblas_symv_impl<T_>(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy); \
    }                                                                                         \
    catch(...)                                                                                \
    {                                                                                         \
        return exception_to_rocblas_status();                                                 \
    }

IMPL(rocblas_ssymv, float);
IMPL(rocblas_dsymv, double);
IMPL(rocblas_csymv, rocblas_float_complex);
IMPL(rocblas_zsymv, rocblas_double_complex);

#undef IMPL

} // extern "C"
