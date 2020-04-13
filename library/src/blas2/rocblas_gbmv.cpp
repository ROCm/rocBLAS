/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_gbmv.hpp"
#include "logging.h"

namespace
{
    template <typename>
    constexpr char rocblas_gbmv_name[] = "unknown";
    template <>
    constexpr char rocblas_gbmv_name<float>[] = "rocblas_sgbmv";
    template <>
    constexpr char rocblas_gbmv_name<double>[] = "rocblas_dgbmv";
    template <>
    constexpr char rocblas_gbmv_name<rocblas_float_complex>[] = "rocblas_cgbmv";
    template <>
    constexpr char rocblas_gbmv_name<rocblas_double_complex>[] = "rocblas_zgbmv";

    template <typename T>
    rocblas_status rocblas_gbmv_impl(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       kl,
                                     rocblas_int       ku,
                                     const T*          alpha,
                                     const T*          A,
                                     rocblas_int       lda,
                                     const T*          x,
                                     rocblas_int       incx,
                                     const T*          beta,
                                     T*                y,
                                     rocblas_int       incy)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto transA_letter = rocblas_transpose_letter(transA);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_gbmv_name<T>,
                              transA,
                              m,
                              n,
                              kl,
                              ku,
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
                              "./rocblas-bench -f gbmv -r",
                              rocblas_precision_string<T>,
                              "--transposeA",
                              transA_letter,
                              "-m",
                              m,
                              "-n",
                              n,
                              "--kl",
                              kl,
                              "--ku",
                              ku,
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
                              rocblas_gbmv_name<T>,
                              transA,
                              m,
                              n,
                              kl,
                              ku,
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
                            rocblas_gbmv_name<T>,
                            "transA",
                            transA_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "kl",
                            kl,
                            "ku",
                            ku,
                            "lda",
                            lda,
                            "incx",
                            incx,
                            "incy",
                            incy);
        }

        if(m < 0 || n < 0 || lda < kl + ku + 1 || !incx || !incy || kl < 0 || ku < 0)
            return rocblas_status_invalid_size;

        if(!m || !n)
            return rocblas_status_success;

        // TODO: LAPACK does a alpha==0 && beta==1 quick return check here.
        //       The following code is up for discussion if we want to include it here
        //       or after the invalid_pointer check as we currently do (in rocblas_gbmv.hpp).

        // if(handle->pointer_mode == rocblas_pointer_mode_host && alpha && beta)
        // {
        //     if(!*alpha && *beta == 1)
        //         return rocblas_status_success;
        // }

        if(!A || !x || !y || !alpha || !beta)
            return rocblas_status_invalid_pointer;

        return rocblas_gbmv_template(handle,
                                     transA,
                                     m,
                                     n,
                                     kl,
                                     ku,
                                     alpha,
                                     A,
                                     0,
                                     lda,
                                     0,
                                     x,
                                     0,
                                     incx,
                                     0,
                                     beta,
                                     y,
                                     0,
                                     incy,
                                     0,
                                     1);
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

rocblas_status rocblas_sgbmv(rocblas_handle    handle,
                             rocblas_operation transA,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       kl,
                             rocblas_int       ku,
                             const float*      alpha,
                             const float*      A,
                             rocblas_int       lda,
                             const float*      x,
                             rocblas_int       incx,
                             const float*      beta,
                             float*            y,
                             rocblas_int       incy)
try
{
    return rocblas_gbmv_impl(handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dgbmv(rocblas_handle    handle,
                             rocblas_operation transA,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       kl,
                             rocblas_int       ku,
                             const double*     alpha,
                             const double*     A,
                             rocblas_int       lda,
                             const double*     x,
                             rocblas_int       incx,
                             const double*     beta,
                             double*           y,
                             rocblas_int       incy)
try
{
    return rocblas_gbmv_impl(handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_cgbmv(rocblas_handle               handle,
                             rocblas_operation            transA,
                             rocblas_int                  m,
                             rocblas_int                  n,
                             rocblas_int                  kl,
                             rocblas_int                  ku,
                             const rocblas_float_complex* alpha,
                             const rocblas_float_complex* A,
                             rocblas_int                  lda,
                             const rocblas_float_complex* x,
                             rocblas_int                  incx,
                             const rocblas_float_complex* beta,
                             rocblas_float_complex*       y,
                             rocblas_int                  incy)
try
{
    return rocblas_gbmv_impl(handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_zgbmv(rocblas_handle                handle,
                             rocblas_operation             transA,
                             rocblas_int                   m,
                             rocblas_int                   n,
                             rocblas_int                   kl,
                             rocblas_int                   ku,
                             const rocblas_double_complex* alpha,
                             const rocblas_double_complex* A,
                             rocblas_int                   lda,
                             const rocblas_double_complex* x,
                             rocblas_int                   incx,
                             const rocblas_double_complex* beta,
                             rocblas_double_complex*       y,
                             rocblas_int                   incy)
try
{
    return rocblas_gbmv_impl(handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
