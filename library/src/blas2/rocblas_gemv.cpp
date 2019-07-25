/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_gemv.hpp"
#include "gemv_device.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_gemv_name[] = "unknown";
    template <>
    constexpr char rocblas_gemv_name<float>[] = "rocblas_sgemv";
    template <>
    constexpr char rocblas_gemv_name<double>[] = "rocblas_dgemv";
    template <>
    constexpr char rocblas_gemv_name<rocblas_float_complex>[] = "rocblas_cgemv";
    template <>
    constexpr char rocblas_gemv_name<rocblas_double_complex>[] = "rocblas_zgemv";

    template <typename T>
    rocblas_status rocblas_gemv_impl(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     rocblas_int       n,
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

        if(!alpha || !beta)
            return rocblas_status_invalid_pointer;

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
                              rocblas_gemv_name<T>,
                              transA,
                              m,
                              n,
                              *alpha,
                              A,
                              lda,
                              x,
                              incx,
                              *beta,
                              y,
                              incy);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::stringstream alphass;
                    alphass << "--alpha " << std::real(*alpha)
                            << (std::imag(*alpha) != 0
                                    ? (" --alphai " + std::to_string(std::imag(*alpha)))
                                    : "");
                    log_bench(handle,
                              "./rocblas-bench -f gemv -r",
                              rocblas_precision_string<T>,
                              "--transposeA",
                              transA_letter,
                              "-m",
                              m,
                              "-n",
                              n,
                              alphass.str(),
                              "--lda",
                              lda,
                              "--incx",
                              incx,
                              "--beta",
                              *beta,
                              "--incy",
                              incy);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_gemv_name<T>,
                              transA,
                              m,
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
                            rocblas_gemv_name<T>,
                            "transA",
                            transA_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "lda",
                            lda,
                            "incx",
                            incx,
                            "incy",
                            incy);
        }

        if(!A || !x || !y)
            return rocblas_status_invalid_pointer;
        if(m < 0 || n < 0 || lda < m || lda < 1 || !incx || !incy)
            return rocblas_status_invalid_size;

        return rocblas_gemv_template(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

rocblas_status rocblas_sgemv(rocblas_handle    handle,
                             rocblas_operation transA,
                             rocblas_int       m,
                             rocblas_int       n,
                             const float*      alpha,
                             const float*      A,
                             rocblas_int       lda,
                             const float*      x,
                             rocblas_int       incx,
                             const float*      beta,
                             float*            y,
                             rocblas_int       incy)
{
    return rocblas_gemv_impl(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

rocblas_status rocblas_dgemv(rocblas_handle    handle,
                             rocblas_operation transA,
                             rocblas_int       m,
                             rocblas_int       n,
                             const double*     alpha,
                             const double*     A,
                             rocblas_int       lda,
                             const double*     x,
                             rocblas_int       incx,
                             const double*     beta,
                             double*           y,
                             rocblas_int       incy)
{
    return rocblas_gemv_impl(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

rocblas_status rocblas_cgemv(rocblas_handle               handle,
                             rocblas_operation            transA,
                             rocblas_int                  m,
                             rocblas_int                  n,
                             const rocblas_float_complex* alpha,
                             const rocblas_float_complex* A,
                             rocblas_int                  lda,
                             const rocblas_float_complex* x,
                             rocblas_int                  incx,
                             const rocblas_float_complex* beta,
                             rocblas_float_complex*       y,
                             rocblas_int                  incy)
{
    return rocblas_gemv_impl(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

rocblas_status rocblas_zgemv(rocblas_handle                handle,
                             rocblas_operation             transA,
                             rocblas_int                   m,
                             rocblas_int                   n,
                             const rocblas_double_complex* alpha,
                             const rocblas_double_complex* A,
                             rocblas_int                   lda,
                             const rocblas_double_complex* x,
                             rocblas_int                   incx,
                             const rocblas_double_complex* beta,
                             rocblas_double_complex*       y,
                             rocblas_int                   incy)
{
    return rocblas_gemv_impl(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

} // extern "C"
