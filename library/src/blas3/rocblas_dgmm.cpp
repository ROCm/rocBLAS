/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_dgmm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_dgmm_name[] = "unknown";
    template <>
    constexpr char rocblas_dgmm_name<float>[] = "rocblas_sdgmm";
    template <>
    constexpr char rocblas_dgmm_name<double>[] = "rocblas_ddgmm";
    template <>
    constexpr char rocblas_dgmm_name<rocblas_float_complex>[] = "rocblas_cdgmm";
    template <>
    constexpr char rocblas_dgmm_name<rocblas_double_complex>[] = "rocblas_zdgmm";

    template <typename T>
    rocblas_status rocblas_dgmm_impl(rocblas_handle handle,
                                     rocblas_side   side,
                                     rocblas_int    m,
                                     rocblas_int    n,
                                     const T*       A,
                                     rocblas_int    lda,
                                     const T*       x,
                                     rocblas_int    incx,
                                     T*             C,
                                     rocblas_int    ldc)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;

        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto side_letter = rocblas_side_letter(side);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle, rocblas_dgmm_name<T>, side, m, n, A, lda, x, incx, C, ldc);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f dgmm -r",
                          rocblas_precision_string<T>,
                          "--side",
                          side_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "--lda",
                          lda,
                          "--incx",
                          incx,
                          "--ldc",
                          ldc);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_dgmm_name<T>,
                            "side",
                            side_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "lda",
                            lda,
                            "incx",
                            incx,
                            "ldc",
                            ldc);
        }

        if(m < 0 || n < 0 || ldc < m || lda < m || incx == 0)
            return rocblas_status_invalid_size;

        if(!m || !n)
            return rocblas_status_success;

        if(!A || !C || !x)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_int    offset_A = 0, offset_x = 0, offset_C = 0, batch_count = 1;
        static constexpr rocblas_stride stride_A = 0, stride_x = 0, stride_C = 0;

        return rocblas_dgmm_template(handle,
                                     side,
                                     m,
                                     n,
                                     A,
                                     offset_A,
                                     lda,
                                     stride_A,
                                     x,
                                     offset_x,
                                     incx,
                                     stride_x,
                                     C,
                                     offset_C,
                                     ldc,
                                     stride_C,
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

#define IMPL(routine_name_, T_)                                                    \
    rocblas_status routine_name_(rocblas_handle handle,                            \
                                 rocblas_side   side,                              \
                                 rocblas_int    m,                                 \
                                 rocblas_int    n,                                 \
                                 const T_*      A,                                 \
                                 rocblas_int    lda,                               \
                                 const T_*      x,                                 \
                                 rocblas_int    incx,                              \
                                 T_*            C,                                 \
                                 rocblas_int    ldc)                               \
    try                                                                            \
    {                                                                              \
        return rocblas_dgmm_impl<T_>(handle, side, m, n, A, lda, x, incx, C, ldc); \
    }                                                                              \
    catch(...)                                                                     \
    {                                                                              \
        return exception_to_rocblas_status();                                      \
    }

IMPL(rocblas_sdgmm, float);
IMPL(rocblas_ddgmm, double);
IMPL(rocblas_cdgmm, rocblas_float_complex);
IMPL(rocblas_zdgmm, rocblas_double_complex);

#undef IMPL

} // extern "C"
