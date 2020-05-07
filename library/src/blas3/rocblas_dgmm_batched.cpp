/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_dgmm.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_dgmm_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_dgmm_batched_name<float>[] = "rocblas_sdgmm_batched";
    template <>
    constexpr char rocblas_dgmm_batched_name<double>[] = "rocblas_ddgmm_batched";
    template <>
    constexpr char rocblas_dgmm_batched_name<rocblas_float_complex>[] = "rocblas_cdgmm_batched";
    template <>
    constexpr char rocblas_dgmm_batched_name<rocblas_double_complex>[] = "rocblas_zdgmm_batched";

    template <typename T>
    rocblas_status rocblas_dgmm_batched_impl(rocblas_handle handle,
                                             rocblas_side   side,
                                             rocblas_int    m,
                                             rocblas_int    n,
                                             const T* const A[],
                                             rocblas_int    lda,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             T* const       C[],
                                             rocblas_int    ldc,
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
            auto side_letter = rocblas_side_letter(side);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_dgmm_batched_name<T>,
                          side,
                          m,
                          n,
                          A,
                          lda,
                          x,
                          incx,
                          C,
                          ldc,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f dgmm_batched -r",
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
                          ldc,
                          "--batch_count",
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_dgmm_batched_name<T>,
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
                            ldc,
                            "--batch_count",
                            batch_count);
        }

        if(m < 0 || n < 0 || ldc < m || lda < m || batch_count < 0 || incx == 0)
            return rocblas_status_invalid_size;

        if(!m || !n || !batch_count)
            return rocblas_status_success;

        if(!A || !x || !C)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_int    offset_a = 0, offset_x = 0, offset_c = 0;
        static constexpr rocblas_stride stride_a = 0, stride_x = 0, stride_c = 0;

        return rocblas_dgmm_template(handle,
                                     side,
                                     m,
                                     n,
                                     A,
                                     offset_a,
                                     lda,
                                     stride_a,
                                     x,
                                     offset_x,
                                     incx,
                                     stride_x,
                                     C,
                                     offset_c,
                                     ldc,
                                     stride_c,
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

#define IMPL(routine_name_, T_)                                            \
    rocblas_status routine_name_(rocblas_handle  handle,                   \
                                 rocblas_side    side,                     \
                                 rocblas_int     m,                        \
                                 rocblas_int     n,                        \
                                 const T_* const A[],                      \
                                 rocblas_int     lda,                      \
                                 const T_* const x[],                      \
                                 rocblas_int     incx,                     \
                                 T_* const       C[],                      \
                                 rocblas_int     ldc,                      \
                                 rocblas_int     batch_count)              \
    {                                                                      \
        try                                                                \
        {                                                                  \
            return rocblas_dgmm_batched_impl<T_>(                          \
                handle, side, m, n, A, lda, x, incx, C, ldc, batch_count); \
        }                                                                  \
        catch(...)                                                         \
        {                                                                  \
            return exception_to_rocblas_status();                          \
        }                                                                  \
    }

IMPL(rocblas_sdgmm_batched, float);
IMPL(rocblas_ddgmm_batched, double);
IMPL(rocblas_cdgmm_batched, rocblas_float_complex);
IMPL(rocblas_zdgmm_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
