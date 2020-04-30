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
    constexpr char rocblas_dgmm_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_dgmm_strided_batched_name<float>[] = "rocblas_sdgmm_strided_batched";
    template <>
    constexpr char rocblas_dgmm_strided_batched_name<double>[] = "rocblas_ddgmm_strided_batched";
    template <>
    constexpr char rocblas_dgmm_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_cdgmm_strided_batched";
    template <>
    constexpr char rocblas_dgmm_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_zdgmm_strided_batched";

    template <typename T>
    rocblas_status rocblas_dgmm_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_side   side,
                                                     rocblas_int    m,
                                                     rocblas_int    n,
                                                     const T*       A,
                                                     rocblas_int    lda,
                                                     rocblas_stride stride_a,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     T*             C,
                                                     rocblas_int    ldc,
                                                     rocblas_stride stride_c,
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
                          rocblas_dgmm_strided_batched_name<T>,
                          side,
                          m,
                          n,
                          A,
                          lda,
                          stride_a,
                          x,
                          incx,
                          stride_x,
                          C,
                          ldc,
                          stride_c,
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f dgmm_strided_batched -r",
                          rocblas_precision_string<T>,
                          "--side",
                          side_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "--lda",
                          lda,
                          "--stride_a",
                          stride_a,
                          "--incx",
                          incx,
                          "--stride_x",
                          stride_x,
                          "--ldc",
                          ldc,
                          "--stride_c",
                          stride_c,
                          "--batch_count",
                          batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                log_profile(handle,
                            rocblas_dgmm_strided_batched_name<T>,
                            "side",
                            side_letter,
                            "M",
                            m,
                            "N",
                            n,
                            "lda",
                            lda,
                            "--stride_a",
                            stride_a,
                            "incx",
                            incx,
                            "--stride_x",
                            stride_x,
                            "ldc",
                            ldc,
                            "--stride_c",
                            stride_c,
                            "--batch_count",
                            batch_count);
            }
        }

        if(m < 0 || n < 0 || ldc < m || lda < m || batch_count < 0 || incx == 0)
            return rocblas_status_invalid_size;

        if(!m || !n || !batch_count)
            return rocblas_status_success;

        if(!A || !C || !x)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_int offset_a = 0, offset_x = 0, offset_c = 0;

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

#define IMPL(routine_name_, T_)                                    \
    rocblas_status routine_name_(rocblas_handle handle,            \
                                 rocblas_side   side,              \
                                 rocblas_int    m,                 \
                                 rocblas_int    n,                 \
                                 const T_*      A,                 \
                                 rocblas_int    lda,               \
                                 rocblas_stride stride_a,          \
                                 const T_*      x,                 \
                                 rocblas_int    incx,              \
                                 rocblas_stride stride_x,          \
                                 T_*            C,                 \
                                 rocblas_int    ldc,               \
                                 rocblas_stride stride_c,          \
                                 rocblas_int    batch_count)       \
    try                                                            \
    {                                                              \
        return rocblas_dgmm_strided_batched_impl<T_>(handle,       \
                                                     side,         \
                                                     m,            \
                                                     n,            \
                                                     A,            \
                                                     lda,          \
                                                     stride_a,     \
                                                     x,            \
                                                     incx,         \
                                                     stride_x,     \
                                                     C,            \
                                                     ldc,          \
                                                     stride_c,     \
                                                     batch_count); \
    }                                                              \
    catch(...)                                                     \
    {                                                              \
        return exception_to_rocblas_status();                      \
    }

IMPL(rocblas_sdgmm_strided_batched, float);
IMPL(rocblas_ddgmm_strided_batched, double);
IMPL(rocblas_cdgmm_strided_batched, rocblas_float_complex);
IMPL(rocblas_zdgmm_strided_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
