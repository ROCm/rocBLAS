/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_tpmv.hpp"
#include "utility.hpp"

namespace
{

    template <typename>
    constexpr char rocblas_tpmv_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_tpmv_strided_batched_name<float>[] = "rocblas_stpmv_strided_batched";
    template <>
    constexpr char rocblas_tpmv_strided_batched_name<double>[] = "rocblas_dtpmv_strided_batched";
    template <>
    constexpr char rocblas_tpmv_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_ctpmv_strided_batched";
    template <>
    constexpr char rocblas_tpmv_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_ztpmv_strided_batched";

    template <typename T>
    rocblas_status rocblas_tpmv_strided_batched_impl(rocblas_handle    handle,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation transa,
                                                     rocblas_diagonal  diag,
                                                     rocblas_int       m,
                                                     const T*          a,
                                                     rocblas_stride    stridea,
                                                     T*                x,
                                                     rocblas_int       incx,
                                                     rocblas_stride    stridex,
                                                     rocblas_int       batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto check_numerics = handle->check_numerics;

        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                auto uplo_letter   = rocblas_fill_letter(uplo);
                auto transa_letter = rocblas_transpose_letter(transa);
                auto diag_letter   = rocblas_diag_letter(diag);
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    log_trace(handle,
                              rocblas_tpmv_strided_batched_name<T>,
                              uplo,
                              transa,
                              diag,
                              m,
                              a,
                              x,
                              incx,
                              stridea,
                              incx,
                              stridex,
                              batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench",
                              "-f",
                              "tpmv_strided_batched",
                              "-r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transa_letter,
                              "--diag",
                              diag_letter,
                              "-m",
                              m,
                              "--stride_a",
                              stridea,
                              "--incx",
                              incx,
                              "--stride_x",
                              stridex,
                              "--batch_count",
                              batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                {
                    log_profile(handle,
                                rocblas_tpmv_strided_batched_name<T>,
                                "uplo",
                                uplo_letter,
                                "transA",
                                transa_letter,
                                "diag",
                                diag_letter,
                                "M",
                                m,
                                "stride_a",
                                stridea,
                                "incx",
                                incx,
                                "stride_x",
                                stridex,
                                "batch_count",
                                batch_count);
                }
            }
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;

        if(m < 0 || !incx || batch_count < 0)
            return rocblas_status_invalid_size;

        if(!m || !batch_count)
        {
            RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
            return rocblas_status_success;
        }

        size_t dev_bytes = m * batch_count * sizeof(T);
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        if(!a || !x)
            return rocblas_status_invalid_pointer;

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
            return rocblas_status_memory_error;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status tpmv_check_numerics_status
                = rocblas_tpmv_check_numerics(rocblas_tpmv_strided_batched_name<T>,
                                              handle,
                                              m,
                                              a,
                                              0,
                                              stridea,
                                              x,
                                              0,
                                              incx,
                                              stridex,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(tpmv_check_numerics_status != rocblas_status_success)
                return tpmv_check_numerics_status;
        }

        rocblas_stride               stridew = m;
        static constexpr rocblas_int NB      = 512;
        static constexpr ptrdiff_t   offseta = 0;
        static constexpr ptrdiff_t   offsetx = 0;
        rocblas_status               status  = rocblas_tpmv_template<NB>(handle,
                                                          uplo,
                                                          transa,
                                                          diag,
                                                          m,
                                                          a,
                                                          offseta,
                                                          stridea,
                                                          x,
                                                          offsetx,
                                                          incx,
                                                          stridex,
                                                          (T*)w_mem,
                                                          stridew,
                                                          batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status tpmv_check_numerics_status
                = rocblas_tpmv_check_numerics(rocblas_tpmv_strided_batched_name<T>,
                                              handle,
                                              m,
                                              a,
                                              0,
                                              stridea,
                                              x,
                                              0,
                                              incx,
                                              stridex,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(tpmv_check_numerics_status != rocblas_status_success)
                return tpmv_check_numerics_status;
        }
        return status;
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

#define IMPL(routine_name_, T_)                                                        \
    rocblas_status routine_name_(rocblas_handle    handle,                             \
                                 rocblas_fill      uplo,                               \
                                 rocblas_operation transA,                             \
                                 rocblas_diagonal  diag,                               \
                                 rocblas_int       m,                                  \
                                 const T_*         A,                                  \
                                 rocblas_stride    stridea,                            \
                                 T_*               x,                                  \
                                 rocblas_int       incx,                               \
                                 rocblas_stride    stridex,                            \
                                 rocblas_int       batch_count)                        \
    try                                                                                \
    {                                                                                  \
        return rocblas_tpmv_strided_batched_impl(                                      \
            handle, uplo, transA, diag, m, A, stridea, x, incx, stridex, batch_count); \
    }                                                                                  \
    catch(...)                                                                         \
    {                                                                                  \
        return exception_to_rocblas_status();                                          \
    }

IMPL(rocblas_stpmv_strided_batched, float);
IMPL(rocblas_dtpmv_strided_batched, double);
IMPL(rocblas_ctpmv_strided_batched, rocblas_float_complex);
IMPL(rocblas_ztpmv_strided_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
