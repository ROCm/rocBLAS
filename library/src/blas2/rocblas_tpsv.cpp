/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_tpsv.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_gemv.hpp"
#include "utility.h"
#include <algorithm>
#include <cstdio>
#include <tuple>

namespace
{
    constexpr rocblas_int STPSV_BLOCK = 128;
    constexpr rocblas_int DTPSV_BLOCK = 128;

    template <typename>
    constexpr char rocblas_tpsv_name[] = "unknown";
    template <>
    constexpr char rocblas_tpsv_name<float>[] = "rocblas_stpsv";
    template <>
    constexpr char rocblas_tpsv_name<double>[] = "rocblas_dtpsv";

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_tpsv_impl(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       n,
                                     const T*          AP,
                                     T*                x,
                                     rocblas_int       incx)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_tpsv_name<T>, uplo, transA, diag, n, AP, x, incx);

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                if(handle->pointer_mode == rocblas_pointer_mode_host)
                    log_bench(handle,
                              "./rocblas-bench -f tpsv -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-n",
                              n,
                              "--incx",
                              incx);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_tpsv_name<T>,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "diag",
                            diag_letter,
                            "N",
                            n,
                            "incx",
                            incx);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(!AP || !x)
            return rocblas_status_invalid_pointer;
        if(n < 0 || !incx)
            return rocblas_status_invalid_size;

        // quick return if possible.
        // return rocblas_status_size_unchanged if device memory size query
        if(!n)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(sizeof(T) * n);

        auto mem = handle->device_malloc(sizeof(T) * n);
        // todo set mem to all 0s

        rocblas_status status = rocblas_tpsv_template<BLOCK, false, T>(
            handle, uplo, transA, diag, n, AP, 0, 0, x, 0, incx, 0, 1, (T*)mem);

        return status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_stpsv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       n,
                             const float*      AP,
                             float*            x,
                             rocblas_int       incx)
try
{
    return rocblas_tpsv_impl<STPSV_BLOCK>(handle, uplo, transA, diag, n, AP, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtpsv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       n,
                             const double*     AP,
                             double*           x,
                             rocblas_int       incx)
try
{
    return rocblas_tpsv_impl<DTPSV_BLOCK>(handle, uplo, transA, diag, n, AP, x, incx);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
