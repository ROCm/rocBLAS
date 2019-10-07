/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_rotg.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_rotg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotg_name<float>[] = "rocblas_srotg";
    template <>
    constexpr char rocblas_rotg_name<double>[] = "rocblas_drotg";
    template <>
    constexpr char rocblas_rotg_name<rocblas_float_complex>[] = "rocblas_crotg";
    template <>
    constexpr char rocblas_rotg_name<rocblas_double_complex>[] = "rocblas_zrotg";

    template <class T, class U>
    rocblas_status rocblas_rotg_impl(rocblas_handle handle, T* a, T* b, U* c, T* s)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotg_name<T>, a, b, c, s);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotg --a_type",
                      rocblas_precision_string<T>,
                      "--b_type",
                      rocblas_precision_string<U>);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotg_name<T>);

        if(!a || !b || !c || !s)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_rotg_template(handle, a, 0, 0, b, 0, 0, c, 0, 0, s, 0, 0, 1);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_srotg(rocblas_handle handle, float* a, float* b, float* c, float* s)
{
    return rocblas_rotg_impl(handle, a, b, c, s);
}

rocblas_status rocblas_drotg(rocblas_handle handle, double* a, double* b, double* c, double* s)
{
    return rocblas_rotg_impl(handle, a, b, c, s);
}

rocblas_status rocblas_crotg(rocblas_handle         handle,
                             rocblas_float_complex* a,
                             rocblas_float_complex* b,
                             float*                 c,
                             rocblas_float_complex* s)
{
    return rocblas_rotg_impl(handle, a, b, c, s);
}

rocblas_status rocblas_zrotg(rocblas_handle          handle,
                             rocblas_double_complex* a,
                             rocblas_double_complex* b,
                             double*                 c,
                             rocblas_double_complex* s)
{
    return rocblas_rotg_impl(handle, a, b, c, s);
}

} // extern "C"
