/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_rotmg.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas/rocblas.h"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_rotmg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotmg_name<float>[] = "rocblas_srotmg";
    template <>
    constexpr char rocblas_rotmg_name<double>[] = "rocblas_drotmg";

    template <class T>
    rocblas_status
        rocblas_rotmg_impl(rocblas_handle handle, T* d1, T* d2, T* x1, const T* y1, T* param)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotmg_name<T>, d1, d2, x1, y1, param);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle, "./rocblas-bench -f rotmg -r", rocblas_precision_string<T>);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotmg_name<T>);

        if(!d1 || !d2 || !x1 || !y1 || !param)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status rotmg_check_numerics_status
                = rocblas_rotmg_check_numerics_template(rocblas_rotmg_name<T>,
                                                        handle,
                                                        1,
                                                        d1,
                                                        0,
                                                        0,
                                                        d2,
                                                        0,
                                                        0,
                                                        x1,
                                                        0,
                                                        0,
                                                        y1,
                                                        0,
                                                        0,
                                                        1,
                                                        check_numerics,
                                                        is_input);
            if(rotmg_check_numerics_status != rocblas_status_success)
                return rotmg_check_numerics_status;
        }

        rocblas_status status = rocblas_rotmg_template(
            handle, d1, 0, 0, d2, 0, 0, x1, 0, 0, y1, 0, 0, param, 0, 0, 1);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status rotmg_check_numerics_status
                = rocblas_rotmg_check_numerics_template(rocblas_rotmg_name<T>,
                                                        handle,
                                                        1,
                                                        d1,
                                                        0,
                                                        0,
                                                        d2,
                                                        0,
                                                        0,
                                                        x1,
                                                        0,
                                                        0,
                                                        y1,
                                                        0,
                                                        0,
                                                        1,
                                                        check_numerics,
                                                        is_input);
            if(rotmg_check_numerics_status != rocblas_status_success)
                return rotmg_check_numerics_status;
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

ROCBLAS_EXPORT rocblas_status rocblas_srotmg(
    rocblas_handle handle, float* d1, float* d2, float* x1, const float* y1, float* param)
try
{
    return rocblas_rotmg_impl(handle, d1, d2, x1, y1, param);
}
catch(...)
{
    return exception_to_rocblas_status();
}

ROCBLAS_EXPORT rocblas_status rocblas_drotmg(
    rocblas_handle handle, double* d1, double* d2, double* x1, const double* y1, double* param)
try
{
    return rocblas_rotmg_impl(handle, d1, d2, x1, y1, param);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
