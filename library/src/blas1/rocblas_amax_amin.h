/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "fetch_template.h"
#include "handle.h"
#include "logging.h"
#include "reduction.h"
#include "rocblas.h"
#include "utility.h"

//
// max by default ?...
//
#ifndef MAX_MIN
#define MAX_MIN max
#endif

//
// max by default again ?...
//

#define QUOTE2(S) #S
#define QUOTE(S) QUOTE2(S)

#define JOIN2(A, B) A##B
#define JOIN(A, B) JOIN2(A, B)


#include "rocblas_iamaxmin_template.h"


//
// Names.
//
template <typename>
static constexpr char rocblas_iamaxmin_name[] = "unknown";

//
// Name specializations.
//
template <>
static constexpr char rocblas_iamaxmin_name<float>[] = "rocblas_isa" QUOTE(MAX_MIN);

template <>
static constexpr char rocblas_iamaxmin_name<double>[] = "rocblas_ida" QUOTE(MAX_MIN);

template <>
static constexpr char rocblas_iamaxmin_name<rocblas_float_complex>[] = "rocblas_ica" QUOTE(MAX_MIN);

template <>
static constexpr char rocblas_iamaxmin_name<rocblas_double_complex>[]
    = "rocblas_iza" QUOTE(MAX_MIN);

//
// allocate workspace inside this API
//
template <typename To, typename Ti>
static rocblas_status rocblas_iamaxmin(
    rocblas_handle handle, rocblas_int n, const Ti* x, rocblas_int incx, rocblas_int* result)
{
    // HIP support up to 1024 threads/work itmes per thread block/work group
    static constexpr int NB = 1024;
    if(!handle)
    {
        return rocblas_status_invalid_handle;
    }

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        log_trace(handle, rocblas_iamaxmin_name<Ti>, n, x, incx);
    }

    if(layer_mode & rocblas_layer_mode_log_bench)
    {
        log_bench(handle,
                  "./rocblas-bench -f ia" QUOTE(MAX_MIN) " -r",
                  rocblas_precision_string<Ti>,
                  "-n",
                  n,
                  "--incx",
                  incx);
    }

    if(layer_mode & rocblas_layer_mode_log_profile)
    {
        log_profile(handle, rocblas_iamaxmin_name<Ti>, "N", n, "incx", incx);
    }

    if(!x || !result)
    {
        return rocblas_status_invalid_pointer;
    }

    // Quick return if possible.
    if(n <= 0 || incx <= 0)
    {
        if(handle->is_device_memory_size_query())
        {
            return rocblas_status_size_unchanged;
        }
        else if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(*result)));
        }
        else
        {
            *result = 0;
        }
        return rocblas_status_success;
    }

    auto blocks = (n - 1) / NB + 1;
    if(handle->is_device_memory_size_query())
    {
        return handle->set_optimal_device_memory_size(sizeof(index_value_t<To>) * blocks);
    }

    auto mem = handle->device_malloc(sizeof(index_value_t<To>) * blocks);
    if(!mem)
    {
        return rocblas_status_memory_error;
    }

    
    return rocblas_reduction_kernel<NB,
                                    rocblas_fetch_amax_amin<To>,
                                    AMAX_AMIN_REDUCTION,
                                    rocblas_finalize_amax_amin>(handle,
                                                                n,
                                                                x,
                                                                incx,
                                                                result,
                                                                (index_value_t<To>*)mem,
                                                                blocks);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status JOIN(rocblas_isa, MAX_MIN)(
    rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result)
{
    return rocblas_iamaxmin<float>(handle, n, x, incx, result);
}

rocblas_status JOIN(rocblas_ida, MAX_MIN)(rocblas_handle handle, // the handle.
                                          rocblas_int    n,
                                          const double*  x,
                                          rocblas_int    incx,
                                          rocblas_int*   result)
{
    return rocblas_iamaxmin<double>(handle, n, x, incx, result);
}

rocblas_status JOIN(rocblas_ica, MAX_MIN)(rocblas_handle               handle,
                                          rocblas_int                  n,
                                          const rocblas_float_complex* x,
                                          rocblas_int                  incx,
                                          rocblas_int*                 result)
{
    return rocblas_iamaxmin<float, rocblas_float_complex>(handle, n, x, incx, result);
}

rocblas_status JOIN(rocblas_iza, MAX_MIN)(rocblas_handle                handle,
                                          rocblas_int                   n,
                                          const rocblas_double_complex* x,
                                          rocblas_int                   incx,
                                          rocblas_int*                  result)
{
    return rocblas_iamaxmin<double, rocblas_double_complex>(handle, n, x, incx, result);
}

} // extern "C"
