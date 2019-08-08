/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

//
// Use the non-batched header.
//
#include "rocblas_amax_amin.h"


//
// Define a trait
//
template <typename Ti>
struct TypeTraits_AMAXMIN
{
public:
    using To = Ti;
};

//
// Specialization of the type trait.
//
template <>
struct TypeTraits_AMAXMIN<rocblas_float_complex>
{
public:
    using To = float;
};

template <>
struct TypeTraits_AMAXMIN<rocblas_double_complex>
{
public:
    using To = double;
};

//
// Extend names.
//
template <typename T>
static constexpr char rocblas_iamaxmin_batched_name[] = rocblas_iamaxmin_name<T> QUOTE(_batched);

//
// 
//
template <typename Ti>
static rocblas_status rocblas_iamaxmin_batched(rocblas_handle handle,
                                               rocblas_int    n,
                                               const Ti*      x[],
                                               rocblas_int    incx,
                                               rocblas_int    batch_count,
                                               rocblas_int*   result[])
{

  //
  // Get the 'T output' type
  //
    using To = typename TypeTraits_AMAXMIN<Ti>::To;

    //
    // HIP support up to 1024 threads/work itmes per thread block/work group
    //
    static constexpr int NB = 1024;
    if(!handle)
    {
        return rocblas_status_invalid_handle;
    }

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        log_trace(handle, rocblas_iamaxmin_batched_name<Ti>, n, x, incx);
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
        log_profile(handle, rocblas_iamaxmin_batched_name<Ti>, "N", n, "incx", incx);
    }

    if(!x || !result)
    {
        return rocblas_status_invalid_pointer;
    }

    //
    // Quick return if possible.
    //
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
                                    rocblas_finalize_amax_amin>(
        handle, n, x[0], incx, result, (index_value_t<To>*)mem, blocks);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status JOIN(rocblas_isa, MAX_MIN)(
    rocblas_handle handle, rocblas_int n, const float* x[], rocblas_int incx, rocblas_int* result[])
{
    return rocblas_iamaxmin_batched(handle, n, x, incx, result);
}

rocblas_status JOIN(rocblas_ida, MAX_MIN)(rocblas_handle handle,
                                          rocblas_int    n,
                                          const double*  x[],
                                          rocblas_int    incx,
                                          rocblas_int*   result[])
{
    return rocblas_iamaxmin_batched(handle, n, x, incx, result);
}

rocblas_status JOIN(rocblas_ica, MAX_MIN)(rocblas_handle               handle,
                                          rocblas_int                  n,
                                          const rocblas_float_complex* x[],
                                          rocblas_int                  incx,
                                          rocblas_int*                 result[])
{
    return rocblas_iamaxmin_batched(handle, n, x, incx, result);
}

rocblas_status JOIN(rocblas_iza, MAX_MIN)(rocblas_handle                handle,
                                          rocblas_int                   n,
                                          const rocblas_double_complex* x[],
                                          rocblas_int                   incx,
                                          rocblas_int*                  result[])
{
    return rocblas_iamaxmin_batched(handle, n, x, incx, result);
}

} // extern "C"
