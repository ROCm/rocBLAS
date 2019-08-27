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
//template <typename T>
// static constexpr char rocblas_iamaxmin_batched_name[] = rocblas_iamaxmin_name<T>;// QUOTE(_batched);

//
// Names.
//
template <typename>
static constexpr char rocblas_iamaxmin_name_batched[] = "unknown";

//
// Name specializations.
//
template <>
static constexpr char rocblas_iamaxmin_name_batched<float>[]
    = "rocblas_isa" QUOTE(MAX_MIN) "_batched";

template <>
static constexpr char rocblas_iamaxmin_name_batched<double>[]
    = "rocblas_ida" QUOTE(MAX_MIN) "_batched";

template <>
static constexpr char rocblas_iamaxmin_name_batched<rocblas_float_complex>[]
    = "rocblas_ica" QUOTE(MAX_MIN) "_batched";

template <>
static constexpr char rocblas_iamaxmin_name_batched<rocblas_double_complex>[]
    = "rocblas_iza" QUOTE(MAX_MIN) "_batched";



//
//
//
template <typename Ti>
static rocblas_status rocblas_iamaxmin_batched(rocblas_handle handle,
                                               rocblas_int    n,
                                               const Ti*      const x[],
                                               rocblas_int    incx,
                                               rocblas_int    batch_count,
                                               rocblas_int*   result)
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
        log_trace(handle, rocblas_iamaxmin_name_batched<Ti>, n, x, incx);
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
        log_profile(handle, rocblas_iamaxmin_name_batched<Ti>, "N", n, "incx", incx);
    }

    if(!x || !result)
    {
        return rocblas_status_invalid_pointer;
    }

    *result = -777;
    //
    // Quick return if possible.
    //
    //    printf("%d %d %d ????\n",n,incx,batch_count);
    if(n <= 0 || incx <= 0 || batch_count <=0)
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



    int status = rocblas_status_success;
    for(int batch_index = 0; batch_index < batch_count; ++batch_index)
    {


      
      status &= rocblas_reduction_kernel<NB,
                                           rocblas_fetch_amax_amin<To>,
                                           AMAX_AMIN_REDUCTION,
                                           rocblas_finalize_amax_amin>
	  (handle, n, x[batch_index], incx, &result[batch_index], (index_value_t<To>*)mem, blocks);
      
      //      printf("results########## %d\n", result[batch_index]);
    }
 
    return rocblas_status_success;
}


//
// C wrapper
//
extern "C" {

rocblas_status JOIN(rocblas_isa, JOIN(MAX_MIN, _batched))(rocblas_handle handle,
                                                          rocblas_int    n,
                                                          const float*   const x[],
                                                          rocblas_int    incx,
                                                          rocblas_int    batch_count,
                                                          rocblas_int*   result)
{
    return rocblas_iamaxmin_batched(handle, n, x, incx, batch_count, result);
}

rocblas_status JOIN(rocblas_ida, JOIN(MAX_MIN, _batched))(rocblas_handle handle,
                                                          rocblas_int    n,
                                                          const double*  const x[],
                                                          rocblas_int    incx,
                                                          rocblas_int    batch_count,
                                                          rocblas_int*   result)
{
    return rocblas_iamaxmin_batched(handle, n, x, incx, batch_count, result);
}

rocblas_status JOIN(rocblas_ica, JOIN(MAX_MIN, _batched))(rocblas_handle               handle,
                                                          rocblas_int                  n,
                                                          const rocblas_float_complex* const x[],
                                                          rocblas_int                  incx,
                                                          rocblas_int                  batch_count,
                                                          rocblas_int*                 result)
{
    return rocblas_iamaxmin_batched(handle, n, x, incx, batch_count, result);
}

rocblas_status JOIN(rocblas_iza, JOIN(MAX_MIN, _batched))(rocblas_handle                handle,
                                                          rocblas_int                   n,
                                                          const rocblas_double_complex* const x[],
                                                          rocblas_int                   incx,
                                                          rocblas_int                   batch_count,
                                                          rocblas_int*                  result)
{
    return rocblas_iamaxmin_batched(handle, n, x, incx, batch_count, result);
}

} // extern "C"
