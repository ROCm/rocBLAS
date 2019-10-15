#pragma once

#ifndef ROCBLAS_BATCHED_SUFFIX
#error Definition of the macro ROCBLAS_BATCHED_SUFFIX is required (_batched, _strided_batched or nothing).
#endif

#include "rocblas_iamaxmin_template.h"
#include "rocblas_utils.h"

//
// Define names.
//
#ifdef ROCBLAS_IAMAXMIN_NAME
#error ROCBLAS_IAMAXMIN_NAME is already defined.
#endif
#define ROCBLAS_IAMAXMIN_NAME JOIN(rocblas_iamaxmin, JOIN(ROCBLAS_BATCHED_SUFFIX, _name))

template <typename>
static constexpr char ROCBLAS_IAMAXMIN_NAME[] = "unknown";

template <>
static constexpr char ROCBLAS_IAMAXMIN_NAME<float>[]
    = "rocblas_isa" QUOTE(MAX_MIN) QUOTE(ROCBLAS_BATCHED_SUFFIX);

template <>
static constexpr char ROCBLAS_IAMAXMIN_NAME<double>[]
    = "rocblas_ida" QUOTE(MAX_MIN) QUOTE(ROCBLAS_BATCHED_SUFFIX);

template <>
static constexpr char ROCBLAS_IAMAXMIN_NAME<rocblas_float_complex>[]
    = "rocblas_ica" QUOTE(MAX_MIN) QUOTE(ROCBLAS_BATCHED_SUFFIX);

template <>
static constexpr char ROCBLAS_IAMAXMIN_NAME<rocblas_double_complex>[]
    = "rocblas_iza" QUOTE(MAX_MIN) QUOTE(ROCBLAS_BATCHED_SUFFIX);

template <typename U>
static rocblas_status rocblas_iamaxmin_impl(rocblas_handle handle,
                                            rocblas_int    n,
                                            U              x,
                                            rocblas_int    incx,
                                            rocblas_stride stridex,
                                            rocblas_int    batch_count,
                                            rocblas_int*   result)
{
    //
    // Get the 'T input' type.
    //
    using Ti = batched_data_t<U>;

    //
    // Get the name of the routine.
    //
    static constexpr const char* const name = ROCBLAS_IAMAXMIN_NAME<Ti>;

    //
    // Get the 'T output' type
    //
    using To = typename reduction_types<Ti>::To;

    //
    // HIP support up to 1024 threads/work times per thread block/work group
    //
    static constexpr int NB = 1024;
    if(!handle)
    {
        return rocblas_status_invalid_handle;
    }

    //
    // Log trace.
    //
    rocblas_reduction_utils::logging_trace(
        handle, n, x, incx, stridex, batch_count, ROCBLAS_IAMAXMIN_NAME<Ti>);

    //
    // Log bench.
    //
    rocblas_reduction_utils::logging_bench(handle,
                                           n,
                                           x,
                                           incx,
                                           stridex,
                                           batch_count,
                                           "ia" QUOTE(MAX_MIN) QUOTE(ROCBLAS_BATCHED_SUFFIX));

    //
    // Log profile.
    //
    rocblas_reduction_utils::logging_profile(
        handle, n, x, incx, stridex, batch_count, ROCBLAS_IAMAXMIN_NAME<Ti>);

    if(!x || !result)
    {
        return rocblas_status_invalid_pointer;
    }

    if(batch_count < 0)
    {
        return rocblas_status_invalid_size;
    }

    const size_t workspace_num_bytes
        = rocblas_reduction_kernel_workspace_size<NB, index_value_t<To>>(n, batch_count);

    if(handle->is_device_memory_size_query())
    {
        return handle->set_optimal_device_memory_size(workspace_num_bytes);
    }

    auto mem = handle->device_malloc(workspace_num_bytes);
    if(!mem)
    {
        return rocblas_status_memory_error;
    }

    return rocblas_iamaxmin_template(handle, n, x, incx, stridex, batch_count, result, (void*)mem);
}

//
// Undefined introduced macro(s).
//
#undef ROCBLAS_IAMAXMIN_NAME
