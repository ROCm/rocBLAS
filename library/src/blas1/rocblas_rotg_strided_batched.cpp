/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_rotg.hpp"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_rotg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotg_name<float>[] = "rocblas_srotg_strided_batched";
    template <>
    constexpr char rocblas_rotg_name<double>[] = "rocblas_drotg_strided_batched";
    template <>
    constexpr char rocblas_rotg_name<rocblas_float_complex>[] = "rocblas_crotg_strided_batched";
    template <>
    constexpr char rocblas_rotg_name<rocblas_double_complex>[] = "rocblas_zrotg_strided_batched";

    template <class T, class U>
    rocblas_status rocblas_rotg_strided_batched_impl(rocblas_handle handle,
                                                     T*             a,
                                                     rocblas_stride stride_a,
                                                     T*             b,
                                                     rocblas_stride stride_b,
                                                     U*             c,
                                                     rocblas_stride stride_c,
                                                     T*             s,
                                                     rocblas_stride stride_s,
                                                     rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotg_name<T>, a, b, c, s, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotg_batched --a_type",
                      rocblas_precision_string<T>,
                      "--b_type",
                      rocblas_precision_string<U>,
                      "--batch",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotg_name<T>, "batch", batch_count);

        if(!a || !b || !c || !s)
            return rocblas_status_invalid_pointer;
        if(batch_count < 0)
            return rocblas_status_invalid_size;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        return rocblas_rotg_template(
            handle, a, 0, stride_a, b, 0, stride_b, c, 0, stride_c, s, 0, stride_s, batch_count);
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_srotg_strided_batched(rocblas_handle handle,
                                             float*         a,
                                             rocblas_stride stride_a,
                                             float*         b,
                                             rocblas_stride stride_b,
                                             float*         c,
                                             rocblas_stride stride_c,
                                             float*         s,
                                             rocblas_stride stride_s,
                                             rocblas_int    batch_count)
{
    return rocblas_rotg_strided_batched_impl(
        handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count);
}

rocblas_status rocblas_drotg_strided_batched(rocblas_handle handle,
                                             double*        a,
                                             rocblas_stride stride_a,
                                             double*        b,
                                             rocblas_stride stride_b,
                                             double*        c,
                                             rocblas_stride stride_c,
                                             double*        s,
                                             rocblas_stride stride_s,
                                             rocblas_int    batch_count)
{
    return rocblas_rotg_strided_batched_impl(
        handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count);
}

rocblas_status rocblas_crotg_strided_batched(rocblas_handle         handle,
                                             rocblas_float_complex* a,
                                             rocblas_stride         stride_a,
                                             rocblas_float_complex* b,
                                             rocblas_stride         stride_b,
                                             float*                 c,
                                             rocblas_stride         stride_c,
                                             rocblas_float_complex* s,
                                             rocblas_stride         stride_s,
                                             rocblas_int            batch_count)
{
    return rocblas_rotg_strided_batched_impl(
        handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count);
}

rocblas_status rocblas_zrotg_strided_batched(rocblas_handle          handle,
                                             rocblas_double_complex* a,
                                             rocblas_stride          stride_a,
                                             rocblas_double_complex* b,
                                             rocblas_stride          stride_b,
                                             double*                 c,
                                             rocblas_stride          stride_c,
                                             rocblas_double_complex* s,
                                             rocblas_stride          stride_s,
                                             rocblas_int             batch_count)
{
    return rocblas_rotg_strided_batched_impl(
        handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count);
}

} // extern "C"
