/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "fetch_template.h"
#include "handle.h"
#include "logging.h"
#include "reduction.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    // HIP support up to 1024 threads/work itmes per thread block/work group
    constexpr int NB = 512;

    template <class To>
    struct rocblas_fetch_nrm2_strided_batched
    {
        template <class Ti>
        __forceinline__ __device__ To operator()(Ti x, ptrdiff_t tid)
        {
            return {fetch_abs2(x)};
        }
    };

    struct rocblas_finalize_nrm2_strided_batched
    {
        template <class To>
        __forceinline__ __host__ __device__ To operator()(To x)
        {
            return sqrt(x);
        }
    };

    template <typename>
    constexpr char rocblas_nrm2_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_nrm2_strided_batched_name<float>[] = "rocblas_snrm2_strided_batched";
    template <>
    constexpr char rocblas_nrm2_strided_batched_name<double>[] = "rocblas_dnrm2_strided_batched";
    template <>
    constexpr char rocblas_nrm2_strided_batched_name<rocblas_half>[]
        = "rocblas_hnrm2_strided_batched";
    template <>
    constexpr char rocblas_nrm2_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_scnrm2_strided_batched";
    template <>
    constexpr char rocblas_nrm2_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_dznrm2_strided_batched";

    // allocate workspace inside this API
    template <typename Ti, typename To>
    rocblas_status rocblas_nrm2_strided_batched(rocblas_handle handle,
                                                rocblas_int    n,
                                                const Ti*      x,
                                                rocblas_int    incx,
                                                rocblas_int    stridex,
                                                To*            result,
                                                rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(
                handle, rocblas_nrm2_strided_batched_name<Ti>, n, x, incx, stridex, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f nrm2_strided_batched -r",
                      rocblas_precision_string<Ti>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--stride_x",
                      stridex,
                      "--batch",
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_nrm2_strided_batched_name<Ti>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "stride_x",
                        stridex,
                        "batch",
                        batch_count);

        if(!x || !result)
            return rocblas_status_invalid_pointer;

        if(stridex < 0 || stridex < n * incx) // negative n or incx rejected later
            return rocblas_status_invalid_size;

        if(batch_count <= 0)
            return rocblas_status_invalid_size;

        // Quick return if possible.
        if(n <= 0 || incx <= 0)
        {
            if(handle->is_device_memory_size_query())
                return rocblas_status_size_unchanged;
            else if(rocblas_pointer_mode_device == handle->pointer_mode)
                RETURN_IF_HIP_ERROR(hipMemset(result, 0, batch_count * sizeof(To)));
            else
            {
                for(int i = 0; i < batch_count; i++)
                    result[i] = 0;
            }
            return rocblas_status_success;
        }

        auto blocks = (n - 1) / NB + 1;
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(sizeof(To) * blocks);

        auto mem = handle->device_malloc(sizeof(To) * blocks);
        if(!mem)
            return rocblas_status_memory_error;

        rocblas_status bstatus;
        for(int i = 0; i < batch_count; i++)
        {
            bstatus = rocblas_reduction_kernel<NB,
                                               rocblas_fetch_nrm2_strided_batched<To>,
                                               rocblas_reduce_sum,
                                               rocblas_finalize_nrm2_strided_batched>(
                handle, n, x + i * stridex, incx, result + i, (To*)mem, blocks);
            if(bstatus != rocblas_status_success)
                return bstatus;
        }
        return bstatus;
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_snrm2_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const float*   x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             float*         result,
                                             rocblas_int    batch_count)
{
    return rocblas_nrm2_strided_batched(handle, n, x, incx, stridex, result, batch_count);
}

rocblas_status rocblas_dnrm2_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const double*  x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             double*        result,
                                             rocblas_int    batch_count)
{
    return rocblas_nrm2_strided_batched(handle, n, x, incx, stridex, result, batch_count);
}

rocblas_status rocblas_scnrm2_strided_batched(rocblas_handle               handle,
                                              rocblas_int                  n,
                                              const rocblas_float_complex* x,
                                              rocblas_int                  incx,
                                              rocblas_int                  stridex,
                                              float*                       result,
                                              rocblas_int                  batch_count)
{
    return rocblas_nrm2_strided_batched(handle, n, x, incx, stridex, result, batch_count);
}

rocblas_status rocblas_dznrm2_strided_batched(rocblas_handle                handle,
                                              rocblas_int                   n,
                                              const rocblas_double_complex* x,
                                              rocblas_int                   incx,
                                              rocblas_int                   stridex,
                                              double*                       result,
                                              rocblas_int                   batch_count)
{
    return rocblas_nrm2_strided_batched(handle, n, x, incx, stridex, result, batch_count);
}

} // extern "C"
