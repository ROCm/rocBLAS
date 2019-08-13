/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "fetch_template.h"
#include "handle.h"
#include "logging.h"
#include "reduction_strided_batched.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <class To>
    struct rocblas_fetch_asum_strided_batched
    {
        template <typename Ti>
        __forceinline__ __device__ To operator()(Ti x, ptrdiff_t)
        {
            return {fetch_asum(x)};
        }
    };

    template <typename>
    constexpr char rocblas_asum_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_asum_strided_batched_name<float>[] = "rocblas_sasum_strided_batched";
    template <>
    constexpr char rocblas_asum_strided_batched_name<double>[] = "rocblas_dasum_strided_batched";
    template <>
    constexpr char rocblas_asum_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_scasum_strided_batched";
    template <>
    constexpr char rocblas_asum_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_dzasum_strided_batched";

    // HIP support up to 1024 threads/work itmes per thread block/work group
    constexpr int NB = 512;

    // allocate workspace inside this API
    template <typename Ti, typename To>
    rocblas_status rocblas_asum_strided_batched(rocblas_handle handle,
                                                rocblas_int    n,
                                                const Ti*      x,
                                                rocblas_int    incx,
                                                rocblas_int    stridex,
                                                To*            results,
                                                rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(
                handle, rocblas_asum_strided_batched_name<Ti>, n, x, incx, stridex, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f asum_strided_batched -r",
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
                        rocblas_asum_strided_batched_name<Ti>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "stride_x",
                        stridex,
                        "batch",
                        batch_count);

        if(!x || !results)
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
                RETURN_IF_HIP_ERROR(hipMemset(results, 0, batch_count * sizeof(To)));
            else
            {
                for(int i = 0; i < batch_count; i++)
                {
                    results[i] = 0;
                }
            }
            return rocblas_status_success;
        }

        auto blocks = (n - 1) / NB + 1;

        // below blocks+1 the +1 is for results when rocblas_pointer_mode_host
        size_t dev_bytes = sizeof(To) * (blocks + 1) * batch_count;

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        auto mem = handle->device_malloc(dev_bytes);
        if(!mem)
            return rocblas_status_memory_error;

        rocblas_status bstatus
            = rocblas_reduction_strided_batched_kernel<NB, rocblas_fetch_asum_strided_batched<To>>(
                handle, n, x, incx, stridex, results, (To*)mem, blocks, batch_count);

        return bstatus;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sasum_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const float*   x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             float*         results,
                                             rocblas_int    batch_count)
{
    return rocblas_asum_strided_batched(handle, n, x, incx, stridex, results, batch_count);
}

rocblas_status rocblas_dasum_strided_batched(rocblas_handle handle,
                                             rocblas_int    n,
                                             const double*  x,
                                             rocblas_int    incx,
                                             rocblas_int    stridex,
                                             double*        results,
                                             rocblas_int    batch_count)
{
    return rocblas_asum_strided_batched(handle, n, x, incx, stridex, results, batch_count);
}

rocblas_status rocblas_scasum_strided_batched(rocblas_handle               handle,
                                              rocblas_int                  n,
                                              const rocblas_float_complex* x,
                                              rocblas_int                  incx,
                                              rocblas_int                  stridex,
                                              float*                       results,
                                              rocblas_int                  batch_count)
{
    return rocblas_asum_strided_batched(handle, n, x, incx, stridex, results, batch_count);
}

rocblas_status rocblas_dzasum_strided_batched(rocblas_handle                handle,
                                              rocblas_int                   n,
                                              const rocblas_double_complex* x,
                                              rocblas_int                   incx,
                                              rocblas_int                   stridex,
                                              double*                       results,
                                              rocblas_int                   batch_count)
{
    return rocblas_asum_strided_batched(handle, n, x, incx, stridex, results, batch_count);
}

} // extern "C"
