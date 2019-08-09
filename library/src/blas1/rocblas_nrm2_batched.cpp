/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "fetch_template.h"
#include "handle.h"
#include "logging.h"
#include "reduction.h"
#include "reduction_batched.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    // HIP support up to 1024 threads/work itmes per thread block/work group
    constexpr int NB = 512;

    // same as non-batched for now
    template <class To>
    struct rocblas_fetch_nrm2_batched
    {
        template <class Ti>
        __forceinline__ __device__ To operator()(Ti x, ptrdiff_t tid)
        {
            return {fetch_abs2(x)};
        }
    };

    struct rocblas_finalize_nrm2_batched
    {
        template <class To>
        __forceinline__ __host__ __device__ To operator()(To x)
        {
            return sqrt(x);
        }
    };

    template <typename>
    constexpr char rocblas_nrm2_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_nrm2_batched_name<float>[] = "rocblas_snrm2_batched";
    template <>
    constexpr char rocblas_nrm2_batched_name<double>[] = "rocblas_dnrm2_batched";
    template <>
    constexpr char rocblas_nrm2_batched_name<rocblas_half>[] = "rocblas_hnrm2_batched";
    template <>
    constexpr char rocblas_nrm2_batched_name<rocblas_float_complex>[] = "rocblas_scnrm2_batched";
    template <>
    constexpr char rocblas_nrm2_batched_name<rocblas_double_complex>[] = "rocblas_dznrm2_batched";

    // allocate workspace inside this API
    template <typename Ti, typename To>
    rocblas_status rocblas_nrm2_batched(rocblas_handle  handle,
                                        rocblas_int     n,
                                        const Ti* const x[],
                                        rocblas_int     incx,
                                        To*             results,
                                        rocblas_int     batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_nrm2_batched_name<Ti>, n, x, incx, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f nrm2_batched -r",
                      rocblas_precision_string<Ti>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--batch",
                      batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(
                handle, rocblas_nrm2_batched_name<Ti>, "N", n, "incx", incx, "batch", batch_count);

        if(!x || !results)
            return rocblas_status_invalid_pointer;

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
                    results[i] = 0;
            }

            return rocblas_status_success;
        }

        auto blocks = (n - 1) / NB + 1;

        // below blocks+1 the +1 is for results when rocblas_pointer_mode_host
        size_t devBytes = sizeof(To) * (blocks+1) * batch_count;

        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(devBytes);

        auto mem = handle->device_malloc(devBytes);
        if(!mem)
            return rocblas_status_memory_error;

        // fetch gpu pointers to cpu for cpu kernel loop use
        Ti* hdx[batch_count];
        RETURN_IF_HIP_ERROR(hipMemcpy(hdx, x, sizeof(Ti*) * batch_count, hipMemcpyDeviceToHost));

        rocblas_status bstatus;

        // for(int i = 0; i < batch_count; i++)
        // {
        //     bstatus = rocblas_reduction_kernel<NB,
        //                                        rocblas_fetch_nrm2_batched<To>,
        //                                        rocblas_reduce_sum,
        //                                        rocblas_finalize_nrm2_batched>(
        //         handle, n, hdx[i], incx, results + i, (To*)mem, blocks);
        //     if(bstatus != rocblas_status_success)
        //         return bstatus;
        // }

        bstatus = rocblas_reduction_batched_kernel<NB,
                                            rocblas_fetch_nrm2_batched<To>,
                                            rocblas_reduce_sum_batched,
                                            rocblas_finalize_nrm2_batched>(
            handle, n, x, incx, results, (To*)mem, blocks, batch_count);

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

rocblas_status rocblas_snrm2_batched(rocblas_handle     handle,
                                     rocblas_int        n,
                                     const float* const x[],
                                     rocblas_int        incx,
                                     float*             results,
                                     rocblas_int        batch_count)
{
    return rocblas_nrm2_batched(handle, n, x, incx, results, batch_count);
}

rocblas_status rocblas_dnrm2_batched(rocblas_handle      handle,
                                     rocblas_int         n,
                                     const double* const x[],
                                     rocblas_int         incx,
                                     double*             results,
                                     rocblas_int         batch_count)
{
    return rocblas_nrm2_batched(handle, n, x, incx, results, batch_count);
}

rocblas_status rocblas_scnrm2_batched(rocblas_handle                     handle,
                                      rocblas_int                        n,
                                      const rocblas_float_complex* const x[],
                                      rocblas_int                        incx,
                                      float*                             results,
                                      rocblas_int                        batch_count)
{
    return rocblas_nrm2_batched(handle, n, x, incx, results, batch_count);
}

rocblas_status rocblas_dznrm2_batched(rocblas_handle                      handle,
                                      rocblas_int                         n,
                                      const rocblas_double_complex* const x[],
                                      rocblas_int                         incx,
                                      double*                             results,
                                      rocblas_int                         batch_count)
{
    return rocblas_nrm2_batched(handle, n, x, incx, results, batch_count);
}

} // extern "C"
