/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "reduction.h"
#include "fetch_template.h"
#include "rocblas_unique_ptr.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {

template <class To>
struct rocblas_fetch_asum
{
    template <typename Ti>
    __forceinline__ __device__ To operator()(Ti x, ssize_t)
    {
        return {fetch_asum(x)};
    }
};

template <typename>
constexpr char rocblas_asum_name[] = "unknown";
template <>
constexpr char rocblas_asum_name<float>[] = "rocblas_sasum";
template <>
constexpr char rocblas_asum_name<double>[] = "rocblas_dasum";
template <>
constexpr char rocblas_asum_name<rocblas_float_complex>[] = "rocblas_scasum";
template <>
constexpr char rocblas_asum_name<rocblas_double_complex>[] = "rocblas_dzasum";

/* ============================================================================================ */

/*! \brief BLAS Level 1 API

    \details
    asum computes the sum of the absolute values of elements of a real vector x,
         or the sum of absolute values of the real and imaginary parts of elements if x is a complex
   vector

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of y.
    @param[inout]
    result
              store the asum product. either on the host CPU or device GPU.
              result is 0 if n <= 0 or incx <= 0.
    ********************************************************************/

// allocate workspace inside this API
template <typename Ti, typename To>
rocblas_status
rocblas_asum(rocblas_handle handle, rocblas_int n, const Ti* x, rocblas_int incx, To* result)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, rocblas_asum_name<Ti>, n, x, incx);

    if(layer_mode & rocblas_layer_mode_log_bench)
        log_bench(handle,
                  "./rocblas-bench -f asum -r",
                  rocblas_precision_string<Ti>,
                  "-n",
                  n,
                  "--incx",
                  incx);

    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle, rocblas_asum_name<Ti>, "N", n, "incx", incx);

    if(!x || !result)
        return rocblas_status_invalid_pointer;

    /*
     * Quick return if possible.
     */
    if(n <= 0 || incx <= 0)
    {
        if(rocblas_pointer_mode_device == handle->pointer_mode)
            RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(*result)));
        else
            *result = 0;
        return rocblas_status_success;
    }

    // HIP support up to 1024 threads/work itmes per thread block/work group
    static constexpr int NB = 512;
    auto blocks             = (n - 1) / NB + 1;
    auto workspace =
        rocblas_unique_ptr{rocblas::device_malloc(sizeof(To) * blocks), rocblas::device_free};
    if(!workspace)
        return rocblas_status_memory_error;

    auto status = rocblas_reduction_kernel<NB, rocblas_fetch_asum<To>>(
        handle, n, x, incx, result, (To*)workspace.get(), blocks);

    return status;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status
rocblas_sasum(rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result)
{
    return rocblas_asum(handle, n, x, incx, result);
}

rocblas_status rocblas_dasum(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result)
{
    return rocblas_asum(handle, n, x, incx, result);
}

#if 0 // complex not supported

rocblas_status rocblas_scasum(rocblas_handle handle,
                              rocblas_int n,
                              const rocblas_float_complex* x,
                              rocblas_int incx,
                              float* result)
{
    return rocblas_asum(handle, n, x, incx, result);
}

rocblas_status rocblas_dzasum(rocblas_handle handle,
                              rocblas_int n,
                              const rocblas_double_complex* x,
                              rocblas_int incx,
                              double* result)
{
    return rocblas_asum(handle, n, x, incx, result);
}

#endif

} // extern "C"
