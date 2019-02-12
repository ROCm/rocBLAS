/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "fetch_template.h"
#include "reduction.h"
#include "rocblas_unique_ptr.hpp"
#include "handle.h"
#include "logging.h"
#include "utility.h"

#define QUOTE2(S) #S
#define QUOTE(S) QUOTE2(S)

#define JOIN2(A, B) A##B
#define JOIN(A, B) JOIN2(A, B)

// pair of value and index
template <typename T>
struct index_value_t
{
    // important: index must come first, so that index_value_t* can be cast to rocblas_int*
    rocblas_int index;
    T value;
};

// Specialization of default_value for index_value_t<T>
template <typename T>
struct default_value<index_value_t<T>>
{
    __forceinline__ __host__ __device__ constexpr auto operator()() const
    {
        index_value_t<T> x;
        x.index = -1;
        return x;
    }
};

// Fetch absolute value
template <typename To>
struct rocblas_fetch_amax_amin
{
    template <typename Ti>
    __forceinline__ __host__ __device__ index_value_t<To> operator()(Ti x, rocblas_int index)
    {
        return {index, fetch_asum(x)};
    }
};

// Replaces x with y if y.value < x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amax
{
    template <typename To>
    __forceinline__ __host__ __device__ void operator()(index_value_t<To>& __restrict__ x,
                                                        const index_value_t<To>& __restrict__ y)
    {
        // If y.index == -1 then y.value is invalid and should not be compared
        if(y.index != -1)
        {
            if(x.index == -1 || y.value > x.value)
                x = y; // if larger or smaller, update max/min and index
            else if(y.index < x.index && x.value == y.value)
                x.index = y.index; // if equal, choose smaller index
        }
    }
};

// Replaces x with y if y.value < x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ void operator()(index_value_t<To>& __restrict__ x,
                                                        const index_value_t<To>& __restrict__ y)
    {
        // If y.index == -1 then y.value is invalid and should not be compared
        if(y.index != -1)
        {
            if(x.index == -1 || y.value < x.value)
                x = y; // if larger or smaller, update max/min and index
            else if(y.index < x.index && x.value == y.value)
                x.index = y.index; // if equal, choose smaller index
        }
    }
};

struct rocblas_finalize_amax_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ rocblas_int operator()(index_value_t<To> x)
    {
        return x.index + 1;
    }
};

template <typename>
constexpr char rocblas_iamaxmin_name[] = "unknown";
template <>
constexpr char rocblas_iamaxmin_name<float>[] = "rocblas_isa" QUOTE(MAX_MIN);
template <>
constexpr char rocblas_iamaxmin_name<double>[] = "rocblas_ida" QUOTE(MAX_MIN);
template <>
constexpr char rocblas_iamaxmin_name<rocblas_float_complex>[] = "rocblas_ica" QUOTE(MAX_MIN);
template <>
constexpr char rocblas_iamaxmin_name<rocblas_double_complex>[] = "rocblas_iza" QUOTE(MAX_MIN);

/* ============================================================================================ */

/*! \brief BLAS Level 1 API

    \details
    iamaxmin finds the first index of the element of maximum magnitude of real vector x
         or the sum of magnitude of the real and imaginary parts of elements if x is a complex
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
              index of max element. either on the host CPU or device GPU.
              return is 0 if n <= 0 or incx <= 0. Note that 1 based indexing
              (Fortran) is used, not 0 based indexing (C).
    ********************************************************************/

// allocate workspace inside this API
template <typename To, typename Ti>
static rocblas_status rocblas_iamaxmin(
    rocblas_handle handle, rocblas_int n, const Ti* x, rocblas_int incx, rocblas_int* result)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    auto layer_mode = handle->layer_mode;

    if(layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, rocblas_iamaxmin_name<Ti>, n, x, incx);

    if(layer_mode & rocblas_layer_mode_log_bench)
        log_bench(handle,
                  "./rocblas-bench -f ia" QUOTE(MAX_MIN) " -r",
                  rocblas_precision_string<Ti>,
                  "-n",
                  n,
                  "--incx",
                  incx);

    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle, rocblas_iamaxmin_name<Ti>, "N", n, "incx", incx);

    if(!x || !result)
        return rocblas_status_invalid_pointer;
    /*
     * Quick return if possible.
     */
    if(n <= 0 || incx <= 0)
    {
        if(handle->pointer_mode == rocblas_pointer_mode_device)
            RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(*result)));
        else
            *result = 0;
        return rocblas_status_success;
    }

    // HIP support up to 1024 threads/work itmes per thread block/work group
    static constexpr int NB = 1024;
    rocblas_int blocks      = (n - 1) / NB + 1;

    auto workspace = rocblas_unique_ptr{rocblas::device_malloc(sizeof(index_value_t<To>) * blocks),
                                        rocblas::device_free};
    if(!workspace)
        return rocblas_status_memory_error;

    auto status = rocblas_reduction_kernel<NB,
                                           rocblas_fetch_amax_amin<To>,
                                           AMAX_AMIN_REDUCTION,
                                           rocblas_finalize_amax_amin>(
        handle, n, x, incx, result, (index_value_t<To>*)workspace.get(), blocks);

    return status;
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

rocblas_status JOIN(rocblas_ida, MAX_MIN)(
    rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result)
{
    return rocblas_iamaxmin<double>(handle, n, x, incx, result);
}

#if 0 // complex not supported

rocblas_status JOIN(rocblas_isca, MAX_MIN)(rocblas_handle handle,
                                           rocblas_int n,
                                           const rocblas_float_complex* x,
                                           rocblas_int incx,
                                           rocblas_int* result)
{
    return rocblas_iamaxmin<float>(handle, n, x, incx, result);
}

rocblas_status JOIN(rocblas_idza, MAX_MIN)(rocblas_handle handle,
                                           rocblas_int n,
                                           const rocblas_double_complex* x,
                                           rocblas_int incx,
                                           rocblas_int* result)
{
    return rocblas_iamaxmin<double>(handle, n, x, incx, result);
}

#endif

} // extern "C"
