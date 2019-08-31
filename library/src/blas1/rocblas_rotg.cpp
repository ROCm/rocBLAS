/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

namespace
{
    template <typename T, typename U, typename std::enable_if<!is_complex<T>, int>::type = 0>
    __device__ __host__ void rotg_calc(T& a, T& b, U& c, T& s)
    {
        T scale = rocblas_abs(a) + rocblas_abs(b);
        if(scale == 0.0)
        {
            c = 1.0;
            s = 0.0;
            a = 0.0;
            b = 0.0;
        }
        else
        {
            T sa  = a / scale;
            T sb  = b / scale;
            T r   = scale * sqrt(sa * sa + sb * sb);
            T roe = rocblas_abs(a) > rocblas_abs(b) ? a : b;
            r     = copysign(r, roe);
            c     = a / r;
            s     = b / r;
            T z   = 1.0;
            if(rocblas_abs(a) > rocblas_abs(b))
                z = s;
            if(rocblas_abs(b) >= rocblas_abs(a) && c != 0.0)
                z = 1.0 / c;
            a = r;
            b = z;
        }
    }

    template <typename T, typename U, typename std::enable_if<is_complex<T>, int>::type = 0>
    __device__ __host__ void rotg_calc(T& a, T& b, U& c, T& s)
    {
        if(!rocblas_abs(a))
        {
            c = 0;
            s = {1, 0};
            a = b;
        }
        else
        {
            auto scale = rocblas_abs(a) + rocblas_abs(b);
            auto sa    = rocblas_abs(a / scale);
            auto sb    = rocblas_abs(b / scale);
            auto norm  = scale * sqrt(sa * sa + sb * sb);
            auto alpha = a / rocblas_abs(a);
            c          = rocblas_abs(a) / norm;
            s          = alpha * conj(b) / norm;
            a          = alpha * norm;
        }
    }

    template <typename T, typename U>
    __global__ void rotg_kernel(T* a, T* b, U* c, T* s)
    {
        rotg_calc(*a, *b, *c, *s);
    }

    template <typename>
    constexpr char rocblas_rotg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotg_name<float>[] = "rocblas_srotg";
    template <>
    constexpr char rocblas_rotg_name<double>[] = "rocblas_drotg";
    template <>
    constexpr char rocblas_rotg_name<rocblas_float_complex>[] = "rocblas_crotg";
    template <>
    constexpr char rocblas_rotg_name<rocblas_double_complex>[] = "rocblas_zrotg";

    template <class T, class U>
    rocblas_status rocblas_rotg(rocblas_handle handle, T* a, T* b, U* c, T* s)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotg_name<T>, a, b, c, s);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle, "./rocblas-bench -f rotg -r", rocblas_precision_string<T>);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle, rocblas_rotg_name<T>);

        if(!a || !b || !c || !s)
            return rocblas_status_invalid_pointer;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        hipStream_t rocblas_stream = handle->rocblas_stream;

        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            hipLaunchKernelGGL(rotg_kernel, 1, 1, 0, rocblas_stream, a, b, c, s);
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(rocblas_stream));
            rotg_calc(*a, *b, *c, *s);
        }

        return rocblas_status_success;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_srotg(rocblas_handle handle, float* a, float* b, float* c, float* s)
{
    return rocblas_rotg(handle, a, b, c, s);
}

rocblas_status rocblas_drotg(rocblas_handle handle, double* a, double* b, double* c, double* s)
{
    return rocblas_rotg(handle, a, b, c, s);
}

rocblas_status rocblas_crotg(rocblas_handle         handle,
                             rocblas_float_complex* a,
                             rocblas_float_complex* b,
                             float*                 c,
                             rocblas_float_complex* s)
{
    return rocblas_rotg(handle, a, b, c, s);
}

rocblas_status rocblas_zrotg(rocblas_handle          handle,
                             rocblas_double_complex* a,
                             rocblas_double_complex* b,
                             double*                 c,
                             rocblas_double_complex* s)
{
    return rocblas_rotg(handle, a, b, c, s);
}

} // extern "C"
