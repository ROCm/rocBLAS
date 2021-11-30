/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_rotm.hpp"

template <typename T, typename U>
__device__ void rotm_kernel_calc(rocblas_int    n,
                                 T              x_in,
                                 rocblas_int    offset_x,
                                 rocblas_int    incx,
                                 rocblas_stride stride_x,
                                 T              y_in,
                                 rocblas_int    offset_y,
                                 rocblas_int    incy,
                                 rocblas_stride stride_y,
                                 U              flag,
                                 U              h11,
                                 U              h21,
                                 U              h12,
                                 U              h22)
{
    auto      x   = load_ptr_batch(x_in, hipBlockIdx_y, offset_x, stride_x);
    auto      y   = load_ptr_batch(y_in, hipBlockIdx_y, offset_y, stride_y);
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n && flag != -2)
    {
        auto ix = tid * incx;
        auto iy = tid * incy;
        auto w  = x[ix];
        auto z  = y[iy];
        if(flag < 0)
        {
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            x[ix] = w * h11 + z * h12;
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            y[iy] = w * h21 + z * h22;
        }
        else if(flag == 0)
        {
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            x[ix] = w + z * h12;
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            y[iy] = w * h21 + z;
        }
        else
        {
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            x[ix] = w * h11 + z;
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            y[iy] = -w + z * h22;
        }
    }
}

template <typename T, typename U>
ROCBLAS_KERNEL void rotm_kernel_batched(rocblas_int    n,
                                        T              x_in,
                                        rocblas_int    offset_x,
                                        rocblas_int    incx,
                                        rocblas_stride stride_x,
                                        T              y_in,
                                        rocblas_int    offset_y,
                                        rocblas_int    incy,
                                        rocblas_stride stride_y,
                                        U              param,
                                        rocblas_int    offset_param,
                                        rocblas_stride stride_param)
{
    auto p    = load_ptr_batch(param, hipBlockIdx_y, offset_param, stride_param);
    auto flag = p[0];
    auto h11  = p[1];
    auto h21  = p[2];
    auto h12  = p[3];
    auto h22  = p[4];
    rotm_kernel_calc(n,
                     x_in,
                     offset_x,
                     incx,
                     stride_x,
                     y_in,
                     offset_y,
                     incy,
                     stride_y,
                     flag,
                     h11,
                     h21,
                     h12,
                     h22);
}

template <typename T, typename U>
ROCBLAS_KERNEL void rotm_kernel_regular(rocblas_int    n,
                                        T*             x_in,
                                        rocblas_int    offset_x,
                                        rocblas_int    incx,
                                        rocblas_stride stride_x,
                                        T*             y_in,
                                        rocblas_int    offset_y,
                                        rocblas_int    incy,
                                        rocblas_stride stride_y,
                                        U              flag,
                                        U              h11,
                                        U              h21,
                                        U              h12,
                                        U              h22)
{
    rotm_kernel_calc(n,
                     x_in,
                     offset_x,
                     incx,
                     stride_x,
                     y_in,
                     offset_y,
                     incy,
                     stride_y,
                     load_scalar(flag),
                     load_scalar(h11),
                     load_scalar(h21),
                     load_scalar(h12),
                     load_scalar(h22));
}

// Workaround to avoid constexpr if - Helper function to quick return when param[0] == -2
template <typename T>
bool quick_return_param(rocblas_handle handle, const T* param, rocblas_stride stride_param)
{
    if(rocblas_pointer_mode_host == handle->pointer_mode)
        if(param[0] == -2 && stride_param == 0)
            return true;
    return false;
}

template <typename T>
bool quick_return_param(rocblas_handle handle, const T* const param[], rocblas_stride stride_param)
{
    return false;
}

template <rocblas_int NB, bool BATCHED_OR_STRIDED, typename T, typename U>
rocblas_status rocblas_rotm_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     T              x,
                                     rocblas_int    offset_x,
                                     rocblas_int    incx,
                                     rocblas_stride stride_x,
                                     T              y,
                                     rocblas_int    offset_y,
                                     rocblas_int    incy,
                                     rocblas_stride stride_y,
                                     U              param,
                                     rocblas_int    offset_param,
                                     rocblas_stride stride_param,
                                     rocblas_int    batch_count)
{
    // Quick return if possible
    if(n <= 0 || batch_count <= 0)
        return rocblas_status_success;

    if(quick_return_param(handle, param, stride_param))
        return rocblas_status_success;

    auto shiftx = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;
    auto shifty = incy < 0 ? offset_y - ptrdiff_t(incy) * (n - 1) : offset_y;

    dim3        blocks((n - 1) / NB + 1, batch_count);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->get_stream();

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL(rotm_kernel_batched,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           x,
                           shiftx,
                           incx,
                           stride_x,
                           y,
                           shifty,
                           incy,
                           stride_y,
                           param,
                           offset_param,
                           stride_param);
    else if(!BATCHED_OR_STRIDED)
        hipLaunchKernelGGL(rotm_kernel_regular,
                           blocks,
                           threads,
                           0,
                           rocblas_stream,
                           n,
                           x,
                           shiftx,
                           incx,
                           stride_x,
                           y,
                           shifty,
                           incy,
                           stride_y,
                           param[0],
                           param[1],
                           param[2],
                           param[3],
                           param[4]);
    else // host mode not implemented for (strided_)batched functions
    {
        // TODO: if desired we can use a host for loop to iterate through
        //       batches in this scenario. Currently simply not implemented.
        return rocblas_status_not_implemented;
    }

    return rocblas_status_success;
}

template <typename T>
rocblas_status rocblas_rotm_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           T              y,
                                           rocblas_int    offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

// If there are any changes in template parameters in the files *rotm*.cpp
// instantiations below will need to be manually updated to match the changes.

// instantiate for rocblas_Xrotm and rocblas_Xrotm_strided_batched
template bool quick_return_param<float>(rocblas_handle, float const*, rocblas_stride);
template bool quick_return_param<double>(rocblas_handle, double const*, rocblas_stride);
// instantiate for rocblas_Xrotm__batched
template bool quick_return_param<float>(rocblas_handle, float const* const*, rocblas_stride);
template bool quick_return_param<double>(rocblas_handle, double const* const*, rocblas_stride);

// clang-format off

#ifdef INSTANTIATE_ROTM_CHECK_NUMERICS
#error INSTANTIATE_ROTM_CHECK_NUMERICS already defined
#endif
#define INSTANTIATE_ROTM_CHECK_NUMERICS(T_)                                         \
template rocblas_status rocblas_rotm_check_numerics <T_>                            \
                                                    (const char*    function_name,  \
                                                     rocblas_handle handle,         \
                                                     rocblas_int    n,              \
                                                     T_             x,              \
                                                     rocblas_int    offset_x,       \
                                                     rocblas_int    inc_x,          \
                                                     rocblas_stride stride_x,       \
                                                     T_             y,              \
                                                     rocblas_int    offset_y,       \
                                                     rocblas_int    inc_y,          \
                                                     rocblas_stride stride_y,       \
                                                     rocblas_int    batch_count,    \
                                                     const int      check_numerics, \
                                                     bool           is_input);

// instantiate for rocblas_Xrotm and rocblas_Xrotm_strided_batched
INSTANTIATE_ROTM_CHECK_NUMERICS(double*)
INSTANTIATE_ROTM_CHECK_NUMERICS( float*)
// instantiate for rocblas_Xrotm__batched
INSTANTIATE_ROTM_CHECK_NUMERICS(double* const*)
INSTANTIATE_ROTM_CHECK_NUMERICS( float* const*)

#undef INSTANTIATE_ROTM_CHECK_NUMERICS

#ifdef INSTANTIATE_ROTM_TEMPLATE
#error INSTANTIATE_ROTM_TEMPLATE already defined
#endif

#define INSTANTIATE_ROTM_TEMPLATE(NB_, BATCHED_OR_STRIDED_, T_, U_)              \
template rocblas_status rocblas_rotm_template <NB_, BATCHED_OR_STRIDED_, T_, U_> \
                                              (rocblas_handle handle,            \
                                               rocblas_int    n,                 \
                                               T_             x,                 \
                                               rocblas_int    offset_x,          \
                                               rocblas_int    incx,              \
                                               rocblas_stride stride_x,          \
                                               T_             y,                 \
                                               rocblas_int    offset_y,          \
                                               rocblas_int    incy,              \
                                               rocblas_stride stride_y,          \
                                               U_             param,             \
                                               rocblas_int    offset_param,      \
                                               rocblas_stride stride_param,      \
                                               rocblas_int    batch_count);

// instantiate for rocblas_Xrotm and rocblas_Xrotm_strided_batched
INSTANTIATE_ROTM_TEMPLATE(512,  true,  float*,         float const*);
INSTANTIATE_ROTM_TEMPLATE(512, false,  float*,         float const*);
INSTANTIATE_ROTM_TEMPLATE(512,  true, double*,        double const*);
INSTANTIATE_ROTM_TEMPLATE(512, false, double*,        double const*);
// instantiate for rocblas_Xrotm__batched
INSTANTIATE_ROTM_TEMPLATE(512,  true,  float* const*,  float const* const*);
INSTANTIATE_ROTM_TEMPLATE(512,  true, double* const*, double const* const*);

#undef INSTANTIATE_ROTM_TEMPLATE

// clang-format on
