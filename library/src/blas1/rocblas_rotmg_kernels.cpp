/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <typename T>
__device__ __host__ void rocblas_rotmg_calc(T& d1, T& d2, T& x1, const T& y1, T* param)
{
    const T gam    = 4096;
    const T gamsq  = gam * gam;
    const T rgamsq = 1 / gamsq;

    T flag = -1;
    T h11 = 0, h21 = 0, h12 = 0, h22 = 0;

    if(d1 < 0)
    {
        d1 = d2 = x1 = 0;
    }
    else
    {
        T p2 = d2 * y1;
        if(p2 == 0)
        {
            flag     = -2;
            param[0] = flag;
            return;
        }
        T p1 = d1 * x1;
        T q2 = p2 * y1;
        T q1 = p1 * x1;
        if(rocblas_abs(q1) > rocblas_abs(q2))
        {
            h21 = -y1 / x1;
            h12 = p2 / p1;
            T u = 1 - h12 * h21;
            if(u > 0)
            {
                flag = 0;
                d1 /= u;
                d2 /= u;
                x1 *= u;
            }
        }
        else
        {
            if(q2 < 0)
            {
                d1 = d2 = x1 = 0;
            }
            else
            {
                flag   = 1;
                h11    = p1 / p2;
                h22    = x1 / y1;
                T u    = 1 + h11 * h22;
                T temp = d2 / u;
                d2     = d1 / u;
                d1     = temp;
                x1     = y1 * u;
            }
        }

        if(d1 != 0)
        {
            while((d1 <= rgamsq) || (d1 >= gamsq))
            {
                if(flag == 0)
                {
                    h11 = h22 = 1;
                    flag      = -1;
                }
                else
                {
                    h21  = -1;
                    h12  = 1;
                    flag = -1;
                }
                if(d1 <= rgamsq)
                {
                    d1 *= gamsq;
                    x1 /= gam;
                    h11 /= gam;
                    h12 /= gam;
                }
                else
                {
                    d1 /= gamsq;
                    x1 *= gam;
                    h11 *= gam;
                    h12 *= gam;
                }
            }
        }

        if(d2 != 0)
        {
            while((rocblas_abs(d2) <= rgamsq) || (rocblas_abs(d2) >= gamsq))
            {
                if(flag == 0)
                {
                    h11 = h22 = 1;
                    flag      = -1;
                }
                else
                {
                    h21  = -1;
                    h12  = 1;
                    flag = -1;
                }
                if(rocblas_abs(d2) <= rgamsq)
                {
                    d2 *= gamsq;
                    h21 /= gam;
                    h22 /= gam;
                }
                else
                {
                    d2 /= gamsq;
                    h21 *= gam;
                    h22 *= gam;
                }
            }
        }
    }

    if(flag < 0)
    {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    }
    else if(flag == 0)
    {
        param[2] = h21;
        param[3] = h12;
    }
    else
    {
        param[1] = h11;
        param[4] = h22;
    }
    param[0] = flag;
}

template <rocblas_int NB, typename T, typename U>
ROCBLAS_KERNEL(NB)
rocblas_rotmg_kernel(T              d1_in,
                     rocblas_stride offset_d1,
                     rocblas_stride stride_d1,
                     T              d2_in,
                     rocblas_stride offset_d2,
                     rocblas_stride stride_d2,
                     T              x1_in,
                     rocblas_stride offset_x1,
                     rocblas_stride stride_x1,
                     U              y1_in,
                     rocblas_stride offset_y1,
                     rocblas_stride stride_y1,
                     T              param,
                     rocblas_stride offset_param,
                     rocblas_stride stride_param,
                     rocblas_int    batch_count)
{
    auto d1 = load_ptr_batch(d1_in, blockIdx.x, offset_d1, stride_d1);
    auto d2 = load_ptr_batch(d2_in, blockIdx.x, offset_d2, stride_d2);
    auto x1 = load_ptr_batch(x1_in, blockIdx.x, offset_x1, stride_x1);
    auto y1 = load_ptr_batch(y1_in, blockIdx.x, offset_y1, stride_y1);
    auto p  = load_ptr_batch(param, blockIdx.x, offset_param, stride_param);
    rocblas_rotmg_calc(*d1, *d2, *x1, *y1, p);
}

template <typename T, typename U>
rocblas_status rocblas_rotmg_template(rocblas_handle handle,
                                      T              d1_in,
                                      rocblas_stride offset_d1,
                                      rocblas_stride stride_d1,
                                      T              d2_in,
                                      rocblas_stride offset_d2,
                                      rocblas_stride stride_d2,
                                      T              x1_in,
                                      rocblas_stride offset_x1,
                                      rocblas_stride stride_x1,
                                      U              y1_in,
                                      rocblas_stride offset_y1,
                                      rocblas_stride stride_y1,
                                      T              param,
                                      rocblas_stride offset_param,
                                      rocblas_stride stride_param,
                                      rocblas_int    batch_count)
{
    if(batch_count <= 0)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();
    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernelGGL(rocblas_rotmg_kernel<1>,
                           batch_count,
                           1,
                           0,
                           rocblas_stream,
                           d1_in,
                           offset_d1,
                           stride_d1,
                           d2_in,
                           offset_d2,
                           stride_d2,
                           x1_in,
                           offset_x1,
                           stride_x1,
                           y1_in,
                           offset_y1,
                           stride_y1,
                           param,
                           offset_param,
                           stride_param,
                           batch_count);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(rocblas_stream));
        // TODO: make this faster for a large number of batches.
        for(int i = 0; i < batch_count; i++)
        {
            auto d1 = load_ptr_batch(d1_in, i, offset_d1, stride_d1);
            auto d2 = load_ptr_batch(d2_in, i, offset_d2, stride_d2);
            auto x1 = load_ptr_batch(x1_in, i, offset_x1, stride_x1);
            auto y1 = load_ptr_batch(y1_in, i, offset_y1, stride_y1);
            auto p  = load_ptr_batch(param, i, offset_param, stride_param);

            rocblas_rotmg_calc(*d1, *d2, *x1, *y1, p);
        }
    }
    return rocblas_status_success;
}

template <typename T, typename U>
ROCBLAS_KERNEL_NO_BOUNDS
    rocblas_rotmg_check_numerics_vector_kernel(T                         d1_in,
                                               rocblas_int               offset_d1,
                                               rocblas_stride            stride_d1,
                                               T                         d2_in,
                                               rocblas_int               offset_d2,
                                               rocblas_stride            stride_d2,
                                               T                         x1_in,
                                               rocblas_int               offset_x1,
                                               rocblas_stride            stride_x1,
                                               U                         y1_in,
                                               rocblas_int               offset_y1,
                                               rocblas_stride            stride_y1,
                                               rocblas_check_numerics_t* abnormal)
{
    auto d1 = load_ptr_batch(d1_in, blockIdx.x, offset_d1, stride_d1);
    auto d2 = load_ptr_batch(d2_in, blockIdx.x, offset_d2, stride_d2);
    auto x1 = load_ptr_batch(x1_in, blockIdx.x, offset_x1, stride_x1);
    auto y1 = load_ptr_batch(y1_in, blockIdx.x, offset_y1, stride_y1);

    //Check every element of the x vector for a NaN/zero/Inf/denormal value
    if(rocblas_iszero(*d1) || rocblas_iszero(*d2) || rocblas_iszero(*x1) || rocblas_iszero(*y1))
        abnormal->has_zero = true;
    if(rocblas_isnan(*d1) || rocblas_isnan(*d2) || rocblas_isnan(*x1) || rocblas_isnan(*y1))
        abnormal->has_NaN = true;
    if(rocblas_isinf(*d1) || rocblas_isinf(*d2) || rocblas_isinf(*x1) || rocblas_isinf(*y1))
        abnormal->has_Inf = true;
    if(rocblas_isdenorm(*d1) || rocblas_isdenorm(*d2) || rocblas_isdenorm(*x1)
       || rocblas_isdenorm(*y1))
        abnormal->has_denorm = true;
}

template <typename T, typename U>
rocblas_status rocblas_rotmg_check_numerics_template(const char*    function_name,
                                                     rocblas_handle handle,
                                                     rocblas_int    n,
                                                     T              d1_in,
                                                     rocblas_stride offset_d1,
                                                     rocblas_stride stride_d1,
                                                     T              d2_in,
                                                     rocblas_stride offset_d2,
                                                     rocblas_stride stride_d2,
                                                     T              x1_in,
                                                     rocblas_stride offset_x1,
                                                     rocblas_stride stride_x1,
                                                     U              y1_in,
                                                     rocblas_stride offset_y1,
                                                     rocblas_stride stride_y1,
                                                     rocblas_int    batch_count,
                                                     const int      check_numerics,
                                                     bool           is_input)
{
    if(!batch_count)
        return rocblas_status_success;

    //Creating structure host object
    rocblas_check_numerics_t h_abnormal;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipStream_t rocblas_stream = handle->get_stream();

        auto d_abnormal = handle->device_malloc(sizeof(rocblas_check_numerics_t));

        //Transferring the rocblas_check_numerics_t structure from host to the device
        RETURN_IF_HIP_ERROR(hipMemcpy((rocblas_check_numerics_t*)d_abnormal,
                                      &h_abnormal,
                                      sizeof(rocblas_check_numerics_t),
                                      hipMemcpyHostToDevice));
        hipLaunchKernelGGL(rocblas_rotmg_check_numerics_vector_kernel,
                           batch_count,
                           1,
                           0,
                           rocblas_stream,
                           d1_in,
                           offset_d1,
                           stride_d1,
                           d2_in,
                           offset_d2,
                           stride_d2,
                           x1_in,
                           offset_x1,
                           stride_x1,
                           y1_in,
                           offset_y1,
                           stride_y1,
                           (rocblas_check_numerics_t*)d_abnormal);

        //Transferring the rocblas_check_numerics_t structure from device to the host
        RETURN_IF_HIP_ERROR(hipMemcpy(&h_abnormal,
                                      (rocblas_check_numerics_t*)d_abnormal,
                                      sizeof(rocblas_check_numerics_t),
                                      hipMemcpyDeviceToHost));
    }
    else
    {
        for(int i = 0; i < batch_count; i++)
        {
            auto d1 = load_ptr_batch(d1_in, i, offset_d1, stride_d1);
            auto d2 = load_ptr_batch(d2_in, i, offset_d2, stride_d2);
            auto x1 = load_ptr_batch(x1_in, i, offset_x1, stride_x1);
            auto y1 = load_ptr_batch(y1_in, i, offset_y1, stride_y1);

            //Check every element of the vectors d1, d2, x1, y1 for a zero/NaN/Inf/denormal value
            if(rocblas_iszero(*d1) || rocblas_iszero(*d2) || rocblas_iszero(*x1)
               || rocblas_iszero(*y1))
                h_abnormal.has_zero = true;
            if(rocblas_isnan(*d1) || rocblas_isnan(*d2) || rocblas_isnan(*x1) || rocblas_isnan(*y1))
                h_abnormal.has_NaN = true;
            if(rocblas_isinf(*d1) || rocblas_isinf(*d2) || rocblas_isinf(*x1) || rocblas_isinf(*y1))
                h_abnormal.has_Inf = true;
            if(rocblas_isdenorm(*d1) || rocblas_isdenorm(*d2) || rocblas_isdenorm(*x1)
               || rocblas_isdenorm(*y1))
                h_abnormal.has_denorm = true;
        }
    }
    return rocblas_check_numerics_abnormal_struct(
        function_name, check_numerics, is_input, &h_abnormal);
}

// If there are any changes in template parameters in the files *rotmg*.cpp
// instantiations below will need to be manually updated to match the changes.

// clang-format off

#ifdef INSTANTIATE_ROTMG_TEMPLATE
#error INSTANTIATE_ROTMG_TEMPLATE already defined
#endif

#define INSTANTIATE_ROTMG_TEMPLATE(T_, U_)                                  \
template rocblas_status rocblas_rotmg_template<T_, U_>                      \
                                              (rocblas_handle handle,       \
                                               T_             d1_in,        \
                                               rocblas_stride offset_d1,    \
                                               rocblas_stride stride_d1,    \
                                               T_             d2_in,        \
                                               rocblas_stride offset_d2,    \
                                               rocblas_stride stride_d2,    \
                                               T_             x1_in,        \
                                               rocblas_stride offset_x1,    \
                                               rocblas_stride stride_x1,    \
                                               U_             y1_in,        \
                                               rocblas_stride offset_y1,    \
                                               rocblas_stride stride_y1,    \
                                               T_             param,        \
                                               rocblas_stride offset_param, \
                                               rocblas_stride stride_param, \
                                               rocblas_int    batch_count);
// instantiate for rocblas_Xrotg and rocblas_Xrotg_strided_batched
INSTANTIATE_ROTMG_TEMPLATE(float*, float const*)
INSTANTIATE_ROTMG_TEMPLATE(double*, double const*)

// instantiate for rocblas_Xrotg_strided_batched
INSTANTIATE_ROTMG_TEMPLATE(float* const*, float const* const*)
INSTANTIATE_ROTMG_TEMPLATE(double* const*, double const* const*)

#undef INSTANTIATE_ROTMG_TEMPLATE

#ifdef INSTANTIATE_ROTMG_CHECK_NUMERICS
#error INSTANTIATE_ROTMG_CHECK_NUMERICS already defined
#endif

#define INSTANTIATE_ROTMG_CHECK_NUMERICS(T_, U_)                               \
template rocblas_status rocblas_rotmg_check_numerics_template <T_, U_>         \
                                               (const char*    function_name,  \
                                                rocblas_handle handle,         \
                                                rocblas_int    n,              \
                                                T_             d1_in,          \
                                                rocblas_stride offset_d1,      \
                                                rocblas_stride stride_d1,      \
                                                T_             d2_in,          \
                                                rocblas_stride offset_d2,      \
                                                rocblas_stride stride_d2,      \
                                                T_             x1_in,          \
                                                rocblas_stride offset_x1,      \
                                                rocblas_stride stride_x1,      \
                                                U_             y1_in,          \
                                                rocblas_stride offset_y1,      \
                                                rocblas_stride stride_y1,      \
                                                rocblas_int    batch_count,    \
                                                const int      check_numerics, \
                                                bool           is_input);

// instantiate for rocblas_Xrotg and rocblas_Xrotg_strided_batched
INSTANTIATE_ROTMG_CHECK_NUMERICS(float*, float const*)
INSTANTIATE_ROTMG_CHECK_NUMERICS(double*, double const*)

// instantiate for rocblas_Xrotg_strided_batched
INSTANTIATE_ROTMG_CHECK_NUMERICS(float* const*, float const* const*)
INSTANTIATE_ROTMG_CHECK_NUMERICS(double* const*, double const* const*)

#undef INSTANTIATE_ROTMG_CHECK_NUMERICS
