/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_rotg.hpp"

template <typename T, typename U, std::enable_if_t<!is_complex<T>, int> = 0>
__device__ __host__ void rocblas_rotg_calc(T& a, T& b, U& c, T& s)
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

template <typename T, typename U, std::enable_if_t<is_complex<T>, int> = 0>
__device__ __host__ void rocblas_rotg_calc(T& a, T& b, U& c, T& s)
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

template <rocblas_int NB, typename T, typename U>
ROCBLAS_KERNEL(NB)
rocblas_rotg_kernel(T              a_in,
                    rocblas_int    offset_a,
                    rocblas_stride stride_a,
                    T              b_in,
                    rocblas_int    offset_b,
                    rocblas_stride stride_b,
                    U              c_in,
                    rocblas_int    offset_c,
                    rocblas_stride stride_c,
                    T              s_in,
                    rocblas_int    offset_s,
                    rocblas_stride stride_s)
{
    auto a = load_ptr_batch(a_in, hipBlockIdx_x, offset_a, stride_a);
    auto b = load_ptr_batch(b_in, hipBlockIdx_x, offset_b, stride_b);
    auto c = load_ptr_batch(c_in, hipBlockIdx_x, offset_c, stride_c);
    auto s = load_ptr_batch(s_in, hipBlockIdx_x, offset_s, stride_s);
    rocblas_rotg_calc(*a, *b, *c, *s);
}

template <typename T, typename U>
rocblas_status rocblas_rotg_template(rocblas_handle handle,
                                     T              a_in,
                                     rocblas_int    offset_a,
                                     rocblas_stride stride_a,
                                     T              b_in,
                                     rocblas_int    offset_b,
                                     rocblas_stride stride_b,
                                     U              c_in,
                                     rocblas_int    offset_c,
                                     rocblas_stride stride_c,
                                     T              s_in,
                                     rocblas_int    offset_s,
                                     rocblas_stride stride_s,
                                     rocblas_int    batch_count)
{
    if(!batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        hipLaunchKernelGGL(rocblas_rotg_kernel<1>,
                           batch_count,
                           1,
                           0,
                           rocblas_stream,
                           a_in,
                           offset_a,
                           stride_a,
                           b_in,
                           offset_b,
                           stride_b,
                           c_in,
                           offset_c,
                           stride_c,
                           s_in,
                           offset_s,
                           stride_s);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(rocblas_stream));
        // TODO: make this faster for a large number of batches.
        for(int i = 0; i < batch_count; i++)
        {
            auto a = load_ptr_batch(a_in, i, offset_a, stride_a);
            auto b = load_ptr_batch(b_in, i, offset_b, stride_b);
            auto c = load_ptr_batch(c_in, i, offset_c, stride_c);
            auto s = load_ptr_batch(s_in, i, offset_s, stride_s);

            rocblas_rotg_calc(*a, *b, *c, *s);
        }
    }

    return rocblas_status_success;
}

template <typename T, typename U>
ROCBLAS_KERNEL_NO_BOUNDS
    rocblas_rotg_check_numerics_vector_kernel(T                         a_in,
                                              rocblas_int               offset_a,
                                              rocblas_stride            stride_a,
                                              T                         b_in,
                                              rocblas_int               offset_b,
                                              rocblas_stride            stride_b,
                                              U                         c_in,
                                              rocblas_int               offset_c,
                                              rocblas_stride            stride_c,
                                              T                         s_in,
                                              rocblas_int               offset_s,
                                              rocblas_stride            stride_s,
                                              rocblas_check_numerics_t* abnormal)
{
    auto a = load_ptr_batch(a_in, hipBlockIdx_x, offset_a, stride_a);
    auto b = load_ptr_batch(b_in, hipBlockIdx_x, offset_b, stride_b);
    auto c = load_ptr_batch(c_in, hipBlockIdx_x, offset_c, stride_c);
    auto s = load_ptr_batch(s_in, hipBlockIdx_x, offset_s, stride_s);

    //Check every element of the vectors a, b, c, s for a zero/NaN/Inf
    if(rocblas_iszero(*a) || rocblas_iszero(*b) || rocblas_iszero(*c) || rocblas_iszero(*s))
        abnormal->has_zero = true;
    if(rocblas_isnan(*a) || rocblas_isnan(*b) || rocblas_isnan(*c) || rocblas_isnan(*s))
        abnormal->has_NaN = true;
    if(rocblas_isinf(*a) || rocblas_isinf(*b) || rocblas_isinf(*c) || rocblas_isinf(*s))
        abnormal->has_Inf = true;
}

template <typename T, typename U>
rocblas_status rocblas_rotg_check_numerics_template(const char*    function_name,
                                                    rocblas_handle handle,
                                                    rocblas_int    n,
                                                    T              a_in,
                                                    rocblas_int    offset_a,
                                                    rocblas_stride stride_a,
                                                    T              b_in,
                                                    rocblas_int    offset_b,
                                                    rocblas_stride stride_b,
                                                    U              c_in,
                                                    rocblas_int    offset_c,
                                                    rocblas_stride stride_c,
                                                    T              s_in,
                                                    rocblas_int    offset_s,
                                                    rocblas_stride stride_s,
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

        hipLaunchKernelGGL(rocblas_rotg_check_numerics_vector_kernel,
                           batch_count,
                           1,
                           0,
                           rocblas_stream,
                           a_in,
                           offset_a,
                           stride_a,
                           b_in,
                           offset_b,
                           stride_b,
                           c_in,
                           offset_c,
                           stride_c,
                           s_in,
                           offset_s,
                           stride_s,
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
            auto a = load_ptr_batch(a_in, i, offset_a, stride_a);
            auto b = load_ptr_batch(b_in, i, offset_b, stride_b);
            auto c = load_ptr_batch(c_in, i, offset_c, stride_c);
            auto s = load_ptr_batch(s_in, i, offset_s, stride_s);

            //Check every element of the x vector for a NaN/zero/Inf
            if(rocblas_iszero(*a) || rocblas_iszero(*b) || rocblas_iszero(*c) || rocblas_iszero(*s))
                h_abnormal.has_zero = true;
            if(rocblas_isnan(*a) || rocblas_isnan(*b) || rocblas_isnan(*c) || rocblas_isnan(*s))
                h_abnormal.has_NaN = true;
            if(rocblas_isinf(*a) || rocblas_isinf(*b) || rocblas_isinf(*c) || rocblas_isinf(*s))
                h_abnormal.has_Inf = true;
        }
    }
    return rocblas_check_numerics_abnormal_struct(
        function_name, check_numerics, is_input, &h_abnormal);
}

// If there are any changes in template parameters in the files *rotg*.cpp
// instantiations below will need to be manually updated to match the changes.

// clang-format off
#ifdef INSTANTIATE_ROTG_TEMPLATE
#error INSTANTIATE_ROTG_TEMPLATE already defined
#endif

#define INSTANTIATE_ROTG_TEMPLATE(T_, U_)                                  \
template rocblas_status rocblas_rotg_template <T_, U_>                     \
                                              (rocblas_handle handle,      \
                                               T_             a_in,        \
                                               rocblas_int    offset_a,    \
                                               rocblas_stride stride_a,    \
                                               T_             b_in,        \
                                               rocblas_int    offset_b,    \
                                               rocblas_stride stride_b,    \
                                               U_             c_in,        \
                                               rocblas_int    offset_c,    \
                                               rocblas_stride stride_c,    \
                                               T_             s_in,        \
                                               rocblas_int    offset_s,    \
                                               rocblas_stride stride_s,    \
                                               rocblas_int    batch_count);

// instantiate for rocblas_Xrotg and rocblas_Xrotg_strided_batched
INSTANTIATE_ROTG_TEMPLATE(float*, float*)
INSTANTIATE_ROTG_TEMPLATE(double*, double*)
INSTANTIATE_ROTG_TEMPLATE(rocblas_float_complex*, float*)
INSTANTIATE_ROTG_TEMPLATE(rocblas_double_complex*, double*)

// instantiate for rocblas_Xrotg_strided_batched
INSTANTIATE_ROTG_TEMPLATE(float* const*, float* const*)
INSTANTIATE_ROTG_TEMPLATE(double* const*, double* const*)
INSTANTIATE_ROTG_TEMPLATE(rocblas_float_complex* const*, float* const*)
INSTANTIATE_ROTG_TEMPLATE(rocblas_double_complex* const*, double* const*)

#undef INSTANTIATE_ROTG_TEMPLATE

#ifdef INSTANTIATE_ROTG_CHECK_NUMERICS
#error INSTANTIATE_ROTG_CHECK_NUMERICS already defined
#endif

#define INSTANTIATE_ROTG_CHECK_NUMERICS(T_, U_)                                             \
template rocblas_status rocblas_rotg_check_numerics_template<T_, U_>                        \
                                                            (const char*    function_name,  \
                                                             rocblas_handle handle,         \
                                                             rocblas_int    n,              \
                                                             T_             a_in,           \
                                                             rocblas_int    offset_a,       \
                                                             rocblas_stride stride_a,       \
                                                             T_             b_in,           \
                                                             rocblas_int    offset_b,       \
                                                             rocblas_stride stride_b,       \
                                                             U_             c_in,           \
                                                             rocblas_int    offset_c,       \
                                                             rocblas_stride stride_c,       \
                                                             T_             s_in,           \
                                                             rocblas_int    offset_s,       \
                                                             rocblas_stride stride_s,       \
                                                             rocblas_int    batch_count,    \
                                                             const int      check_numerics, \
                                                             bool           is_input);

// instantiate for rocblas_Xrotg and rocblas_Xrotg_strided_batched
INSTANTIATE_ROTG_CHECK_NUMERICS(float*, float*)
INSTANTIATE_ROTG_CHECK_NUMERICS(double*, double*)
INSTANTIATE_ROTG_CHECK_NUMERICS(rocblas_float_complex*, float*)
INSTANTIATE_ROTG_CHECK_NUMERICS(rocblas_double_complex*, double*)

// instantiate for rocblas_Xrotg_strided_batched
INSTANTIATE_ROTG_CHECK_NUMERICS(float* const*, float* const*)
INSTANTIATE_ROTG_CHECK_NUMERICS(double* const*, double* const*)
INSTANTIATE_ROTG_CHECK_NUMERICS(rocblas_float_complex* const*, float* const*)
INSTANTIATE_ROTG_CHECK_NUMERICS(rocblas_double_complex* const*, double* const*)

#undef INSTANTIATE_ROTG_CHECK_NUMERICS

// clang-format off
