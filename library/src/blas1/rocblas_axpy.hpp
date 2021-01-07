/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"

//!
//! @brief General kernel (batched, strided batched) of axpy.
//!
template <rocblas_int NB, typename Tex, typename Ta, typename Tx, typename Ty>
__global__ __launch_bounds__(NB) void axpy_kernel(rocblas_int    n,
                                                  Ta             alpha_device_host,
                                                  Tx             x,
                                                  rocblas_int    incx,
                                                  ptrdiff_t      offsetx,
                                                  rocblas_stride stridex,
                                                  Ty             y,
                                                  rocblas_int    incy,
                                                  ptrdiff_t      offsety,
                                                  rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host);
    if(!alpha)
    {
        return;
    }

    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        auto tx = load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
        auto ty = load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);

        *ty = (*ty) + Tex(alpha) * (*tx);
    }
}

//!
//! @brief Large batch size kernel (batched, strided batched) of axpy.
//!
template <int DIM_X, int DIM_Y, typename Tex, typename Ta, typename Tx, typename Ty>
__global__ __launch_bounds__(DIM_X* DIM_Y) void axpy_kernel_batched(rocblas_int n,
                                                                    Ta          alpha_device_host,
                                                                    Tx          x,
                                                                    rocblas_int incx,
                                                                    ptrdiff_t   offsetx,
                                                                    rocblas_stride stridex,
                                                                    Ty             y,
                                                                    rocblas_int    incy,
                                                                    ptrdiff_t      offsety,
                                                                    rocblas_stride stridey,
                                                                    rocblas_int    batch_count)
{
    auto alpha = load_scalar(alpha_device_host);
    if(!alpha)
    {
        return;
    }
    Tex ex_alph = Tex(alpha);

    ptrdiff_t tid = hipBlockIdx_x * DIM_X + hipThreadIdx_x;
    int       bid = 4 * (hipBlockIdx_y * DIM_Y + hipThreadIdx_y);
    if(tid < n)
    {
        offsetx += tid * incx;
        offsety += tid * incy;

        for(int i = 0; i < 4; i++)
        {
            if(bid + i < batch_count)
            {
                auto tx = load_ptr_batch(x, bid + i, offsetx, stridex);
                auto ty = load_ptr_batch(y, bid + i, offsety, stridey);

                *ty += ex_alph * (*tx);
            }
        }
    }
}

//!
//! @brief Optimized kernel for the remaining part of 8 half floating points.
//! @remark Increment are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename Ta, typename Tx, typename Ty>
__global__ __launch_bounds__(NB) void haxpy_mod_8_kernel(rocblas_int    n_mod_8,
                                                         Ta             alpha_device_host,
                                                         Tx             x,
                                                         ptrdiff_t      offsetx,
                                                         rocblas_stride stridex,
                                                         Ty             y,
                                                         ptrdiff_t      offsety,
                                                         rocblas_stride stridey)
{
    auto      alpha = load_scalar(alpha_device_host);
    ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n_mod_8)
    {
        auto tx = load_ptr_batch(x, hipBlockIdx_y, offsetx + tid, stridex);
        auto ty = load_ptr_batch(y, hipBlockIdx_y, offsety + tid, stridey);
        *ty += alpha * (*tx);
    }
}

//!
//! @brief Optimized kernel for the groups of 8 half floating points.
//!
template <rocblas_int NB, typename Ta, typename Tx, typename Ty>
__global__ __launch_bounds__(NB) void haxpy_mlt_8_kernel(rocblas_int    n_mlt_8,
                                                         Ta             alpha_device_host,
                                                         Tx             x,
                                                         rocblas_stride stridex,
                                                         Ty             y,
                                                         rocblas_stride stridey)
{

    union
    {
        rocblas_half2 value;
        uint32_t      data;
    } alpha_h2 = {load_scalar(alpha_device_host)};

    if(!(alpha_h2.data & 0x7fff))
    {
        return;
    }

    ptrdiff_t t8id = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    rocblas_half2 y0, y1, y2, y3;
    rocblas_half2 x0, x1, x2, x3;
    rocblas_half2 z0, z1, z2, z3;

    auto tid = t8id * 8;
    if(tid < n_mlt_8)
    {
        //
        // Cast to rocblas_half8.
        // The reason rocblas_half8 does not appear in the signature
        // is due to the generalization of the non-batched/batched/strided batched case.
        // But the purpose of this routine is to specifically doing calculation with rocblas_half8 but also being general.
        // Then we can consider it is acceptable.
        //
        const rocblas_half8* ax
            = (const rocblas_half8*)load_ptr_batch(x, hipBlockIdx_y, tid, stridex);
        rocblas_half8* ay = (rocblas_half8*)load_ptr_batch(y, hipBlockIdx_y, tid, stridey);

        y0[0] = (*ay)[0];
        y0[1] = (*ay)[1];
        y1[0] = (*ay)[2];
        y1[1] = (*ay)[3];
        y2[0] = (*ay)[4];
        y2[1] = (*ay)[5];
        y3[0] = (*ay)[6];
        y3[1] = (*ay)[7];

        x0[0] = (*ax)[0];
        x0[1] = (*ax)[1];
        x1[0] = (*ax)[2];
        x1[1] = (*ax)[3];
        x2[0] = (*ax)[4];
        x2[1] = (*ax)[5];
        x3[0] = (*ax)[6];
        x3[1] = (*ax)[7];

        z0 = rocblas_fmadd_half2(alpha_h2.value, x0, y0);
        z1 = rocblas_fmadd_half2(alpha_h2.value, x1, y1);
        z2 = rocblas_fmadd_half2(alpha_h2.value, x2, y2);
        z3 = rocblas_fmadd_half2(alpha_h2.value, x3, y3);

        (*ay)[0] = z0[0];
        (*ay)[1] = z0[1];
        (*ay)[2] = z1[0];
        (*ay)[3] = z1[1];
        (*ay)[4] = z2[0];
        (*ay)[5] = z2[1];
        (*ay)[6] = z3[0];
        (*ay)[7] = z3[1];
    }
}

//!
//! @brief General template to compute y = a * x + y.
//!
template <int NB, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_axpy_template(rocblas_handle handle,
                                                             rocblas_int    n,
                                                             const Ta*      alpha,
                                                             Tx             x,
                                                             rocblas_int    incx,
                                                             rocblas_stride stridex,
                                                             Ty             y,
                                                             rocblas_int    incy,
                                                             rocblas_stride stridey,
                                                             rocblas_int    batch_count)
{
    // Temporarily change the thread's default device ID to the handle's device ID
    auto saved_device_id = handle->push_device_id();

    //
    // Using rocblas_half ?
    //
    static constexpr bool using_rocblas_half
        = std::is_same<Ta, rocblas_half>::value && std::is_same<Tex, rocblas_half>::value;

    if(n <= 0 || batch_count <= 0) // Quick return if possible. Not Argument error
    {
        return rocblas_status_success;
    }

    //
    // If not using rocblas_half otherwise only if incx == 1  && incy == 1.
    //
    bool non_unit_inc = (incx != 1 || incy != 1);
    if(!using_rocblas_half || non_unit_inc)
    {
        ptrdiff_t offsetx = (incx < 0) ? ptrdiff_t(incx) * (1 - n) : 0;
        ptrdiff_t offsety = (incy < 0) ? ptrdiff_t(incy) * (1 - n) : 0;

        if(batch_count < 8192 || !std::is_same<Ta, float>::value)
        {
            // Default calculation
            dim3 blocks((n - 1) / (NB) + 1, batch_count);
            dim3 threads(NB);
            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                hipLaunchKernelGGL((axpy_kernel<NB, Tex>),
                                   blocks,
                                   threads,
                                   0,
                                   handle->get_stream(),
                                   n,
                                   alpha,
                                   x,
                                   incx,
                                   offsetx,
                                   stridex,
                                   y,
                                   incy,
                                   offsety,
                                   stridey);
            }
            else
            {
                hipLaunchKernelGGL((axpy_kernel<NB, Tex>),
                                   blocks,
                                   threads,
                                   0,
                                   handle->get_stream(),
                                   n,
                                   *alpha,
                                   x,
                                   incx,
                                   offsetx,
                                   stridex,
                                   y,
                                   incy,
                                   offsety,
                                   stridey);
            }
        }
        else
        {
            constexpr int DIM_X = 128;
            constexpr int DIM_Y = 8;

            dim3 blocks((n - 1) / (DIM_X) + 1, (batch_count - 1) / (DIM_Y * 4) + 1);
            dim3 threads(DIM_X, DIM_Y);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                hipLaunchKernelGGL((axpy_kernel_batched<DIM_X, DIM_Y, Tex>),
                                   blocks,
                                   threads,
                                   0,
                                   handle->get_stream(),
                                   n,
                                   alpha,
                                   x,
                                   incx,
                                   offsetx,
                                   stridex,
                                   y,
                                   incy,
                                   offsety,
                                   stridey,
                                   batch_count);
            }
            else
            {
                hipLaunchKernelGGL((axpy_kernel_batched<DIM_X, DIM_Y, Tex>),
                                   blocks,
                                   threads,
                                   0,
                                   handle->get_stream(),
                                   n,
                                   *alpha,
                                   x,
                                   incx,
                                   offsetx,
                                   stridex,
                                   y,
                                   incy,
                                   offsety,
                                   stridey,
                                   batch_count);
            }
        }
    }
    else // it is using rocblas_half with increments equal to 1.
    {

        //
        // Optimized version of rocblas_half, where incx == 1 and incy == 1.
        // TODO: always use an optimized version.
        //
        //
        // Note: Do not use pointer arithmetic with x and y when passing parameters.
        // The kernel will do the cast if needed.
        //
        rocblas_int n_mod_8 = n & 7; // n mod 8
        rocblas_int n_mlt_8 = n & ~(rocblas_int)7; // multiple of 8
        int         blocks  = (n / 8 - 1) / NB + 1;
        dim3        grid(blocks, batch_count);
        dim3        threads(NB);
        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((haxpy_mlt_8_kernel<NB>),
                               grid,
                               threads,
                               0,
                               handle->get_stream(),
                               n_mlt_8,
                               (const rocblas_half2*)alpha,
                               x,
                               stridex,
                               y,
                               stridey);

            if(n_mod_8)
            {
                //
                // cleanup non-multiple of 8
                //
                hipLaunchKernelGGL((haxpy_mod_8_kernel<NB>),
                                   dim3(1, batch_count),
                                   n_mod_8,
                                   0,
                                   handle->get_stream(),
                                   n_mod_8,
                                   alpha,
                                   x,
                                   n_mlt_8,
                                   stridex,
                                   y,
                                   n_mlt_8,
                                   stridey);
            }
        }
        else
        {
            hipLaunchKernelGGL((haxpy_mlt_8_kernel<NB>),
                               grid,
                               threads,
                               0,
                               handle->get_stream(),
                               n_mlt_8,
                               load_scalar((const rocblas_half2*)alpha),
                               x,
                               stridex,
                               y,
                               stridey);

            if(n_mod_8)
            {
                hipLaunchKernelGGL((haxpy_mod_8_kernel<NB>),
                                   dim3(1, batch_count),
                                   n_mod_8,
                                   0,
                                   handle->get_stream(),
                                   n_mod_8,
                                   *alpha,
                                   x,
                                   n_mlt_8,
                                   stridex,
                                   y,
                                   n_mlt_8,
                                   stridey);
            }
        }
    }

    return rocblas_status_success;
}

template <typename T, typename U>
rocblas_status rocblas_axpy_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_int    offset_y,
                                           rocblas_int    inc_y,
                                           rocblas_stride stride_y,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status = rocblas_check_numerics_vector_template(function_name,
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

    check_numerics_status = rocblas_check_numerics_vector_template(function_name,
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
