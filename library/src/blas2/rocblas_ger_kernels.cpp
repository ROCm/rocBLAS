/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_ger.hpp"

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int WIN,
          bool        CONJ,
          typename T,
          typename V,
          typename U,
          typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_ger_kernel(rocblas_int    m,
                   rocblas_int    n,
                   V              alpha_device_host,
                   rocblas_stride stride_alpha,
                   const U __restrict__ xa,
                   rocblas_stride shiftx,
                   rocblas_int    incx,
                   rocblas_stride stridex,
                   const U __restrict__ ya,
                   rocblas_stride shifty,
                   rocblas_int    incy,
                   rocblas_stride stridey,
                   W __restrict__ Aa,
                   rocblas_stride shifta,
                   rocblas_int    lda,
                   rocblas_stride strideA)
{
    __shared__ T xdata[DIM_X];
    __shared__ T ydata[DIM_Y * WIN];

    auto alpha = load_scalar(alpha_device_host, blockIdx.z, stride_alpha);
    if(!alpha)
        return;

    const T* __restrict__ x = load_ptr_batch(xa, blockIdx.z, shiftx, stridex);
    const T* __restrict__ y = load_ptr_batch(ya, blockIdx.z, shifty, stridey);

    T* __restrict__ A = load_ptr_batch(Aa, blockIdx.z, shifta, strideA);

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    ty *= WIN;

    // shared data base index
    int tyi = threadIdx.y * WIN;

    if(threadIdx.y == 0)
    {
        xdata[threadIdx.x] = tx < m ? x[tx * incx] : 0;
    }

    if(threadIdx.x < WIN)
    {
        ydata[tyi + threadIdx.x]
            = (ty + ptrdiff_t(threadIdx.x) < n) ? y[(ty + ptrdiff_t(threadIdx.x)) * incy] : 0;
    }

    __syncthreads();

    if(tx < m)
    {
        T x_value = alpha * xdata[threadIdx.x];

        for(int i = 0; i < WIN; i++)
        {
            int yi = ty + i;
            if(yi < n)
                A[tx + size_t(lda) * yi]
                    += x_value * (CONJ ? conj(ydata[tyi + i]) : ydata[tyi + i]);
        }
    }
}

//optimized kernel for SGER
template <rocblas_int DIM_X, typename T, typename V, typename U, typename W>
ROCBLAS_KERNEL(DIM_X)
rocblas_sger_kernel(rocblas_int    m,
                    rocblas_int    n,
                    V              alpha_device_host,
                    rocblas_stride stride_alpha,
                    const U __restrict__ xa,
                    rocblas_stride shiftx,
                    rocblas_int    incx,
                    rocblas_stride stridex,
                    const U __restrict__ ya,
                    rocblas_stride shifty,
                    rocblas_int    incy,
                    rocblas_stride stridey,
                    W __restrict__ Aa,
                    rocblas_stride shifta,
                    rocblas_int    lda,
                    rocblas_stride strideA)
{
    rocblas_int tx  = threadIdx.x;
    rocblas_int col = blockIdx.x;

    auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);

    if(!alpha)
        return;

    const T* __restrict__ x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
    const T* __restrict__ y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

    T* __restrict__ A = load_ptr_batch(Aa, blockIdx.y, shifta, strideA);

    if(tx < m)
        A += tx;

    //Each blockIdx.x takes care of the computation of each column of matrix 'A'
    A += col * size_t(lda);

    const T res_y = y[col * incy] * alpha;

    //scalar-vector-vector product and add the result to a Hermitian matrix 'A'.
    //If m > DIM_X, then the threads are reused and the multiplied values will be accumalated to Hermitian matrix 'A'.

    for(rocblas_int i = 0; tx + i < m; i += DIM_X)
    {
        A[i] += res_y * x[(tx + i) * incx];
    }
}

//optimized double buffered load kernel for GER
template <bool        CONJ,
          rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int elements_per_thread,
          typename T,
          typename TStruct,
          typename U,
          typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_ger_double_buffered_kernel(bool           host_ptr_mode,
                                   rocblas_int    m,
                                   rocblas_int    n,
                                   TStruct        alpha_device_host,
                                   rocblas_stride stride_alpha,
                                   const U __restrict__ xa,
                                   rocblas_stride shiftx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const U __restrict__ ya,
                                   rocblas_stride shifty,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   W __restrict__ Aa,
                                   rocblas_stride shifta,
                                   rocblas_int    lda,
                                   rocblas_stride strideA)
{
    auto alpha              = host_ptr_mode ? alpha_device_host.value
                                            : load_scalar(alpha_device_host.ptr, blockIdx.z, stride_alpha);
    const T* __restrict__ x = load_ptr_batch(xa, blockIdx.z, shiftx, stridex);
    const T* __restrict__ y = load_ptr_batch(ya, blockIdx.z, shifty, stridey);

    T* __restrict__ A = load_ptr_batch(Aa, blockIdx.z, shifta, strideA);

    if(!alpha)
        return;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);

    T x_reg_upper = 0.0;
    T x_reg_lower = 0.0;
    T areg_upper[elements_per_thread];
    T areg_lower[elements_per_thread];
    T y_reg[elements_per_thread];

    // Advance 'A'
    A += DIM_X * bx;
    A += by * DIM_X * size_t(lda);

    // Advance 'x'
    x += (bx * DIM_X) * incx;

    // Advance 'y'
    y += (by * DIM_X) * incy;

    const int j = ty_ * elements_per_thread * lda + tx_;

    x_reg_upper = x[tx_ * incx] * alpha;
    x_reg_lower = x[((DIM_X / 2) + tx_) * incx] * alpha;

// read upper
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        areg_upper[k] = A[j + k * lda];

// read lower
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
    {
        areg_lower[k] = A[(DIM_X / 2) + j + k * lda];
        y_reg[k]      = y[(ty_ * elements_per_thread + k) * incy];
    }

// compute upper
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        areg_upper[k] += x_reg_upper * (CONJ ? conj(y_reg[k]) : y_reg[k]);

// store upper
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        A[j + k * lda] = areg_upper[k];

// compute lower
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        areg_lower[k] += x_reg_lower * (CONJ ? conj(y_reg[k]) : y_reg[k]);

// store lower
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        A[(DIM_X / 2) + j + k * lda] = areg_lower[k];
}

template <bool CONJ, typename T, typename U, typename V, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_ger_template(rocblas_handle handle,
                                  rocblas_int    m,
                                  rocblas_int    n,
                                  const V*       alpha,
                                  rocblas_stride stride_alpha,
                                  const U*       x,
                                  rocblas_stride offsetx,
                                  rocblas_int    incx,
                                  rocblas_stride stridex,
                                  const U*       y,
                                  rocblas_stride offsety,
                                  rocblas_int    incy,
                                  rocblas_stride stridey,
                                  W*             A,
                                  rocblas_stride offsetA,
                                  rocblas_int    lda,
                                  rocblas_stride strideA,
                                  rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (m - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    //Identifying the precision to have an appropriate optimization
    static constexpr bool is_float         = std::is_same<T, float>{};
    static constexpr bool is_double        = std::is_same<T, double>{};
    static constexpr bool is_complex_float = std::is_same<T, rocblas_float_complex>{};

    bool is_gfx90a = handle->getArch() == 910 ? true : false;

#define ger_KARGS(alpha_)                                                                  \
    ger_grid, ger_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, x, shiftx, incx, \
        stridex, y, shifty, incy, stridey, A, offsetA, lda, strideA

    //optimized double buffered loads kernel for float, double and float_complex precisions in gfx90a
    if(is_gfx90a && (m > 2000) && (m == n)
       && ((m % 64 == 0 && (is_double || is_complex_float)) || ((m % 128 == 0) && is_float)))
    {
        //The following rocblas_ger_double_buffered_kernel is only valid for the multiples of DIM_X
        static constexpr int DIM_X               = is_float ? 128 : 64;
        static constexpr int DIM_Y               = is_float ? 8 : 16;
        static constexpr int elements_per_thread = DIM_X / (2 * DIM_Y);

        const int block_x = m / DIM_X;
        const int block_y = n / DIM_X;
        dim3      ger_threads(DIM_X, DIM_Y);
        dim3      ger_grid(block_x, block_y, batch_count);

        bool host_ptr_mode = handle->pointer_mode == rocblas_pointer_mode_host;
        rocblas_internal_val_ptr<V> alpha_device_host(host_ptr_mode, alpha);

        hipLaunchKernelGGL(
            (rocblas_ger_double_buffered_kernel<CONJ, DIM_X, DIM_Y, elements_per_thread, T>),
            ger_grid,
            ger_threads,
            0,
            rocblas_stream,
            host_ptr_mode,
            m,
            n,
            alpha_device_host,
            stride_alpha,
            x,
            shiftx,
            incx,
            stridex,
            y,
            shifty,
            incy,
            stridey,
            A,
            offsetA,
            lda,
            strideA);
    }
    else if(is_float && m > 1024)
    {
        static constexpr int DIM_X = 1024;
        dim3                 ger_grid(n, batch_count);
        dim3                 ger_threads(DIM_X);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((rocblas_sger_kernel<DIM_X, T>), ger_KARGS(alpha));
        }
        else
        {
            hipLaunchKernelGGL((rocblas_sger_kernel<DIM_X, T>), ger_KARGS(*alpha));
        }
    }
    else
    {
        static constexpr int DIM_X   = 32;
        static constexpr int DIM_Y   = 32;
        static constexpr int WIN     = 2; // work item number of elements to process
        rocblas_int          blocksX = (m - 1) / DIM_X + 1;
        rocblas_int          blocksY = (n - 1) / (DIM_Y * WIN) + 1; // WIN columns/work item

        dim3 ger_grid(blocksX, blocksY, batch_count);
        dim3 ger_threads(DIM_X, DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            hipLaunchKernelGGL((rocblas_ger_kernel<DIM_X, DIM_Y, WIN, CONJ, T>), ger_KARGS(alpha));
        }
        else
        {
            hipLaunchKernelGGL((rocblas_ger_kernel<DIM_X, DIM_Y, WIN, CONJ, T>), ger_KARGS(*alpha));
        }
    }
#undef ger_KARGS
    return rocblas_status_success;
}

template <typename T, typename U>
rocblas_status rocblas_ger_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          rocblas_int    m,
                                          rocblas_int    n,
                                          U              A,
                                          rocblas_stride offset_a,
                                          rocblas_int    lda,
                                          rocblas_stride stride_a,
                                          T              x,
                                          rocblas_stride offset_x,
                                          rocblas_int    inc_x,
                                          rocblas_stride stride_x,
                                          T              y,
                                          rocblas_stride offset_y,
                                          rocblas_int    inc_y,
                                          rocblas_stride stride_y,
                                          rocblas_int    batch_count,
                                          const int      check_numerics,
                                          bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          n,
                                                          A,
                                                          offset_a,
                                                          lda,
                                                          stride_a,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    if(is_input)
    {
        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                                handle,
                                                                                m,
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
    }
    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *ger*.cpp

// clang-format off

#ifdef INSTANTIATE_GER_TEMPLATE
#error INSTANTIATE_GER_TEMPLATE already defined
#endif

#define INSTANTIATE_GER_TEMPLATE(CONJ_, T_, U_, V_, W_)                                \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_ger_template \
                                 <CONJ_, T_, U_, V_, W_>                               \
                                 (rocblas_handle handle,                               \
                                  rocblas_int    m,                                    \
                                  rocblas_int    n,                                    \
                                  V_ const *      alpha,                               \
                                  rocblas_stride stride_alpha,                         \
                                  U_ const *      x,                                   \
                                  rocblas_stride    offsetx,                              \
                                  rocblas_int    incx,                                 \
                                  rocblas_stride    stridex,                              \
                                  U_ const *      y,                                   \
                                  rocblas_stride    offsety,                              \
                                  rocblas_int    incy,                                 \
                                  rocblas_stride    stridey,                              \
                                  W_*            A,                                    \
                                  rocblas_stride    offsetA,                              \
                                  rocblas_int    lda,                                  \
                                  rocblas_stride    strideA,                              \
                                  rocblas_int    batch_count);

INSTANTIATE_GER_TEMPLATE(false, float, float, float, float)
INSTANTIATE_GER_TEMPLATE(false, double, double, double, double)
INSTANTIATE_GER_TEMPLATE(false, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_GER_TEMPLATE(false, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex)
INSTANTIATE_GER_TEMPLATE(true, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_GER_TEMPLATE(true, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex, rocblas_double_complex)
INSTANTIATE_GER_TEMPLATE(false, float, float const*, float, float* const)
INSTANTIATE_GER_TEMPLATE(false, double, double const*, double, double* const)
INSTANTIATE_GER_TEMPLATE(false, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex, rocblas_float_complex* const)
INSTANTIATE_GER_TEMPLATE(false, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex, rocblas_double_complex* const)
INSTANTIATE_GER_TEMPLATE(true, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex, rocblas_float_complex* const)
INSTANTIATE_GER_TEMPLATE(true, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex, rocblas_double_complex* const)

#undef INSTANTIATE_GER_TEMPLATE

#ifdef INSTANTIATE_GER_NUMERICS
#error INSTANTIATE_GER_NUMERICS already defined
#endif

#define INSTANTIATE_GER_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_ger_check_numerics<T_, U_>               \
                                         (const char*    function_name,  \
                                          rocblas_handle handle,         \
                                          rocblas_int    m,              \
                                          rocblas_int    n,              \
                                          U_             A,              \
                                          rocblas_stride offset_a,       \
                                          rocblas_int    lda,            \
                                          rocblas_stride stride_a,       \
                                          T_             x,              \
                                          rocblas_stride offset_x,       \
                                          rocblas_int    inc_x,          \
                                          rocblas_stride stride_x,       \
                                          T_             y,              \
                                          rocblas_stride offset_y,       \
                                          rocblas_int    inc_y,          \
                                          rocblas_stride stride_y,       \
                                          rocblas_int    batch_count,    \
                                          const int      check_numerics, \
                                          bool           is_input);

INSTANTIATE_GER_NUMERICS(float const*, float*)
INSTANTIATE_GER_NUMERICS(double const*, double*)
INSTANTIATE_GER_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_GER_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_GER_NUMERICS(float const* const*, float* const*)
INSTANTIATE_GER_NUMERICS(double const* const*, double* const*)
INSTANTIATE_GER_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_GER_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_GER_NUMERICS

// clang-format on
