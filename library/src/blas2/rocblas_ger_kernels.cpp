/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_ger.hpp"

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int WIN,
          bool        CONJ,
          typename T_lda,
          typename T,
          typename V,
          typename U,
          typename W>
ROCBLAS_KERNEL __launch_bounds__(DIM_X* DIM_Y) void ger_kernel(rocblas_int    m,
                                                               rocblas_int    n,
                                                               V              alpha_device_host,
                                                               rocblas_stride stride_alpha,
                                                               const U __restrict__ xa,
                                                               ptrdiff_t      shiftx,
                                                               rocblas_int    incx,
                                                               rocblas_stride stridex,
                                                               const U __restrict__ ya,
                                                               ptrdiff_t      shifty,
                                                               rocblas_int    incy,
                                                               rocblas_stride stridey,
                                                               W __restrict__ Aa,
                                                               ptrdiff_t      shifta,
                                                               T_lda          lda,
                                                               rocblas_stride strideA)
{
    __shared__ T xdata[DIM_X];
    __shared__ T ydata[DIM_Y * WIN];

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_z, stride_alpha);
    if(!alpha)
        return;

    const T* __restrict__ x = load_ptr_batch(xa, hipBlockIdx_z, shiftx, stridex);
    const T* __restrict__ y = load_ptr_batch(ya, hipBlockIdx_z, shifty, stridey);

    T* __restrict__ A = load_ptr_batch(Aa, hipBlockIdx_z, shifta, strideA);

    int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    ty *= WIN;

    // shared data base index
    int tyi = hipThreadIdx_y * WIN;

    if(hipThreadIdx_y == 0)
    {
        xdata[hipThreadIdx_x] = tx < m ? x[tx * incx] : 0;
    }

    if(hipThreadIdx_x < WIN)
    {
        ydata[tyi + hipThreadIdx_x]
            = (ty + hipThreadIdx_x < n) ? y[(ty + hipThreadIdx_x) * incy] : 0;
    }

    __syncthreads();

    if(tx < m)
    {
        T x_value = alpha * xdata[hipThreadIdx_x];

        for(int i = 0; i < WIN; i++)
        {
            int yi = ty + i;
            if(yi < n)
                A[tx + lda * yi] += x_value * (CONJ ? conj(ydata[tyi + i]) : ydata[tyi + i]);
        }
    }
}

//optimized kernel for SGER
template <rocblas_int DIM_X, typename T_lda, typename T, typename V, typename U, typename W>
ROCBLAS_KERNEL __launch_bounds__(DIM_X) void sger_kernel(rocblas_int    m,
                                                         rocblas_int    n,
                                                         V              alpha_device_host,
                                                         rocblas_stride stride_alpha,
                                                         const U __restrict__ xa,
                                                         ptrdiff_t      shiftx,
                                                         rocblas_int    incx,
                                                         rocblas_stride stridex,
                                                         const U __restrict__ ya,
                                                         ptrdiff_t      shifty,
                                                         rocblas_int    incy,
                                                         rocblas_stride stridey,
                                                         W __restrict__ Aa,
                                                         ptrdiff_t      shifta,
                                                         T_lda          lda,
                                                         rocblas_stride strideA)
{
    rocblas_int tx  = hipThreadIdx_x;
    rocblas_int col = hipBlockIdx_x;

    auto alpha = load_scalar(alpha_device_host, hipBlockIdx_y, stride_alpha);

    if(!alpha)
        return;

    const T* __restrict__ x = load_ptr_batch(xa, hipBlockIdx_y, shiftx, stridex);
    const T* __restrict__ y = load_ptr_batch(ya, hipBlockIdx_y, shifty, stridey);

    T* __restrict__ A = load_ptr_batch(Aa, hipBlockIdx_y, shifta, strideA);

    if(tx < m)
        A += tx;

    //Each hipBlockIdx_x takes care of the computation of each column of matrix 'A'
    A += col * lda;

    const T res_y = y[col * incy] * alpha;

    //scalar-vector-vector product and add the result to a Hermitian matrix 'A'.
    //If m > DIM_X, then the threads are reused and the multiplied values will be accumalated to Hermitian matrix 'A'.

    for(rocblas_int i = 0; tx + i < m; i += DIM_X)
    {
        A[i] += res_y * x[(tx + i) * incx];
    }
}

template <bool CONJ, typename T, typename U, typename V, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_ger_template(rocblas_handle handle,
                                  rocblas_int    m,
                                  rocblas_int    n,
                                  const V*       alpha,
                                  rocblas_stride stride_alpha,
                                  const U*       x,
                                  rocblas_int    offsetx,
                                  rocblas_int    incx,
                                  rocblas_int    stridex,
                                  const U*       y,
                                  rocblas_int    offsety,
                                  rocblas_int    incy,
                                  rocblas_int    stridey,
                                  W*             A,
                                  rocblas_int    offsetA,
                                  rocblas_int    lda,
                                  rocblas_int    strideA,
                                  rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (m - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    bool i64_indices = n * size_t(lda) > std::numeric_limits<rocblas_int>::max();

    //Identifying the precision to have an appropriate optimization
    bool is_float = std::is_same<T, float>{};

#define ger_KARGS(alpha_)                                                                  \
    ger_grid, ger_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, x, shiftx, incx, \
        stridex, y, shifty, incy, stridey, A, offsetA, lda, strideA

    //optimized sger kernel
    if(is_float && m > 1024)
    {
        static constexpr int DIM_X = 1024;
        dim3                 ger_grid(n, batch_count);
        dim3                 ger_threads(DIM_X);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            if(i64_indices)
                hipLaunchKernelGGL((sger_kernel<DIM_X, size_t, T>), ger_KARGS(alpha));
            else
                hipLaunchKernelGGL((sger_kernel<DIM_X, rocblas_int, T>), ger_KARGS(alpha));
        }
        else
        {
            if(i64_indices)
                hipLaunchKernelGGL((sger_kernel<DIM_X, size_t, T>), ger_KARGS(*alpha));
            else
                hipLaunchKernelGGL((sger_kernel<DIM_X, rocblas_int, T>), ger_KARGS(*alpha));
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
            if(i64_indices)
                hipLaunchKernelGGL((ger_kernel<DIM_X, DIM_Y, WIN, CONJ, size_t, T>),
                                   ger_KARGS(alpha));
            else
                hipLaunchKernelGGL((ger_kernel<DIM_X, DIM_Y, WIN, CONJ, rocblas_int, T>),
                                   ger_KARGS(alpha));
        }
        else
        {
            if(i64_indices)
                hipLaunchKernelGGL((ger_kernel<DIM_X, DIM_Y, WIN, CONJ, size_t, T>),
                                   ger_KARGS(*alpha));
            else
                hipLaunchKernelGGL((ger_kernel<DIM_X, DIM_Y, WIN, CONJ, rocblas_int, T>),
                                   ger_KARGS(*alpha));
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
                                          rocblas_int    offset_a,
                                          rocblas_int    lda,
                                          rocblas_stride stride_a,
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
        = rocblas_internal_check_numerics_ge_matrix_template(function_name,
                                                             handle,
                                                             rocblas_operation_none,
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
                                  rocblas_int    offsetx,                              \
                                  rocblas_int    incx,                                 \
                                  rocblas_int    stridex,                              \
                                  U_ const *      y,                                   \
                                  rocblas_int    offsety,                              \
                                  rocblas_int    incy,                                 \
                                  rocblas_int    stridey,                              \
                                  W_*            A,                                    \
                                  rocblas_int    offsetA,                              \
                                  rocblas_int    lda,                                  \
                                  rocblas_int    strideA,                              \
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
                                          rocblas_int    offset_a,       \
                                          rocblas_int    lda,            \
                                          rocblas_stride stride_a,       \
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
