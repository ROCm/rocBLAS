/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas/rocblas.h"
#include "rocblas_syr2.hpp"

template <typename T>
__device__ void syr2_kernel_calc(bool        upper,
                                 rocblas_int n,
                                 T           alpha,
                                 const T*    x,
                                 rocblas_int incx,
                                 const T*    y,
                                 rocblas_int incy,
                                 T*          A,
                                 rocblas_int lda)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(upper ? ty < n && tx <= ty : tx < n && ty <= tx)
        A[tx + ty * lda]
            += alpha * x[tx * incx] * y[ty * incy] + alpha * y[tx * incy] * x[ty * incx];
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL __launch_bounds__(DIM_X* DIM_Y) void rocblas_syr2_kernel(bool           upper,
                                                                        rocblas_int    n,
                                                                        TScal          alphaa,
                                                                        TConstPtr      xa,
                                                                        ptrdiff_t      shift_x,
                                                                        rocblas_int    incx,
                                                                        rocblas_stride stride_x,
                                                                        TConstPtr      ya,
                                                                        ptrdiff_t      shift_y,
                                                                        rocblas_int    incy,
                                                                        rocblas_stride stride_y,
                                                                        TPtr           Aa,
                                                                        rocblas_int    lda,
                                                                        ptrdiff_t      shift_A,
                                                                        rocblas_stride stride_A)
{
    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the number of threads as template parameters indicate.

    auto alpha = load_scalar(alphaa);
    if(!alpha)
        return;

    auto*       A = load_ptr_batch(Aa, hipBlockIdx_z, shift_A, stride_A);
    const auto* x = load_ptr_batch(xa, hipBlockIdx_z, shift_x, stride_x);
    const auto* y = load_ptr_batch(ya, hipBlockIdx_z, shift_y, stride_y);

    syr2_kernel_calc(upper, n, alpha, x, incx, y, incy, A, lda);
}

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the bast type (float or double)
 */
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   TScal          alpha,
                                   TConstPtr      x,
                                   rocblas_int    offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   TConstPtr      y,
                                   rocblas_int    offset_y,
                                   rocblas_int    incy,
                                   rocblas_stride stride_y,
                                   TPtr           A,
                                   rocblas_int    lda,
                                   rocblas_int    offset_A,
                                   rocblas_stride stride_A,
                                   rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc, shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;
    ptrdiff_t shift_y = incy < 0 ? offset_y - ptrdiff_t(incy) * (n - 1) : offset_y;

    static constexpr int SYR2_DIM_X = 128;
    static constexpr int SYR2_DIM_Y = 8;
    rocblas_int          blocksX    = (n - 1) / SYR2_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / SYR2_DIM_Y + 1;

    dim3 syr2_grid(blocksX, blocksY, batch_count);
    dim3 syr2_threads(SYR2_DIM_X, SYR2_DIM_Y);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        hipLaunchKernelGGL((rocblas_syr2_kernel<SYR2_DIM_X, SYR2_DIM_Y>),
                           syr2_grid,
                           syr2_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           y,
                           shift_y,
                           incy,
                           stride_y,
                           A,
                           lda,
                           offset_A,
                           stride_A);
    else
        hipLaunchKernelGGL((rocblas_syr2_kernel<SYR2_DIM_X, SYR2_DIM_Y>),
                           syr2_grid,
                           syr2_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           *alpha,
                           x,
                           shift_x,
                           incx,
                           stride_x,
                           y,
                           shift_y,
                           incy,
                           stride_y,
                           A,
                           lda,
                           offset_A,
                           stride_A);

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_sy_matrix_template for checking Matrix `A` which is a Symmetric Matrix
template <typename T, typename U>
rocblas_status rocblas_syr2_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    n,
                                           T              A,
                                           rocblas_int    offset_a,
                                           rocblas_int    lda,
                                           rocblas_stride stride_a,
                                           U              x,
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

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *syr2*.cpp

// clang-format off

#ifdef INSTANTIATE_SYR2_TEMPLATE
#error INSTANTIATE_SYR2_TEMPLATE already defined
#endif

#define INSTANTIATE_SYR2_TEMPLATE(TScal_, TConstPtr_, TPtr_)    \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status        \
    rocblas_internal_syr2_template<TScal_, TConstPtr_, TPtr_>   \
                                  (rocblas_handle handle,       \
                                   rocblas_fill   uplo,         \
                                   rocblas_int    n,            \
                                   TScal_         alpha,        \
                                   TConstPtr_     x,            \
                                   rocblas_int    offset_x,     \
                                   rocblas_int    incx,         \
                                   rocblas_stride stride_x,     \
                                   TConstPtr_     y,            \
                                   rocblas_int    offset_y,     \
                                   rocblas_int    incy,         \
                                   rocblas_stride stride_y,     \
                                   TPtr_          A,            \
                                   rocblas_int    lda,          \
                                   rocblas_int    offset_A,     \
                                   rocblas_stride stride_A,     \
                                   rocblas_int    batch_count);

INSTANTIATE_SYR2_TEMPLATE(float const*, float const*, float*)
INSTANTIATE_SYR2_TEMPLATE(double const*, double const*, double*)
INSTANTIATE_SYR2_TEMPLATE(rocblas_float_complex const*, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_SYR2_TEMPLATE(float const*, float const* const*, float* const*)
INSTANTIATE_SYR2_TEMPLATE(double const*, double const* const*, double* const*)
INSTANTIATE_SYR2_TEMPLATE(rocblas_float_complex const*, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYR2_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_SYR2_TEMPLATE

#ifdef INSTANTIATE_SYR2_NUMERICS
#error INSTANTIATE_SYR2_NUMERICS already defined
#endif

#define INSTANTIATE_SYR2_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_syr2_check_numerics<T_, U_>               \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           rocblas_int    n,              \
                                           T_             A,              \
                                           rocblas_int    offset_a,       \
                                           rocblas_int    lda,            \
                                           rocblas_stride stride_a,       \
                                           U_             x,              \
                                           rocblas_int    offset_x,       \
                                           rocblas_int    inc_x,          \
                                           rocblas_stride stride_x,       \
                                           U_             y,              \
                                           rocblas_int    offset_y,       \
                                           rocblas_int    inc_y,          \
                                           rocblas_stride stride_y,       \
                                           rocblas_int    batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_SYR2_NUMERICS(float*, float const*)
INSTANTIATE_SYR2_NUMERICS(double*, double const*)
INSTANTIATE_SYR2_NUMERICS(rocblas_double_complex*, rocblas_double_complex const*)
INSTANTIATE_SYR2_NUMERICS(float* const*, float const* const*)
INSTANTIATE_SYR2_NUMERICS(double* const*, double const* const*)
INSTANTIATE_SYR2_NUMERICS(rocblas_float_complex*, rocblas_float_complex const*)
INSTANTIATE_SYR2_NUMERICS(rocblas_float_complex* const*, rocblas_float_complex const* const*)
INSTANTIATE_SYR2_NUMERICS(rocblas_double_complex* const*, rocblas_double_complex const* const*)

#undef INSTANTIATE_SYR2_NUMERICS

// clang-format on
