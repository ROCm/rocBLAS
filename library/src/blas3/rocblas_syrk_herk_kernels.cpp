/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "handle.hpp"
#include "herk_scale_device.hpp"
#include "rocblas_syrk_herk.hpp"

template <typename T, typename U>
ROCBLAS_KERNEL_ILF void syrk_scale_device(bool upper, rocblas_int n, T beta, U* C, rocblas_int ldc)
{
    auto tx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;

    int from = upper ? tx : ty;
    int to   = upper ? ty : tx;

    if(tx < n && ty < n && from <= to)
    {
        C[ty * size_t(ldc) + tx] = beta ? beta * C[ty * size_t(ldc) + tx] : 0;
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <int DIM_X, int DIM_Y, typename U, typename V>
ROCBLAS_KERNEL __launch_bounds__(DIM_X* DIM_Y) void syrk_scale_kernel(bool        upper,
                                                                      rocblas_int n,
                                                                      U           beta_host_device,
                                                                      V           CP_array,
                                                                      ptrdiff_t   shift_c,
                                                                      rocblas_int ldc,
                                                                      rocblas_stride stride_c)
{
    auto beta = load_scalar(beta_host_device);

    if(beta == 1)
        return;

    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);
    syrk_scale_device(upper, n, beta, C, ldc);
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <typename U, typename V>
ROCBLAS_KERNEL void herk_scale_kernel(bool           upper,
                                      rocblas_int    n,
                                      rocblas_int    k,
                                      U              alpha_host_device,
                                      U              beta_host_device,
                                      V              CP_array,
                                      ptrdiff_t      shift_c,
                                      rocblas_int    ldc,
                                      rocblas_stride stride_c)
{

    auto C     = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);
    auto alpha = load_scalar(alpha_host_device);
    auto beta  = load_scalar(beta_host_device);

    if(beta == 1 && (k == 0 || alpha == 0)) // if alpha not zero we need imaginary clear on diagonal
        return;

    herk_scale_device(upper, n, beta, C, ldc);
}

/**
  * kernel
  */
template <bool HERM, bool TRANSA, rocblas_int TILE_NK, typename T, typename U>
ROCBLAS_KERNEL_ILF void syrk_herk_mult_add_device(bool        upper,
                                                  rocblas_int n,
                                                  rocblas_int k,
                                                  U           alpha,
                                                  const T* __restrict__ A,
                                                  rocblas_int lda,
                                                  T* __restrict__ C,
                                                  rocblas_int ldc)
{
    __shared__ T atile[TILE_NK][TILE_NK];
    __shared__ T btile[TILE_NK][TILE_NK];

    int col_pos = blockIdx.y * TILE_NK;
    int row_pos = blockIdx.x * TILE_NK;

    int tilefrom = upper ? row_pos : col_pos;
    int tileto   = upper ? col_pos : row_pos;
    if(!alpha || tilefrom > tileto)
    {
        // any overlap of tile and output
        return;
    }

    int a_cols = !TRANSA ? k : n;
    int a_rows = !TRANSA ? n : k;

    int row = row_pos + threadIdx.x;
    int col = col_pos + threadIdx.y;

    int from = upper ? row : col;
    int to   = upper ? col : row;

    for(int k_pos = 0; k_pos < k; k_pos += TILE_NK)
    {
        // tiling over dimension K

        int row_loc, col_loc;
        int r, c;

        // fetch tile of matrix A
        row_loc = row_pos + threadIdx.x;
        col_loc = k_pos + threadIdx.y;
        r       = TRANSA ? col_loc : row_loc; // true A = A^T, false A = A
        c       = TRANSA ? row_loc : col_loc;

        atile[threadIdx.x][threadIdx.y]
            = (r < a_rows && c < a_cols)
                  ? (HERM && TRANSA ? conj(A[c * size_t(lda) + r]) : A[c * size_t(lda) + r])
                  : 0;

        // fetch tile of matrix B
        row_loc = k_pos + threadIdx.x;
        col_loc = col_pos + threadIdx.y;
        r       = TRANSA ? row_loc : col_loc; // true B = A, false B = A^T
        c       = TRANSA ? col_loc : row_loc;

        btile[threadIdx.x][threadIdx.y]
            = (c < a_cols && r < a_rows)
                  ? (HERM && !TRANSA ? conj(A[c * size_t(lda) + r]) : A[c * size_t(lda) + r])
                  : 0;

        __syncthreads();

        // n x n symmetric/Hermitian output, tile zero where invalid
        if(row < n && col < n && from <= to)
        {
            T sum = T(0);
            for(int ki = 0; ki < TILE_NK; ++ki)
            {
                sum += atile[threadIdx.x][ki] * btile[ki][threadIdx.y];
            }
            C[col * size_t(ldc) + row] += alpha * sum;
        }

        __syncthreads();

    } // k_pos

    // if(HERM && row == col && row < n)
    // {
    //     // zero imaginary in case of numerical drift
    //     C[col * size_t(ldc) + row].y = 0;
    // }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <bool        HERM,
          bool        TRANS,
          rocblas_int DIM_XYT,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL_INSTANTIATE
    __launch_bounds__(DIM_XYT* DIM_XYT) void syrk_herk_kernel(bool              upper,
                                                              rocblas_operation transA,
                                                              rocblas_int       n,
                                                              rocblas_int       k,
                                                              TScal             alpha_host_device,
                                                              TConstPtr         AP_array,
                                                              ptrdiff_t         shift_a,
                                                              rocblas_int       lda,
                                                              rocblas_stride    stride_a,
                                                              TPtr              CP_array,
                                                              ptrdiff_t         shift_c,
                                                              rocblas_int       ldc,
                                                              rocblas_stride    stride_c)
{
    auto alpha = load_scalar(alpha_host_device);
    if(alpha == 0)
        return;

    // compute A^T * A or A * A^T and accumulate on the fly into C
    // when HERM does A^H in place of A^T
    auto A = load_ptr_batch(AP_array, hipBlockIdx_z, shift_a, stride_a);
    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);

    syrk_herk_mult_add_device<HERM, TRANS, DIM_XYT>(upper, n, k, alpha, A, lda, C, ldc);
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   TScal             alpha,
                                   TConstPtr         AP,
                                   rocblas_int       offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   TScal             beta,
                                   TPtr              CP,
                                   rocblas_int       offsetC,
                                   rocblas_int       ldc,
                                   rocblas_stride    strideC,
                                   rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    static constexpr int SYRK_SCALE_DIM_X = 128;
    static constexpr int SYRK_SCALE_DIM_Y = 8;
    rocblas_int          gx               = (n - 1) / (SYRK_SCALE_DIM_X) + 1;
    rocblas_int          gy               = (n - 1) / (SYRK_SCALE_DIM_Y) + 1;
    dim3                 syrk_scale_grid(gx, gy, batch_count);
    dim3                 syrk_scale_threads(SYRK_SCALE_DIM_X, SYRK_SCALE_DIM_Y);

    static constexpr int SYRK_DIM_XY = 32;
    rocblas_int          bx          = (n - 1) / (SYRK_DIM_XY) + 1;
    rocblas_int          by          = (n - 1) / (SYRK_DIM_XY) + 1;
    dim3                 syrk_grid(bx, by, batch_count);
    dim3                 syrk_threads(SYRK_DIM_XY, SYRK_DIM_XY);

    // Launch a herk kernel for syrk.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syrk_scale_kernel<SYRK_SCALE_DIM_X, SYRK_SCALE_DIM_Y>),
                           syrk_scale_grid,
                           syrk_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(transA == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syrk_herk_kernel<false, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syrk_herk_kernel<false, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }
    else
    {
        if((!*alpha || k == 0) && *beta == 1)
            return rocblas_status_success;

        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syrk_scale_kernel<SYRK_SCALE_DIM_X, SYRK_SCALE_DIM_Y>),
                           syrk_scale_grid,
                           syrk_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           *beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(transA == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syrk_herk_kernel<false, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syrk_herk_kernel<false, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }

    return rocblas_status_success;
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   TScal             alpha,
                                   TConstPtr         AP,
                                   rocblas_int       offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   TScal             beta,
                                   TPtr              CP,
                                   rocblas_int       offsetC,
                                   rocblas_int       ldc,
                                   rocblas_stride    strideC,
                                   rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    static constexpr int HERK_SCALE_DIM_X = 128;
    static constexpr int HERK_SCALE_DIM_Y = 8;
    rocblas_int          gx               = (n - 1) / (HERK_SCALE_DIM_X) + 1;
    rocblas_int          gy               = (n - 1) / (HERK_SCALE_DIM_Y) + 1;
    dim3                 herk_scale_grid(gx, gy, batch_count);
    dim3                 herk_scale_threads(HERK_SCALE_DIM_X, HERK_SCALE_DIM_Y);

    // Uses a syrk kernel in Hermitian mode
    static constexpr int  SYRK_DIM_XY = 32;
    rocblas_int           bx          = (n - 1) / (SYRK_DIM_XY) + 1;
    rocblas_int           by          = (n - 1) / (SYRK_DIM_XY) + 1;
    dim3                  syrk_grid(bx, by, batch_count);
    dim3                  syrk_threads(SYRK_DIM_XY, SYRK_DIM_XY);
    static constexpr bool Hermitian = true;

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // scale C so we can use directly for output without work buffer, zeros diag imaginary
        hipLaunchKernelGGL((herk_scale_kernel),
                           herk_scale_grid,
                           herk_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           k,
                           alpha,
                           beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(transA == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syrk_herk_kernel<Hermitian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syrk_herk_kernel<Hermitian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }
    else
    {
        if((!*alpha || k == 0) && *beta == 1)
            return rocblas_status_success;

        // scale C so we can use directly for output without work buffer, zeros diag imaginary
        hipLaunchKernelGGL((herk_scale_kernel),
                           herk_scale_grid,
                           herk_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           k,
                           *alpha,
                           *beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(transA == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syrk_herk_kernel<Hermitian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syrk_herk_kernel<Hermitian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               transA,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }

    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syrk*.cpp or herk*.cpp

// clang-format off
#undef INSTANTIATE_SYRK_HERK_KERNEL

#ifdef INSTANTIATE_SYRK_TEMPLATE
#error INSTANTIATE_SYRK_TEMPLATE already defined
#endif

#define INSTANTIATE_SYRK_TEMPLATE(TScal_, TConstPtr_, TPtr_)                             \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                 \
rocblas_internal_syrk_template<TScal_, TConstPtr_, TPtr_>(rocblas_handle    handle,      \
                                                          rocblas_fill      uplo,        \
                                                          rocblas_operation transA,      \
                                                          rocblas_int       n,           \
                                                          rocblas_int       k,           \
                                                          TScal_             alpha,      \
                                                          TConstPtr_         AP,         \
                                                          rocblas_int       offsetA,     \
                                                          rocblas_int       lda,         \
                                                          rocblas_stride    strideA,     \
                                                          TScal_             beta,       \
                                                          TPtr_              CP,         \
                                                          rocblas_int       offsetC,     \
                                                          rocblas_int       ldc,         \
                                                          rocblas_stride    strideC,     \
                                                          rocblas_int       batch_count);

INSTANTIATE_SYRK_TEMPLATE( float const*, float const*,  float*)
INSTANTIATE_SYRK_TEMPLATE(double const*, double const*, double*)
INSTANTIATE_SYRK_TEMPLATE( rocblas_float_complex const*,  rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_SYRK_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_SYRK_TEMPLATE( float const*,  float const* const*,  float* const*)
INSTANTIATE_SYRK_TEMPLATE(double const*, double const* const*, double* const*)
INSTANTIATE_SYRK_TEMPLATE( rocblas_float_complex const*,  rocblas_float_complex const* const*,  rocblas_float_complex* const*)
INSTANTIATE_SYRK_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_SYRK_TEMPLATE

#ifdef INSTANTIATE_HERK_TEMPLATE
#error INSTANTIATE_HERK_TEMPLATE already defined
#endif

#define INSTANTIATE_HERK_TEMPLATE(Tscal_, TConstPtr_, TPtr_)                             \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                 \
rocblas_internal_herk_template<Tscal_, TConstPtr_, TPtr_>(rocblas_handle    handle,      \
                                                          rocblas_fill      uplo,        \
                                                          rocblas_operation transA,      \
                                                          rocblas_int       n,           \
                                                          rocblas_int       k,           \
                                                          Tscal_            alpha,       \
                                                          TConstPtr_        AP,          \
                                                          rocblas_int       offsetA,     \
                                                          rocblas_int       lda,         \
                                                          rocblas_stride    strideA,     \
                                                          Tscal_            beta,        \
                                                          TPtr_             CP,          \
                                                          rocblas_int       offsetC,     \
                                                          rocblas_int       ldc,         \
                                                          rocblas_stride    strideC,     \
                                                          rocblas_int       batch_count);

INSTANTIATE_HERK_TEMPLATE( float const*, rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_HERK_TEMPLATE( float const*, rocblas_float_complex const* const*,  rocblas_float_complex* const*)
INSTANTIATE_HERK_TEMPLATE( double const*, rocblas_double_complex const*,  rocblas_double_complex*)
INSTANTIATE_HERK_TEMPLATE( double const*, rocblas_double_complex const* const*,  rocblas_double_complex* const*)

#undef INSTANTIATE_HERK_TEMPLATE

// clang-format on
