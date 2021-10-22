/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "handle.hpp"
#include "herk_scale_device.hpp"
#include "rocblas_syr2k.hpp"

template <typename T, typename U>
ROCBLAS_KERNEL_ILF void syr2k_scale_device(bool upper, rocblas_int n, T beta, U* C, rocblas_int ldc)
{
    auto tx = blockIdx.x * blockDim.x + threadIdx.x;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;

    int from = upper ? tx : ty;
    int to   = upper ? ty : tx;

    if(tx < n && ty < n && from <= to)
        C[ty * ldc + tx] = beta ? beta * C[ty * ldc + tx] : 0;
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <int DIM_X, int DIM_Y, typename U, typename V>
ROCBLAS_KERNEL __launch_bounds__(DIM_X* DIM_Y) void syr2k_scale_kernel(bool        upper,
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
    syr2k_scale_device(upper, n, beta, C, ldc);
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <int DIM_X, int DIM_Y, typename U, typename V, typename W>
ROCBLAS_KERNEL __launch_bounds__(DIM_X* DIM_Y) void her2k_scale_kernel(bool        upper,
                                                                       rocblas_int n,
                                                                       rocblas_int k,
                                                                       U         alpha_host_device,
                                                                       V         beta_host_device,
                                                                       W         CP_array,
                                                                       ptrdiff_t shift_c,
                                                                       rocblas_int    ldc,
                                                                       rocblas_stride stride_c)
{
    auto alpha = load_scalar(alpha_host_device);
    auto beta  = load_scalar(beta_host_device);

    if(beta == 1 && (k == 0 || alpha == 0)) // if alpha not zero we need imaginary clear on diagonal
        return;

    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);
    herk_scale_device(upper, n, beta, C, ldc);
}

/** helper for complex support */
template <typename T>
ROCBLAS_KERNEL_ILF void syr2k_her2k_zero_imaginary(T&)
{
}

template <typename T>
ROCBLAS_KERNEL_ILF void syr2k_her2k_zero_imaginary(rocblas_complex_num<T>& a)
{
    a.imag(0);
}

/**
  * kernel
  */
template <bool TWOK, bool HERM, bool trans, rocblas_int TILE_NK, typename T, typename U>
ROCBLAS_KERNEL_ILF void syr2k_her2k_mult_add_device(bool        upper,
                                                    rocblas_int n,
                                                    rocblas_int k,
                                                    U           alpha,
                                                    const T* __restrict__ A,
                                                    rocblas_int lda,
                                                    const T* __restrict__ B,
                                                    rocblas_int ldb,
                                                    T* __restrict__ C,
                                                    rocblas_int ldc)
{
    // if !alpha this function isn't called

    __shared__ T atile[TILE_NK][TILE_NK];
    __shared__ T btile[TILE_NK][TILE_NK];

    int col_pos = blockIdx.y * TILE_NK;
    int row_pos = blockIdx.x * TILE_NK;

    int tilefrom = upper ? row_pos : col_pos;
    int tileto   = upper ? col_pos : row_pos;
    if(tilefrom > tileto)
    {
        // any overlap of tile and output
        return;
    }

    int ab_rows = !trans ? n : k;
    int ab_cols = !trans ? k : n;

    int row = row_pos + threadIdx.x;
    int col = col_pos + threadIdx.y;

    int from = upper ? row : col;
    int to   = upper ? col : row;

    for(int k_pos = 0; k_pos < k; k_pos += TILE_NK)
    {
        // tiling over dimension K

        int row_loc, col_loc;
        int r, c;

        // first matrix mult: alpha*op(A)*op(B)^T
        // when HERM ^H instead of ^T

        // fetch tile of matrix A
        row_loc = row_pos + threadIdx.x;
        col_loc = k_pos + threadIdx.y;
        r       = trans ? col_loc : row_loc; // trans A = A^T, else A = A
        c       = trans ? row_loc : col_loc;

        atile[threadIdx.x][threadIdx.y]
            = (r < ab_rows && c < ab_cols) ? (HERM && trans ? conj(A[c * lda + r]) : A[c * lda + r])
                                           : 0;

        // fetch tile of matrix B
        row_loc = k_pos + threadIdx.x;
        col_loc = col_pos + threadIdx.y;
        r       = trans ? row_loc : col_loc; // trans B = B, else B = B^T
        c       = trans ? col_loc : row_loc;

        btile[threadIdx.x][threadIdx.y]
            = (c < ab_cols && r < ab_rows)
                  ? (HERM && !trans ? conj(B[c * ldb + r]) : B[c * ldb + r])
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
            C[col * ldc + row] += alpha * sum;
        }

        __syncthreads();

        // second matrix mult: alpha*op(B)*op(A)^T, if HERM conj(alpha) and ^H
        if(TWOK)
        {
            // fetch tile of matrix B  into tileA
            row_loc = row_pos + threadIdx.x;
            col_loc = k_pos + threadIdx.y;
            r       = trans ? col_loc : row_loc; // trans B = B^T, else B = B
            c       = trans ? row_loc : col_loc;

            atile[threadIdx.x][threadIdx.y]
                = (r < ab_rows && c < ab_cols)
                      ? (HERM && trans ? conj(B[c * ldb + r]) : B[c * ldb + r])
                      : 0;

            // fetch tile of matrix A into tileB
            row_loc = k_pos + threadIdx.x;
            col_loc = col_pos + threadIdx.y;
            r       = trans ? row_loc : col_loc; // trans A = A, else A = A^T
            c       = trans ? col_loc : row_loc;

            btile[threadIdx.x][threadIdx.y]
                = (c < ab_cols && r < ab_rows)
                      ? (HERM && !trans ? conj(A[c * lda + r]) : A[c * lda + r])
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
                C[col * ldc + row] += (HERM ? conj(alpha) : alpha) * sum;
            }

            __syncthreads();
        }

    } // k_pos

    if(!TWOK && HERM && row == col && row < n)
    {
        // zero imaginary for cases when A*B aren't true Hermitian
        syr2k_her2k_zero_imaginary(C[col * ldc + row]);
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <bool        TWOK,
          bool        HERM,
          bool        TRANS,
          rocblas_int DIM_XYT,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL_INSTANTIATE
    __launch_bounds__(DIM_XYT* DIM_XYT) void syr2k_her2k_kernel(bool              upper,
                                                                rocblas_operation trans,
                                                                rocblas_int       n,
                                                                rocblas_int       k,
                                                                TScal             alpha_host_device,
                                                                TConstPtr         AP_array,
                                                                ptrdiff_t         shift_a,
                                                                rocblas_int       lda,
                                                                rocblas_stride    stride_a,
                                                                TConstPtr         BP_array,
                                                                ptrdiff_t         shift_b,
                                                                rocblas_int       ldb,
                                                                rocblas_stride    stride_b,
                                                                TPtr              CP_array,
                                                                ptrdiff_t         shift_c,
                                                                rocblas_int       ldc,
                                                                rocblas_stride    stride_c)
{
    auto alpha = load_scalar(alpha_host_device);
    if(alpha == 0)
        return;

    auto A = load_ptr_batch(AP_array, hipBlockIdx_z, shift_a, stride_a);
    auto B = load_ptr_batch(BP_array, hipBlockIdx_z, shift_b, stride_b);
    auto C = load_ptr_batch(CP_array, hipBlockIdx_z, shift_c, stride_c);

    // compute matrix multiplies and accumulate on the fly into C
    // when HERM does ^H in place of ^T
    syr2k_her2k_mult_add_device<TWOK, HERM, TRANS, DIM_XYT>(
        upper, n, k, alpha, A, lda, B, ldb, C, ldc);
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool TWOK, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2k_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal             alpha,
                                    TConstPtr         AP,
                                    rocblas_int       offsetA,
                                    rocblas_int       lda,
                                    rocblas_stride    strideA,
                                    TConstPtr         BP,
                                    rocblas_int       offsetB,
                                    rocblas_int       ldb,
                                    rocblas_stride    strideB,
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

    static constexpr int syr2k_SCALE_DIM_X = 128;
    static constexpr int syr2k_SCALE_DIM_Y = 8;
    rocblas_int          gx                = (n - 1) / (syr2k_SCALE_DIM_X) + 1;
    rocblas_int          gy                = (n - 1) / (syr2k_SCALE_DIM_Y) + 1;
    dim3                 syr2k_scale_grid(gx, gy, batch_count);
    dim3                 syr2k_scale_threads(syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y);

    static constexpr int syr2k_DIM_XY = 32;
    rocblas_int          bx           = (n - 1) / (syr2k_DIM_XY) + 1;
    rocblas_int          by           = (n - 1) / (syr2k_DIM_XY) + 1;
    dim3                 syr2k_grid(bx, by, batch_count);
    dim3                 syr2k_threads(syr2k_DIM_XY, syr2k_DIM_XY);

    // Launch a herk kernel for syr2k.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syr2k_scale_kernel<syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y>),
                           syr2k_scale_grid,
                           syr2k_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }
    else
    {
        if(*beta == 1 && (*alpha == 0 || k == 0))
            return rocblas_status_success;

        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syr2k_scale_kernel<syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y>),
                           syr2k_scale_grid,
                           syr2k_scale_threads,
                           0,
                           handle->get_stream(),
                           uplo == rocblas_fill_upper,
                           n,
                           *beta,
                           CP,
                           offsetC,
                           ldc,
                           strideC);

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, false, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
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
template <bool TWOK, typename TScal, typename TConstPtr, typename UScal, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2k_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal             alpha,
                                    TConstPtr         AP,
                                    rocblas_int       offsetA,
                                    rocblas_int       lda,
                                    rocblas_stride    strideA,
                                    TConstPtr         BP,
                                    rocblas_int       offsetB,
                                    rocblas_int       ldb,
                                    rocblas_stride    strideB,
                                    UScal             beta,
                                    TPtr              CP,
                                    rocblas_int       offsetC,
                                    rocblas_int       ldc,
                                    rocblas_stride    strideC,
                                    rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    static constexpr int her2k_SCALE_DIM_X = 128;
    static constexpr int her2k_SCALE_DIM_Y = 8;
    rocblas_int          gx                = (n - 1) / (her2k_SCALE_DIM_X) + 1;
    rocblas_int          gy                = (n - 1) / (her2k_SCALE_DIM_Y) + 1;
    dim3                 her2k_scale_grid(gx, gy, batch_count);
    dim3                 her2k_scale_threads(her2k_SCALE_DIM_X, her2k_SCALE_DIM_Y);

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
        hipLaunchKernelGGL((her2k_scale_kernel<her2k_SCALE_DIM_X, her2k_SCALE_DIM_Y>),
                           her2k_scale_grid,
                           her2k_scale_threads,
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

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, Hermitian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, Hermitian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }
    else
    {
        if(*beta == 1 && (*alpha == 0 || k == 0))
            return rocblas_status_success;

        // scale C so we can use directly for output without work buffer, zeros diag imaginary
        hipLaunchKernelGGL((her2k_scale_kernel<her2k_SCALE_DIM_X, her2k_SCALE_DIM_Y>),
                           her2k_scale_grid,
                           her2k_scale_threads,
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

        if(k == 0)
            return rocblas_status_success;

        if(trans == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, Hermitian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
        else
        {
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, Hermitian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->get_stream(),
                               uplo == rocblas_fill_upper,
                               trans,
                               n,
                               k,
                               *alpha,
                               AP,
                               offsetA,
                               lda,
                               strideA,
                               BP,
                               offsetB,
                               ldb,
                               strideB,
                               CP,
                               offsetC,
                               ldc,
                               strideC);
        }
    }

    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syr2k*.cpp or her2k*.cpp

// clang-format off

#ifdef INSTANTIATE_SYR2K_TEMPLATE
#error INSTANTIATE_SYR2K_TEMPLATE already defined
#endif

#define INSTANTIATE_SYR2K_TEMPLATE(TWOK_, TScal_, TConstPtr_, TPtr_)                     \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_syr2k_template \
                                <TWOK_, TScal_, TConstPtr_, TPtr_>                       \
				(rocblas_handle    handle,                               \
                                 rocblas_fill      uplo,                                 \
                                 rocblas_operation trans,                                \
                                 rocblas_int       n,                                    \
                                 rocblas_int       k,                                    \
                                 TScal_            alpha,                                \
                                 TConstPtr_        AP,                                   \
                                 rocblas_int       offsetA,                              \
                                 rocblas_int       lda,                                  \
                                 rocblas_stride    strideA,                              \
                                 TConstPtr_        BP,                                   \
                                 rocblas_int       offsetB,                              \
                                 rocblas_int       ldb,                                  \
                                 rocblas_stride    strideB,                              \
                                 TScal_            beta,                                 \
                                 TPtr_             CP,                                   \
                                 rocblas_int       offsetC,                              \
                                 rocblas_int       ldc,                                  \
                                 rocblas_stride    strideC,                              \
                                 rocblas_int       batch_count);

INSTANTIATE_SYR2K_TEMPLATE(true, float const*, float const*, float*)
INSTANTIATE_SYR2K_TEMPLATE(true, double const*, double const*, double*)
INSTANTIATE_SYR2K_TEMPLATE(true, rocblas_float_complex const*, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2K_TEMPLATE(true, rocblas_double_complex const*, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_SYR2K_TEMPLATE(true, float const*, float const* const*, float* const*)
INSTANTIATE_SYR2K_TEMPLATE(true, double const*, double const* const*, double* const*)
INSTANTIATE_SYR2K_TEMPLATE(true, rocblas_float_complex const*, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYR2K_TEMPLATE(true, rocblas_double_complex const*, rocblas_double_complex const* const*, rocblas_double_complex* const*)
INSTANTIATE_SYR2K_TEMPLATE(false, float const*, float const*, float*)
INSTANTIATE_SYR2K_TEMPLATE(false, double const*, double const*, double*)
INSTANTIATE_SYR2K_TEMPLATE(false, rocblas_float_complex const*, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SYR2K_TEMPLATE(false, rocblas_double_complex const*, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_SYR2K_TEMPLATE(false, float const*, float const* const*, float* const*)
INSTANTIATE_SYR2K_TEMPLATE(false, double const*, double const* const*, double* const*)
INSTANTIATE_SYR2K_TEMPLATE(false, rocblas_float_complex const*, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SYR2K_TEMPLATE(false, rocblas_double_complex const*, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_SYR2K_TEMPLATE

#ifdef INSTANTIATE_HER2K_TEMPLATE
#error INSTANTIATE_HER2K_TEMPLATE already defined
#endif

#define INSTANTIATE_HER2K_TEMPLATE(TWOK_, TScal_, TConstPtr_, UScal_, TPtr_) \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                     \
rocblas_internal_her2k_template<TWOK_, TScal_, TConstPtr_, UScal_, TPtr_>    \
                               (rocblas_handle    handle,                    \
                                rocblas_fill      uplo,                      \
                                rocblas_operation trans,                     \
                                rocblas_int       n,                         \
                                rocblas_int       k,                         \
                                TScal_            alpha,                     \
                                TConstPtr_        AP,                        \
                                rocblas_int       offsetA,                   \
                                rocblas_int       lda,                       \
                                rocblas_stride    strideA,                   \
                                TConstPtr_        BP,                        \
                                rocblas_int       offsetB,                   \
                                rocblas_int       ldb,                       \
                                rocblas_stride    strideB,                   \
                                UScal_            beta,                      \
                                TPtr_             CP,                        \
                                rocblas_int       offsetC,                   \
                                rocblas_int       ldc,                       \
                                rocblas_stride    strideC,                   \
                                rocblas_int       batch_count);

INSTANTIATE_HER2K_TEMPLATE(true, rocblas_float_complex const*, rocblas_float_complex const*, float const*, rocblas_float_complex*)
INSTANTIATE_HER2K_TEMPLATE(true, rocblas_double_complex const*, rocblas_double_complex const*, double const*, rocblas_double_complex*)
INSTANTIATE_HER2K_TEMPLATE(true, rocblas_float_complex const*, rocblas_float_complex const* const*, float const*, rocblas_float_complex* const*)
INSTANTIATE_HER2K_TEMPLATE(true, rocblas_double_complex const*, rocblas_double_complex const* const*, double const*, rocblas_double_complex* const*)
INSTANTIATE_HER2K_TEMPLATE(false, rocblas_float_complex const*, rocblas_float_complex const*, float const*, rocblas_float_complex*)
INSTANTIATE_HER2K_TEMPLATE(false, rocblas_double_complex const*, rocblas_double_complex const*, double const*, rocblas_double_complex*)
INSTANTIATE_HER2K_TEMPLATE(false, rocblas_float_complex const*, rocblas_float_complex const* const*, float const*, rocblas_float_complex* const*)
INSTANTIATE_HER2K_TEMPLATE(false, rocblas_double_complex const*, rocblas_double_complex const* const*, double const*, rocblas_double_complex* const*)

#undef INSTANTIATE_HER2K_TEMPLATE

// clang-format on
