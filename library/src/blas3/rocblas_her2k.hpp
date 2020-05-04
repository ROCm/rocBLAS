/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "herk_scale_device.hpp"
#include "rocblas_syr2k.hpp"

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <typename U, typename V, typename W>
__global__ void her2k_scale_kernel(bool           upper,
                                   rocblas_int    n,
                                   rocblas_int    k,
                                   U              alpha_host_device,
                                   V              beta_host_device,
                                   W              CP_array,
                                   ptrdiff_t      shift_c,
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

template <typename TScal, typename TConstPtr, typename UScal, typename TPtr>
inline rocblas_status rocblas_her2k_arg_check(rocblas_handle    handle,
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
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(trans != rocblas_operation_none && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n
       || (trans == rocblas_operation_none && (lda < n || ldb < n))
       || (trans != rocblas_operation_none && (lda < k || ldb < k)))
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if((k > 0 && (!AP || !BP || !alpha)) || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool TWOK, typename TScal, typename TConstPtr, typename UScal, typename TPtr>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_her2k_template(rocblas_handle    handle,
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

    // Uses a syrk kernel in hermitian mode
    static constexpr int  SYRK_DIM_XY = 32;
    rocblas_int           bx          = (n - 1) / (SYRK_DIM_XY) + 1;
    rocblas_int           by          = (n - 1) / (SYRK_DIM_XY) + 1;
    dim3                  syrk_grid(bx, by, batch_count);
    dim3                  syrk_threads(SYRK_DIM_XY, SYRK_DIM_XY);
    static constexpr bool hermetian = true;

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // scale C so we can use directly for output without work buffer, zeros diag imaginary
        hipLaunchKernelGGL(her2k_scale_kernel,
                           her2k_scale_grid,
                           her2k_scale_threads,
                           0,
                           handle->rocblas_stream,
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
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, hermetian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
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
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, hermetian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
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
        hipLaunchKernelGGL(her2k_scale_kernel,
                           her2k_scale_grid,
                           her2k_scale_threads,
                           0,
                           handle->rocblas_stream,
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
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, hermetian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
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
            hipLaunchKernelGGL((syr2k_her2k_kernel<TWOK, hermetian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
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
