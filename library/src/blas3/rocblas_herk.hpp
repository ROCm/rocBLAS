/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "herk_scale_device.hpp"
#include "rocblas_syrk.hpp"

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <typename U, typename V>
__global__ void herk_scale_kernel(bool           upper,
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

template <typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_herk_arg_check(rocblas_handle    handle,
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
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;
    if(transA != rocblas_operation_none && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n || (transA == rocblas_operation_none && lda < n)
       || (transA != rocblas_operation_none && lda < k))
        return rocblas_status_invalid_size;
    if(!n || !batch_count)
        return rocblas_status_success;
    if((k > 0 && (!AP || !alpha)) || !CP || !beta)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_herk_template(rocblas_handle    handle,
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
        hipLaunchKernelGGL((herk_scale_kernel),
                           herk_scale_grid,
                           herk_scale_threads,
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

        if(transA == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syrk_herk_kernel<hermetian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
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
            hipLaunchKernelGGL((syrk_herk_kernel<hermetian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
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

        if(transA == rocblas_operation_none)
        {
            hipLaunchKernelGGL((syrk_herk_kernel<hermetian, false, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
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
            hipLaunchKernelGGL((syrk_herk_kernel<hermetian, true, SYRK_DIM_XY>),
                               syrk_grid,
                               syrk_threads,
                               0,
                               handle->rocblas_stream,
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
