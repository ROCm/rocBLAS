/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_SYRKX_HPP__
#define __ROCBLAS_SYRKX_HPP__

#include "handle.h"
#include "rocblas.h"
#include "rocblas_syr2k.hpp"
#include "utility.h"

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_syrkx_template(rocblas_handle    handle,
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
        hipLaunchKernelGGL((syr2k_scale_kernel),
                           syr2k_scale_grid,
                           syr2k_scale_threads,
                           0,
                           handle->rocblas_stream,
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
            hipLaunchKernelGGL((syr2k_her2k_kernel<false, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
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
            hipLaunchKernelGGL((syr2k_her2k_kernel<false, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
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

        // first scale C so we can use directly for output without work buffer
        hipLaunchKernelGGL((syr2k_scale_kernel),
                           syr2k_scale_grid,
                           syr2k_scale_threads,
                           0,
                           handle->rocblas_stream,
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
            hipLaunchKernelGGL((syr2k_her2k_kernel<false, false, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
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
            hipLaunchKernelGGL((syr2k_her2k_kernel<false, true, syr2k_DIM_XY>),
                               syr2k_grid,
                               syr2k_threads,
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

#endif
