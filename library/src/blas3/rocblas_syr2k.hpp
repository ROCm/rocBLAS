/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "handle.hpp"

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
ROCBLAS_KERNEL __launch_bounds__(DIM_XYT* DIM_XYT) void syr2k_her2k_kernel(bool              upper,
                                                                           rocblas_operation trans,
                                                                           rocblas_int       n,
                                                                           rocblas_int       k,
                                                                           TScal alpha_host_device,
                                                                           TConstPtr      AP_array,
                                                                           ptrdiff_t      shift_a,
                                                                           rocblas_int    lda,
                                                                           rocblas_stride stride_a,
                                                                           TConstPtr      BP_array,
                                                                           ptrdiff_t      shift_b,
                                                                           rocblas_int    ldb,
                                                                           rocblas_stride stride_b,
                                                                           TPtr           CP_array,
                                                                           ptrdiff_t      shift_c,
                                                                           rocblas_int    ldc,
                                                                           rocblas_stride stride_c);

template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_syr2k_arg_check(rocblas_handle    handle,
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
                                       rocblas_int       batch_count);

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
                                    rocblas_int       batch_count);
