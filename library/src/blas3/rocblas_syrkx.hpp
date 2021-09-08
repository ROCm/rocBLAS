/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "Tensile/gemm.hpp"
#include "definitions.hpp"
#include "rocblas_syr2k.hpp"

template <int  MIN_NB,
          bool BATCHED,
          typename T,
          typename TScal,
          typename TPtr,
          typename TConstPtr,
          typename TLd>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrkx_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal*            alpha,
                                    TConstPtr*        da,
                                    TLd               offset_a,
                                    TLd               lda,
                                    rocblas_stride    stride_a,
                                    TConstPtr*        db,
                                    TLd               offset_b,
                                    TLd               ldb,
                                    rocblas_stride    stride_b,
                                    TScal*            beta,
                                    TPtr*             dc,
                                    TLd               offset_c,
                                    TLd               ldc,
                                    rocblas_stride    stride_c,
                                    rocblas_int       batch_count);
