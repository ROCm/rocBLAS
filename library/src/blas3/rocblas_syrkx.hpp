/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "Tensile/gemm.hpp"
#include "definitions.hpp"
#include "rocblas_syr2k_her2k.hpp"

template <int MIN_NB, bool BATCHED, typename T, typename TScal, typename TPtr, typename TConstPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrkx_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
                                    rocblas_int       n,
                                    rocblas_int       k,
                                    TScal*            alpha,
                                    TConstPtr*        da,
                                    rocblas_int       offset_a,
                                    rocblas_int       lda,
                                    rocblas_stride    stride_a,
                                    TConstPtr*        db,
                                    rocblas_int       offset_b,
                                    rocblas_int       ldb,
                                    rocblas_stride    stride_b,
                                    TScal*            beta,
                                    TPtr*             dc,
                                    rocblas_int       offset_c,
                                    rocblas_int       ldc,
                                    rocblas_stride    stride_c,
                                    rocblas_int       batch_count);
