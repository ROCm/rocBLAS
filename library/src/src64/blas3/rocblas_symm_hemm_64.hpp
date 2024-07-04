/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include "handle.hpp"

template <bool HERM, typename T>
rocblas_status rocblas_internal_symm_hemm_launcher_64(rocblas_handle handle,
                                                      rocblas_side   side,
                                                      rocblas_fill   uplo,
                                                      int64_t        m_64,
                                                      int64_t        n_64,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_stride offsetA,
                                                      int64_t        lda_64,
                                                      rocblas_stride strideA,
                                                      const T*       B,
                                                      rocblas_stride offsetB,
                                                      int64_t        ldb_64,
                                                      rocblas_stride strideB,
                                                      const T*       beta,
                                                      T*             C,
                                                      rocblas_stride offsetC,
                                                      int64_t        ldc_64,
                                                      rocblas_stride strideC,
                                                      int64_t        batch_count_64);

template <bool HERM, typename T_>
rocblas_status rocblas_internal_symm_hemm_batched_launcher_64(rocblas_handle   handle,
                                                              rocblas_side     side,
                                                              rocblas_fill     uplo,
                                                              int64_t          m_64,
                                                              int64_t          n_64,
                                                              const T_*        alpha,
                                                              const T_* const* A,
                                                              rocblas_stride   offsetA,
                                                              int64_t          lda_64,
                                                              rocblas_stride   strideA,
                                                              const T_* const* B,
                                                              rocblas_stride   offsetB,
                                                              int64_t          ldb_64,
                                                              rocblas_stride   strideB,
                                                              const T_*        beta,
                                                              T_* const*       C,
                                                              rocblas_stride   offsetC,
                                                              int64_t          ldc_64,
                                                              rocblas_stride   strideC,
                                                              int64_t          batch_count_64);

template <typename T_>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symm_template_64(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      int64_t        m_64,
                                      int64_t        n_64,
                                      const T_*      alpha,
                                      const T_*      A,
                                      rocblas_stride offsetA,
                                      int64_t        lda_64,
                                      rocblas_stride strideA,
                                      const T_*      B,
                                      rocblas_stride offsetB,
                                      int64_t        ldb_64,
                                      rocblas_stride strideB,
                                      const T_*      beta,
                                      T_*            C,
                                      rocblas_stride offsetC,
                                      int64_t        ldc_64,
                                      rocblas_stride strideC,
                                      int64_t        batch_count_64);

template <typename T_>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symm_batched_template_64(rocblas_handle   handle,
                                              rocblas_side     side,
                                              rocblas_fill     uplo,
                                              int64_t          m_64,
                                              int64_t          n_64,
                                              const T_*        alpha,
                                              const T_* const* A,
                                              rocblas_stride   offsetA,
                                              int64_t          lda_64,
                                              rocblas_stride   strideA,
                                              const T_* const* B,
                                              rocblas_stride   offsetB,
                                              int64_t          ldb_64,
                                              rocblas_stride   strideB,
                                              const T_*        beta,
                                              T_* const*       C,
                                              rocblas_stride   offsetC,
                                              int64_t          ldc_64,
                                              rocblas_stride   strideC,
                                              int64_t          batch_count_64);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemm_template_64(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      int64_t        m_64,
                                      int64_t        n_64,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_stride offsetA,
                                      int64_t        lda_64,
                                      rocblas_stride strideA,
                                      const T*       B,
                                      rocblas_stride offsetB,
                                      int64_t        ldb_64,
                                      rocblas_stride strideB,
                                      const T*       beta,
                                      T*             C,
                                      rocblas_stride offsetC,
                                      int64_t        ldc_64,
                                      rocblas_stride strideC,
                                      int64_t        batch_count_64);

template <typename T_>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemm_batched_template_64(rocblas_handle   handle,
                                              rocblas_side     side,
                                              rocblas_fill     uplo,
                                              int64_t          m_64,
                                              int64_t          n_64,
                                              const T_*        alpha,
                                              const T_* const* A,
                                              rocblas_stride   offsetA,
                                              int64_t          lda_64,
                                              rocblas_stride   strideA,
                                              const T_* const* B,
                                              rocblas_stride   offsetB,
                                              int64_t          ldb_64,
                                              rocblas_stride   strideB,
                                              const T_*        beta,
                                              T_* const*       C,
                                              rocblas_stride   offsetC,
                                              int64_t          ldc_64,
                                              rocblas_stride   strideC,
                                              int64_t          batch_count_64);
