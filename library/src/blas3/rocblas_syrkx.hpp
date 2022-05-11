/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
                                    rocblas_stride    offset_a,
                                    rocblas_int       lda,
                                    rocblas_stride    stride_a,
                                    TConstPtr*        db,
                                    rocblas_stride    offset_b,
                                    rocblas_int       ldb,
                                    rocblas_stride    stride_b,
                                    TScal*            beta,
                                    TPtr*             dc,
                                    rocblas_stride    offset_c,
                                    rocblas_int       ldc,
                                    rocblas_stride    stride_c,
                                    rocblas_int       batch_count);
