/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#pragma once
#ifndef _SUPPLEMENTARY_BLAS_
#define _SUPPLEMENTARY_BLAS_

#include "rocblas.h"

/*!\file
 * \brief Used to provided CPU reference code for BLAS functions not normally
          available (primarily specialized data format types), for testing
          purposes only, and not part of the GPU library
 */

template <typename Tin, typename Tout>
void cblas_gemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int m,
                rocblas_int n,
                rocblas_int k,
                Tout alpha,
                Tin* A,
                rocblas_int lda,
                Tout* B,
                rocblas_int ldb,
                Tin beta,
                Tout* C,
                rocblas_int ldc);

/* ============================================================================================ */

#endif /* _SUPPLEMENTARY_BLAS__ */
