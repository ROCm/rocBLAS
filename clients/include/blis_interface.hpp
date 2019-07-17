/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#ifndef _BLIS_INTERFACE_
#define _BLIS_INTERFACE_


#include "rocblas.h"

void setup_blis();

void blis_dgemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       k,
                double            alpha,
                double*           A,
                rocblas_int       lda,
                double*           B,
                rocblas_int       ldb,
                double            beta,
                double*           C,
                rocblas_int       ldc);

void blis_sgemm(rocblas_operation transA,
                rocblas_operation transB,
                rocblas_int       m,
                rocblas_int       n,
                rocblas_int       k,
                float             alpha,
                float*            A,
                rocblas_int       lda,
                float*            B,
                rocblas_int       ldb,
                float             beta,
                float*            C,
                rocblas_int       ldc);

template <typename T>
void (*blis_gemm)(rocblas_operation transA,
                  rocblas_operation transB,
                  rocblas_int       m,
                  rocblas_int       n,
                  rocblas_int       k,
                  T                 alpha,
                  T*                A,
                  rocblas_int       lda,
                  T*                B,
                  rocblas_int       ldb,
                  T                 beta,
                  T*                C,
                  rocblas_int       ldc);

template <>
static constexpr auto blis_gemm<float> = blis_sgemm;

template <>
static constexpr auto blis_gemm<double> = blis_dgemm;

#endif /* _BLIS_INTERFACE_ */