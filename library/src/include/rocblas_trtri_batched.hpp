/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas.h"
#include "status.h"

template <typename T>
rocblas_status rocblas_trtri_batched_template(rocblas_handle handle,
                                              rocblas_fill uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int n,
                                              const T* A,
                                              rocblas_int lda,
                                              rocblas_int bsa,
                                              T* invA,
                                              rocblas_int ldinvA,
                                              rocblas_int bsinvA,
                                              rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_strtri_batched(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      const float* A,
                                      rocblas_int lda,
                                      rocblas_int bsa,
                                      float* invA,
                                      rocblas_int ldinvA,
                                      rocblas_int bsinvA,
                                      rocblas_int batch_count);

ROCBLAS_EXPORT rocblas_status rocblas_dtrtri_batched(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      const double* A,
                                      rocblas_int lda,
                                      rocblas_int bsa,
                                      double* invA,
                                      rocblas_int ldinvA,
                                      rocblas_int bsinvA,
                                      rocblas_int batch_count);