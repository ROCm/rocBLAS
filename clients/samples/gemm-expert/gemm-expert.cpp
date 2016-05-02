// ==================================================
// Copyright 2016 Advanced Micro Devices, Inc.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// @file
/// @brief Sample file demonstrating use of the expert GEMM interface
///

#include "hip_runtime.h"
#include "rocblas.h"

int main( int argc, char* argv[] )
{
  rocblas_matrix alpha;
  rocblas_matrix A;
  rocblas_matrix B;
  rocblas_matrix beta;
  rocblas_matrix C;

  rocblas_control control;

  rocblas_status stat = rocblas_gemm( &alpha, &A, &B, &beta, &C, &control );

  return 0;
}
