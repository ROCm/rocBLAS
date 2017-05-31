/**************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 ************************************************************************** */

#include <hip/hip_runtime.h>
#include <sys/time.h>
#include "rocblas.h"
#include "Tensile.h"
#include "gemm.h"
#include "definitions.h"
#include "handle.h"


/*******************************************************************************
 * API Args
 ******************************************************************************/
#define ARGS(TYPE) \
  rocblas_handle handle, \
  rocblas_operation trans_a, \
  rocblas_operation trans_b, \
  rocblas_int m, \
  rocblas_int n, \
  rocblas_int k, \
  const TYPE *alpha, \
  const TYPE *A, \
  rocblas_int ld_a, \
  const TYPE *B, \
  rocblas_int ld_b, \
  const TYPE *beta, \
  TYPE *C, \
  rocblas_int ld_c

#define ARGS_BATCHED(TYPE) \
  rocblas_handle handle, \
  rocblas_operation trans_a, \
  rocblas_operation trans_b, \
  rocblas_int m, \
  rocblas_int n, \
  rocblas_int k, \
  const TYPE *alpha, \
  const TYPE *A, \
  rocblas_int ld_a, \
  rocblas_int bs_a, \
  const TYPE *B, \
  rocblas_int ld_b, \
  rocblas_int bs_b, \
  const TYPE *beta, \
  TYPE *C, \
  rocblas_int ld_c, \
  rocblas_int bs_c, \
  rocblas_int b_c


/*******************************************************************************
 * Preamble Code
 ******************************************************************************/
#define PREAMBLE \
  rocblas_int b_c = 1; \
  rocblas_int bs_c; \
  rocblas_int bs_a; \
  rocblas_int bs_b; \
  infer_batch_strides( trans_a, trans_b, m, n, k, \
      ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c ); \
  rocblas_status validArgs = validateArgs( handle, trans_a, trans_b, \
    m, n, k, alpha, A, ld_a, bs_a, B, ld_b, bs_b, beta, C, ld_c, bs_c, b_c); \
  if (validArgs != rocblas_status_success) return validArgs; \
  \
  unsigned int strideC1 = static_cast<unsigned int>(ld_c); \
  unsigned int strideC2 = static_cast<unsigned int>(bs_c); \
  unsigned int strideA1 = static_cast<unsigned int>(ld_a); \
  unsigned int strideA2 = static_cast<unsigned int>(bs_a); \
  unsigned int strideB1 = static_cast<unsigned int>(ld_b); \
  unsigned int strideB2 = static_cast<unsigned int>(bs_b); \
  unsigned int sizeI    = static_cast<unsigned int>(m); \
  unsigned int sizeJ    = static_cast<unsigned int>(n); \
  unsigned int sizeK    = b_c; \
  unsigned int sizeL    = static_cast<unsigned int>(k);

#define PREAMBLE_BATCHED \
  rocblas_status validArgs = validateArgs( handle, trans_a, trans_b, \
    m, n, k, alpha, A, ld_a, bs_a, B, ld_b, bs_b, beta, C, ld_c, bs_c, b_c); \
  if (validArgs != rocblas_status_success) return validArgs; \
  \
  unsigned int strideC1 = static_cast<unsigned int>(ld_c); \
  unsigned int strideC2 = static_cast<unsigned int>(bs_c); \
  unsigned int strideA1 = static_cast<unsigned int>(ld_a); \
  unsigned int strideA2 = static_cast<unsigned int>(bs_a); \
  unsigned int strideB1 = static_cast<unsigned int>(ld_b); \
  unsigned int strideB2 = static_cast<unsigned int>(bs_b); \
  unsigned int sizeI    = static_cast<unsigned int>(m); \
  unsigned int sizeJ    = static_cast<unsigned int>(n); \
  unsigned int sizeK    = static_cast<unsigned int>(b_c); \
  unsigned int sizeL    = static_cast<unsigned int>(k);



/*******************************************************************************
 * Calling Tensile
 ******************************************************************************/
#ifndef NDEBUG

#define PRINT_SOLUTION_NAME(PREC,TRANS) \
  std::cout << "Solution Name: " << tensileGetSolutionName_ ## TRANS ## _ ## PREC ## B(  \
      strideC1, strideC2, strideA1, strideA2, \
      strideB1, strideB2, sizeI, sizeJ, sizeK, sizeL, \
      handle->rocblas_stream ) << std::endl;

#define PRINT_RETURN_STATUS \
  std::cout << "Return Status: " << status << std::endl;

#else
#define PRINT_SOLUTION_NAME(PREC,TRANS)
#define PRINT_RETURN_STATUS
#endif

#define CALL_TENSILE(PREC, TRANS) \
  PRINT_SOLUTION_NAME(PREC,TRANS) \
  status = tensile_ ## TRANS ## _ ## PREC ## B( C, A, B, *alpha, *beta, \
      0, 0, 0, strideC1, strideC2, strideA1, strideA2, \
      strideB1, strideB2, sizeI, sizeJ, sizeK, sizeL, \
      handle->rocblas_stream, 0, nullptr, nullptr); \
  PRINT_RETURN_STATUS

/*******************************************************************************
 * Handle Transposes
 ******************************************************************************/
#define TENSILE_TRANSPOSES(PREC) \
  hipError_t status; \
  if ( trans_a == rocblas_operation_none) { \
    if (trans_b == rocblas_operation_none) { /*NN*/ \
      CALL_TENSILE(PREC,Cijk_Ailk_Bljk) \
    } else { /*NT*/ \
      CALL_TENSILE(PREC,Cijk_Ailk_Bjlk) \
    } \
  } else { /*TN*/ \
    if (trans_b == rocblas_operation_none) { \
      CALL_TENSILE(PREC,Cijk_Alik_Bljk) \
    } else { /*TT*/ \
      CALL_TENSILE(PREC,Cijk_Alik_Bjlk) \
    } \
  } \
  return get_rocblas_status_for_hip_status( status );


/*******************************************************************************
 * Batched vs Non
 ******************************************************************************/
#define GEMM_API(prec, PREC, TYPE) \
  rocblas_status rocblas_ ## prec ## gemm( ARGS(TYPE) ) { \
    PREAMBLE \
    TENSILE_TRANSPOSES(PREC) \
  }

#define GEMM_API_BATCHED(prec, PREC, TYPE) \
  rocblas_status rocblas_ ## prec ## gemm_strided_batched( ARGS_BATCHED(TYPE) ) { \
    PREAMBLE_BATCHED \
    TENSILE_TRANSPOSES(PREC) \
  }

/*******************************************************************************
 * GEMM APIs
 ******************************************************************************/
GEMM_API(s, S, float )
GEMM_API(d, D, double )
GEMM_API_BATCHED(s, S, float )
GEMM_API_BATCHED(d, D, double )

