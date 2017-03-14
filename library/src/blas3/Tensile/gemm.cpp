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
  rocblas_order order, \
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
  rocblas_order order, \
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
  infer_batch_strides( order, trans_a, trans_b, m, n, k, \
      ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c ); \
  rocblas_status validArgs = validateArgs( handle, order, trans_a, trans_b, \
    m, n, k, alpha, A, ld_a, bs_a, B, ld_b, bs_b, beta, C, ld_c, bs_c, b_c); \
  if (validArgs != rocblas_status_success) return validArgs; \
  \
  unsigned int strideC1 = static_cast<unsigned int>(ld_c); \
  unsigned int strideC2 = static_cast<unsigned int>(bs_c); \
  unsigned int strideA1 = static_cast<unsigned int>(ld_a); \
  unsigned int strideA2 = static_cast<unsigned int>(bs_a); \
  unsigned int strideB1 = static_cast<unsigned int>(ld_b); \
  unsigned int strideB2 = static_cast<unsigned int>(bs_b); \
  unsigned int sizeI    = (order==rocblas_order_column_major) \
      ? static_cast<unsigned int>(m) : static_cast<unsigned int>(n) ; \
  unsigned int sizeJ    = (order==rocblas_order_column_major) \
      ? static_cast<unsigned int>(n) : static_cast<unsigned int>(m) ; \
  unsigned int sizeK    = b_c; \
  unsigned int sizeL    = static_cast<unsigned int>(k);

#define PREAMBLE_BATCHED \
  rocblas_status validArgs = validateArgs( handle, order, trans_a, trans_b, \
    m, n, k, alpha, A, ld_a, bs_a, B, ld_b, bs_b, beta, C, ld_c, bs_c, b_c); \
  if (validArgs != rocblas_status_success) return validArgs; \
  \
  unsigned int strideC1 = static_cast<unsigned int>(ld_c); \
  unsigned int strideC2 = static_cast<unsigned int>(bs_c); \
  unsigned int strideA1 = static_cast<unsigned int>(ld_a); \
  unsigned int strideA2 = static_cast<unsigned int>(bs_a); \
  unsigned int strideB1 = static_cast<unsigned int>(ld_b); \
  unsigned int strideB2 = static_cast<unsigned int>(bs_b); \
  unsigned int sizeI    = (order==rocblas_order_column_major) \
      ? static_cast<unsigned int>(m) : static_cast<unsigned int>(n) ; \
  unsigned int sizeJ    = (order==rocblas_order_column_major) \
      ? static_cast<unsigned int>(n) : static_cast<unsigned int>(m) ; \
  unsigned int sizeK    = static_cast<unsigned int>(b_c); \
  unsigned int sizeL    = static_cast<unsigned int>(k);



/*******************************************************************************
 * Calling Tensile
 ******************************************************************************/
#define TENSILE_NN(SCHEDULE, PREC) \
  tensile_ ## SCHEDULE ##_Cijk_Ailk_Bljk_ ## PREC ## B
#define TENSILE_NT(SCHEDULE, PREC) \
  tensile_ ## SCHEDULE ##_Cijk_Ailk_Bjlk_ ## PREC ## B
#define TENSILE_TN(SCHEDULE, PREC) \
  tensile_ ## SCHEDULE ##_Cijk_Alik_Bljk_ ## PREC ## B
#define TENSILE_TT(SCHEDULE, PREC) \
  tensile_ ## SCHEDULE ##_Cijk_Alik_Bjlk_ ## PREC ## B

#define TENSILE_CALLS(SCHEDULE, PREC) \
  hipError_t status; \
  if ( trans_a == rocblas_operation_none) { \
    if (trans_b == rocblas_operation_none) { /*NN*/ \
      status = TENSILE_NN(SCHEDULE,PREC)( C, A, B, *alpha, *beta, \
          0, 0, 0, strideC1, strideC2, strideA1, strideA2, \
          strideB1, strideB2, sizeI, sizeJ, sizeK, sizeL, \
          handle->rocblas_stream, 0, nullptr, nullptr); \
    } else { /*NT*/ \
      status = TENSILE_NT(SCHEDULE,PREC)( C, A, B, *alpha, *beta, \
          0, 0, 0, strideC1, strideC2, strideA1, strideA2, \
          strideB1, strideB2, sizeI, sizeJ, sizeK, sizeL, \
          handle->rocblas_stream, 0, nullptr, nullptr); \
    } \
  } else { /*TN*/ \
    if (trans_b == rocblas_operation_none) { \
      status = TENSILE_TN(SCHEDULE,PREC)( C, A, B, *alpha, *beta, \
          0, 0, 0, strideC1, strideC2, strideA1, strideA2, \
          strideB1, strideB2, sizeI, sizeJ, sizeK, sizeL, \
          handle->rocblas_stream, 0, nullptr, nullptr); \
    } else { /*TT*/ \
      status = TENSILE_TT(SCHEDULE,PREC)( C, A, B, *alpha, *beta, \
          0, 0, 0, strideC1, strideC2, strideA1, strideA2, \
          strideB1, strideB2, sizeI, sizeJ, sizeK, sizeL, \
          handle->rocblas_stream, 0, nullptr, nullptr); \
    } \
  } \
  return get_rocblas_status_for_hip_status( status );


/*******************************************************************************
 * GEMM APIs
 ******************************************************************************/
#define GEMM_API(prec, PREC, TYPE, SCHEDULE) \
  rocblas_status rocblas_ ## prec ## gemm( ARGS(TYPE) ) { \
    PREAMBLE \
    TENSILE_CALLS(SCHEDULE, PREC) \
  }

#define GEMM_API_BATCHED(prec, PREC, TYPE, SCHEDULE) \
  rocblas_status rocblas_ ## prec ## gemm_batched( ARGS_BATCHED(TYPE) ) { \
    PREAMBLE_BATCHED \
    TENSILE_CALLS(SCHEDULE, PREC) \
  }

GEMM_API(s, S, float, Fiji )
GEMM_API(d, D, double, Fiji )
GEMM_API_BATCHED(s, S, float, Fiji )
GEMM_API_BATCHED(d, D, double, Fiji )

#if 0

#define COMPLEX 0

/*******************************************************************************
 * GEMM wrapper around Tensile
 ******************************************************************************/
rocblas_status xgemm_tensile(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    TensileDataType type_alpha, const void *alpha,
    TensileDataType     type_a, const void *a,     rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    TensileDataType     type_b, const void *b,     rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    TensileDataType  type_beta, const void *beta,
    TensileDataType     type_c,       void *c,     rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int b_c ) {

    // quick return 0 is valid in BLAS
    if ( m == 0 || n == 0 || k == 0 || b_c == 0) {
        return rocblas_status_success;
    }

    // sizes must not be negative
    if ( m < 0 || n < 0 || k < 0 || b_c < 0) {
        return rocblas_status_invalid_size;
    }

    // handle must be valid
    if (handle == nullptr) {
        return rocblas_status_invalid_handle;
    }

    // pointers must be valid
    if ( c == nullptr
        || a == nullptr
        || b == nullptr
        || alpha == nullptr
        || beta == nullptr ) {
            return rocblas_status_invalid_pointer;
    }

    // tensor dimensions in rows/cols
    int num_cols_c = n;
    int num_rows_c = m;
    int num_cols_a = (trans_a == rocblas_operation_none) ? k : m;
    int num_rows_a = (trans_a == rocblas_operation_none) ? m : k;
    int num_cols_b = (trans_b == rocblas_operation_none) ? n : k;
    int num_rows_b = (trans_b == rocblas_operation_none) ? k : n;

    if(order==rocblas_order_column_major){
        if( num_rows_a > ld_a
            || num_rows_b > ld_b
            || num_rows_c > ld_c) {
                return rocblas_status_invalid_size;
            }
    }

    /* create tensile tensors
    * - translates rows/cols into dim0/dim1
    * - dim0 is dimension with shortest stride (necessary for performance)
    * - dim1 is other matrix dimension
    * - dim2 is batch dimension
    */

    // create tensor c
    TensileTensor tensor_c;
    tensor_c.dataType = type_c;
    tensor_c.numDimensions = (b_c > 1) ? 3 : 2;
    tensor_c.dimensions[0].stride = ls_c;
    tensor_c.dimensions[0].size   = (order==rocblas_order_column_major) ? num_rows_c : num_cols_c;
    tensor_c.dimensions[1].stride = ld_c;
    tensor_c.dimensions[1].size   = (order==rocblas_order_column_major) ? num_cols_c : num_rows_c;
    tensor_c.dimensions[2].stride = bs_c;
    tensor_c.dimensions[2].size   = b_c;
    // validate tensor c
    if (tensor_c.dimensions[0].stride < 1) {
        // user gave invalid ls_c
        return rocblas_status_invalid_size;
    }
    if (tensor_c.dimensions[1].stride < tensor_c.dimensions[0].stride * tensor_c.dimensions[0].size ) {
        // user gave invalid ld_c
        return rocblas_status_invalid_size;
    }
    if (tensor_c.dimensions[2].stride < tensor_c.dimensions[1].stride * tensor_c.dimensions[1].size ) {
        // user gave invalid bs_c
        return rocblas_status_invalid_size;
    }

    // create tensor a
    TensileTensor tensor_a;
    tensor_a.dataType = conjugate_if_necessary( type_a, trans_a );
    tensor_a.numDimensions = (b_c > 1) ? 3 : 2;
    tensor_a.dimensions[0].stride = ls_a;
    tensor_a.dimensions[0].size   = (order==rocblas_order_column_major) ? num_rows_a : num_cols_a;
    tensor_a.dimensions[1].stride = ld_a;
    tensor_a.dimensions[1].size   = (order==rocblas_order_column_major) ? num_cols_a : num_rows_a;
    tensor_a.dimensions[2].stride = bs_a;
    tensor_a.dimensions[2].size   = b_c;
    // validate tensor a
    if (tensor_a.dimensions[0].stride < 1) {
        // user gave invalid ls_a
        return rocblas_status_invalid_size;
    }
    if (tensor_a.dimensions[1].stride < tensor_a.dimensions[0].stride * tensor_a.dimensions[0].size ) {
        // user gave invalid ld_a
        return rocblas_status_invalid_size;
    }
    if (tensor_a.dimensions[2].stride < tensor_a.dimensions[1].stride * tensor_a.dimensions[1].size ) {
        // user gave invalid bs_a
        return rocblas_status_invalid_size;
    }

    // create tensor b
    TensileTensor tensor_b;
    tensor_b.dataType = conjugate_if_necessary( type_b, trans_b );
    tensor_b.numDimensions = (b_c > 1) ? 3 : 2;
    tensor_b.dimensions[0].stride = ls_b;
    tensor_b.dimensions[0].size   = (order==rocblas_order_column_major) ? num_rows_b : num_cols_b;
    tensor_b.dimensions[1].stride = ld_b;
    tensor_b.dimensions[1].size   = (order==rocblas_order_column_major) ? num_cols_b : num_rows_b;
    tensor_b.dimensions[2].stride = bs_b;
    tensor_b.dimensions[2].size   = b_c;
    // validate tensor b
    if (tensor_b.dimensions[0].stride < 1) {
        // user gave invalid ls_b
        return rocblas_status_invalid_size;
    }
    if (tensor_b.dimensions[1].stride < tensor_b.dimensions[0].stride * tensor_b.dimensions[0].size ) {
        // user gave invalid ld_b
        return rocblas_status_invalid_size;
    }
    if (tensor_b.dimensions[2].stride < tensor_b.dimensions[1].stride * tensor_b.dimensions[1].size ) {
        // user gave invalid bs_b
        return rocblas_status_invalid_size;
    }


    // TODO - do assignments depend on order?
    // index assignments
    unsigned int index_assignments_a[3];
    unsigned int index_assignments_b[3];
    if ( b_c > 1) {
        index_assignments_a[0] = trans_a == rocblas_operation_none ? 0 : 3;
        index_assignments_a[1] = trans_a == rocblas_operation_none ? 3 : 0;
        index_assignments_a[2] = 2;
        index_assignments_b[0] = trans_b == rocblas_operation_none ? 3 : 1;
        index_assignments_b[1] = trans_b == rocblas_operation_none ? 1 : 3;
        index_assignments_b[2] = 2;
    } else {
        index_assignments_a[0] = trans_a == rocblas_operation_none ? 0 : 2;
        index_assignments_a[1] = trans_a == rocblas_operation_none ? 2 : 0;
        index_assignments_b[0] = trans_b == rocblas_operation_none ? 2 : 1;
        index_assignments_b[1] = trans_b == rocblas_operation_none ? 1 : 2;
    }

    #ifndef NDEBUG
    printf("creating problem \n");
    #endif
    // create problem
    TensileProblem problem;
    // DONOT SIMPLY return a TENSILE error, the return type is rocblas_status, otherwise cause stalling
    PRINT_IF_TENSILE_ERROR( tensileCreateProblem(
        &problem,
        tensor_c,
        tensor_a,
        tensor_b,
        index_assignments_a,
        index_assignments_b,
        tensileOperationTypeContraction,
        type_alpha,
        type_beta,
        false, // Use offsets? No. Only OpenCL needed them for generality; HIP doesn't.
        handle->tensile_device_profile) );

    #ifndef NDEBUG
    // thorough validation that problem was created correctly
    PRINT_IF_TENSILE_ERROR( tensileValidateProblem(problem) );
    #endif

    #ifndef NDEBUG
    // lookup solution
    printf("looking up solution \n");
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double begin = (tv.tv_sec * 1000 * 1000) + tv.tv_usec ;
    #endif
    
    TensileSolution solution;
    PRINT_IF_TENSILE_ERROR( tensileGetSolutionForProblem( &solution, problem ) );

    #ifndef NDEBUG
    gettimeofday(&tv, NULL);
    double end = (tv.tv_sec * 1000 * 1000) + tv.tv_usec ;
    double time_used_in_us =  (end - begin);
    printf("It takes %f us to get the solution \n", time_used_in_us);
    #endif

    // wrap pointers and enqueue solution
    TensileTensorData      tensor_data_c{ c, 0 };
    TensileTensorDataConst tensor_data_a{ a, 0 };
    TensileTensorDataConst tensor_data_b{ b, 0 };
    TensileScalarData      scalar_data_alpha{ alpha };
    TensileScalarData      scalar_data_beta{ beta };
    PRINT_IF_TENSILE_ERROR( tensileEnqueueSolution(
        solution,
        tensor_data_c,
        tensor_data_a,
        tensor_data_b,
        scalar_data_alpha,
        scalar_data_beta,
        &handle->tensile_control) );

    // cleanup
    PRINT_IF_TENSILE_ERROR( tensileDestroyProblem(problem) );
    PRINT_IF_TENSILE_ERROR( tensileDestroySolution(solution) );

    // success
    return rocblas_status_success;
}



/*******************************************************************************
 * API Functions :
 ******************************************************************************/
#if 0
 rocblas_status rocblas_hgemm(
     rocblas_handle handle,
     rocblas_order order,
     rocblas_operation trans_a, rocblas_operation trans_b,
     rocblas_int m, rocblas_int n, rocblas_int k,
     const rocblas_half *alpha,
     const rocblas_half *A, rocblas_int ld_a,
     const rocblas_half *B, rocblas_int ld_b,
     const rocblas_half *beta,
           rocblas_half *C, rocblas_int ld_c) {

     TensileDataType type_c     = tensileDataTypeHalf;
     TensileDataType type_a     = tensileDataTypeHalf;
     TensileDataType type_b     = tensileDataTypeHalf;
     TensileDataType type_alpha = tensileDataTypeHalf;
     TensileDataType type_beta  = tensileDataTypeHalf;

     rocblas_int ls_c = 1;
     rocblas_int ls_a = 1;
     rocblas_int ls_b = 1;

     rocblas_int bs_c;
     rocblas_int bs_a;
     rocblas_int bs_b;

     infer_batch_strides( order, trans_a, trans_b, m, n, k,
       ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
     rocblas_int b_c = 1;

     return xgemm_tensile( handle, order, trans_a, trans_b,
         m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
         type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
         type_c, C, ls_c, ld_c, bs_c, b_c );
}
#endif


rocblas_status rocblas_dgemm( ARGS(double) ) {
}

// rocblas_status rocblas_qgemm(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation trans_a, rocblas_operation trans_b,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half_complex *alpha,
//     const rocblas_half_complex *A, rocblas_int ld_a,
//     const rocblas_half_complex *B, rocblas_int ld_b,
//     const rocblas_half_complex *beta,
//           rocblas_half_complex *C, rocblas_int ld_c) {

  //   TensileDataType type_c     = tensileDataTypeComplexHalf;
  //   TensileDataType type_a     = tensileDataTypeComplexHalf;
  //   TensileDataType type_b     = tensileDataTypeComplexHalf;
  //   TensileDataType type_alpha = tensileDataTypeComplexHalf;
  //   TensileDataType type_beta  = tensileDataTypeComplexHalf;

  //   rocblas_int ls_c = 1;
  //   rocblas_int ls_a = 1;
  //   rocblas_int ls_b = 1;

  //   rocblas_int bs_c;
  //   rocblas_int bs_a;
  //   rocblas_int bs_b;

  //   infer_batch_strides( order, trans_a, trans_b, m, n, k,
  //     ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  //   rocblas_int b_c = 1;

  //   return xgemm_tensile( handle, order, trans_a, trans_b,
  //       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
  //       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
  //       type_c, C, ls_c, ld_c, bs_c, b_c );
// }

#if COMPLEX
rocblas_status rocblas_cgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int ld_a,
    const rocblas_float_complex *B, rocblas_int ld_b,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ld_c) {

    TensileDataType type_c     = tensileDataTypeComplexSingle;
    TensileDataType type_a     = tensileDataTypeComplexSingle;
    TensileDataType type_b     = tensileDataTypeComplexSingle;
    TensileDataType type_alpha = tensileDataTypeComplexSingle;
    TensileDataType type_beta  = tensileDataTypeComplexSingle;

    rocblas_int ls_c = 1;
    rocblas_int ls_a = 1;
    rocblas_int ls_b = 1;

    rocblas_int bs_c;
    rocblas_int bs_a;
    rocblas_int bs_b;

    infer_batch_strides( order, trans_a, trans_b, m, n, k,
        ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
        rocblas_int b_c = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
          m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
          type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
          type_c, C, ls_c, ld_c, bs_c, b_c );
}

rocblas_status rocblas_zgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int ld_a,
    const rocblas_double_complex *B, rocblas_int ld_b,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ld_c) {

    TensileDataType type_c     = tensileDataTypeComplexDouble;
    TensileDataType type_a     = tensileDataTypeComplexDouble;
    TensileDataType type_b     = tensileDataTypeComplexDouble;
    TensileDataType type_alpha = tensileDataTypeComplexDouble;
    TensileDataType type_beta  = tensileDataTypeComplexDouble;

    rocblas_int ls_c = 1;
    rocblas_int ls_a = 1;
    rocblas_int ls_b = 1;

    rocblas_int bs_c;
    rocblas_int bs_a;
    rocblas_int bs_b;

    infer_batch_strides( order, trans_a, trans_b, m, n, k,
      ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
    rocblas_int b_c = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}
#endif

    /***************************************************************************
     * strided - specify leading stride
     * ls_a - non-1 leading stride of a
     * ls_b - non-1 leading stride of b
     * ls_c - non-1 leading stride of c
     **************************************************************************/
// rocblas_status rocblas_hgemm_strided(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation trans_a, rocblas_operation trans_b,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half *alpha,
//     const rocblas_half *A, rocblas_int ls_a, rocblas_int ld_a,
//     const rocblas_half *B, rocblas_int ls_b, rocblas_int ld_b,
//     const rocblas_half *beta,
//           rocblas_half *C, rocblas_int ls_c, rocblas_int ld_c) {
//
  //   TensileDataType type_c     = tensileDataTypeHalf;
  //   TensileDataType type_a     = tensileDataTypeHalf;
  //   TensileDataType type_b     = tensileDataTypeHalf;
  //   TensileDataType type_alpha = tensileDataTypeHalf;
  //   TensileDataType type_beta  = tensileDataTypeHalf;
  //
  //   rocblas_int bs_c;
  //   rocblas_int bs_a;
  //   rocblas_int bs_b;
  //
  //   infer_batch_strides( order, trans_a, trans_b, m, n, k,
  //     ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  //   rocblas_int b_c = 1;
  //
  //   return xgemm_tensile( handle, order, trans_a, trans_b,
  //       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
  //       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
  //       type_c, C, ls_c, ld_c, bs_c, b_c );
// }

rocblas_status rocblas_sgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int ls_a, rocblas_int ld_a,
    const float *B, rocblas_int ls_b, rocblas_int ld_b,
    const float *beta,
          float *C, rocblas_int ls_c, rocblas_int ld_c) {

    TensileDataType type_c     = tensileDataTypeSingle;
    TensileDataType type_a     = tensileDataTypeSingle;
    TensileDataType type_b     = tensileDataTypeSingle;
    TensileDataType type_alpha = tensileDataTypeSingle;
    TensileDataType type_beta  = tensileDataTypeSingle;

    rocblas_int bs_c;
    rocblas_int bs_a;
    rocblas_int bs_b;

    infer_batch_strides( order, trans_a, trans_b, m, n, k,
      ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
    rocblas_int b_c = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

rocblas_status rocblas_dgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int ls_a, rocblas_int ld_a,
    const double *B, rocblas_int ls_b, rocblas_int ld_b,
    const double *beta,
          double *C, rocblas_int ls_c, rocblas_int ld_c) {

    TensileDataType type_c     = tensileDataTypeDouble;
    TensileDataType type_a     = tensileDataTypeDouble;
    TensileDataType type_b     = tensileDataTypeDouble;
    TensileDataType type_alpha = tensileDataTypeDouble;
    TensileDataType type_beta  = tensileDataTypeDouble;

    rocblas_int bs_c;
    rocblas_int bs_a;
    rocblas_int bs_b;

    infer_batch_strides( order, trans_a, trans_b, m, n, k,
      ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
    rocblas_int b_c = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

// rocblas_status rocblas_qgemm_strided(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation trans_a, rocblas_operation trans_b,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half_complex *alpha,
//     const rocblas_half_complex *A, rocblas_int ls_a, rocblas_int ld_a,
//     const rocblas_half_complex *B, rocblas_int ls_b, rocblas_int ld_b,
//     const rocblas_half_complex *beta,
//           rocblas_half_complex *C, rocblas_int ls_c, rocblas_int ld_c) {

//   TensileDataType type_c     = tensileDataTypeComplexHalf;
//   TensileDataType type_a     = tensileDataTypeComplexHalf;
//   TensileDataType type_b     = tensileDataTypeComplexHalf;
//   TensileDataType type_alpha = tensileDataTypeComplexHalf;
//   TensileDataType type_beta  = tensileDataTypeComplexHalf;

//   rocblas_int bs_c;
//   rocblas_int bs_a;
//   rocblas_int bs_b;

//   infer_batch_strides( order, trans_a, trans_b, m, n, k,
//     ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
//   rocblas_int b_c = 1;

//   return xgemm_tensile( handle, order, trans_a, trans_b,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, b_c );
// }

#if COMPLEX
rocblas_status rocblas_cgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int ls_a, rocblas_int ld_a,
    const rocblas_float_complex *B, rocblas_int ls_b, rocblas_int ld_b,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ls_c, rocblas_int ld_c) {

    TensileDataType type_c     = tensileDataTypeComplexSingle;
    TensileDataType type_a     = tensileDataTypeComplexSingle;
    TensileDataType type_b     = tensileDataTypeComplexSingle;
    TensileDataType type_alpha = tensileDataTypeComplexSingle;
    TensileDataType type_beta  = tensileDataTypeComplexSingle;

    rocblas_int bs_c;
    rocblas_int bs_a;
    rocblas_int bs_b;

    infer_batch_strides( order, trans_a, trans_b, m, n, k,
      ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
    rocblas_int b_c = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

rocblas_status rocblas_zgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int ls_a, rocblas_int ld_a,
    const rocblas_double_complex *B, rocblas_int ls_b, rocblas_int ld_b,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ls_c, rocblas_int ld_c) {

    TensileDataType type_c     = tensileDataTypeComplexDouble;
    TensileDataType type_a     = tensileDataTypeComplexDouble;
    TensileDataType type_b     = tensileDataTypeComplexDouble;
    TensileDataType type_alpha = tensileDataTypeComplexDouble;
    TensileDataType type_beta  = tensileDataTypeComplexDouble;

    rocblas_int bs_c;
    rocblas_int bs_a;
    rocblas_int bs_b;

    infer_batch_strides( order, trans_a, trans_b, m, n, k,
      ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
    rocblas_int b_c = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}
#endif

    /***************************************************************************
     * batched
     * bs_a - "batch stride a": stride from the start of one "A" matrix to the next
     * bs_b
     * bs_c
     * b_c - numbers of gemm's in the batch
     **************************************************************************/
// rocblas_status rocblas_hgemm_batched(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation trans_a, rocblas_operation trans_b,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half *alpha,
//     const rocblas_half *A, rocblas_int ld_a, rocblas_int bs_a,
//     const rocblas_half *B, rocblas_int ld_b, rocblas_int bs_b,
//     const rocblas_half *beta,
//           rocblas_half *C, rocblas_int ld_c, rocblas_int bs_c,
//     rocblas_int b_c ) {
//
//   TensileDataType type_c     = tensileDataTypeHalf;
//   TensileDataType type_a     = tensileDataTypeHalf;
//   TensileDataType type_b     = tensileDataTypeHalf;
//   TensileDataType type_alpha = tensileDataTypeHalf;
//   TensileDataType type_beta  = tensileDataTypeHalf;
//
//   rocblas_int ls_c = 1;
//   rocblas_int ls_a = 1;
//   rocblas_int ls_b = 1;
//
//   return xgemm_tensile( handle, order, trans_a, trans_b,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, b_c );
// }

rocblas_status rocblas_sgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int ld_a, rocblas_int bs_a,
    const float *B, rocblas_int ld_b, rocblas_int bs_b,
    const float *beta,
          float *C, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int b_c ) {

    TensileDataType type_c     = tensileDataTypeSingle;
    TensileDataType type_a     = tensileDataTypeSingle;
    TensileDataType type_b     = tensileDataTypeSingle;
    TensileDataType type_alpha = tensileDataTypeSingle;
    TensileDataType type_beta  = tensileDataTypeSingle;

    rocblas_int ls_c = 1;
    rocblas_int ls_a = 1;
    rocblas_int ls_b = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

rocblas_status rocblas_dgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int ld_a, rocblas_int bs_a,
    const double *B, rocblas_int ld_b, rocblas_int bs_b,
    const double *beta,
          double *C, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int b_c ) {

    TensileDataType type_c     = tensileDataTypeDouble;
    TensileDataType type_a     = tensileDataTypeDouble;
    TensileDataType type_b     = tensileDataTypeDouble;
    TensileDataType type_alpha = tensileDataTypeDouble;
    TensileDataType type_beta  = tensileDataTypeDouble;

    rocblas_int ls_c = 1;
    rocblas_int ls_a = 1;
    rocblas_int ls_b = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

// rocblas_status rocblas_qgemm_batched(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation trans_a, rocblas_operation trans_b,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half_complex *alpha,
//     const rocblas_half_complex *A, rocblas_int ld_a, rocblas_int bs_a,
//     const rocblas_half_complex *B, rocblas_int ld_b, rocblas_int bs_b,
//     const rocblas_half_complex *beta,
//           rocblas_half_complex *C, rocblas_int ld_c, rocblas_int bs_c,
//     rocblas_int b_c ) {

  //   TensileDataType type_c     = tensileDataTypeComplexHalf;
  //   TensileDataType type_a     = tensileDataTypeComplexHalf;
  //   TensileDataType type_b     = tensileDataTypeComplexHalf;
  //   TensileDataType type_alpha = tensileDataTypeComplexHalf;
  //   TensileDataType type_beta  = tensileDataTypeComplexHalf;

  //   rocblas_int ls_c = 1;
  //   rocblas_int ls_a = 1;
  //   rocblas_int ls_b = 1;

  //   return xgemm_tensile( handle, order, trans_a, trans_b,
  //       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
  //       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
  //       type_c, C, ls_c, ld_c, bs_c, b_c );
// }

#if COMPLEX
rocblas_status rocblas_cgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int ld_a, rocblas_int bs_a,
    const rocblas_float_complex *B, rocblas_int ld_b, rocblas_int bs_b,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int b_c ) {

    TensileDataType type_c     = tensileDataTypeComplexSingle;
    TensileDataType type_a     = tensileDataTypeComplexSingle;
    TensileDataType type_b     = tensileDataTypeComplexSingle;
    TensileDataType type_alpha = tensileDataTypeComplexSingle;
    TensileDataType type_beta  = tensileDataTypeComplexSingle;

    rocblas_int ls_c = 1;
    rocblas_int ls_a = 1;
    rocblas_int ls_b = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

rocblas_status rocblas_zgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int ld_a, rocblas_int bs_a,
    const rocblas_double_complex *B, rocblas_int ld_b, rocblas_int bs_b,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int b_c ) {

    TensileDataType type_c     = tensileDataTypeComplexDouble;
    TensileDataType type_a     = tensileDataTypeComplexDouble;
    TensileDataType type_b     = tensileDataTypeComplexDouble;
    TensileDataType type_alpha = tensileDataTypeComplexDouble;
    TensileDataType type_beta  = tensileDataTypeComplexDouble;

    rocblas_int ls_c = 1;
    rocblas_int ls_a = 1;
    rocblas_int ls_b = 1;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

#endif
    /***************************************************************************
     * strided & batched
     * ls_a - non-1 leading stride of a
     * ls_b - non-1 leading stride of b
     * ls_c - non-1 leading stride of c
     * bs_a - "batch stride a": stride from the start of one "A" matrix to the next
     * bs_b
     * bs_c
     * b_c - numbers of gemm's in the batch
     **************************************************************************/
// rocblas_status rocblas_hgemm_strided_batched(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation trans_a, rocblas_operation trans_b,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half *alpha,
//     const rocblas_half *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
//     const rocblas_half *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
//     const rocblas_half *beta,
//           rocblas_half *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
//     rocblas_int b_c ) {
//
  //   TensileDataType type_c     = tensileDataTypeHalf;
  //   TensileDataType type_a     = tensileDataTypeHalf;
  //   TensileDataType type_b     = tensileDataTypeHalf;
  //   TensileDataType type_alpha = tensileDataTypeHalf;
  //   TensileDataType type_beta  = tensileDataTypeHalf;
  //
  //   return xgemm_tensile( handle, order, trans_a, trans_b,
  //       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
  //       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
  //       type_c, C, ls_c, ld_c, bs_c, b_c );
// }

rocblas_status rocblas_sgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const float *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const float *beta,
          float *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int b_c ) {

    TensileDataType type_c     = tensileDataTypeSingle;
    TensileDataType type_a     = tensileDataTypeSingle;
    TensileDataType type_b     = tensileDataTypeSingle;
    TensileDataType type_alpha = tensileDataTypeSingle;
    TensileDataType type_beta  = tensileDataTypeSingle;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

rocblas_status rocblas_dgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const double *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const double *beta,
          double *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int b_c ) {

    TensileDataType type_c     = tensileDataTypeDouble;
    TensileDataType type_a     = tensileDataTypeDouble;
    TensileDataType type_b     = tensileDataTypeDouble;
    TensileDataType type_alpha = tensileDataTypeDouble;
    TensileDataType type_beta  = tensileDataTypeDouble;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

// rocblas_status rocblas_qgemm_strided_batched(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation trans_a, rocblas_operation trans_b,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half_complex *alpha,
//     const rocblas_half_complex *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
//     const rocblas_half_complex *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
//     const rocblas_half_complex *beta,
//           rocblas_half_complex *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
//     rocblas_int b_c ) {

  //   TensileDataType type_c     = tensileDataTypeComplexHalf;
  //   TensileDataType type_a     = tensileDataTypeComplexHalf;
  //   TensileDataType type_b     = tensileDataTypeComplexHalf;
  //   TensileDataType type_alpha = tensileDataTypeComplexHalf;
  //   TensileDataType type_beta  = tensileDataTypeComplexHalf;

  //   return xgemm_tensile( handle, order, trans_a, trans_b,
  //       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
  //       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
  //       type_c, C, ls_c, ld_c, bs_c, b_c );
// }

#if COMPLEX
rocblas_status rocblas_cgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const rocblas_float_complex *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int b_c ) {

    TensileDataType type_c     = tensileDataTypeComplexSingle;
    TensileDataType type_a     = tensileDataTypeComplexSingle;
    TensileDataType type_b     = tensileDataTypeComplexSingle;
    TensileDataType type_alpha = tensileDataTypeComplexSingle;
    TensileDataType type_beta  = tensileDataTypeComplexSingle;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}

rocblas_status rocblas_zgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation trans_a, rocblas_operation trans_b,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const rocblas_double_complex *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int b_c ) {

    TensileDataType type_c     = tensileDataTypeComplexDouble;
    TensileDataType type_a     = tensileDataTypeComplexDouble;
    TensileDataType type_b     = tensileDataTypeComplexDouble;
    TensileDataType type_alpha = tensileDataTypeComplexDouble;
    TensileDataType type_beta  = tensileDataTypeComplexDouble;

    return xgemm_tensile( handle, order, trans_a, trans_b,
        m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
        type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
        type_c, C, ls_c, ld_c, bs_c, b_c );
}
// eliminating complex
#endif


// eliminating old code
#endif
