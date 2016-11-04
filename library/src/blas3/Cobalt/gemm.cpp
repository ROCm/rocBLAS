/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "Cobalt.h"
#include "gemm.h"
#include "definitions.h"
#include "handle.h"

/*******************************************************************************
 * GEMM wrapper around Cobalt
 ******************************************************************************/
rocblas_status xgemm_cobalt(
  rocblas_handle handle,
  rocblas_order order,
  rocblas_operation trans_a, rocblas_operation trans_b,
  rocblas_int m, rocblas_int n, rocblas_int k,
  CobaltDataType type_alpha, const void *alpha,
  CobaltDataType     type_a, const void *a,     rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
  CobaltDataType     type_b, const void *b,     rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
  CobaltDataType  type_beta, const void *beta,
  CobaltDataType     type_c,       void *c,     rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
  rocblas_int batch_count ) {

  // sizes must not be negative
  if ( m < 0 || n < 0 || k < 0 ) {
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
  int num_cols_c = m;
  int num_rows_c = n;
  int num_cols_a = (trans_a == rocblas_operation_none) ? k : m;
  int num_rows_a = (trans_a == rocblas_operation_none) ? m : k;
  int num_cols_b = (trans_b == rocblas_operation_none) ? n : k;
  int num_rows_b = (trans_b == rocblas_operation_none) ? k : n;


  /* create cobalt tensors
   * - translates rows/cols into dim0/dim1
   * - dim0 is dimension with shortest stride (necessary for performance)
   * - dim1 is other matrix dimension
   * - dim2 is batch dimension
   */

  // create tensor c
  CobaltTensor tensor_c;
  tensor_c.dataType = type_c;
  tensor_c.numDimensions = (batch_count > 1) ? 3 : 2;
  tensor_c.dimensions[0].stride = ls_c;
  tensor_c.dimensions[0].size   = (order==rocblas_order_column_major) ? num_rows_c : num_cols_c;
  tensor_c.dimensions[1].stride = ld_c;
  tensor_c.dimensions[1].size   = (order==rocblas_order_column_major) ? num_cols_c : num_rows_c;
  tensor_c.dimensions[2].stride = bs_c;
  tensor_c.dimensions[2].size   = batch_count;
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
  CobaltTensor tensor_a;
  tensor_a.dataType = conjugate_if_necessary( type_a, trans_a );
  tensor_a.numDimensions = (batch_count > 1) ? 3 : 2;
  tensor_a.dimensions[0].stride = ls_a;
  tensor_a.dimensions[0].size   = (order==rocblas_order_column_major) ? num_rows_a : num_cols_a;
  tensor_a.dimensions[1].stride = ld_a;
  tensor_a.dimensions[1].size   = (order==rocblas_order_column_major) ? num_cols_a : num_rows_a;
  tensor_a.dimensions[2].stride = bs_a;
  tensor_a.dimensions[2].size   = batch_count;
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
  CobaltTensor tensor_b;
  tensor_b.dataType = conjugate_if_necessary( type_b, trans_b );
  tensor_b.numDimensions = (batch_count > 1) ? 3 : 2;
  tensor_b.dimensions[0].stride = ls_b;
  tensor_b.dimensions[0].size   = (order==rocblas_order_column_major) ? num_rows_b : num_cols_b;
  tensor_b.dimensions[1].stride = ld_b;
  tensor_b.dimensions[1].size   = (order==rocblas_order_column_major) ? num_cols_b : num_rows_b;
  tensor_b.dimensions[2].stride = bs_b;
  tensor_b.dimensions[2].size   = batch_count;
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
  if ( batch_count > 1) {
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

  // create problem
  CobaltProblem problem;
  RETURN_IF_COBALT_ERROR( cobaltCreateProblem(
      &problem,
      tensor_c,
      tensor_a,
      tensor_b,
      index_assignments_a,
      index_assignments_b,
      cobaltOperationTypeContraction,
      type_alpha,
      type_beta,
      false, // Use offsets? No. Only OpenCL needed them for generality; HIP doesn't.
      handle->cobalt_device_profile) );

#ifdef _DEBUG
  // thorough validation that problem was created correctly
  RETURN_IF_COBALT_ERROR( cobaltValidateProblem(problem) );
#endif

  // lookup solution
  CobaltSolution solution;
  RETURN_IF_COBALT_ERROR( cobaltGetSolutionForProblem( &solution, problem ) );

  // wrap pointers and enqueue solution
  CobaltTensorData      tensor_data_c{ c, 0 };
  CobaltTensorDataConst tensor_data_a{ a, 0 };
  CobaltTensorDataConst tensor_data_b{ b, 0 };
  CobaltScalarData      scalar_data_alpha{ alpha };
  CobaltScalarData      scalar_data_beta{ beta };
  RETURN_IF_COBALT_ERROR( cobaltEnqueueSolution(
      solution,
      tensor_data_c,
      tensor_data_a,
      tensor_data_b,
      scalar_data_alpha,
      scalar_data_beta,
      &handle->cobalt_control) );

  // cleanup
  RETURN_IF_COBALT_ERROR( cobaltDestroyProblem(problem) );
  RETURN_IF_COBALT_ERROR( cobaltDestroySolution(solution) );

  // TODO - put events into handle, if necessary

  // success
  return rocblas_status_success;
}



/*******************************************************************************
 * API Functions :
 ******************************************************************************/
// fp16 (hgemm) is available
// rocblas_status rocblas_hgemm(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation transa, rocblas_operation transb,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half *alpha,
//     const rocblas_half *A, rocblas_int ld_a,
//     const rocblas_half *B, rocblas_int ld_b,
//     const rocblas_half *beta,
//           rocblas_half *C, rocblas_int ld_c) {
//
//   CobaltDataType type_c     = cobaltDataTypeHalf;
//   CobaltDataType type_a     = cobaltDataTypeHalf;
//   CobaltDataType type_b     = cobaltDataTypeHalf;
//   CobaltDataType type_alpha = cobaltDataTypeHalf;
//   CobaltDataType type_beta  = cobaltDataTypeHalf;
//
//   rocblas_int ls_c = 1;
//   rocblas_int ls_a = 1;
//   rocblas_int ls_b = 1;
//
//   rocblas_int bs_c;
//   rocblas_int bs_a;
//   rocblas_int bs_b;
//
//   infer_batch_strides( order, transa, transb, m, n, k,
//     ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
//   rocblas_int batch_count = 1;
//
//   return xgemm_cobalt( handle, order, transa, transb,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, batch_count );
// }

rocblas_status rocblas_sgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int ld_a,
    const float *B, rocblas_int ld_b,
    const float *beta,
          float *C, rocblas_int ld_c) {

  CobaltDataType type_c     = cobaltDataTypeSingle;
  CobaltDataType type_a     = cobaltDataTypeSingle;
  CobaltDataType type_b     = cobaltDataTypeSingle;
  CobaltDataType type_alpha = cobaltDataTypeSingle;
  CobaltDataType type_beta  = cobaltDataTypeSingle;

  rocblas_int ls_c = 1;
  rocblas_int ls_a = 1;
  rocblas_int ls_b = 1;

  rocblas_int bs_c;
  rocblas_int bs_a;
  rocblas_int bs_b;

  infer_batch_strides( order, transa, transb, m, n, k,
    ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

rocblas_status rocblas_dgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int ld_a,
    const double *B, rocblas_int ld_b,
    const double *beta,
          double *C, rocblas_int ld_c) {

  CobaltDataType type_c     = cobaltDataTypeDouble;
  CobaltDataType type_a     = cobaltDataTypeDouble;
  CobaltDataType type_b     = cobaltDataTypeDouble;
  CobaltDataType type_alpha = cobaltDataTypeDouble;
  CobaltDataType type_beta  = cobaltDataTypeDouble;

  rocblas_int ls_c = 1;
  rocblas_int ls_a = 1;
  rocblas_int ls_b = 1;

  rocblas_int bs_c;
  rocblas_int bs_a;
  rocblas_int bs_b;

  infer_batch_strides( order, transa, transb, m, n, k,
    ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

// rocblas_status rocblas_qgemm(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation transa, rocblas_operation transb,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half_complex *alpha,
//     const rocblas_half_complex *A, rocblas_int ld_a,
//     const rocblas_half_complex *B, rocblas_int ld_b,
//     const rocblas_half_complex *beta,
//           rocblas_half_complex *C, rocblas_int ld_c) {

//   CobaltDataType type_c     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_a     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_b     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_alpha = cobaltDataTypeComplexHalf;
//   CobaltDataType type_beta  = cobaltDataTypeComplexHalf;

//   rocblas_int ls_c = 1;
//   rocblas_int ls_a = 1;
//   rocblas_int ls_b = 1;

//   rocblas_int bs_c;
//   rocblas_int bs_a;
//   rocblas_int bs_b;

//   infer_batch_strides( order, transa, transb, m, n, k,
//     ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
//   rocblas_int batch_count = 1;

//   return xgemm_cobalt( handle, order, transa, transb,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, batch_count );
// }

rocblas_status rocblas_cgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int ld_a,
    const rocblas_float_complex *B, rocblas_int ld_b,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ld_c) {

  CobaltDataType type_c     = cobaltDataTypeComplexSingle;
  CobaltDataType type_a     = cobaltDataTypeComplexSingle;
  CobaltDataType type_b     = cobaltDataTypeComplexSingle;
  CobaltDataType type_alpha = cobaltDataTypeComplexSingle;
  CobaltDataType type_beta  = cobaltDataTypeComplexSingle;

  rocblas_int ls_c = 1;
  rocblas_int ls_a = 1;
  rocblas_int ls_b = 1;

  rocblas_int bs_c;
  rocblas_int bs_a;
  rocblas_int bs_b;

  infer_batch_strides( order, transa, transb, m, n, k,
    ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

rocblas_status rocblas_zgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int ld_a,
    const rocblas_double_complex *B, rocblas_int ld_b,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ld_c) {

  CobaltDataType type_c     = cobaltDataTypeComplexDouble;
  CobaltDataType type_a     = cobaltDataTypeComplexDouble;
  CobaltDataType type_b     = cobaltDataTypeComplexDouble;
  CobaltDataType type_alpha = cobaltDataTypeComplexDouble;
  CobaltDataType type_beta  = cobaltDataTypeComplexDouble;

  rocblas_int ls_c = 1;
  rocblas_int ls_a = 1;
  rocblas_int ls_b = 1;

  rocblas_int bs_c;
  rocblas_int bs_a;
  rocblas_int bs_b;

  infer_batch_strides( order, transa, transb, m, n, k,
    ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}


    /***************************************************************************
     * strided - specify leading stride
     * ls_a - non-1 leading stride of a
     * ls_b - non-1 leading stride of b
     * ls_c - non-1 leading stride of c
     **************************************************************************/
// rocblas_status rocblas_hgemm_strided(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation transa, rocblas_operation transb,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half *alpha,
//     const rocblas_half *A, rocblas_int ls_a, rocblas_int ld_a,
//     const rocblas_half *B, rocblas_int ls_b, rocblas_int ld_b,
//     const rocblas_half *beta,
//           rocblas_half *C, rocblas_int ls_c, rocblas_int ld_c) {
//
//   CobaltDataType type_c     = cobaltDataTypeHalf;
//   CobaltDataType type_a     = cobaltDataTypeHalf;
//   CobaltDataType type_b     = cobaltDataTypeHalf;
//   CobaltDataType type_alpha = cobaltDataTypeHalf;
//   CobaltDataType type_beta  = cobaltDataTypeHalf;
//
//   rocblas_int bs_c;
//   rocblas_int bs_a;
//   rocblas_int bs_b;
//
//   infer_batch_strides( order, transa, transb, m, n, k,
//     ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
//   rocblas_int batch_count = 1;
//
//   return xgemm_cobalt( handle, order, transa, transb,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, batch_count );
// }

rocblas_status rocblas_sgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int ls_a, rocblas_int ld_a,
    const float *B, rocblas_int ls_b, rocblas_int ld_b,
    const float *beta,
          float *C, rocblas_int ls_c, rocblas_int ld_c) {

  CobaltDataType type_c     = cobaltDataTypeSingle;
  CobaltDataType type_a     = cobaltDataTypeSingle;
  CobaltDataType type_b     = cobaltDataTypeSingle;
  CobaltDataType type_alpha = cobaltDataTypeSingle;
  CobaltDataType type_beta  = cobaltDataTypeSingle;

  rocblas_int bs_c;
  rocblas_int bs_a;
  rocblas_int bs_b;

  infer_batch_strides( order, transa, transb, m, n, k,
    ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

rocblas_status rocblas_dgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int ls_a, rocblas_int ld_a,
    const double *B, rocblas_int ls_b, rocblas_int ld_b,
    const double *beta,
          double *C, rocblas_int ls_c, rocblas_int ld_c) {

  CobaltDataType type_c     = cobaltDataTypeDouble;
  CobaltDataType type_a     = cobaltDataTypeDouble;
  CobaltDataType type_b     = cobaltDataTypeDouble;
  CobaltDataType type_alpha = cobaltDataTypeDouble;
  CobaltDataType type_beta  = cobaltDataTypeDouble;

  rocblas_int bs_c;
  rocblas_int bs_a;
  rocblas_int bs_b;

  infer_batch_strides( order, transa, transb, m, n, k,
    ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

// rocblas_status rocblas_qgemm_strided(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation transa, rocblas_operation transb,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half_complex *alpha,
//     const rocblas_half_complex *A, rocblas_int ls_a, rocblas_int ld_a,
//     const rocblas_half_complex *B, rocblas_int ls_b, rocblas_int ld_b,
//     const rocblas_half_complex *beta,
//           rocblas_half_complex *C, rocblas_int ls_c, rocblas_int ld_c) {

//   CobaltDataType type_c     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_a     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_b     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_alpha = cobaltDataTypeComplexHalf;
//   CobaltDataType type_beta  = cobaltDataTypeComplexHalf;

//   rocblas_int bs_c;
//   rocblas_int bs_a;
//   rocblas_int bs_b;

//   infer_batch_strides( order, transa, transb, m, n, k,
//     ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
//   rocblas_int batch_count = 1;

//   return xgemm_cobalt( handle, order, transa, transb,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, batch_count );
// }

rocblas_status rocblas_cgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int ls_a, rocblas_int ld_a,
    const rocblas_float_complex *B, rocblas_int ls_b, rocblas_int ld_b,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ls_c, rocblas_int ld_c) {

  CobaltDataType type_c     = cobaltDataTypeComplexSingle;
  CobaltDataType type_a     = cobaltDataTypeComplexSingle;
  CobaltDataType type_b     = cobaltDataTypeComplexSingle;
  CobaltDataType type_alpha = cobaltDataTypeComplexSingle;
  CobaltDataType type_beta  = cobaltDataTypeComplexSingle;

  rocblas_int bs_c;
  rocblas_int bs_a;
  rocblas_int bs_b;

  infer_batch_strides( order, transa, transb, m, n, k,
    ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

rocblas_status rocblas_zgemm_strided(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int ls_a, rocblas_int ld_a,
    const rocblas_double_complex *B, rocblas_int ls_b, rocblas_int ld_b,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ls_c, rocblas_int ld_c) {

  CobaltDataType type_c     = cobaltDataTypeComplexDouble;
  CobaltDataType type_a     = cobaltDataTypeComplexDouble;
  CobaltDataType type_b     = cobaltDataTypeComplexDouble;
  CobaltDataType type_alpha = cobaltDataTypeComplexDouble;
  CobaltDataType type_beta  = cobaltDataTypeComplexDouble;

  rocblas_int bs_c;
  rocblas_int bs_a;
  rocblas_int bs_b;

  infer_batch_strides( order, transa, transb, m, n, k,
    ld_a, &bs_a, ld_b, &bs_b, ld_c, &bs_c );
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

    /***************************************************************************
     * batched
     * bs_a - "batch stride a": stride from the start of one "A" matrix to the next
     * bs_b
     * bs_c
     * batch_count - numbers of gemm's in the batch
     **************************************************************************/
// rocblas_status rocblas_hgemm_batched(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation transa, rocblas_operation transb,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half *alpha,
//     const rocblas_half *A, rocblas_int ld_a, rocblas_int bs_a,
//     const rocblas_half *B, rocblas_int ld_b, rocblas_int bs_b,
//     const rocblas_half *beta,
//           rocblas_half *C, rocblas_int ld_c, rocblas_int bs_c,
//     rocblas_int batch_count ) {
//
//   CobaltDataType type_c     = cobaltDataTypeHalf;
//   CobaltDataType type_a     = cobaltDataTypeHalf;
//   CobaltDataType type_b     = cobaltDataTypeHalf;
//   CobaltDataType type_alpha = cobaltDataTypeHalf;
//   CobaltDataType type_beta  = cobaltDataTypeHalf;
//
//   rocblas_int ls_c = 1;
//   rocblas_int ls_a = 1;
//   rocblas_int ls_b = 1;
//
//   return xgemm_cobalt( handle, order, transa, transb,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, batch_count );
// }

rocblas_status rocblas_sgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int ld_a, rocblas_int bs_a,
    const float *B, rocblas_int ld_b, rocblas_int bs_b,
    const float *beta,
          float *C, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeSingle;
  CobaltDataType type_a     = cobaltDataTypeSingle;
  CobaltDataType type_b     = cobaltDataTypeSingle;
  CobaltDataType type_alpha = cobaltDataTypeSingle;
  CobaltDataType type_beta  = cobaltDataTypeSingle;

  rocblas_int ls_c = 1;
  rocblas_int ls_a = 1;
  rocblas_int ls_b = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

rocblas_status rocblas_dgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int ld_a, rocblas_int bs_a,
    const double *B, rocblas_int ld_b, rocblas_int bs_b,
    const double *beta,
          double *C, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeDouble;
  CobaltDataType type_a     = cobaltDataTypeDouble;
  CobaltDataType type_b     = cobaltDataTypeDouble;
  CobaltDataType type_alpha = cobaltDataTypeDouble;
  CobaltDataType type_beta  = cobaltDataTypeDouble;

  rocblas_int ls_c = 1;
  rocblas_int ls_a = 1;
  rocblas_int ls_b = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

// rocblas_status rocblas_qgemm_batched(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation transa, rocblas_operation transb,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half_complex *alpha,
//     const rocblas_half_complex *A, rocblas_int ld_a, rocblas_int bs_a,
//     const rocblas_half_complex *B, rocblas_int ld_b, rocblas_int bs_b,
//     const rocblas_half_complex *beta,
//           rocblas_half_complex *C, rocblas_int ld_c, rocblas_int bs_c,
//     rocblas_int batch_count ) {

//   CobaltDataType type_c     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_a     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_b     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_alpha = cobaltDataTypeComplexHalf;
//   CobaltDataType type_beta  = cobaltDataTypeComplexHalf;

//   rocblas_int ls_c = 1;
//   rocblas_int ls_a = 1;
//   rocblas_int ls_b = 1;

//   return xgemm_cobalt( handle, order, transa, transb,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, batch_count );
// }

rocblas_status rocblas_cgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int ld_a, rocblas_int bs_a,
    const rocblas_float_complex *B, rocblas_int ld_b, rocblas_int bs_b,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexSingle;
  CobaltDataType type_a     = cobaltDataTypeComplexSingle;
  CobaltDataType type_b     = cobaltDataTypeComplexSingle;
  CobaltDataType type_alpha = cobaltDataTypeComplexSingle;
  CobaltDataType type_beta  = cobaltDataTypeComplexSingle;

  rocblas_int ls_c = 1;
  rocblas_int ls_a = 1;
  rocblas_int ls_b = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

rocblas_status rocblas_zgemm_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int ld_a, rocblas_int bs_a,
    const rocblas_double_complex *B, rocblas_int ld_b, rocblas_int bs_b,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexDouble;
  CobaltDataType type_a     = cobaltDataTypeComplexDouble;
  CobaltDataType type_b     = cobaltDataTypeComplexDouble;
  CobaltDataType type_alpha = cobaltDataTypeComplexDouble;
  CobaltDataType type_beta  = cobaltDataTypeComplexDouble;

  rocblas_int ls_c = 1;
  rocblas_int ls_a = 1;
  rocblas_int ls_b = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}


    /***************************************************************************
     * strided & batched
     * ls_a - non-1 leading stride of a
     * ls_b - non-1 leading stride of b
     * ls_c - non-1 leading stride of c
     * bs_a - "batch stride a": stride from the start of one "A" matrix to the next
     * bs_b
     * bs_c
     * batch_count - numbers of gemm's in the batch
     **************************************************************************/
// rocblas_status rocblas_hgemm_strided_batched(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation transa, rocblas_operation transb,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half *alpha,
//     const rocblas_half *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
//     const rocblas_half *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
//     const rocblas_half *beta,
//           rocblas_half *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
//     rocblas_int batch_count ) {
//
//   CobaltDataType type_c     = cobaltDataTypeHalf;
//   CobaltDataType type_a     = cobaltDataTypeHalf;
//   CobaltDataType type_b     = cobaltDataTypeHalf;
//   CobaltDataType type_alpha = cobaltDataTypeHalf;
//   CobaltDataType type_beta  = cobaltDataTypeHalf;
//
//   return xgemm_cobalt( handle, order, transa, transb,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, batch_count );
// }

rocblas_status rocblas_sgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const float *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const float *beta,
          float *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeSingle;
  CobaltDataType type_a     = cobaltDataTypeSingle;
  CobaltDataType type_b     = cobaltDataTypeSingle;
  CobaltDataType type_alpha = cobaltDataTypeSingle;
  CobaltDataType type_beta  = cobaltDataTypeSingle;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

rocblas_status rocblas_dgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const double *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const double *beta,
          double *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeDouble;
  CobaltDataType type_a     = cobaltDataTypeDouble;
  CobaltDataType type_b     = cobaltDataTypeDouble;
  CobaltDataType type_alpha = cobaltDataTypeDouble;
  CobaltDataType type_beta  = cobaltDataTypeDouble;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

// rocblas_status rocblas_qgemm_strided_batched(
//     rocblas_handle handle,
//     rocblas_order order,
//     rocblas_operation transa, rocblas_operation transb,
//     rocblas_int m, rocblas_int n, rocblas_int k,
//     const rocblas_half_complex *alpha,
//     const rocblas_half_complex *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
//     const rocblas_half_complex *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
//     const rocblas_half_complex *beta,
//           rocblas_half_complex *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
//     rocblas_int batch_count ) {

//   CobaltDataType type_c     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_a     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_b     = cobaltDataTypeComplexHalf;
//   CobaltDataType type_alpha = cobaltDataTypeComplexHalf;
//   CobaltDataType type_beta  = cobaltDataTypeComplexHalf;

//   return xgemm_cobalt( handle, order, transa, transb,
//       m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
//       type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
//       type_c, C, ls_c, ld_c, bs_c, batch_count );
// }

rocblas_status rocblas_cgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const rocblas_float_complex *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexSingle;
  CobaltDataType type_a     = cobaltDataTypeComplexSingle;
  CobaltDataType type_b     = cobaltDataTypeComplexSingle;
  CobaltDataType type_alpha = cobaltDataTypeComplexSingle;
  CobaltDataType type_beta  = cobaltDataTypeComplexSingle;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}

rocblas_status rocblas_zgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const rocblas_double_complex *B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexDouble;
  CobaltDataType type_a     = cobaltDataTypeComplexDouble;
  CobaltDataType type_b     = cobaltDataTypeComplexDouble;
  CobaltDataType type_alpha = cobaltDataTypeComplexDouble;
  CobaltDataType type_beta  = cobaltDataTypeComplexDouble;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, ls_a, ld_a, bs_a,
      type_b, B, ls_b, ld_b, bs_b, type_beta, beta,
      type_c, C, ls_c, ld_c, bs_c, batch_count );
}


/*******************************************************************************
 * Helper Functions
 ******************************************************************************/
CobaltDataType conjugate_if_necessary( CobaltDataType type, rocblas_operation trans ) {
  if ( trans == rocblas_operation_conjugate_transpose ) {
    switch ( type ) {
    // case cobaltDataTypeComplexHalf:
    //   return cobaltDataTypeComplexConjugateHalf;
    case cobaltDataTypeComplexSingle:
      return cobaltDataTypeComplexConjugateSingle;
    case cobaltDataTypeComplexDouble:
      return cobaltDataTypeComplexConjugateDouble;
    default:
      // code should never come here, so create an error
      return cobaltDataTypeNone;
    }
  } else {
    // not conjugate transposing
    return type;
  }
}

/*******************************************************************************
 * Infer Batch Strides
 ******************************************************************************/
void infer_batch_strides(
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    rocblas_int ld_a, rocblas_int *bs_a,
    rocblas_int ld_b, rocblas_int *bs_b,
    rocblas_int ld_c, rocblas_int *bs_c ) {

  int num_cols_a = (transa == rocblas_operation_none ? k : m);
  int num_rows_a = (transa == rocblas_operation_none ? m : k);
  int num_cols_b = (transb == rocblas_operation_none ? n : k);
  int num_rows_b = (transb == rocblas_operation_none ? k : n);
  int num_cols_c = m;
  int num_rows_c = n;

  int dim1_size_a = (order==rocblas_order_column_major) ? num_cols_a : num_rows_a;
  int dim1_size_b = (order==rocblas_order_column_major) ? num_cols_b : num_rows_b;
  int dim1_size_c = (order==rocblas_order_column_major) ? num_cols_c : num_rows_c;

  *bs_a = ld_a * dim1_size_a;
  *bs_b = ld_b * dim1_size_b;
  *bs_c = ld_c * dim1_size_c;
}
