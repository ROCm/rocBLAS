/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "definitions.h"
#include "status.h"
#include "rocblas.h"
#include "handle.h"
#include "hip_runtime.h"
#include "Cobalt.h"
#include "gemm.h"

// TODO - if trans and complex, make cobaltDataType Conjugate
rocblas_status
xgemm_cobalt(
  rocblas_handle handle,
  rocblas_order order,
  rocblas_operation trans_a, rocblas_operation trans_b,
  rocblas_int m, rocblas_int n, rocblas_int k,
  CobaltDataType type_alpha, const void *alpha,
  CobaltDataType     type_a, const void *a,     rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
  CobaltDataType     type_b, const void *b,     rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
  CobaltDataType  type_beta, const void *beta,
  CobaltDataType     type_c,       void *c,     rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
  rocblas_int batch_count ) {
  // ensure sizes positive
  if ( m < 0 || n < 0 || k < 0 ) {
    return rocblas_status_invalid_size;
  }

  // C already correct
  if ( m == 0 || n == 0 ) {
    return rocblas_status_success;
  }

  // handle must be valid
  if (handle == nullptr) {
    return rocblas_status_invalid_size;
  }

  bool use_offsets = false;

  // TODO - handle order parameter
  // matrix dimensions
  int num_cols_c = m;
  int num_rows_c = n;
  int num_cols_a = trans_a != rocblas_operation_none ? k : m;
  int num_rows_a = trans_a != rocblas_operation_none ? m : k;
  int num_cols_b = trans_b != rocblas_operation_none ? n : k;
  int num_rows_b = trans_b != rocblas_operation_none ? k : n;
  unsigned int matrix_stride_c = m*ldc;
  unsigned int matrix_stride_a = num_cols_a*lda;
  unsigned int matrix_stride_b = num_cols_b*ldb;

  // ensure strides positive
  if ( ldc < num_cols_c || lda < num_cols_a || ldb < num_cols_b ) {
    return rocblas_status_invalid_size;
  }

  // create tensors
  CobaltTensor tensor_c;
  CobaltTensor tensor_a;
  CobaltTensor tensor_b;
  initialize_tensor_for_gemm(
      tensor_c,
      type_c,
      lsc,              // row stride
      num_cols_c,       // row size (num cols)
      ldc,              // col stride
      num_rows_c,       // col size (num rows)
      matrix_stride_c,  // matrix stride
      batch_count );    // num matrices
  initialize_tensor_for_gemm(
      tensor_a,
      type_a,
      lsa,              // row stride
      num_cols_a,       // row size (num cols)
      lda,              // col stride
      num_rows_a,       // col size (num rows)
      matrix_stride_a,  // matrix stride
      batch_count );    // num matrices
  initialize_tensor_for_gemm(
      tensor_b,
      type_b,
      lsb,              // row stride
      num_cols_b,       // row size (num cols)
      ldb,              // col stride
      num_rows_b,       // col size (num rows)
      matrix_stride_b,  // matrix stride
      batch_count );    // num matrices

  // index assignments
  unsigned int index_assignments_a[3];
  unsigned int index_assignments_b[3];
  if ( batch_count > 1) {
    index_assignments_a[0] = trans_a != rocblas_operation_none ? 3 : 0;
    index_assignments_a[1] = trans_a != rocblas_operation_none ? 0 : 3;
    index_assignments_a[2] = 2;
    index_assignments_b[0] = trans_b != rocblas_operation_none ? 1 : 3;
    index_assignments_b[1] = trans_b != rocblas_operation_none ? 3 : 1;
    index_assignments_b[2] = 2;
  } else {
    index_assignments_a[0] = trans_a != rocblas_operation_none ? 2 : 0;
    index_assignments_a[1] = trans_a != rocblas_operation_none ? 0 : 2;
    index_assignments_b[0] = trans_b != rocblas_operation_none ? 1 : 2;
    index_assignments_b[1] = trans_b != rocblas_operation_none ? 2 : 1;
  }

  // create problem
  CobaltStatus status;
  CobaltProblem problem = cobaltCreateProblem(
      tensor_c,
      tensor_a,
      tensor_b,
      index_assignments_a,
      index_assignments_b,
      cobaltOperationTypeContraction,
      type_alpha,
      type_beta,
      use_offsets,
      handle->cobalt_device_profile,
      &status );
  cobaltStatusCheck(status);
  if (status != cobaltStatusSuccess) {
    return rocblas_status_invalid_size;
  }

#ifdef _DEBUG
  // validate that problem was created correctly
  CobaltStatus validationStatus = cobaltValidateProblem( problem );
  cobaltStatusCheck(validationStatus);
  if (validationStatus != cobaltStatusSuccess) {
    return rocblas_status_invalid_size;
  }
#endif

  // lookup solution
  CobaltSolution solution = cobaltGetSolutionForProblem( problem, &status );
  cobaltStatusCheck(status);
  if (status != cobaltStatusSuccess) {
    return rocblas_status_internal_error;
  }

  // ensure pointers are valid
  if ( c == nullptr
      || a == nullptr
      || b == nullptr
      || alpha == nullptr
      || beta == nullptr ) {
    return rocblas_status_invalid_pointer;
  }

  // pointers
  CobaltTensorData      tensor_data_c{ c, 0 };
  CobaltTensorDataConst tensor_data_a{ a, 0 };
  CobaltTensorDataConst tensor_data_b{ b, 0 };
  CobaltScalarData      scalar_data_alpha{ alpha };
  CobaltScalarData      scalar_data_beta{ beta };

  // enqueue solution
  cobaltEnqueueSolution(
      solution,
      tensor_data_c,
      tensor_data_a,
      tensor_data_b,
      scalar_data_alpha,
      scalar_data_beta,
      &handle->cobalt_control );

  // problem cleanup
  status = cobaltDestroyProblem( problem );
  cobaltStatusCheck(status);
  RETURN_IF_COBALT_ERROR(status);

  // solution cleanup
  status = cobaltDestroySolution( solution );
  cobaltStatusCheck(status);
  RETURN_IF_COBALT_ERROR(status);

  // TODO put events into handle, if necessary

  // success
  return rocblas_status_success;
}



void initialize_tensor_for_gemm(
    CobaltTensor & tensor,
    CobaltDataType data_type,
    int stride0,      // stride from one row to another
    int size0,        // num columns
    int stride1,      // stride from one column to next column
    int size1,        // num rows
    unsigned int stride_batch, // stride from one matrix to another
    int batch_count // batch size (num matrices)
) {
  tensor.dataType = data_type;
  tensor.numDimensions = (batch_count > 1) ? 3 : 2;
  tensor.dimensions[0].stride = static_cast<unsigned int>(stride0);
  tensor.dimensions[0].size   = static_cast<unsigned int>(size0);
  tensor.dimensions[1].stride = static_cast<unsigned int>(stride1); 
  tensor.dimensions[1].size   = static_cast<unsigned int>(size1);
  tensor.dimensions[2].stride = static_cast<unsigned int>(stride_batch);
  tensor.dimensions[2].size   = static_cast<unsigned int>(batch_count);
}   





rocblas_status rocblas_hgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half *alpha,
    const rocblas_half *A, rocblas_int lda,
    const rocblas_half *B, rocblas_int ldb,
    const rocblas_half *beta,
          rocblas_half *C, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeHalf;
  CobaltDataType type_a     = cobaltDataTypeHalf;
  CobaltDataType type_b     = cobaltDataTypeHalf;
  CobaltDataType type_alpha = cobaltDataTypeHalf;
  CobaltDataType type_beta  = cobaltDataTypeHalf;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_sgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int lda,
    const float *B, rocblas_int ldb,
    const float *beta,
          float *C, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeSingle;
  CobaltDataType type_a     = cobaltDataTypeSingle;
  CobaltDataType type_b     = cobaltDataTypeSingle;
  CobaltDataType type_alpha = cobaltDataTypeSingle;
  CobaltDataType type_beta  = cobaltDataTypeSingle;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_dgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int lda,
    const double *B, rocblas_int ldb,
    const double *beta,
          double *C, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeDouble;
  CobaltDataType type_a     = cobaltDataTypeDouble;
  CobaltDataType type_b     = cobaltDataTypeDouble;
  CobaltDataType type_alpha = cobaltDataTypeDouble;
  CobaltDataType type_beta  = cobaltDataTypeDouble;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_qgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lda,
    const rocblas_half_complex *B, rocblas_int ldb,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeComplexHalf;
  CobaltDataType type_a     = cobaltDataTypeComplexHalf;
  CobaltDataType type_b     = cobaltDataTypeComplexHalf;
  CobaltDataType type_alpha = cobaltDataTypeComplexHalf;
  CobaltDataType type_beta  = cobaltDataTypeComplexHalf;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_cgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda,
    const rocblas_float_complex *B, rocblas_int ldb,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeComplexSingle;
  CobaltDataType type_a     = cobaltDataTypeComplexSingle;
  CobaltDataType type_b     = cobaltDataTypeComplexSingle;
  CobaltDataType type_alpha = cobaltDataTypeComplexSingle;
  CobaltDataType type_beta  = cobaltDataTypeComplexSingle;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_zgemm(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda,
    const rocblas_double_complex *B, rocblas_int ldb,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeComplexDouble;
  CobaltDataType type_a     = cobaltDataTypeComplexDouble;
  CobaltDataType type_b     = cobaltDataTypeComplexDouble;
  CobaltDataType type_alpha = cobaltDataTypeComplexDouble;
  CobaltDataType type_beta  = cobaltDataTypeComplexDouble;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}


    /***************************************************************************
     * strided - specify leading stride
     * lsa - non-1 leading stride of a
     * lsb - non-1 leading stride of b
     * lsc - non-1 leading stride of c
     **************************************************************************/
rocblas_status rocblas_hgemm_s(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half *alpha,
    const rocblas_half *A, rocblas_int lsa, rocblas_int lda,
    const rocblas_half *B, rocblas_int lsb, rocblas_int ldb,
    const rocblas_half *beta,
          rocblas_half *C, rocblas_int lsc, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeHalf;
  CobaltDataType type_a     = cobaltDataTypeHalf;
  CobaltDataType type_b     = cobaltDataTypeHalf;
  CobaltDataType type_alpha = cobaltDataTypeHalf;
  CobaltDataType type_beta  = cobaltDataTypeHalf;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_sgemm_s(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int lsa, rocblas_int lda,
    const float *B, rocblas_int lsb, rocblas_int ldb,
    const float *beta,
          float *C, rocblas_int lsc, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeSingle;
  CobaltDataType type_a     = cobaltDataTypeSingle;
  CobaltDataType type_b     = cobaltDataTypeSingle;
  CobaltDataType type_alpha = cobaltDataTypeSingle;
  CobaltDataType type_beta  = cobaltDataTypeSingle;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_dgemm_s(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int lsa, rocblas_int lda,
    const double *B, rocblas_int lsb, rocblas_int ldb,
    const double *beta,
          double *C, rocblas_int lsc, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeDouble;
  CobaltDataType type_a     = cobaltDataTypeDouble;
  CobaltDataType type_b     = cobaltDataTypeDouble;
  CobaltDataType type_alpha = cobaltDataTypeDouble;
  CobaltDataType type_beta  = cobaltDataTypeDouble;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_qgemm_s(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lsa, rocblas_int lda,
    const rocblas_half_complex *B, rocblas_int lsb, rocblas_int ldb,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int lsc, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeComplexHalf;
  CobaltDataType type_a     = cobaltDataTypeComplexHalf;
  CobaltDataType type_b     = cobaltDataTypeComplexHalf;
  CobaltDataType type_alpha = cobaltDataTypeComplexHalf;
  CobaltDataType type_beta  = cobaltDataTypeComplexHalf;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_cgemm_s(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lsa, rocblas_int lda,
    const rocblas_float_complex *B, rocblas_int lsb, rocblas_int ldb,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int lsc, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeComplexSingle;
  CobaltDataType type_a     = cobaltDataTypeComplexSingle;
  CobaltDataType type_b     = cobaltDataTypeComplexSingle;
  CobaltDataType type_alpha = cobaltDataTypeComplexSingle;
  CobaltDataType type_beta  = cobaltDataTypeComplexSingle;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_zgemm_s(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lsa, rocblas_int lda,
    const rocblas_double_complex *B, rocblas_int lsb, rocblas_int ldb,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int lsc, rocblas_int ldc) {

  CobaltDataType type_c     = cobaltDataTypeComplexDouble;
  CobaltDataType type_a     = cobaltDataTypeComplexDouble;
  CobaltDataType type_b     = cobaltDataTypeComplexDouble;
  CobaltDataType type_alpha = cobaltDataTypeComplexDouble;
  CobaltDataType type_beta  = cobaltDataTypeComplexDouble;

  rocblas_int bsc = 1;
  rocblas_int bsa = 1;
  rocblas_int bsb = 1;
  rocblas_int batch_count = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

    /***************************************************************************
     * batched
     * bsa - "batch stride a": stride from the start of one "A" matrix to the next
     * bsb
     * bsc
     * batch_count - numbers of gemm's in the batch
     **************************************************************************/
rocblas_status rocblas_hgemm_b(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half *alpha,
    const rocblas_half *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_half *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_half *beta,
          rocblas_half *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeHalf;
  CobaltDataType type_a     = cobaltDataTypeHalf;
  CobaltDataType type_b     = cobaltDataTypeHalf;
  CobaltDataType type_alpha = cobaltDataTypeHalf;
  CobaltDataType type_beta  = cobaltDataTypeHalf;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_sgemm_b(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int lda, rocblas_int bsa,
    const float *B, rocblas_int ldb, rocblas_int bsb,
    const float *beta,
          float *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeSingle;
  CobaltDataType type_a     = cobaltDataTypeSingle;
  CobaltDataType type_b     = cobaltDataTypeSingle;
  CobaltDataType type_alpha = cobaltDataTypeSingle;
  CobaltDataType type_beta  = cobaltDataTypeSingle;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_dgemm_b(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int lda, rocblas_int bsa,
    const double *B, rocblas_int ldb, rocblas_int bsb,
    const double *beta,
          double *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeDouble;
  CobaltDataType type_a     = cobaltDataTypeDouble;
  CobaltDataType type_b     = cobaltDataTypeDouble;
  CobaltDataType type_alpha = cobaltDataTypeDouble;
  CobaltDataType type_beta  = cobaltDataTypeDouble;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_qgemm_b(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_half_complex *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexHalf;
  CobaltDataType type_a     = cobaltDataTypeComplexHalf;
  CobaltDataType type_b     = cobaltDataTypeComplexHalf;
  CobaltDataType type_alpha = cobaltDataTypeComplexHalf;
  CobaltDataType type_beta  = cobaltDataTypeComplexHalf;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_cgemm_b(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_float_complex *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexSingle;
  CobaltDataType type_a     = cobaltDataTypeComplexSingle;
  CobaltDataType type_b     = cobaltDataTypeComplexSingle;
  CobaltDataType type_alpha = cobaltDataTypeComplexSingle;
  CobaltDataType type_beta  = cobaltDataTypeComplexSingle;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

  CobaltDataType type_c     = cobaltDataTypeHalf;
  CobaltDataType type_a     = cobaltDataTypeHalf;
  CobaltDataType type_b     = cobaltDataTypeHalf;
  CobaltDataType type_alpha = cobaltDataTypeHalf;
  CobaltDataType type_beta  = cobaltDataTypeHalf;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

rocblas_status rocblas_zgemm_b(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lda, rocblas_int bsa,
    const rocblas_double_complex *B, rocblas_int ldb, rocblas_int bsb,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexDouble;
  CobaltDataType type_a     = cobaltDataTypeComplexDouble;
  CobaltDataType type_b     = cobaltDataTypeComplexDouble;
  CobaltDataType type_alpha = cobaltDataTypeComplexDouble;
  CobaltDataType type_beta  = cobaltDataTypeComplexDouble;

  rocblas_int lsc = 1;
  rocblas_int lsa = 1;
  rocblas_int lsb = 1;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}


    /***************************************************************************
     * strided & batched
     * lsa - non-1 leading stride of a
     * lsb - non-1 leading stride of b
     * lsc - non-1 leading stride of c
     * bsa - "batch stride a": stride from the start of one "A" matrix to the next
     * bsb
     * bsc
     * batch_count - numbers of gemm's in the batch
     **************************************************************************/
rocblas_status rocblas_hgemm_sb(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half *alpha,
    const rocblas_half *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const rocblas_half *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const rocblas_half *beta,
          rocblas_half *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeHalf;
  CobaltDataType type_a     = cobaltDataTypeHalf;
  CobaltDataType type_b     = cobaltDataTypeHalf;
  CobaltDataType type_alpha = cobaltDataTypeHalf;
  CobaltDataType type_beta  = cobaltDataTypeHalf;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_sgemm_sb(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float *alpha,
    const float *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const float *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const float *beta,
          float *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeSingle;
  CobaltDataType type_a     = cobaltDataTypeSingle;
  CobaltDataType type_b     = cobaltDataTypeSingle;
  CobaltDataType type_alpha = cobaltDataTypeSingle;
  CobaltDataType type_beta  = cobaltDataTypeSingle;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_dgemm_sb(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const double *alpha,
    const double *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const double *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const double *beta,
          double *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeDouble;
  CobaltDataType type_a     = cobaltDataTypeDouble;
  CobaltDataType type_b     = cobaltDataTypeDouble;
  CobaltDataType type_alpha = cobaltDataTypeDouble;
  CobaltDataType type_beta  = cobaltDataTypeDouble;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_qgemm_sb(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_half_complex *alpha,
    const rocblas_half_complex *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const rocblas_half_complex *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const rocblas_half_complex *beta,
          rocblas_half_complex *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexHalf;
  CobaltDataType type_a     = cobaltDataTypeComplexHalf;
  CobaltDataType type_b     = cobaltDataTypeComplexHalf;
  CobaltDataType type_alpha = cobaltDataTypeComplexHalf;
  CobaltDataType type_beta  = cobaltDataTypeComplexHalf;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_cgemm_sb(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_float_complex *alpha,
    const rocblas_float_complex *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const rocblas_float_complex *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const rocblas_float_complex *beta,
          rocblas_float_complex *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexSingle;
  CobaltDataType type_a     = cobaltDataTypeComplexSingle;
  CobaltDataType type_b     = cobaltDataTypeComplexSingle;
  CobaltDataType type_alpha = cobaltDataTypeComplexSingle;
  CobaltDataType type_beta  = cobaltDataTypeComplexSingle;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}

rocblas_status rocblas_zgemm_sb(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const rocblas_double_complex *alpha,
    const rocblas_double_complex *A, rocblas_int lsa, rocblas_int lda, rocblas_int bsa,
    const rocblas_double_complex *B, rocblas_int lsb, rocblas_int ldb, rocblas_int bsb,
    const rocblas_double_complex *beta,
          rocblas_double_complex *C, rocblas_int lsc, rocblas_int ldc, rocblas_int bsc,
    rocblas_int batch_count ) {

  CobaltDataType type_c     = cobaltDataTypeComplexDouble;
  CobaltDataType type_a     = cobaltDataTypeComplexDouble;
  CobaltDataType type_b     = cobaltDataTypeComplexDouble;
  CobaltDataType type_alpha = cobaltDataTypeComplexDouble;
  CobaltDataType type_beta  = cobaltDataTypeComplexDouble;

  return xgemm_cobalt( handle, order, transa, transb,
      m, n, k, type_alpha, alpha, type_a, A, lsa, lda, bsa,
      type_b, B, lsb, ldb, bsb, type_beta, beta,
      type_c, C, lsc, ldc, bsc, batch_count );
}


