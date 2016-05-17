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

// rocblas_gemm
// rocblas_gemm_leadingstride
// rocblas_gemm_batched
// rocblas_gemm_leadingstride_batched

// only accept column-major input
rocblas_status
rocblas_sgemm(
    rocblas_handle handle,
    rocblas_operation trans_a,
    rocblas_operation trans_b,
    rocblas_int m,
    rocblas_int n,
    rocblas_int k,
    const float *alpha,
    const float *a,
    rocblas_int lda,
    const float *b,
    rocblas_int ldb,
    const float *beta,
          float *c,
    rocblas_int ldc )
{
  // ensure sizes positive
  if ( m < 0 || n < 0 || k < 0 ) {
    return rocblas_status_invalid_value;
  }

  // C already correct
  if ( m == 0 || n == 0 ) {
    return rocblas_status_success;
  }
  if ( *beta == 1.f && ( *alpha == 0.f || k == 0) ) {
    return rocblas_status_success;
  }

  // handle must be valid
  if (handle == nullptr) {
    return rocblas_status_invalid_value;
  }

  // sgemm
  CobaltDataType data_type = cobaltDataTypeSingle;
  CobaltDataType alpha_type = data_type;
  CobaltDataType beta_type = data_type;
  bool useOffsets = false;

  // unbatched
  unsigned int size_batch = 1;

  // initialStride1
  int initial_stride_c = 1;
  int initial_stride_a = 1;
  int initial_stride_b = 1;

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
    return rocblas_status_invalid_value;
  }

  // create tensors
  CobaltTensor tensor_c;
  CobaltTensor tensor_a;
  CobaltTensor tensor_b;
  initialize_tensor_for_gemm(
      tensor_c,
      data_type,
      initial_stride_c, // row stride
      num_cols_c,       // row size (num cols)
      ldc,            // col stride
      num_rows_c,       // col size (num rows)
      matrix_stride_c,  // matrix stride
      size_batch );    // num matrices
  initialize_tensor_for_gemm(
      tensor_a,
      data_type,
      initial_stride_a, // row stride
      num_cols_a,       // row size (num cols)
      lda,            // col stride
      num_rows_a,       // col size (num rows)
      matrix_stride_a,  // matrix stride
      size_batch );    // num matrices
  initialize_tensor_for_gemm(
      tensor_b,
      data_type,
      initial_stride_b, // row stride
      num_cols_b,       // row size (num cols)
      ldb,            // col stride
      num_rows_b,       // col size (num rows)
      matrix_stride_b,  // matrix stride
      size_batch );    // num matrices

  // index assignments
  unsigned int index_assignments_a[3];
  unsigned int index_assignments_b[3];
  if ( size_batch > 1) {
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
      alpha_type,
      beta_type,
      useOffsets,
      handle->cobalt_device_profile,
      &status );
  cobaltStatusCheck(status);
  if (status != cobaltStatusSuccess) {
    return rocblas_status_invalid_value;
  }

#ifdef _DEBUG
  // validate that problem was created correctly
  CobaltStatus validationStatus = cobaltValidateProblem( problem );
  cobaltStatusCheck(validationStatus);
  if (validationStatus != cobaltStatusSuccess) {
    return rocblas_status_invalid_value;
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
    return rocblas_status_invalid_value;
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
    int size_batch     // batch size (num matrices)
) {
  tensor.dataType = data_type;
  tensor.numDimensions = (size_batch > 1) ? 3 : 2;
  tensor.dimensions[0].stride = static_cast<unsigned int>(stride0);
  tensor.dimensions[0].size   = static_cast<unsigned int>(size0);
  tensor.dimensions[1].stride = static_cast<unsigned int>(stride1); 
  tensor.dimensions[1].size   = static_cast<unsigned int>(size1);
  tensor.dimensions[2].stride = static_cast<unsigned int>(stride_batch);
  tensor.dimensions[2].size   = static_cast<unsigned int>(size_batch);
}   

