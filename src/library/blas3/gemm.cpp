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
    rocblas_transpose transA,
    rocblas_transpose transB,
    rocblas_int M,
    rocblas_int N,
    rocblas_int K,
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
  if ( M < 1 || N < 1 || K < 1 ) {
    return rocblas_status_invalid_parameter;
  }

  // handle must be valid
  if (handle == nullptr) {
    return rocblas_status_invalid_parameter;
  }

  // sgemm
  CobaltDataType dataType = cobaltDataTypeSingle;
  CobaltDataType alphaType = dataType;
  CobaltDataType betaType = dataType;
  bool useOffsets = false;

  // unbatched
  unsigned int sizeBatch = 1;

  // initialStride1
  int initialStrideC = 1;
  int initialStrideA = 1;
  int initialStrideB = 1;

  // matrix dimensions
  int numColsC = M;
  int numRowsC = N;
  int numColsA = transA != rocblas_transpose_none ? K : M;
  int numRowsA = transA != rocblas_transpose_none ? M : K;
  int numColsB = transB != rocblas_transpose_none ? N : K;
  int numRowsB = transB != rocblas_transpose_none ? K : N;
  unsigned int matrixStrideC = M*ldc;
  unsigned int matrixStrideA = numColsA*lda;
  unsigned int matrixStrideB = numColsB*ldb;

  // ensure strides positive
  if ( ldc < numColsC || lda < numColsA || ldb < numColsB ) {
    return rocblas_status_invalid_parameter;
  }

  // create tensors
  CobaltTensor tensorC;
  CobaltTensor tensorA;
  CobaltTensor tensorB;
  initializeTensorForGEMM(
      tensorC,
      dataType,
      initialStrideC, // row stride
      numColsC,       // row size (num cols)
      ldc,            // col stride
      numRowsC,       // col size (num rows)
      matrixStrideC,  // matrix stride
      sizeBatch );    // num matrices
  initializeTensorForGEMM(
      tensorA,
      dataType,
      initialStrideA, // row stride
      numColsA,       // row size (num cols)
      lda,            // col stride
      numRowsA,       // col size (num rows)
      matrixStrideA,  // matrix stride
      sizeBatch );    // num matrices
  initializeTensorForGEMM(
      tensorB,
      dataType,
      initialStrideB, // row stride
      numColsB,       // row size (num cols)
      ldb,            // col stride
      numRowsB,       // col size (num rows)
      matrixStrideB,  // matrix stride
      sizeBatch );    // num matrices

  // index assignments
  unsigned int indexAssignmentsA[3];
  unsigned int indexAssignmentsB[3];
  if ( sizeBatch > 1) {
    indexAssignmentsA[0] = transA != rocblas_transpose_none ? 3 : 0;
    indexAssignmentsA[1] = transA != rocblas_transpose_none ? 0 : 3;
    indexAssignmentsA[2] = 2;
    indexAssignmentsB[0] = transB != rocblas_transpose_none ? 1 : 3;
    indexAssignmentsB[1] = transB != rocblas_transpose_none ? 3 : 1;
    indexAssignmentsB[2] = 2;
  } else {
    indexAssignmentsA[0] = transA != rocblas_transpose_none ? 2 : 0;
    indexAssignmentsA[1] = transA != rocblas_transpose_none ? 0 : 2;
    indexAssignmentsB[0] = transB != rocblas_transpose_none ? 1 : 2;
    indexAssignmentsB[1] = transB != rocblas_transpose_none ? 2 : 1;
  }

  // create problem
  CobaltStatus status;
  CobaltProblem problem = cobaltCreateProblem(
      tensorC,
      tensorA,
      tensorB,
      indexAssignmentsA,
      indexAssignmentsB,
      cobaltOperationTypeContraction,
      alphaType,
      betaType,
      useOffsets,
      handle->cobaltDeviceProfile,
      &status );
  cobaltStatusCheck(status);
  if (status != cobaltStatusSuccess) {
    return rocblas_status_invalid_parameter;
  }

#ifdef _DEBUG
  // validate that problem was created correctly
  CobaltStatus validationStatus = cobaltValidateProblem( problem );
  cobaltStatusCheck(validationStatus);
  if (validationStatus != cobaltStatusSuccess) {
    return rocblas_status_invalid_parameter;
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
    return rocblas_status_invalid_parameter;
  }

  // pointers
  CobaltTensorData tensorDataC{ c, 0 };
  CobaltTensorDataConst tensorDataA{ a, 0 };
  CobaltTensorDataConst tensorDataB{ b, 0 };
  CobaltScalarData scalarDataAlpha{ alpha };
  CobaltScalarData scalarDataBeta{ beta };

  // enqueue solution
  cobaltEnqueueSolution(
      solution,
      tensorDataC,
      tensorDataA,
      tensorDataB,
      scalarDataAlpha,
      scalarDataBeta,
      &handle->cobaltControl );

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



void initializeTensorForGEMM(
    CobaltTensor & tensor,
    CobaltDataType dataType,
    int stride0,      // stride from one row to another
    int size0,        // num columns
    int stride1,      // stride from one column to next column
    int size1,        // num rows
    unsigned int strideBatch, // stride from one matrix to another
    int sizeBatch     // batch size (num matrices)
) {
  tensor.dataType = dataType;
  tensor.numDimensions = (sizeBatch > 1) ? 3 : 2;
  tensor.dimensions[0].stride = static_cast<unsigned int>(stride0);
  tensor.dimensions[0].size   = static_cast<unsigned int>(size0);
  tensor.dimensions[1].stride = static_cast<unsigned int>(stride1); 
  tensor.dimensions[1].size   = static_cast<unsigned int>(size1);
  tensor.dimensions[2].stride = static_cast<unsigned int>(strideBatch);
  tensor.dimensions[2].size   = static_cast<unsigned int>(sizeBatch);
}   

