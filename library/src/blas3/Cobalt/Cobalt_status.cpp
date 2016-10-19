
#include "status.h"
#include <hip_runtime_api.h>
#include "Cobalt.h"

/*******************************************************************************
 * \brief convert CobaltStatus to rocblas_status
 ******************************************************************************/
rocblas_status
get_rocblas_status_for_cobalt_status( CobaltStatus status ) {
  switch(status) {

  case cobaltStatusSuccess:
    return rocblas_status_success;

  case cobaltStatusControlInvalid:
    return rocblas_status_invalid_handle;

  case cobaltStatusTensorNumDimensionsInvalid:
  case cobaltStatusTensorDimensionOrderInvalid:
  case cobaltStatusTensorDimensionStrideInvalid:
  case cobaltStatusTensorDimensionSizeInvalid:
  case cobaltStatusOperandNumDimensionsMismatch:
  case cobaltStatusOperationOperandNumIndicesMismatch:
  case cobaltStatusOperationIndexAssignmentInvalidA:
  case cobaltStatusOperationIndexAssignmentInvalidB:
  case cobaltStatusOperationIndexAssignmentDuplicateA:
  case cobaltStatusOperationIndexAssignmentDuplicateB:
  case cobaltStatusOperationNumFreeIndicesInvalid:
  case cobaltStatusOperationNumSummationIndicesInvalid:
  case cobaltStatusOperationIndexUnassigned:
  case cobaltStatusOperationSummationIndexAssignmentsInvalid:
  case cobaltStatusDeviceProfileNumDevicesInvalid:
  case cobaltStatusDeviceProfileNotSupported: // cobalt should return a default implementation
  case cobaltStatusProblemNotSupported: //
  case cobaltStatusInvalidParameter:
  default:
    return rocblas_status_internal_error;
  }
}


