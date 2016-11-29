
#include "status.h"
#include <hip/hip_runtime_api.h>
#include "Tensile.h"

/*******************************************************************************
 * \brief convert TensileStatus to rocblas_status
 ******************************************************************************/
rocblas_status
get_rocblas_status_for_tensile_status( TensileStatus status ) {
    switch(status) {
        case tensileStatusSuccess:
          return rocblas_status_success;

        case tensileStatusControlInvalid:
          return rocblas_status_invalid_handle;

        case tensileStatusTensorNumDimensionsInvalid:
        case tensileStatusTensorDimensionOrderInvalid:
        case tensileStatusTensorDimensionStrideInvalid:
        case tensileStatusTensorDimensionSizeInvalid:
        case tensileStatusOperandNumDimensionsMismatch:
        case tensileStatusOperationOperandNumIndicesMismatch:
        case tensileStatusOperationIndexAssignmentInvalidA:
        case tensileStatusOperationIndexAssignmentInvalidB:
        case tensileStatusOperationIndexAssignmentDuplicateA:
        case tensileStatusOperationIndexAssignmentDuplicateB:
        case tensileStatusOperationNumFreeIndicesInvalid:
        case tensileStatusOperationNumSummationIndicesInvalid:
        case tensileStatusOperationIndexUnassigned:
        case tensileStatusOperationSummationIndexAssignmentsInvalid:
        case tensileStatusDeviceProfileNumDevicesInvalid:
        case tensileStatusDeviceProfileNotSupported: // tensile should return a default implementation
        case tensileStatusProblemNotSupported: //
        case tensileStatusInvalidParameter:
        default:
          return rocblas_status_internal_error;
    }
}
