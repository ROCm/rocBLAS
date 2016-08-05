#ifndef DEFINITIONS_H
#define DEFINITIONS_H
/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/

#define RETURN_IF_COBALT_ERROR(INPUT_STATUS_FOR_CHECK) \
  cobaltStatus TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
  if (TMP_STATUS_FOR_CHECK != cobaltStatusSuccess) { \
    cobaltStatusCheck( TMP_STATUS_FOR_CHECK ); \
    return get_rocblas_status_for_cobalt_status(TMP_STATUS_FOR_CHECK); \
  }
#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK) \
  hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
  if (TMP_STATUS_FOR_CHECK != hipSuccess ) { \
    return get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
  }
#define RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK) \
  rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
  if (TMP_STATUS_FOR_CHECK != rocblas_status_success) { \
    return TMP_STATUS_FOR_CHECK; \
  }

#define THROW_IF_COBALT_ERROR(INPUT_STATUS_FOR_CHECK) \
  cobaltStatus TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
  if (TMP_STATUS_FOR_CHECK != cobaltStatusSuccess) { \
    cobaltStatusCheck( TMP_STATUS_FOR_CHECK ); \
    throw get_rocblas_status_for_cobalt_status(TMP_STATUS_FOR_CHECK); \
  }
#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK) \
  hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
  if (TMP_STATUS_FOR_CHECK != hipSuccess ) { \
    throw get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
  }
#define THROW_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK) \
  rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
  if (TMP_STATUS_FOR_CHECK != rocblas_status_success) { \
    throw TMP_STATUS_FOR_CHECK; \
  }

#endif
