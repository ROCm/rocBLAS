#ifndef DEFINITIONS_H
#define DEFINITIONS_H
/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/

#define RETURN_IF_COBALT_ERROR(STATUS) \
  if (STATUS != cobaltStatusSuccess) { \
    cobaltStatusCheck( STATUS ); \
    return get_rocblas_status_for_cobalt_status(STATUS); \
  }
#define RETURN_IF_HIP_ERROR(STATUS) \
  if (STATUS != hipSuccess ) { \
    return get_rocblas_status_for_hip_status(STATUS); \
  }
#define RETURN_IF_ROCBLAS_ERROR(STATUS) \
  if (STATUS != rocblas_status_success) { \
    return STATUS; \
  }

#define THROW_IF_COBALT_ERROR(STATUS) \
  if (STATUS != cobaltStatusSuccess) { \
    cobaltStatusCheck( STATUS ); \
    throw get_rocblas_status_for_cobalt_status(STATUS); \
  }
#define THROW_IF_HIP_ERROR(STATUS) \
  if (STATUS != hipSuccess ) { \
    throw get_rocblas_status_for_hip_status(STATUS); \
  }
#define THROW_IF_ROCBLAS_ERROR(STATUS) \
  if (STATUS != rocblas_status_success) { \
    throw STATUS; \
  }

#endif
