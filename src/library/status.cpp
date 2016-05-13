#include "status.h"
#include "hip_runtime.h"
#include "Cobalt.h"

/*******************************************************************************
 * \brief convert CobaltStatus to rocblas_status
 ******************************************************************************/
rocblas_status 
get_rocblas_status_for_cobalt_status( CobaltStatus status ) {
  switch(status) {
  case cobaltStatusSuccess:
    return rocblas_status_success;
  default:
    return rocblas_status_success;
  }
}


/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 ******************************************************************************/
rocblas_status
get_rocblas_status_for_hip_status( hipError_t status ) {
  switch(status) {
  case hipSuccess:
    return rocblas_status_success;
  default:
    return rocblas_status_success;
  }
}

