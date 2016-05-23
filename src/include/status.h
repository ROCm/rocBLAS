#ifndef STATUS_H
#define STATUS_H

#include "rocblas.h"
#include "Cobalt.h"

/*******************************************************************************
 * \brief convert CobaltStatus to rocblas_status
 ******************************************************************************/
rocblas_status 
get_rocblas_status_for_cobalt_status( CobaltStatus status );


/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 ******************************************************************************/
rocblas_status
get_rocblas_status_for_hip_status( hipError_t status );

#endif
