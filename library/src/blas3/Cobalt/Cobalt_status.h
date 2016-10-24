#ifndef COBALT_STATUS_H
#define COBALT_STATUS_H

#include "rocblas.h"
#include "Cobalt.h"

/*******************************************************************************
 * \brief convert CobaltStatus to rocblas_status
 ******************************************************************************/
rocblas_status 
get_rocblas_status_for_cobalt_status( CobaltStatus status );


#endif
