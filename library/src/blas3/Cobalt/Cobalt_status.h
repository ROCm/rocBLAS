#ifndef TENSILE_STATUS_H
#define TENSILE_STATUS_H

#include "rocblas.h"
#include "Tensile.h"

/*******************************************************************************
 * \brief convert TensileStatus to rocblas_status
 ******************************************************************************/
rocblas_status 
get_rocblas_status_for_tensile_status( TensileStatus status );


#endif
