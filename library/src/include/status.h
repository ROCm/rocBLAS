/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef STATUS_H
#define STATUS_H

#include "rocblas.h"

/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 ******************************************************************************/
rocblas_status get_rocblas_status_for_hip_status(hipError_t status);

#endif
