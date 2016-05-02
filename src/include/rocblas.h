/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*!\file
 * \brief rocblas.h includes other *.h and exposes a common interface
 */


#pragma once
#ifndef _ROCBLAS_H_
#define _ROCBLAS_H_

#include <stdbool.h>

/* version */
//#include "rocblas-version.h"

/* Data type */
#include "rocblas_types.h"
#include "rocblas_hip.h"

/* Publis APIs */
#include "rocblas_auxilary.h"
#include "rocblas_template_api.h"
#include "rocblas_netlib.h"
#include "rocblas_netlib_batched.h"

/* Advance expert interfaces */
#include "rocblas_expert.h"


#endif // _ROCBLAS_H_
