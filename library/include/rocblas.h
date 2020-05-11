/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*!\file
 * \brief rocblas.h includes other *.h and exposes a common interface
 */

#ifndef _ROCBLAS_H_
#define _ROCBLAS_H_

#if __clang__
#define ROCBLAS_CLANG_STATIC static
#else
#define ROCBLAS_CLANG_STATIC
#endif

/* library headers */
#include "rocblas-auxiliary.h"
#include "rocblas-export.h"
#include "rocblas-functions.h"
#include "rocblas-types.h"
#include "rocblas-version.h"

#endif // _ROCBLAS_H_
