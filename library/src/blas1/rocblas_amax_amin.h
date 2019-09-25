/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "fetch_template.h"
#include "handle.h"
#include "logging.h"
#include "reduction_strided_batched.h"
#include "rocblas.h"
#include "utility.h"

#ifndef MAX_MIN
#error undefined macro MAX_MIN
#endif

//
// Specify which suffix to use: _batched, _strided_batched or nothing.
// Here nothing.
//
#define ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX

//
// Include the template.
//
#include "rocblas_iamaxmin_impl.h"

//
// C wrapper
//
extern "C" {

#ifdef ROCBLAS_IAMAXMIN_HEADER
#error existing macro ROCBLAS_IAMAXMIN_BATCHED_HEADER
#endif
  
#ifdef ROCBLAS_IAMAXMIN_CIMPL
#error existing macro ROCBLAS_IAMAXMIN_STRIDED_BATCHED_CIMPL
#endif

  //
  // Define the C header.
  //
#define ROCBLAS_IAMAXMIN_HEADER(name) JOIN(name, MAX_MIN )

#define ROCBLAS_IAMAXMIN_CIMPL(name, type)				\
  rocblas_status ROCBLAS_IAMAXMIN_HEADER(name) (rocblas_handle  handle,		\
						rocblas_int     n,	\
						const type*     x,	\
						rocblas_int     incx,	\
						rocblas_int*    result)	\
  {									\
    return rocblas_iamaxmin_impl(handle,				\
				 n,					\
				 x,					\
				 incx,					\
				 0,					\
				 1,					\
				 result);				\
  }
  
  ROCBLAS_IAMAXMIN_CIMPL( rocblas_isa , float);
  ROCBLAS_IAMAXMIN_CIMPL( rocblas_ida , double);
  ROCBLAS_IAMAXMIN_CIMPL( rocblas_ica , rocblas_float_complex);
  ROCBLAS_IAMAXMIN_CIMPL( rocblas_iza , rocblas_double_complex);
  
  //
  // Undefined introduced macros.
  //
  
#undef ROCBLAS_IAMAXMIN_CIMPL
#undef ROCBLAS_IAMAXMIN_HEADER
  
} // extern "C"

#undef ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX
