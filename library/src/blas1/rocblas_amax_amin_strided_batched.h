/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "fetch_template.h"
#include "handle.h"
#include "logging.h"
#include "reduction_strided_batched.h"
#include "rocblas.h"
#include "utility.h"

#ifndef MAX_MIN
#define undefined macro MAX_MIN
#endif

//
// Specify which suffix to use: _batched, _strided_batched or nothing.
// Here _strided_batched.
//
#define ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX _strided_batched

#include "rocblas_iamaxmin_impl.h"

//
// C wrapper
//
extern "C" {

#ifdef ROCBLAS_IAMAXMIN_STRIDED_BATCHED_HEADER
#error existing macro ROCBLAS_IAMAXMIN_STRIDED_BATCHED_HEADER
#endif
#ifdef ROCBLAS_IAMAXMIN_STRIDED_BATCHED_CIMPL
#error existing macro ROCBLAS_IAMAXMIN_STRIDED_BATCHED_CIMPL
#endif
  
  //
  // Define the C header.
  //
#define ROCBLAS_IAMAXMIN_STRIDED_BATCHED_HEADER(name)		\
  JOIN(name, JOIN(MAX_MIN, ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX) )



#define ROCBLAS_IAMAXMIN_STRIDED_BATCHED_CIMPL(name, type)		\
  rocblas_status ROCBLAS_IAMAXMIN_STRIDED_BATCHED_HEADER(name) (rocblas_handle  handle, \
								rocblas_int     n, \
								const type*     x, \
								rocblas_int     incx, \
								rocblas_stride  stridex, \
								rocblas_int     batch_count, \
								rocblas_int*    result) \
  {									\
    return rocblas_iamaxmin_impl(handle,				\
				 n,					\
				 x,					\
				 incx,					\
				 stridex,				\
				 batch_count,				\
				 result);				\
  }
  
ROCBLAS_IAMAXMIN_STRIDED_BATCHED_CIMPL( rocblas_isa , float);
ROCBLAS_IAMAXMIN_STRIDED_BATCHED_CIMPL( rocblas_ida , double);
ROCBLAS_IAMAXMIN_STRIDED_BATCHED_CIMPL( rocblas_ica , rocblas_float_complex);
ROCBLAS_IAMAXMIN_STRIDED_BATCHED_CIMPL( rocblas_iza , rocblas_double_complex);
  
  //
  // Undefined introduced macros.
  //
#undef ROCBLAS_IAMAXMIN_STRIDED_BATCHED_CIMPL
#undef ROCBLAS_IAMAXMIN_STRIDED_BATCHED_HEADER

  
} // extern "C"

#undef ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX

