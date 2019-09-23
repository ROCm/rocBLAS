/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

//
// Use the non-batched header.
//
#include "rocblas_amax_amin.h"
#include "rocblas_iamaxmin_template.h"

//
// C wrapper
//
extern "C" {
  
#ifdef ROCBLAS_IAMAXMIN_BATCHED_HEADER
#error existing macro ROCBLAS_IAMAXMIN_BATCHED_HEADER
#endif
#ifdef ROCBLAS_IAMAXMIN_BATCHED_CIMPL
#error existing macro ROCBLAS_IAMAXMIN_BATCHED_CIMPL
#endif
  
  //
  // Define the C header.
  //
#define ROCBLAS_IAMAXMIN_BATCHED_HEADER(name)	\
  JOIN(name, JOIN(MAX_MIN, _batched))
  
#define ROCBLAS_IAMAXMIN_BATCHED_CIMPL(name, type)			\
  rocblas_status ROCBLAS_IAMAXMIN_BATCHED_HEADER(name) (rocblas_handle                handle, \
							rocblas_int                   n, \
							const type* const             x[], \
							rocblas_int                   incx, \
							rocblas_int                   batch_count, \
							rocblas_int*                  result, \
							rocblas_int                   incresult) \
  {									\
    static constexpr rocblas_int stridex = 0;				\
    return rocblas_iamaxmin_impl(handle,				\
				 n,					\
				 x,					\
				 incx,					\
				 stridex,				\
				 result,				\
				 incresult,				\
				 1,					\
				 batch_count,				\
				 QUOTE(ROCBLAS_IAMAXMIN_BATCHED_HEADER(name))); \
  }
  
  ROCBLAS_IAMAXMIN_BATCHED_CIMPL( rocblas_isa , float);
  ROCBLAS_IAMAXMIN_BATCHED_CIMPL( rocblas_ida , double);
  ROCBLAS_IAMAXMIN_BATCHED_CIMPL( rocblas_ica , rocblas_float_complex);
  ROCBLAS_IAMAXMIN_BATCHED_CIMPL( rocblas_iza , rocblas_double_complex);
  
  
  //
  // Undefined introduced macro.
  //
#undef ROCBLAS_IAMAXMIN_BATCHED_CIMPL
#undef ROCBLAS_IAMAXMIN_BATCHED_HEADER
  
} // extern "C"


