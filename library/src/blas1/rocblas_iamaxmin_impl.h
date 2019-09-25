#pragma once

#include "rocblas_iamaxmin_template.h"

//
// Define names.
//
#define ROCBLAS_IAMAXMIN_NAME JOIN(rocblas_iamaxmin,JOIN(ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX,_name))
  
template <typename>
constexpr char ROCBLAS_IAMAXMIN_NAME[] = "unknown";

template <>
constexpr char ROCBLAS_IAMAXMIN_NAME<float>[]
= "rocblas_is" QUOTE(MAX_MIN) QUOTE(ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX);

template <>
constexpr char ROCBLAS_IAMAXMIN_NAME<double>[]
= "rocblas_id" QUOTE(MAX_MIN) QUOTE(ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX);

template <>
constexpr char ROCBLAS_IAMAXMIN_NAME<rocblas_float_complex>[]
= "rocblas_ic" QUOTE(MAX_MIN) QUOTE(ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX);

template <>
constexpr char ROCBLAS_IAMAXMIN_NAME<rocblas_double_complex>[]
= "rocblas_iz" QUOTE(MAX_MIN) QUOTE(ROCBLAS_IAMAXMIN_GROUPKIND_SUFFIX);


template <typename U>
static rocblas_status rocblas_iamaxmin_impl(rocblas_handle handle,
					    rocblas_int    n,
					    U              x,
					    rocblas_int    incx,
					    rocblas_stride stridex,
					    rocblas_int    batch_count,
					    rocblas_int*   result)
{
  //
  // Get the 'T input' type.
  //
  using Ti = batched_data_t<U>;
      
  //
  // Get the name of the routine.
  //
  static constexpr const char * const name =  ROCBLAS_IAMAXMIN_NAME<Ti>;
      
  //
  // Get the 'T output' type
  //
  using To = typename reduction_types<Ti>::To;
      
  //
  // HIP support up to 1024 threads/work times per thread block/work group
  //
  static constexpr int NB = 1024;
  if(!handle)
    {
      return rocblas_status_invalid_handle;
    }
      
  auto layer_mode = handle->layer_mode;
  if(layer_mode & rocblas_layer_mode_log_trace)
    {
      log_trace(handle, name, n, x, incx, stridex, batch_count);
    }

  if(layer_mode & rocblas_layer_mode_log_bench)
    {
      log_bench(handle,
		"./rocblas-bench -f",
		&name[8], // skip 'rocblas_'
		"-r",
		rocblas_precision_string<Ti>,
		"-n",
		n,
		"--incx",
		incx,
		"--stride_x",
		stridex,
		"--batch",
		batch_count);
    }
  
  if(layer_mode & rocblas_layer_mode_log_profile)
    {
      log_profile(handle,
		  name,
		  "N",
		  n,
		  "incx",
		  incx,
		  "stride_x",
		  stridex,
		  "batch",
		  batch_count);
    }
  
  if(!x || !result)
    {
      return rocblas_status_invalid_pointer;
    }
  
  if(batch_count < 0)
    {
      return rocblas_status_invalid_size;
    }

  const size_t workspace_num_bytes
    = rocblas_reduction_kernel_workspace_size<NB, index_value_t<To> >(n, batch_count);

  if(handle->is_device_memory_size_query())
    {
      return handle->set_optimal_device_memory_size(workspace_num_bytes);
    }
  
  auto mem = handle->device_malloc(workspace_num_bytes);
  if(!mem)
    {
      return rocblas_status_memory_error;
    }
  
  return rocblas_iamaxmin_template(handle,
				   n,
				   x,
				   incx,
				   stridex,
				   batch_count,
				   result,
				   (void*)mem);
}

#undef ROCBLAS_IAMAXMIN_NAME
