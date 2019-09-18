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
// Define the name and its specializations.
//
#define TOKEN_ROCBLAS_IAMAXMIN_NAME_BATCHED QUOTE(MAX_MIN) "_batched"

template <typename>
static constexpr char rocblas_iamaxmin_name_batched[] = "unknown";

template <>
static constexpr char rocblas_iamaxmin_name_batched<float>[]
= "rocblas_isa" TOKEN_ROCBLAS_IAMAXMIN_NAME_BATCHED;

template <>
static constexpr char rocblas_iamaxmin_name_batched<double>[]
= "rocblas_ida" TOKEN_ROCBLAS_IAMAXMIN_NAME_BATCHED;

template <>
static constexpr char rocblas_iamaxmin_name_batched<rocblas_float_complex>[]
= "rocblas_ica" TOKEN_ROCBLAS_IAMAXMIN_NAME_BATCHED;

template <>
static constexpr char rocblas_iamaxmin_name_batched<rocblas_double_complex>[]
= "rocblas_iza" TOKEN_ROCBLAS_IAMAXMIN_NAME_BATCHED;

#undef TOKEN_ROCBLAS_IAMAXMIN_NAME_BATCHED


#if 0
//
//
//
template <typename Ti>
static rocblas_status rocblas_iamaxmin_batched(rocblas_handle handle,
                                               rocblas_int    n,
                                               const Ti*      const x[],
                                               rocblas_int    incx,
                                               rocblas_int    batch_count,
                                               rocblas_int*   result)
{
    //
    // Get the 'T output' type
    //
    using To = typename TypeTraits_AMAXMIN<Ti>::To;

    //
    // HIP support up to 1024 threads/work itmes per thread block/work group
    //
    static constexpr int NB = 1024;
    static constexpr int P = 8;
    if(!handle)
    {
        return rocblas_status_invalid_handle;
    }

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        log_trace(handle, rocblas_iamaxmin_name_batched<Ti>, n, x, incx);
    }

    if(layer_mode & rocblas_layer_mode_log_bench)
    {
        log_bench(handle,
                  "./rocblas-bench -f ia" QUOTE(MAX_MIN) " -r",
                  rocblas_precision_string<Ti>,
                  "-n",
                  n,
                  "--incx",
                  incx);
    }

    if(layer_mode & rocblas_layer_mode_log_profile)
    {
        log_profile(handle, rocblas_iamaxmin_name_batched<Ti>, "N", n, "incx", incx);
    }

    if(!x || !result)
    {
        return rocblas_status_invalid_pointer;
    }

    *result = -777;
    
    //
    // Quick return if possible.
    //
    if(n <= 0 || incx <= 0 || batch_count <=0)
    {
        if(handle->is_device_memory_size_query())
        {
	  return rocblas_status_size_unchanged;
        }
        else if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
	  RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(*result)));
        }
        else
        {
	  *result = 0;
        }
        return rocblas_status_success;
    }
    

    //
    // 
    //
    //
    // block_m =  P
    // block_n =  NumThreadsPerBlock / P
    // mblock = batch_count / block_m
    // nblock = n / block_n
    //
    //


    
    //
    //
    //
    //
    
    //
    // batch_count x n matrix.
    //
    // 1 x n matrix.
    // A_i = A(1, i * NumThreadsPerBlock:(i+1)NumThreadsPerBlock)
    //
    // batch_count x n matrix.
    // A_i = A(:, i * NumThreadsPerBlock:(i+1)NumThreadsPerBlock)
    //
    // (batch_count/P) * P x n matrix.
    // A_i,p = A( p * P : (p+1)*P, i * (NumThreadsPerBlock/P) :(i+1)(NumThreadsPerBlock/P) )
    //
    //
    // num
    //
    // block_m =  P
    // block_n =  NumThreadsPerBlock / P
    //
    //
    //
    // NumThreadsPerBlock = 1024
    // NumBlocks = (n-1) / NumThreadsPerBlock + 1
    //
    // NP = 4
    // NUMBATCH = (batch_count-1) / NP + 1
    // NumMatrixPerBlock = NumThreadsPerBlock / NP
    //
    //
    //
    //
    // V = 1,  matrix 1024 blocks of 1x1024   (1 matrix at a time)
    // V = 2,  matrix 1024 blocks of 2x512    (2 matrices at a time) ?
    // V = 4,  matrix 1024 blocks of 4x256    (4 matrices at a time) ?
    // V = 8,  matrix 1024 blocks of 8x256    (8 matrices at a time) ?
    // V = 16, matrix 1024 blocks of 16x128  (16 matrices at a time) ?
    //
    // size on X:  NumThreadsPerBlock / V
    //
    //
    // P = 1, V = 1,  1 matrix 1024 blocks of 1x1024   (1 matrix at a time)
    // P = 1, V = 2,  1 matrix 1024 blocks of 2x512    (2 matrices at a time) ?
    // P = 1, V = 4,  1 matrix 1024 blocks of 4x256    (4 matrices at a time) ?
    // P = 1, V = 8,  1 matrix 1024 blocks of 8x128    (8 matrices at a time) ?
    // P = 1, V = 1,  1 matrix NumBlocks blocks of 1xNumThreadsPerBlock    (8 matrices at a time) ?
    // P = 1, V = 1,  batch_count,  2*NumBlocks blocks of 2x(NumThreadsPerBlock/2)    (8 matrices at a time) ?
    // P = 1, V = 1,  batch_count = 2,  2*NumBlocks blocks of 2x(NumThreadsPerBlock/2)    (8 matrices at a time) ?

    // P = 1, V = 16, (batch_count / V) * [ (V x NumBlocks) blocks of Vx(NumThreadsPerBlock/V) ]  (V matrices at a time) ?
    //
    //
    //
    //
    //
    // P = 2, V = 1,  2 matrices P*1024 blocks of 1x1024   (1 matrix at a time)
    // P = 2, V = 2,  2 matrices P*1024 blocks of 2x512    (2 matrices at a time) ?
    // P = 2, V = 4,  2 matrices P*1024 blocks of 4x256    (4 matrices at a time) ?
    // P = 2, V = 8,  2 matrices P*1024 blocks of 8x256    (8 matrices at a time) ?
    // P = 2, V = 16, 2 matrices P*1024 blocks of 16x128  (16 matrices at a time) ?
    //
    //
    // batch_count X n
    //
    // batch_count X n
    //  
    // One block can treat a piece of 
    //
    //
    //
    // B[i] = MAX(X_i[j])   i * NumThreadsPerBlock <= j < (i+1)*NumThreadsPerBlock
    // B[i] = MAX(X_i[j])   i * NumThreadsPerBlock <= j < (i+1)*NumThreadsPerBlock
    //
    //
    // Now we treat p components
    //
    // B[i] = MAX(X_i[j])   i * NumThreadsPerBlock <= j < (i+1)*NumThreadsPerBlock
    // B[i] is a vector of batch_count components.
    //
    //
    // 
    //
    //
    //
    auto blocks = (n - 1) / NB + 1;
    auto blocks = (n - 1) / NB + 1;
    if(handle->is_device_memory_size_query())
    {
        return handle->set_optimal_device_memory_size(sizeof(index_value_t<To>) * blocks);
    }

    auto mem = handle->device_malloc(sizeof(index_value_t<To>) * blocks);
    if(!mem)
    {
        return rocblas_status_memory_error;
    }

    //
    // Ask for a temporary memory of 
    //
    // P vectors in a compact format.
    //
    // Build a matrix V of size PxN 
    //
    //
    
    int status = rocblas_status_success;

    //
    //
    // batch_count X n
    //
    //
    status &= rocblas_batched_reduction_kernel<NB,
                                       rocblas_fetch_amax_amin<To>,
                                       AMAX_AMIN_REDUCTION,
                                       rocblas_finalize_amax_amin>
	  (handle, n, x[batch_index], incx, &result[batch_index], (index_value_t<To>*)mem, blocks);
      
    // printf("results########## %d\n", result[batch_index]);
 
    return rocblas_status_success;
}



template <typename Ti>
static rocblas_status rocblas_iamaxmin_batched_trivial(rocblas_handle handle,
						       rocblas_int    n,
						       const Ti*      const x[],
						       rocblas_int    incx,
						       rocblas_int    batch_count,
						       rocblas_int*   result)
{
    //
    // Get the 'T output' type
    //
    using To = typename TypeTraits_AMAXMIN<Ti>::To;

    //
    // HIP support up to 1024 threads/work itmes per thread block/work group
    //
    static constexpr int NB = 1024;
    if(!handle)
    {
        return rocblas_status_invalid_handle;
    }

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        log_trace(handle, rocblas_iamaxmin_name_batched<Ti>, n, x, incx);
    }

    if(layer_mode & rocblas_layer_mode_log_bench)
    {
        log_bench(handle,
                  "./rocblas-bench -f ia" QUOTE(MAX_MIN) " -r",
                  rocblas_precision_string<Ti>,
                  "-n",
                  n,
                  "--incx",
                  incx);
    }

    if(layer_mode & rocblas_layer_mode_log_profile)
    {
        log_profile(handle, rocblas_iamaxmin_name_batched<Ti>, "N", n, "incx", incx);
    }

    if(!x || !result)
    {
        return rocblas_status_invalid_pointer;
    }

    *result = -777;
    //
    // Quick return if possible.
    //
    //    printf("%d %d %d ????\n",n,incx,batch_count);
    if(n <= 0 || incx <= 0 || batch_count <=0)
    {
        if(handle->is_device_memory_size_query())
        {
	  return rocblas_status_size_unchanged;
        }
        else if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
	  RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(*result)));
        }
        else
        {
	  *result = 0;
        }
        return rocblas_status_success;
    }
    
    auto blocks = (n - 1) / NB + 1;
    if(handle->is_device_memory_size_query())
    {
        return handle->set_optimal_device_memory_size(sizeof(index_value_t<To>) * blocks);
    }

    auto mem = handle->device_malloc(sizeof(index_value_t<To>) * blocks);
    if(!mem)
    {
        return rocblas_status_memory_error;
    }

    int status = rocblas_status_success;
    for(int batch_index = 0; batch_index < batch_count; ++batch_index)
    {
      status &= rocblas_reduction_kernel<NB,
                                           rocblas_fetch_amax_amin<To>,
                                           AMAX_AMIN_REDUCTION,
                                           rocblas_finalize_amax_amin>
	  (handle, n, x[batch_index], incx, &result[batch_index], (index_value_t<To>*)mem, blocks);
      
      //      printf("results########## %d\n", result[batch_index]);
    }
 
    return rocblas_status_success;
}


#endif




#if 0
template <typename Ti>
static rocblas_status rocblas_iamaxmin_batched_impl(rocblas_handle handle,
						    rocblas_int    n,
						    const Ti*      const x[],
						    rocblas_int    incx,
						    rocblas_int    batch_count,
						    rocblas_int*   result)
{
    //
    // Get the 'T output' type
    //
    using To = typename TypeTraits_AMAXMIN<Ti>::To;

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
        log_trace(handle, rocblas_iamaxmin_name_batched<Ti>, n, x, incx);
    }

    if(layer_mode & rocblas_layer_mode_log_bench)
    {
        log_bench(handle,
                  "./rocblas-bench -f ia" QUOTE(MAX_MIN) " -r",
                  rocblas_precision_string<Ti>,
                  "-n",
                  n,
                  "--incx",
                  incx);
    }

    if(layer_mode & rocblas_layer_mode_log_profile)
    {
        log_profile(handle, rocblas_iamaxmin_name_batched<Ti>, "N", n, "incx", incx);
    }

    if(!x || !result)
    {
        return rocblas_status_invalid_pointer;
    }

    *result = -777;
    //
    // Quick return if possible.
    //
    if(n <= 0 || incx <= 0 || batch_count <=0)
    {
        if(handle->is_device_memory_size_query())
        {
	  return rocblas_status_size_unchanged;
        }
        else if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
	  RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(*result)));
        }
        else
        {
	  *result = 0;
        }
        return rocblas_status_success;
    }
    
    auto blocks = (n - 1) / NB + 1;
    if(handle->is_device_memory_size_query())
    {
        return handle->set_optimal_device_memory_size(sizeof(index_value_t<To>) * blocks);
    }

    auto mem = handle->device_malloc(sizeof(index_value_t<To>) * blocks);
    if(!mem)
    {
        return rocblas_status_memory_error;
    }




    
    int status = rocblas_status_success;
    for(int batch_index = 0; batch_index < batch_count; ++batch_index)
    {
      status &= rocblas_reduction_kernel<NB,
                                           rocblas_fetch_amax_amin<To>,
                                           AMAX_AMIN_REDUCTION,
                                           rocblas_finalize_amax_amin>
	  (handle, n, x[batch_index], incx, &result[batch_index], (index_value_t<To>*)mem, blocks);     
    }



    
    return rocblas_status_success;
}
#endif





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
							rocblas_int*                  result) \
  {									\
  static constexpr rocblas_int stridex = 0;				\
  return rocblas_iamaxmin_template(handle,				\
				   n,					\
				   x,					\
				   incx,				\
				   stridex,				\
				   result,				\
				   1,					\
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


