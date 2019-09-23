#pragma once
#include "rocblas_batched_arrays.h"
#ifndef AMAX_AMIN_REDUCTION
#define AMAX_AMIN_REDUCTION rocblas_reduce_amax
#endif


//
// Define a type trait and its specializations.
//
template <typename Ti>
struct TypeTraits_AMAXMIN
{
public:
    using To = Ti;
};

template <>
struct TypeTraits_AMAXMIN<rocblas_float_complex>
{
 public: using To = float;
};

template <>
struct TypeTraits_AMAXMIN<rocblas_double_complex>
{
 public: using To = double;
};

//
// Extension of a rocblas_int as a pair of index and value.
// As an extension the index must be stored first.
//
template <typename T>
struct index_value_t
{
    rocblas_int index;
    T           value;
};


template <typename T>
std::ostream& operator<<(std::ostream& out,const index_value_t<T>& h)
{
  out << "(" << h.index << "," << h.value << ")" << std::endl;
  return out;
};

//
// Specialization of default_value for index_value_t<T>.
//
template <typename T>
struct default_value<index_value_t<T>>
{
    __forceinline__ __host__ __device__ constexpr auto operator()() const
    {
        index_value_t<T> x;
        x.index = -1;
        return x;
    }
};

//
// Fetch absolute value.
//
template <typename To>
struct rocblas_fetch_amax_amin
{
    template <typename Ti>
    __forceinline__ __host__ __device__ index_value_t<To> operator()(Ti x, rocblas_int index)
  {
    return {index, fetch_asum(x)};
    }
};




//
// Replaces x with y if y.value < x.value or y.value == x.value and y.index < x.index.
//
struct rocblas_reduce_amax
{
    template <typename To>
    __forceinline__ __host__ __device__ void operator()(index_value_t<To>& __restrict__ x,
                                                        const index_value_t<To>& __restrict__ y)
    {

        //
        // If y.index == -1 then y.value is invalid and should not be compared.
        //
        if(y.index != -1)
        {
            if(-1 == x.index || y.value > x.value)
            {
                x = y; // if larger or smaller, update max/min and index.
            }
            else if(y.index < x.index && x.value == y.value)
            {
                x.index = y.index; // if equal, choose smaller index.
            }
        }
    }
};

// Replaces x with y if y.value < x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ void operator()(index_value_t<To>& __restrict__ x,
                                                        const index_value_t<To>& __restrict__ y)
    {
        // If y.index == -1 then y.value is invalid and should not be compared
        if(y.index != -1)
        {
            if(x.index == -1 || y.value < x.value)
            {
                x = y; // if larger or smaller, update max/min and index
            }
            else if(y.index < x.index && x.value == y.value)
            {
                x.index = y.index; // if equal, choose smaller index
            }
        }
    }
};

struct rocblas_finalize_amax_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ auto operator()(const index_value_t<To>& x)
    {
        return x.index + 1;
    }
};


#if 0
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

template <typename U>
struct rocblas_iamaxmin_name_template
{
  static constexpr char name[] = "unknown";
};

#endif

template <typename U>
static rocblas_status rocblas_iamaxmin_impl(rocblas_handle handle,
					    rocblas_int    n,
					    U              x,
					    rocblas_int    incx,
					    rocblas_stride stridex,
					    rocblas_int*   r,
					    rocblas_int    incr,
					    rocblas_stride strider,
					    rocblas_int    batch_count,
					    const char     name[])
{
  //
  // Get the 'T input' type
  //
  using Ti = typename batched_arrays_traits<U>::base_t;
  
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
  
  if(!x || !r)
    {
      return rocblas_status_invalid_pointer;
    }
  
  if(batch_count < 0)
    {
      return rocblas_status_invalid_size;
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
  
  return rocblas_iamaxmin_template(handle,
				   n,
				   x,
				   incx,
				   stridex,
				   r,
				   incr,
				   strider,
				   batch_count,
				   (void*)mem,
				   name);
}


template <typename U, typename R>
static rocblas_status rocblas_iamaxmin_template(rocblas_handle handle,
						rocblas_int    n,
						U              x,
						rocblas_int    incx,
						rocblas_stride stridex,
						R*             r,
						rocblas_int    incr,
						rocblas_stride strider,
						rocblas_int    batch_count,
						void *         workspace,
						const char     name[])
{
  //
  // Get the 'T input' type
  //
  using Ti = typename batched_arrays_traits<U>::base_t;
  
  //
  // Get the 'T output' type
  //
  using To = typename TypeTraits_AMAXMIN<Ti>::To;
  
  //
  // HIP support up to 1024 threads/work times per thread block/work group
  //
  static constexpr int NB = 1024;

  //
  // Quick return if possible.
  //
  if(n <= 0 || incx <= 0 || batch_count == 0)
    {
      if(handle->is_device_memory_size_query())
        {
	  return rocblas_status_size_unchanged;
        }
      else if(handle->pointer_mode == rocblas_pointer_mode_device && batch_count > 0)
        {
	  RETURN_IF_HIP_ERROR(hipMemset(r, 0, batch_count * sizeof(R)));
        }
      else
        {
	  //
	  // On host.
	  //
	  for(int i = 0; i < batch_count; i++)
            {
	      r[i] = R(0);
            }
        }
      return rocblas_status_success;
    }
  auto blocks = (n - 1) / NB + 1;
  
    int status = rocblas_status_success;
    for(int batch_index = 0; batch_index < batch_count; ++batch_index)
      {
      auto h = load_batched_ptr(x,batch_index,stridex);
      status &= rocblas_reduction_kernel<NB,
                                           rocblas_fetch_amax_amin<To>,
                                           AMAX_AMIN_REDUCTION,
                                           rocblas_finalize_amax_amin>
	  (handle, n, h, incx, &r[batch_index], (index_value_t<To>*)workspace, blocks);     
    }
    
    return rocblas_status_success;
}
