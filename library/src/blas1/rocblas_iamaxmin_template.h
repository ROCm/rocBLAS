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



#if 0
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
#endif

template <typename U>
struct rocblas_iamaxmin_name_template
{
  static constexpr char name[] = "unknown";
};



template <typename T>
struct toto
{

};
#if 0
template <typename T>
struct toto<T>
{
  static constexpr char c[] = "dd";
};
#endif
template <typename T>
struct rocblas_iamaxmin_name_template< const_batched_arrays<T> >
{
  static constexpr char name[] = "rocblas_" toto<T>::c;// "_batched";
};


#endif
template <typename U>
static rocblas_status rocblas_iamaxmin_template(rocblas_handle handle,
						rocblas_int    n,
						U              x,
						rocblas_int    incx,
						rocblas_int    stridex,
						rocblas_int*   r,
						rocblas_int    incr,
						rocblas_int    strider,
						rocblas_int    batch_count,
						const char name[])
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
      log_trace(handle, name, n, x, incx);
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
      log_profile(handle, name, "N", n, "incx", incx);
    }
  
  if(!x || !r)
    {
      return rocblas_status_invalid_pointer;
    }
  
  *r = -777;
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
	  RETURN_IF_HIP_ERROR(hipMemset(r, 0, sizeof(*r)));
        }
        else
        {
	  *r = 0;
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
      auto h = load_batched_ptr(x,batch_index,stridex);
      status &= rocblas_reduction_kernel<NB,
                                           rocblas_fetch_amax_amin<To>,
                                           AMAX_AMIN_REDUCTION,
                                           rocblas_finalize_amax_amin>
	  (handle, n, h, incx, &r[batch_index], (index_value_t<To>*)mem, blocks);     
    }
    
    return rocblas_status_success;
}
