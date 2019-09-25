#pragma once

#include "rocblas_batched_arrays.h"

#ifndef AMAX_AMIN_REDUCTION
#error undefined macro AMAX_AMIN_REDUCTION
#endif

#define QUOTE2(S) #S
#define QUOTE(S) QUOTE2(S)

#define JOIN2(A, B) A##B
#define JOIN(A, B) JOIN2(A, B)

//
//
// Template on the reduction types.
// 
//
template <typename T>
struct reduction_types
{
public:
    using Ti = T;
    using To = T;
};

//
// Specialization of the reduction types for the float complex.
//
template <>
struct reduction_types<rocblas_float_complex>
{
 public: using Ti = rocblas_float_complex;
 public: using To = float;
};

//
// Specialization of the reduction types for the double complex.
//
template <>
struct reduction_types<rocblas_double_complex>
{
 public: using Ti = rocblas_double_complex;
 public: using To = double;
};


//
// Extension of a rocblas_int as a pair of index and value.
// As an extension the index must be stored first.
// This struct is the working type, i.e. the intermediate data type of the min/max routine.
//
template <typename T>
struct index_value_t
{
  rocblas_int index;
  T           value;
};

//
// Overload the output stream operator for the intermediate data type.
//
template <typename T>
std::ostream& operator<<(std::ostream& out,const index_value_t<T>& index_value)
{
  out << "(" << index_value.index << "," << index_value.value << ")" << std::endl;
  return out;
};


// #############################################################
// DEFINITION OF ACTIONS TO EXECUTE DURING THE MIN/MAX ALGORITHM
// #############################################################

//
// Specialization of default_value for index_value_t<T>.
//
template <typename T>
struct rocblas_default_value<index_value_t<T>>
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

//
// Finalizer.
//
struct rocblas_finalize_amax_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ auto operator()(const index_value_t<To>& x)
    {
        return x.index + 1;
    }
};

template <typename U>
static rocblas_status rocblas_iamaxmin_template(rocblas_handle handle,
						rocblas_int    n,
						U              x,
						rocblas_int    incx,
						rocblas_stride stridex,
						rocblas_int    batch_count,
						rocblas_int *  result,
						void *         workspace)
{
  //
  // Get the 'T input' type
  //
  using Ti = batched_data_t<U>;
  
  //
  // Get the 'T output' type
  //
  using To = typename reduction_types<Ti>::To;
  
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
	  RETURN_IF_HIP_ERROR(hipMemset(result, 0, batch_count * sizeof(rocblas_int)));
        }
      else
        {
	  //
	  // On host.
	  //
	  for(int i = 0; i < batch_count; i++)
            {
	      result[i] = rocblas_int(0);
            }
        }
      return rocblas_status_success;
    }
  
  return rocblas_reduction_strided_batched_kernel
    < NB,
    Ti,
    rocblas_fetch_amax_amin<To>,
    AMAX_AMIN_REDUCTION,
    rocblas_finalize_amax_amin>(handle, n, x, 0, incx, stridex, batch_count, (index_value_t<To>*)workspace, result);
  
}
