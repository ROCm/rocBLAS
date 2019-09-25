/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

// CBLAS does not have a cblas_iamin function, so we write our own version of it

template <typename T>
void template_testing_iamax_iamin_strided_batched_bad_arg(const Arguments& arg,
								   rocblas_iamax_iamin_strided_batched_t<T> func)
{
  rocblas_int
    N           = 100,
    incx        = 1,
    batch_count = 5;
  
  static const size_t safe_size   = 100;
  
  rocblas_local_handle handle;
  
  //
  // allocate memory on device
  //
  T* dx;
  hipMalloc(&dx, batch_count * sizeof(T));
  if(nullptr == dx)
    {
      CHECK_HIP_ERROR(hipErrorOutOfMemory);
      return;
    }
  
  rocblas_int h_rocblas_result;
  
  EXPECT_ROCBLAS_STATUS(func(handle, N, nullptr, incx, incx * N, batch_count, &h_rocblas_result),
			rocblas_status_invalid_pointer);
  
  EXPECT_ROCBLAS_STATUS(func(handle, N, dx, incx, incx * N, batch_count, nullptr),
			rocblas_status_invalid_pointer);
  
  EXPECT_ROCBLAS_STATUS(func(nullptr, N, dx, incx, incx * N, batch_count, &h_rocblas_result),
			rocblas_status_invalid_handle);
}

template <typename T,
	  void CBLAS_FUNC(rocblas_int, const T*, rocblas_int, rocblas_int*)>
void testing_iamax_iamin_strided_batched(const Arguments& arg, rocblas_iamax_iamin_strided_batched_t<T> func)
{

  rocblas_int
    N           = arg.N,
    incx        = arg.incx,
    batch_count = arg.batch_count;
  
  rocblas_stride stridex     = arg.stride_x;
  
  double
    rocblas_error_1,
    rocblas_error_2;
  
  rocblas_local_handle handle;
  if(batch_count < 0)
    {
      static const size_t safe_size = 100; //  arbitrarily set to zero
      device_vector<T>   dx(safe_size);
      device_vector<rocblas_int>   d_rocblas_result(std::abs(batch_count));
      if(!dx || !d_rocblas_result)
        {
	  CHECK_HIP_ERROR(hipErrorOutOfMemory);
	  return;
        }
      
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
      EXPECT_ROCBLAS_STATUS((func(handle, N, dx, incx, stridex, batch_count, d_rocblas_result)),
                              rocblas_status_invalid_size);
        return;
    }

  // check to prevent undefined memory allocation error
  if(N <= 0 || incx <= 0 || batch_count == 0)
    {
      static const size_t safe_size = 100; //  arbitrarily set to zero
      device_vector<T>   dx(safe_size);
      device_vector<rocblas_int>   d_rocblas_result( std::max( size_t(batch_count), safe_size) );
      if(!dx || !d_rocblas_result)
        {
	  CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }
      
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
      CHECK_ROCBLAS_ERROR((func(handle, N, dx, incx, stridex, batch_count, d_rocblas_result)));
      return;
    }
  
  rocblas_int
    rocblas_result_1[batch_count],
    rocblas_result_2[batch_count],
    cpu_result[batch_count];

  size_t size_input = std::abs(stridex) * (batch_count-1) + N * incx;

  // allocate memory on device
  device_vector<T> dx(size_input);
  device_vector<rocblas_int> d_rocblas_result_2(batch_count);
  if(!dx || !d_rocblas_result_2)
    {
      CHECK_HIP_ERROR(hipErrorOutOfMemory);
      return;
    }

  // Naming: dx is in GPU (device) memory. hx is in CPU (host) memory, plz follow this practice
  host_vector<T> hx(size_input);

  //
  // Initialize data on CPU
  //
  rocblas_seedrand();
  rocblas_init(hx, 1, N, incx, std::abs(stridex), batch_count);

  // copy data from CPU to device, does not work for incx != 1
  CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_input, hipMemcpyHostToDevice));

  T
    *ptrdx = (stridex > 0)
    ? ((T*)dx)
    : ((T*)dx) + (1 - batch_count) * stridex,

    *ptrhx = (stridex > 0)
    ? ((T*)hx)
    : ((T*)hx) + (1 - batch_count) * stridex;
  
  double gpu_time_used, cpu_time_used;

  if(arg.unit_check || arg.norm_check)
    {
      //
      // GPU BLAS, rocblas_pointer_mode_host
      //
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
      CHECK_ROCBLAS_ERROR((func(handle, N, ptrdx, incx, stridex, batch_count, rocblas_result_1)));
      
      //
      // GPU BLAS, rocblas_pointer_mode_device
      //
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
      CHECK_ROCBLAS_ERROR((func(handle, N, ptrdx, incx, stridex, batch_count, d_rocblas_result_2)));

      //
      // Copy result back to host.
      //
      CHECK_HIP_ERROR(hipMemcpy(rocblas_result_2, d_rocblas_result_2, batch_count * sizeof(rocblas_int), hipMemcpyDeviceToHost));

      //
      // COMPARE WITH CPU BLAS
      //

      //
      // Time to execution
      //
      cpu_time_used = get_time_us();
      for(int batch_index = 0; batch_index < batch_count; ++batch_index)
	{
	  CBLAS_FUNC(N, ptrhx + batch_index * stridex, incx, cpu_result + batch_index);
	  *(cpu_result + batch_index) += 1;
	}
      cpu_time_used = get_time_us() - cpu_time_used;

      //
      // Check the results
      //
      if(arg.unit_check)
	{
	  unit_check_general<rocblas_int>(batch_count, 1, 1, cpu_result, rocblas_result_1);
	  unit_check_general<rocblas_int>(batch_count, 1, 1, cpu_result, rocblas_result_2);
	}
	
      if(arg.norm_check)
	{
	  rocblas_error_1 = rocblas_result_1 - cpu_result;
	  rocblas_error_2 = rocblas_result_2 - cpu_result;
	}

    }

  if(arg.timing)
    {
      int number_cold_calls = 2;
      int number_hot_calls  = 100;
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

      for(int iter = 0; iter < number_cold_calls; iter++)
        {
	  func(handle, N, ptrdx, incx, stridex, batch_count, rocblas_result_2);
        }

      gpu_time_used = get_time_us(); // in microseconds

      for(int iter = 0; iter < number_hot_calls; iter++)
        {
	  func(handle, N, ptrdx, incx, stridex, batch_count, rocblas_result_2);
        }

      gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

      std::cout << "N,incx,stridex,batch_count,rocblas(us)";

      if(arg.norm_check)
	std::cout << ",CPU(us),error_host_ptr,error_dev_ptr";

      std::cout << std::endl;
      std::cout << N << "," << incx << "," << stridex << "," << batch_count << ","
		<< gpu_time_used;

      if(arg.norm_check)
	std::cout << "," << cpu_time_used << "," << rocblas_error_1 << "," << rocblas_error_2;

      std::cout << std::endl;
    }
}

template <typename T>
void testing_iamax_strided_batched_bad_arg(const Arguments& arg)
{
  template_testing_iamax_iamin_strided_batched_bad_arg<T>
    (arg, rocblas_iamax_strided_batched<T>);
}

template <typename T>
void testing_iamax_strided_batched(const Arguments& arg)
{
  testing_iamax_iamin_strided_batched<T, rocblas_cblas::iamax<T> >
    (arg, rocblas_iamax_strided_batched<T>);
}

template <typename T>
void testing_iamin_strided_batched_bad_arg(const Arguments& arg)
{
  template_testing_iamax_iamin_strided_batched_bad_arg<T>
    (arg, rocblas_iamin_strided_batched<T>);
}

template <typename T>
void testing_iamin_strided_batched(const Arguments& arg)
{
  testing_iamax_iamin_strided_batched<T, rocblas_cblas::iamin<T> >
    (arg, rocblas_iamin_strided_batched<T>);
}

