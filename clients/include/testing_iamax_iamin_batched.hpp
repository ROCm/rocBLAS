/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

template <typename T, typename U = T>
void testing_iamax_iamin_batched(const Arguments& arg)
{  
  rocblas_int
    N           = arg.N,
    incx        = arg.incx,
    batch_count = arg.batch_count;

  U           h_alpha     = arg.get_alpha<U>();

  rocblas_local_handle handle;

  size_t size_x = N * size_t(incx);
  
  
  // argument sanity check before allocating invalid memory
  if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
      static const size_t safe_size = 100; // arbitrarily set to 100
      T**                 dx;
      hipMalloc(&dx, sizeof(T*));
      if(!dx)
	{
	  CHECK_HIP_ERROR(hipErrorOutOfMemory);
	  return;
	}

      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
      CHECK_ROCBLAS_ERROR(
			  (rocblas_iamax_iamin_batched<T, U>(handle, N, &h_alpha, dx, incx, batch_count)));
      CHECK_HIP_ERROR(hipFree(dx));
      return;
    }

  // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice

  // Device-arrays of pointers to device memory
  T
    ** dx_1,
    ** dx_2;
  
  device_vector<U> d_alpha(1);
  hipMalloc(&dx_1, batch_count * sizeof(T*));
  hipMalloc(&dx_2, batch_count * sizeof(T*));
  if(!dx_1 || !dx_2 || !d_alpha)
    {
      CHECK_HIP_ERROR(hipErrorOutOfMemory);
      return;
    }

  // Host-arrays of pointers to host memory
  host_vector<T>
    hx_1[batch_count],
    hx_2[batch_count],
    hx_gold[batch_count];

  // Host-arrays of pointers to device memory
  // (intermediate arrays used for the transfers)
  T
    * x_1[batch_count],
    * x_2[batch_count];
  
  for(int i = 0; i < batch_count; i++)
    {
      hx_1[i]    = host_vector<T>(size_x);
      hx_2[i]    = host_vector<T>(size_x);
      hx_gold[i] = host_vector<T>(size_x);

      hipMalloc(&x_1[i], size_x * sizeof(T));
      hipMalloc(&x_2[i], size_x * sizeof(T));
    }

  int last = batch_count - 1;
  if((!x_1[last] && size_x) || (!x_2[last] && size_x))
    {
      CHECK_HIP_ERROR(hipErrorOutOfMemory);
      return;
    }

  // Initial Data on CPU
  rocblas_seedrand();
  for(int i = 0; i < batch_count; i++)
    {
      rocblas_init<T>(hx_1[i], 1, N, incx);

      hx_2[i]    = hx_1[i];
      hx_gold[i] = hx_1[i];
    }

  // copy data from CPU to device, does not work for incx != 1
  // 1. User intermediate arrays to access device memory from host
  for(int i = 0; i < batch_count; i++)
    {
      CHECK_HIP_ERROR(hipMemcpy(x_1[i], hx_1[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
    }
  // 2. Copy intermediate arrays into device arrays
  CHECK_HIP_ERROR(hipMemcpy(dx_1, x_1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

  double
    gpu_time_used,
    cpu_time_used,
    rocblas_gflops,
    cblas_gflops,
    rocblas_bandwidth,
    rocblas_error_1 = double(0.0),
    rocblas_error_2 = double(0.0);

  if(arg.unit_check || arg.norm_check)
    {
      for(int i = 0; i < batch_count; i++)
	{
	  CHECK_HIP_ERROR(hipMemcpy(x_2[i], hx_2[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
	}
      CHECK_HIP_ERROR(hipMemcpy(dx_2, x_2, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
      CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));

      // GPU BLAS, rocblas_pointer_mode_host
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
      CHECK_ROCBLAS_ERROR(
			  (rocblas_iamax_iamin_batched<T, U>(handle, N, &h_alpha, dx_1, incx, batch_count)));

      // GPU BLAS, rocblas_pointer_mode_device
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
      CHECK_ROCBLAS_ERROR(
			  (rocblas_iamax_iamin_batched<T, U>(handle, N, d_alpha, dx_2, incx, batch_count)));

      // copy output from device to CPU
      for(int i = 0; i < batch_count; i++)
	{
	  CHECK_HIP_ERROR(hipMemcpy(hx_1[i], x_1[i], sizeof(T) * size_x, hipMemcpyDeviceToHost));
	  CHECK_HIP_ERROR(hipMemcpy(hx_2[i], x_2[i], sizeof(T) * size_x, hipMemcpyDeviceToHost));
	}

      // CPU BLAS
      cpu_time_used = get_time_us();
      for(int i = 0; i < batch_count; i++)
	{
	  cblas_iamax_iamin<T, U>(N, h_alpha, hx_gold[i], incx);
	}
      cpu_time_used = get_time_us() - cpu_time_used;
      cblas_gflops  = axpy_gflop_count<T>(N) / cpu_time_used * 1e6 * 1;

      if(arg.unit_check)
	{
	  unit_check_general<T>(1, N, batch_count, incx, hx_gold, hx_1);
	  unit_check_general<T>(1, N, batch_count, incx, hx_gold, hx_2);
	}

      if(arg.norm_check)
	{
	  rocblas_error_1 = norm_check_general<T>('F', 1, N, incx, batch_count, hx_gold, hx_1);
	  rocblas_error_2 = norm_check_general<T>('F', 1, N, incx, batch_count, hx_gold, hx_2);
	}

    } // end of if unit/norm check

  if(arg.timing)
    {
      int number_cold_calls = 2;
      int number_hot_calls  = 100;
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

      for(int iter = 0; iter < number_cold_calls; iter++)
	{
	  rocblas_iamax_iamin_batched<T, U>(handle, N, &h_alpha, dx_1, incx, batch_count);
	}

      gpu_time_used = get_time_us(); // in microseconds

      for(int iter = 0; iter < number_hot_calls; iter++)
	{
	  rocblas_iamax_iamin_batched<T, U>(handle, N, &h_alpha, dx_1, incx, batch_count);
	}

      gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
      rocblas_gflops    = axpy_gflop_count<T>(N) / gpu_time_used * 1e6 * 1;
      rocblas_bandwidth = (2.0 * N) * sizeof(T) / gpu_time_used / 1e3;

      std::cout << "N,alpha,incx,rocblas-Gflops,rocblas-GB/s,rocblas-us";

      if(arg.norm_check)
	std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";

      std::cout << std::endl;

      std::cout << N << "," << h_alpha << "," << incx << "," << rocblas_gflops << ","
		<< rocblas_bandwidth << "," << gpu_time_used;

      if(arg.norm_check)
	std::cout << cblas_gflops << ',' << rocblas_error_1 << ',' << rocblas_error_2;

      std::cout << std::endl;
    }

  for(int i = 0; i < batch_count; i++)
    {
      CHECK_HIP_ERROR(hipFree(x_1[i]));
      CHECK_HIP_ERROR(hipFree(x_2[i]));
    }
  CHECK_HIP_ERROR(hipFree(dx_1));
  CHECK_HIP_ERROR(hipFree(dx_2));
}

template <typename T, typename U = T>
void testing_iamax_iamin_batched_bad_arg(const Arguments& arg)
{
  rocblas_int
    N           = 100,
    incx        = 1,
    batch_count = 5;

  U           h_alpha     = U(1.0);
  rocblas_local_handle handle;
  size_t size_x = N * size_t(incx);

  // allocate memory on device
  T** dx;
  hipMalloc(&dx, batch_count * sizeof(T));
  if(!dx)
    {
      CHECK_HIP_ERROR(hipErrorOutOfMemory);
      return;
    }

  EXPECT_ROCBLAS_STATUS((rocblas_iamax_iamin_batched<T, U>)(handle, N, nullptr, dx, incx, batch_count),
			rocblas_status_invalid_pointer);
  EXPECT_ROCBLAS_STATUS(
			(rocblas_iamax_iamin_batched<T, U>)(handle, N, &h_alpha, nullptr, incx, batch_count),
			rocblas_status_invalid_pointer);

  CHECK_HIP_ERROR(hipFree(dx));
}


