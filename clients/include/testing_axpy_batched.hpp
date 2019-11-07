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

/* ============================================================================================ */
template <typename T>
void testing_axpy_batched_bad_arg(const Arguments& arg)
{

  rocblas_int
    N           = 100,
    incx        = 1,
    incy        = 1,
    batch_count = arg.batch_count;

  static const size_t safe_size   = 100;
  float               alpha_float = 0.6;
  T                   alpha;
  
  if(std::is_same<T, rocblas_half>{})
    alpha = float_to_half(alpha_float);
  else
    alpha = alpha_float;
  
  rocblas_local_handle handle;
  device_batch_vector<T>
    dx(safe_size,incx,2),
    dy(safe_size,incy,2);
  
  CHECK_HIP_ERROR(dx.memcheck());
  CHECK_HIP_ERROR(dy.memcheck());
  
  EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched<T>(handle,
						N,
						&alpha,
						nullptr,
						incx,
						dy.ptr_on_device(),
						incy,
						batch_count),
			rocblas_status_invalid_pointer);
  
  EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched<T>(handle, N, &alpha, dx.ptr_on_device(), incx,  nullptr, incy,  batch_count),
			rocblas_status_invalid_pointer);
  EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched<T>(handle, N, nullptr, dx.ptr_on_device(), incx,  dy.ptr_on_device(), incy,  batch_count),
			rocblas_status_invalid_pointer);
  EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched<T>(nullptr, N, &alpha, dx.ptr_on_device(), incx,  dy.ptr_on_device(), incy,  batch_count),
			rocblas_status_invalid_handle);

}



template <typename T>
void testing_axpy_batched(const Arguments& arg)
{
  rocblas_int
    N       = arg.N,
    incx    = arg.incx,
    incy    = arg.incy,
    batch_count = arg.batch_count;
  
  
  T h_alpha = arg.get_alpha<T>();
  rocblas_local_handle handle;

  // argument sanity check before allocating invalid memory
  if(N <= 0 || batch_count <= 0)
    {
      static const size_t safe_size = 100; // arbitrarily set to 100
      device_batch_vector<T>
	dx(safe_size,1,3),
	dy(safe_size,1,3);
      
      CHECK_HIP_ERROR(dx.memcheck());
      CHECK_HIP_ERROR(dy.memcheck());
      
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
      EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched<T>(handle, N, &h_alpha, dx.ptr_on_device(), incx,  dy.ptr_on_device(), incy,  batch_count),
			    (N > 0 && batch_count < 0) ? rocblas_status_invalid_size
			    : rocblas_status_success);
      return;
    }

  rocblas_int
    abs_incx = std::abs(incx),
    abs_incy = std::abs(incy);
  //
  // Host memory.
  //
  host_batch_vector<T>
    hx(N,incx,batch_count),
    hy(N,incy,batch_count),
    hsolution(N,incy,batch_count);
  host_vector<T>
    halpha(1);
  

  device_batch_vector<T>
    dx(N,incx,batch_count),
    dy(N,incy,batch_count);
  device_vector<T>
    dalpha(1);

  CHECK_HIP_ERROR(hx.memcheck());
  CHECK_HIP_ERROR(hy.memcheck());
  CHECK_HIP_ERROR(halpha.memcheck());
  CHECK_HIP_ERROR(hsolution.memcheck());
  CHECK_HIP_ERROR(dx.memcheck());
  CHECK_HIP_ERROR(dy.memcheck());
  CHECK_HIP_ERROR(dalpha.memcheck());
  
  halpha[0] = h_alpha;
    
  //
  // Initialize host memory.
  // TODO: add NaN testing when roblas_isnan(arg.alpha) returns true.
  //

  rocblas_init(hx,true);
  rocblas_init(hy,false);

  //
  // Copy to host solution.
  //
  hsolution.copy_from(hy);

  //
  // Device memory.
  //

  double gpu_time_used, cpu_time_used;
  double rocblas_gflops, cblas_gflops, rocblas_bandwidth;
  double rocblas_error_1 = 0.0;
  double rocblas_error_2 = 0.0;
	
  if(arg.unit_check || arg.norm_check)
    {

      //
      // CPU BLAS
      //
      {
        cpu_time_used = get_time_us();
	
	//
	// Compute the host solution.
	//
	for (rocblas_int batch_index = 0;batch_index < batch_count;++batch_index)
	  {
	    cblas_axpy<T>(N,
			  h_alpha,
			  hx[batch_index],
			  incx,
			  hsolution[batch_index],
			  incy);
	  }	
        cpu_time_used = get_time_us() - cpu_time_used;
        cblas_gflops  = axpy_gflop_count<T>(N * batch_count) / cpu_time_used * 1e6;
      }

      //
      // Transfer host to device
      //
      CHECK_HIP_ERROR(dx.transfer_from(hx));

      //
      // Call routine with pointer mode on host.
      //
      {
	//
	// Transfer host to device
	//
	host_batch_vector<T> htmp(N,incy,batch_count);
	htmp.copy_from(hy);
	CHECK_HIP_ERROR(dy.transfer_from(hy));

	//
	// Pointer mode.
	//
	CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

	//
	// Call routine.
	//
	CHECK_ROCBLAS_ERROR(rocblas_axpy_batched<T>(handle,
							    N,
							    halpha,
							    dx.ptr_on_device(),
							    incx,
							    
							    dy.ptr_on_device(),
							    incy,
							    
							    batch_count));
	
	//
	// Transfer from device to host.
	//
	CHECK_HIP_ERROR(hy.transfer_from(dy));

	//
	// Compare with with hsolution.
	//	
        if(arg.unit_check)
	  {
	    for (rocblas_int batch_index=0;batch_index<batch_count;++batch_index)
	      {
		unit_check_general<T>(1,
				      N,
				      abs_incy,
				      hsolution[batch_index],
				      hy[batch_index]);
	      }
	  }
	
        if(arg.norm_check)
	  {
	    for (rocblas_int batch_index=0;batch_index<batch_count;++batch_index)
	      {
	      	double batch_norm = norm_check_general<T>('I', 1, N, abs_incy, hsolution[batch_index], hy[batch_index]);
		rocblas_error_1 = std::max(rocblas_error_1,batch_norm);
	      }
	  }

	//
	// Reinitialize y.
	//
	hy.copy_from(htmp);
      }

      //
      // Call routine with pointer mode on device.
      //
      {
	host_batch_vector<T> htmp(N,incy,batch_count);
	htmp.copy_from(hy);
	CHECK_HIP_ERROR(dy.transfer_from(hy));
	CHECK_HIP_ERROR(dalpha.transfer_from(halpha));
	
	//
	// Pointer mode.
	//
	CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
	
	//
	// Call routine.
	//
	CHECK_ROCBLAS_ERROR(rocblas_axpy_batched<T>(handle,
							    N,
							    dalpha,
							    dx.ptr_on_device(),
							    incx,
							    
							    dy.ptr_on_device(),
							    incy,
							    
							    batch_count));
	
	//
	// Transfer from device to host.
	//
	CHECK_HIP_ERROR(hy.transfer_from(dy));
	
	//
	// Compare with with hsolution.
	//	
        if(arg.unit_check)
	  {
	    for (rocblas_int batch_index=0;batch_index<batch_count;++batch_index)
	      {
		unit_check_general<T>(1, N, abs_incy, hsolution[batch_index], hy[batch_index]);
	      }
	  }
	
        if(arg.norm_check)
	  {
	    for (rocblas_int batch_index=0;batch_index<batch_count;++batch_index)
	      {
		double batch_norm = norm_check_general<T>('I', 1, N, abs_incy, hsolution[batch_index], hy[batch_index]);
		rocblas_error_2 = std::max(rocblas_error_2,batch_norm);
	      }
	  }
	
	//
	// Reinitialize y.
	//
	hy.copy_from(htmp);
      }

    }

  if(arg.timing)
    {
      int number_cold_calls = 2;
      int number_hot_calls  = 100;
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
      	
      //
      // Transfer from host to device.
      //
      CHECK_HIP_ERROR(dy.transfer_from(hy));
	
      //
      // Cold.
      //
      for(int iter = 0; iter < number_cold_calls; iter++)
        {
	  rocblas_axpy_batched<T>(handle, N, &h_alpha, dx.ptr_on_device(), incx,  dy.ptr_on_device(), incy, batch_count);
        }

      //
      // Transfer from host to device.
      //
      CHECK_HIP_ERROR(dy.transfer_from(hy));
      
      gpu_time_used = get_time_us(); // in microseconds      
      for(int iter = 0; iter < number_hot_calls; iter++)
	{
	  rocblas_axpy_batched<T>(handle, N, &h_alpha, dx.ptr_on_device(), incx,  dy.ptr_on_device(), incy, batch_count);
	}      
      gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
      rocblas_gflops    = axpy_gflop_count<T>(N*batch_count) / gpu_time_used * 1e6 * 1;
      rocblas_bandwidth = (3.0 * N) * sizeof(T) / gpu_time_used / 1e3;

      //
      // Report.
      //
      std::cout << "N,alpha,incx,incy,batch,rocblas-Gflops,rocblas-GB/s,rocblas-us";      
      if(arg.norm_check)
	std::cout << "CPU-Gflops,norm_error_host_ptr,norm_error_dev_ptr";      
      std::cout << std::endl;
      std::cout << N << "," << h_alpha << "," << incx << "," << incy  << "," << batch_count << "," << rocblas_gflops
		<< "," << rocblas_bandwidth << "," << gpu_time_used;      
      if(arg.norm_check)
	std::cout << "," << cblas_gflops << ',' << rocblas_error_1 << ',' << rocblas_error_2;      
      std::cout << std::endl;
      
    }

}
