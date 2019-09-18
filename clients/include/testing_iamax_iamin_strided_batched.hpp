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

namespace rocblas_cblas3
{
    template <typename T>
    T asum(T x)
    {
        return x < 0 ? -x : x;
    }

    rocblas_half asum(rocblas_half x)
    {
        return x & 0x7fff;
    }

    template <typename T>
    bool lessthan(T x, T y)
    {
        return x < y;
    }

    bool lessthan(rocblas_half x, rocblas_half y)
    {
        return half_to_float(x) < half_to_float(y);
    }

    template <typename T>
    void cblas_iamin(rocblas_int N, const T* X, rocblas_int incx, rocblas_int* result)
    {
        rocblas_int minpos = -1;
        if(N > 0 && incx > 0)
        {
            auto min = asum(X[0]);
            minpos   = 0;
            for(size_t i = 1; i < N; ++i)
            {
                auto a = asum(X[i * incx]);
                if(lessthan(a, min))
                {
                    min    = a;
                    minpos = i;
                }
            }
        }
        *result = minpos;
    }

} // namespace rocblas_cblas

template <typename T>
void testing_iamax_iamin_strided_batched_bad_arg(const Arguments& arg,
						 rocblas_iamax_iamin_strided_batched_t<T> func)
{
  rocblas_int         N           = 100;
  rocblas_int         incx        = 1;
  rocblas_int         batch_count = 5;
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
  
  EXPECT_ROCBLAS_STATUS(func(handle, N, nullptr, incx, incx * N, batch_count, &h_rocblas_result, 1),
			rocblas_status_invalid_pointer);
  
  EXPECT_ROCBLAS_STATUS(func(handle, N, dx, incx, incx * N, batch_count, nullptr, 1),
			  rocblas_status_invalid_pointer);
  
  EXPECT_ROCBLAS_STATUS(func(nullptr, N, dx, incx, incx * N, batch_count, &h_rocblas_result, 1),
			rocblas_status_invalid_handle);
}

template <typename T>
void testing_iamax_strided_batched_bad_arg(const Arguments& arg)
{
  testing_iamax_iamin_strided_batched_bad_arg<T>(arg, rocblas_iamax_strided_batched<T>);
}

template <typename T>
void testing_iamin_strided_batched_bad_arg(const Arguments& arg)
{
  testing_iamax_iamin_strided_batched_bad_arg<T>(arg, rocblas_iamin_strided_batched<T>);
}

template <typename T,
	  void CBLAS_FUNC(rocblas_int, const T*, rocblas_int, rocblas_int*)>
void testing_iamax_iamin_strided_batched(const Arguments& arg, rocblas_iamax_iamin_strided_batched_t<T> func)
{
#if 0
    rocblas_int N    = arg.N;
    rocblas_int incx = arg.incx;
    rocblas_int batch_count = arg.batch_count;
    rocblas_int h_rocblas_result_1;
    rocblas_int h_rocblas_result_2;

    rocblas_int rocblas_error_1;
    rocblas_int rocblas_error_2;

    rocblas_local_handle handle;
    T** dxx = nullptr;
    
    //
    // Argument sanity check before allocating invalid memory.
    //
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        static const size_t safe_size = 100; // arbritrarily set to 100

        hipMalloc(&dxx, sizeof(T*));
        if(nullptr == dxx)
	  {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
	  }
	
	// printf("CHECK INVALID: N = %d, incx = %d, batch_count = %d\n", N, incx, batch_count);

	//
	// Only need to provide a non-null address.
	//
        CHECK_ROCBLAS_ERROR(func(handle, N, dxx, incx, batch_count, &h_rocblas_result_1));
	
#ifdef GOOGLE_TEST
        EXPECT_EQ(h_rocblas_result_1, 0);
#endif
        return;
    }
        
    size_t size_x = size_t(N) * incx;

    //
    // Allocate the device jagged array.
    //
    hipMalloc(&dxx, sizeof(T*) * batch_count);
    if(nullptr == dxx)
      {
	CHECK_HIP_ERROR(hipErrorOutOfMemory);
	return;
      }
    
    for (int batch_index=0;batch_index<batch_count;++batch_index)
      {
	CHECK_HIP_ERROR(hipMalloc(&dxx[batch_index], sizeof(T) * size_x));	
      }

    //
    // Init the seed rand.
    //
    rocblas_seedrand();

    //
    // Naming: dxx is in GPU (device) memory. hx is in CPU (host) memory, plz
    // follow this practice
    //
    T**hxx = new T*[batch_count];    
    for (int batch_index=0;batch_index<batch_count;++batch_index)
      {
	hxx[batch_index] = new T[size_x];

	//
	// 1,N,incx
	//
	for (int i=0;i<N;++i)
	  {
	    hxx[batch_index][i*incx] = random_generator<T>();
	  }
      }


    //    printf("CHECK INVALID: N = %d, incx = %d, batch_count = %d\n", N, incx, batch_count);
    
    //    host_vector<T> hx(size_x);
    host_vector<rocblas_int> hresults(batch_count);
    //    rocblas_int * hresults = new rocblas_int[batch_count];
    host_vector<rocblas_int> hresults2(batch_count);
    device_vector<rocblas_int> dresults(batch_count);
    if(nullptr == dresults)
      {
	CHECK_HIP_ERROR(hipErrorOutOfMemory);
	return;
      }

    //
    // Initial Data on CPU.
    //

    //
    // Copy data from CPU to device.
    //
    for (int batch_index=0;batch_index<batch_count;++batch_index)
      {
	CHECK_HIP_ERROR(hipMemcpy(dxx[batch_index], hxx[batch_index], sizeof(T) * size_x, hipMemcpyHostToDevice));
      }

    double gpu_time_used, cpu_time_used;
    if(arg.unit_check || arg.norm_check)
    {      
      // GPU BLAS rocblas_pointer_mode_host      
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));      
      CHECK_ROCBLAS_ERROR(func(handle, N, dxx, incx, batch_count, (rocblas_int*)hresults));
      
      // GPU BLAS, rocblas_pointer_mode_device
      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
      CHECK_ROCBLAS_ERROR(func(handle, N, dxx, incx, batch_count, (rocblas_int*)dresults));
      CHECK_HIP_ERROR(hipMemcpy((rocblas_int*)hresults2, (rocblas_int*)dresults, batch_count * sizeof(rocblas_int), hipMemcpyDeviceToHost));
      
      for (int i=0;i<batch_count;++i)
	{
	  rocblas_int cpu_result;
      
	  // CPU BLAS
	  CBLAS_FUNC(N, hxx[i], incx, &cpu_result);
	  cpu_result += 1;
	  if(arg.unit_check)
	    {
#if 0
	      if ( (cpu_result != hresults[i]) ||  (cpu_result != hresults2[i]) )
		{
		  int nn = (N < (cpu_result+1)) ? N : (cpu_result+1);
		  for (int ii=0;ii<nn;++ii)
		    {
		      std::cout << "BATCHED INPUT[" << i << "]["<< ii << "]" << hxx[i][ii*incx] << "    " << std::endl;
		    }
		}
#endif	      
#if 1
	      if (cpu_result != hresults[i])
		{
#if 0
		  printf("MODE HOST N %d incx %d batch_count = %d cpu_result %d hresult[%d] = %d\n",
			 N,
			 incx,
			 batch_count,
			 cpu_result,
			 i,
			 hresults[i]);
#endif

#if 0
		  int nn  = (N < (cpu_result+1)) ? N : (cpu_result+1);
		  for (int j=0;j<nn;++j)
		    {
		      std::cout << hxx[i][j] << std::endl;
		    }
#endif
		}
#endif
#if 1
	      if (cpu_result != hresults2[i])
		{
#if 0
		  printf("MODE DEVICE N %d incx %d batch_count = %d cpu_result %d hresult[%d] = %d\n",
			 N,
			 incx,
			 batch_count,
			 cpu_result,
			 i,
			 hresults2[i]);
#endif
#if 0
		  int nn  = (N < 17) ? N : 17;
		  for (int j=0;j<nn;++j)
		    {
		      std::cout << hxx[i][j] << std::endl;
		    }
#endif
		}
#endif
	      unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &hresults[i]); // leads to an error on amin	      
	      unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &hresults2[i]);
	    }
	  
	}
      
#if 0
	cpu_time_used = get_time_us();
        rocblas_int cpu_result;
	
        // CPU BLAS
        CBLAS_FUNC(N, hx, incx, &cpu_result);

        cpu_time_used = get_time_us() - cpu_time_used;
        cpu_result += 1; // make index 1 based as in Fortran BLAS, not 0 based as in CBLAS

        if(arg.unit_check)
        {
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &h_rocblas_result_1);
            unit_check_general<rocblas_int>(1, 1, 1, &cpu_result, &h_rocblas_result_2);
        }

        if(arg.norm_check)
        {
            rocblas_error_1 = h_rocblas_result_1 - cpu_result;
            rocblas_error_2 = h_rocblas_result_2 - cpu_result;
        }
	
#endif
	
    }

    //    hipFree(dresults);
    
#if 0  

    if(arg.timing)
    {
#if 0
        int number_cold_calls = 2;
        int number_hot_calls  = 100;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
	  func(handle, N, dxx, incx, batch_count, d_rocblas_result);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
	  func(handle, N, dxx, incx, batch_count, d_rocblas_result);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        std::cout << "N,incx,rocblas-us";

        if(arg.norm_check)
            std::cout << ",cpu_time_used,rocblas_error_host_ptr,rocblas_error_dev_ptr";

        std::cout << std::endl;

        std::cout << (int)N << "," << incx << "," << gpu_time_used;

        if(arg.norm_check)
            std::cout << "," << cpu_time_used << "," << rocblas_error_1 << "," << rocblas_error_2;

        std::cout << std::endl;
#endif
    }
#endif


    
    
    for (int batch_index=0;batch_index<batch_count;++batch_index)
      {
	hipFree(dxx[batch_index]);
      }
    hipFree(dxx);
#endif
}



template <typename T>
void testing_iamax_strided_batched(const Arguments& arg)
{
  testing_iamax_iamin_strided_batched<T, cblas_iamax<T> >(arg,
							  rocblas_iamax_strided_batched<T>);
}

template <typename T>
void testing_iamin_strided_batched(const Arguments& arg)
{
  testing_iamax_iamin_strided_batched<T, rocblas_cblas3::cblas_iamin<T> >(arg,
									  rocblas_iamin_strided_batched<T>);
}
