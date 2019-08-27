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
namespace rocblas_cblas2
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






template <typename T,
          rocblas_status (&FUNC)(rocblas_handle, rocblas_int, const T* const [], rocblas_int, rocblas_int, rocblas_int*)>
void testing_iamax_iamin_batched_bad_arg(const Arguments& arg)
{
    rocblas_int         N         = 100;
    rocblas_int         incx      = 1;
    rocblas_int         batch_count = 5;
    static const size_t safe_size = 100;

    rocblas_local_handle handle;

    //
    // allocate memory on device
    //
    T** dx;
    hipMalloc(&dx, batch_count * sizeof(T*));
    if(nullptr == dx)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }
    
    rocblas_int h_rocblas_result;

    EXPECT_ROCBLAS_STATUS(FUNC(handle, N, nullptr, incx, batch_count, &h_rocblas_result),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(FUNC(handle, N, dx, incx, batch_count, nullptr),
			  rocblas_status_invalid_pointer);
    
    EXPECT_ROCBLAS_STATUS(FUNC(nullptr, N, dx, incx, batch_count, &h_rocblas_result),
                          rocblas_status_invalid_handle);

}

template <typename T>
void testing_iamax_batched_bad_arg(const Arguments& arg)
{
    testing_iamax_iamin_batched_bad_arg<T, rocblas_iamax_batched<T> >(arg);
}

template <typename T>
void testing_iamin_batched_bad_arg(const Arguments& arg)
{
    testing_iamax_iamin_batched_bad_arg<T, rocblas_iamin_batched<T>>(arg);
}




template <typename T,
          rocblas_status (&FUNC)(rocblas_handle, rocblas_int, const T * const[], rocblas_int, rocblas_int, rocblas_int*),
	  void CBLAS_FUNC(rocblas_int, const T*, rocblas_int, rocblas_int*)>
void testing_iamax_iamin_batched(const Arguments& arg)
{
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
        CHECK_ROCBLAS_ERROR(FUNC(handle, N, dxx, incx, batch_count, &h_rocblas_result_1));
	
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

//
//	for (int i=0;i<N;++i)
//	  {
//	    if (0 == hxx[batch_index][i*incx])
//	      {
//		printf("hxxxxxxxxxxxxxxxx zero %d\n",i);
//	      }
//	  }
//
	
	//     	rocblas_init<T>(hxx[batch_index], 1, N, incx);
      }
    
    //    host_vector<T> hx(size_x);
    host_vector<rocblas_int> hresults(batch_count);
    //    rocblas_int * hresults = new rocblas_int[batch_count];
    host_vector<rocblas_int> hresults2(batch_count);
#if 1

    device_vector<rocblas_int> dresults(batch_count);
    if(nullptr == dresults)
      {
	CHECK_HIP_ERROR(hipErrorOutOfMemory);
	return;
      }
#endif    
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
      CHECK_ROCBLAS_ERROR(FUNC(handle, N, dxx, incx, batch_count, (rocblas_int*)hresults));
      
      // GPU BLAS, rocblas_pointer_mode_device
#if 1

      CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
      CHECK_ROCBLAS_ERROR(FUNC(handle, N, dxx, incx, batch_count, (rocblas_int*)dresults));
      CHECK_HIP_ERROR(hipMemcpy((rocblas_int*)hresults2, (rocblas_int*)dresults, batch_count * sizeof(rocblas_int), hipMemcpyDeviceToHost));
      
#endif
      
      for (int i=0;i<batch_count;++i)
	{
	  rocblas_int cpu_result;
      
	  // CPU BLAS
	  CBLAS_FUNC(N, hxx[i], incx, &cpu_result);
	  cpu_result += 1;
	  if(arg.unit_check)
	    {
#if 0
	      if (cpu_result != hresults[i])
		{
		  printf("diff N %d incx %d batch_count = %d cpu_result %d hresult[%d] = %d hresult2[%d] = %d\n",
			 N,
			 incx,
			 batch_count,
			 cpu_result,
			 i,
			 hresults[i],
			 i,
			 hresults2[i]);
		  
		  for (int j=0;j<30;++j)
		    {
		      std::cout << hxx[i][j] << std::endl; // printf("%f\n",hxx[i]);
		    }
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
	  FUNC(handle, N, dxx, incx, batch_count, d_rocblas_result);
        }

        gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
	  FUNC(handle, N, dxx, incx, batch_count, d_rocblas_result);
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
}



template <typename T>
void testing_iamax_batched(const Arguments& arg)
{
  testing_iamax_iamin_batched<T, rocblas_iamax_batched<T>, cblas_iamax<T> >(arg);
}



template <typename T>
void testing_iamin_batched(const Arguments& arg)
{
  testing_iamax_iamin_batched<T, rocblas_iamin_batched<T>, rocblas_cblas2::cblas_iamin<T> >(arg);
}




//
//template <typename T, typename U = T>
//void testing_iamax_iamin_batchedOOOOOO(const Arguments& arg)
//{
//  rocblas_int N = arg.N, incx = arg.incx, batch_count = arg.batch_count;
//  
//  U h_alpha = arg.get_alpha<U>();
//
//    rocblas_local_handle handle;
//    
//    size_t size_x = N * size_t(incx);
//    
//    // argument sanity check before allocating invalid memory
//    if(N <= 0 || incx <= 0 || batch_count <= 0)
//    {
//        static const size_t safe_size = 100; // arbitrarily set to 100
//        T**                 dx;
//        hipMalloc(&dx, sizeof(T*));
//        if(!dx)
//        {
//            CHECK_HIP_ERROR(hipErrorOutOfMemory);
//            return;
//        }
//
//        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
//        CHECK_ROCBLAS_ERROR(
//			    (rocblas_iamax_batched<T>(handle, N, &h_alpha, dx, incx, batch_count, nullptr)));
//        CHECK_HIP_ERROR(hipFree(dx));
//        return;
//    }
//
//    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
//
//    // Device-arrays of pointers to device memory
//    T **dx_1, **dx_2;
//
//    device_vector<U> d_alpha(1);
//    hipMalloc(&dx_1, batch_count * sizeof(T*));
//    hipMalloc(&dx_2, batch_count * sizeof(T*));
//    if(!dx_1 || !dx_2 || !d_alpha)
//    {
//        CHECK_HIP_ERROR(hipErrorOutOfMemory);
//        return;
//    }
//
//    // Host-arrays of pointers to host memory
//    host_vector<T> hx_1[batch_count], hx_2[batch_count], hx_gold[batch_count];
//
//    // Host-arrays of pointers to device memory
//    // (intermediate arrays used for the transfers)
//    T *x_1[batch_count], *x_2[batch_count];
//
//    for(int i = 0; i < batch_count; i++)
//    {
//        hx_1[i]    = host_vector<T>(size_x);
//        hx_2[i]    = host_vector<T>(size_x);
//        hx_gold[i] = host_vector<T>(size_x);
//
//        hipMalloc(&x_1[i], size_x * sizeof(T));
//        hipMalloc(&x_2[i], size_x * sizeof(T));
//    }
//
//    int last = batch_count - 1;
//    if((!x_1[last] && size_x) || (!x_2[last] && size_x))
//    {
//        CHECK_HIP_ERROR(hipErrorOutOfMemory);
//        return;
//    }
//
//    // Initial Data on CPU
//    rocblas_seedrand();
//    for(int i = 0; i < batch_count; i++)
//    {
//        rocblas_init<T>(hx_1[i], 1, N, incx);
//
//        hx_2[i]    = hx_1[i];
//        hx_gold[i] = hx_1[i];
//    }
//
//    // copy data from CPU to device, does not work for incx != 1
//    // 1. User intermediate arrays to access device memory from host
//    for(int i = 0; i < batch_count; i++)
//    {
//        CHECK_HIP_ERROR(hipMemcpy(x_1[i], hx_1[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
//    }
//    // 2. Copy intermediate arrays into device arrays
//    CHECK_HIP_ERROR(hipMemcpy(dx_1, x_1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
//
//    double gpu_time_used, cpu_time_used, rocblas_gflops, cblas_gflops, rocblas_bandwidth,
//        rocblas_error_1 = double(0.0), rocblas_error_2 = double(0.0);
//
//    if(arg.unit_check || arg.norm_check)
//    {
//        for(int i = 0; i < batch_count; i++)
//        {
//            CHECK_HIP_ERROR(hipMemcpy(x_2[i], hx_2[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
//        }
//        CHECK_HIP_ERROR(hipMemcpy(dx_2, x_2, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
//        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));
//
//        // GPU BLAS, rocblas_pointer_mode_host
//        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
//        CHECK_ROCBLAS_ERROR(
//            (rocblas_iamax_batched<T>(handle, N, &h_alpha, dx_1, incx, batch_count, nullptr)));
//
//        // GPU BLAS, rocblas_pointer_mode_device
//        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
//        CHECK_ROCBLAS_ERROR(
//            (rocblas_iamax_batched<T>(handle, N, d_alpha, dx_2, incx, batch_count, nullptr)));
//
//        // copy output from device to CPU
//        for(int i = 0; i < batch_count; i++)
//        {
//            CHECK_HIP_ERROR(hipMemcpy(hx_1[i], x_1[i], sizeof(T) * size_x, hipMemcpyDeviceToHost));
//            CHECK_HIP_ERROR(hipMemcpy(hx_2[i], x_2[i], sizeof(T) * size_x, hipMemcpyDeviceToHost));
//        }
//
//        // CPU BLAS
//	
//        cpu_time_used = get_time_us();
//#if 0
//        for(int i = 0; i < batch_count; i++)
//        {
//            cblas_iamax_iamin<T, U>(N, h_alpha, hx_gold[i], incx);
//        }
//        cpu_time_used = get_time_us() - cpu_time_used;
//        cblas_gflops  = axpy_gflop_count<T>(N) / cpu_time_used * 1e6 * 1;
//#endif
//
//        if(arg.unit_check)
//        {
//            unit_check_general<T>(1, N, batch_count, incx, hx_gold, hx_1);
//            unit_check_general<T>(1, N, batch_count, incx, hx_gold, hx_2);
//        }
//
//        if(arg.norm_check)
//        {
//            rocblas_error_1 = norm_check_general<T>('F', 1, N, incx, batch_count, hx_gold, hx_1);
//            rocblas_error_2 = norm_check_general<T>('F', 1, N, incx, batch_count, hx_gold, hx_2);
//        }
//
//    } // end of if unit/norm check
//
//    if(arg.timing)
//    {
//        int number_cold_calls = 2;
//        int number_hot_calls  = 100;
//        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
//
//        for(int iter = 0; iter < number_cold_calls; iter++)
//        {
//            rocblas_iamax_batched<T>(handle, N, &h_alpha, dx_1, incx, batch_count, nullptr);
//        }
//
//        gpu_time_used = get_time_us(); // in microseconds
//
//        for(int iter = 0; iter < number_hot_calls; iter++)
//        {
//            rocblas_iamax_batched<T>(handle, N, &h_alpha, dx_1, incx, batch_count, nullptr);
//        }
//
//        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
//	//        rocblas_gflops    = axpy_gflop_count<T>(N) / gpu_time_used * 1e6 * 1;
//        rocblas_bandwidth = (2.0 * N) * sizeof(T) / gpu_time_used / 1e3;
//
//        std::cout << "N,alpha,incx,rocblas-Gflops,rocblas-GB/s,rocblas-us";
//
//        if(arg.norm_check)
//            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";
//
//        std::cout << std::endl;
//
//        std::cout << N << "," << h_alpha << "," << incx << "," << rocblas_gflops << ","
//                  << rocblas_bandwidth << "," << gpu_time_used;
//
//        if(arg.norm_check)
//            std::cout << cblas_gflops << ',' << rocblas_error_1 << ',' << rocblas_error_2;
//
//        std::cout << std::endl;
//    }
//
//    for(int i = 0; i < batch_count; i++)
//    {
//        CHECK_HIP_ERROR(hipFree(x_1[i]));
//        CHECK_HIP_ERROR(hipFree(x_2[i]));
//    }
//    CHECK_HIP_ERROR(hipFree(dx_1));
//    CHECK_HIP_ERROR(hipFree(dx_2));
//}
//
//
//
//
//
//


//// CBLAS does not have a cblas_iamin function, so we write our own version of it
//namespace rocblas_cblas
//{
//    template <typename T>
//    T asum(T x)
//    {
//        return x < 0 ? -x : x;
//    }
//
//    rocblas_half asum(rocblas_half x)
//    {
//        return x & 0x7fff;
//    }
//
//    template <typename T>
//    bool lessthan(T x, T y)
//    {
//        return x < y;
//    }
//
//    bool lessthan(rocblas_half x, rocblas_half y)
//    {
//        return half_to_float(x) < half_to_float(y);
//    }
//
//    template <typename T>
//    void cblas_iamin(rocblas_int N, const T* X, rocblas_int incx, rocblas_int* result)
//    {
//        rocblas_int minpos = -1;
//        if(N > 0 && incx > 0)
//        {
//            auto min = asum(X[0]);
//            minpos   = 0;
//            for(size_t i = 1; i < N; ++i)
//            {
//                auto a = asum(X[i * incx]);
//                if(lessthan(a, min))
//                {
//                    min    = a;
//                    minpos = i;
//                }
//            }
//        }
//        *result = minpos;
//    }
//
//} // namespace rocblas_cblas
//


///* ************************************************************************
// * Copyright 2018-2019 Advanced Micro Devices, Inc.
// * ************************************************************************ */
//
//#include "cblas_interface.hpp"
//#include "flops.hpp"
//#include "norm.hpp"
//#include "rocblas.hpp"
//#include "rocblas_init.hpp"
//#include "rocblas_math.hpp"
//#include "rocblas_random.hpp"
//#include "rocblas_test.hpp"
//#include "rocblas_vector.hpp"
//#include "unit.hpp"
//#include "utility.hpp"
//
//// #include "rocblas_iamax_iamin_batched.h"
//
//template <typename T, typename U = T>
//void testing_iamax_iamin_batched(const Arguments& arg)
//{
//  rocblas_int N = arg.N, incx = arg.incx, batch_count = arg.batch_count;
//  
//  U h_alpha = arg.get_alpha<U>();
//
//    rocblas_local_handle handle;
//    
//    size_t size_x = N * size_t(incx);
//    
//    // argument sanity check before allocating invalid memory
//    if(N <= 0 || incx <= 0 || batch_count <= 0)
//    {
//        static const size_t safe_size = 100; // arbitrarily set to 100
//        T**                 dx;
//        hipMalloc(&dx, sizeof(T*));
//        if(!dx)
//        {
//            CHECK_HIP_ERROR(hipErrorOutOfMemory);
//            return;
//        }
//
//        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
//        CHECK_ROCBLAS_ERROR(
//			    (rocblas_iamax_iamin_batched<T>(handle, N, &h_alpha, dx, incx, batch_count, nullptr)));
//        CHECK_HIP_ERROR(hipFree(dx));
//        return;
//    }
//
//    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
//
//    // Device-arrays of pointers to device memory
//    T **dx_1, **dx_2;
//
//    device_vector<U> d_alpha(1);
//    hipMalloc(&dx_1, batch_count * sizeof(T*));
//    hipMalloc(&dx_2, batch_count * sizeof(T*));
//    if(!dx_1 || !dx_2 || !d_alpha)
//    {
//        CHECK_HIP_ERROR(hipErrorOutOfMemory);
//        return;
//    }
//
//    // Host-arrays of pointers to host memory
//    host_vector<T> hx_1[batch_count], hx_2[batch_count], hx_gold[batch_count];
//
//    // Host-arrays of pointers to device memory
//    // (intermediate arrays used for the transfers)
//    T *x_1[batch_count], *x_2[batch_count];
//
//    for(int i = 0; i < batch_count; i++)
//    {
//        hx_1[i]    = host_vector<T>(size_x);
//        hx_2[i]    = host_vector<T>(size_x);
//        hx_gold[i] = host_vector<T>(size_x);
//
//        hipMalloc(&x_1[i], size_x * sizeof(T));
//        hipMalloc(&x_2[i], size_x * sizeof(T));
//    }
//
//    int last = batch_count - 1;
//    if((!x_1[last] && size_x) || (!x_2[last] && size_x))
//    {
//        CHECK_HIP_ERROR(hipErrorOutOfMemory);
//        return;
//    }
//
//    // Initial Data on CPU
//    rocblas_seedrand();
//    for(int i = 0; i < batch_count; i++)
//    {
//        rocblas_init<T>(hx_1[i], 1, N, incx);
//
//        hx_2[i]    = hx_1[i];
//        hx_gold[i] = hx_1[i];
//    }
//
//    // copy data from CPU to device, does not work for incx != 1
//    // 1. User intermediate arrays to access device memory from host
//    for(int i = 0; i < batch_count; i++)
//    {
//        CHECK_HIP_ERROR(hipMemcpy(x_1[i], hx_1[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
//    }
//    // 2. Copy intermediate arrays into device arrays
//    CHECK_HIP_ERROR(hipMemcpy(dx_1, x_1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
//
//    double gpu_time_used, cpu_time_used, rocblas_gflops, cblas_gflops, rocblas_bandwidth,
//        rocblas_error_1 = double(0.0), rocblas_error_2 = double(0.0);
//
//    if(arg.unit_check || arg.norm_check)
//    {
//        for(int i = 0; i < batch_count; i++)
//        {
//            CHECK_HIP_ERROR(hipMemcpy(x_2[i], hx_2[i], sizeof(T) * size_x, hipMemcpyHostToDevice));
//        }
//        CHECK_HIP_ERROR(hipMemcpy(dx_2, x_2, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
//        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));
//
//        // GPU BLAS, rocblas_pointer_mode_host
//        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
//        CHECK_ROCBLAS_ERROR(
//            (rocblas_iamax_iamin_batched<T>(handle, N, &h_alpha, dx_1, incx, batch_count, nullptr)));
//
//        // GPU BLAS, rocblas_pointer_mode_device
//        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
//        CHECK_ROCBLAS_ERROR(
//            (rocblas_iamax_iamin_batched<T>(handle, N, d_alpha, dx_2, incx, batch_count, nullptr)));
//
//        // copy output from device to CPU
//        for(int i = 0; i < batch_count; i++)
//        {
//            CHECK_HIP_ERROR(hipMemcpy(hx_1[i], x_1[i], sizeof(T) * size_x, hipMemcpyDeviceToHost));
//            CHECK_HIP_ERROR(hipMemcpy(hx_2[i], x_2[i], sizeof(T) * size_x, hipMemcpyDeviceToHost));
//        }
//
//        // CPU BLAS
//        cpu_time_used = get_time_us();
//        for(int i = 0; i < batch_count; i++)
//        {
//            cblas_iamax_iamin<T, U>(N, h_alpha, hx_gold[i], incx);
//        }
//        cpu_time_used = get_time_us() - cpu_time_used;
//        cblas_gflops  = axpy_gflop_count<T>(N) / cpu_time_used * 1e6 * 1;
//
//        if(arg.unit_check)
//        {
//            unit_check_general<T>(1, N, batch_count, incx, hx_gold, hx_1);
//            unit_check_general<T>(1, N, batch_count, incx, hx_gold, hx_2);
//        }
//
//        if(arg.norm_check)
//        {
//            rocblas_error_1 = norm_check_general<T>('F', 1, N, incx, batch_count, hx_gold, hx_1);
//            rocblas_error_2 = norm_check_general<T>('F', 1, N, incx, batch_count, hx_gold, hx_2);
//        }
//
//    } // end of if unit/norm check
//
//    if(arg.timing)
//    {
//        int number_cold_calls = 2;
//        int number_hot_calls  = 100;
//        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
//
//        for(int iter = 0; iter < number_cold_calls; iter++)
//        {
//            rocblas_iamax_iamin_batched<T>(handle, N, &h_alpha, dx_1, incx, batch_count, nullptr);
//        }
//
//        gpu_time_used = get_time_us(); // in microseconds
//
//        for(int iter = 0; iter < number_hot_calls; iter++)
//        {
//            rocblas_iamax_iamin_batched<T>(handle, N, &h_alpha, dx_1, incx, batch_count, nullptr);
//        }
//
//        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
//        rocblas_gflops    = axpy_gflop_count<T>(N) / gpu_time_used * 1e6 * 1;
//        rocblas_bandwidth = (2.0 * N) * sizeof(T) / gpu_time_used / 1e3;
//
//        std::cout << "N,alpha,incx,rocblas-Gflops,rocblas-GB/s,rocblas-us";
//
//        if(arg.norm_check)
//            std::cout << ",CPU-Gflops,norm_error_host_ptr,norm_error_device_ptr";
//
//        std::cout << std::endl;
//
//        std::cout << N << "," << h_alpha << "," << incx << "," << rocblas_gflops << ","
//                  << rocblas_bandwidth << "," << gpu_time_used;
//
//        if(arg.norm_check)
//            std::cout << cblas_gflops << ',' << rocblas_error_1 << ',' << rocblas_error_2;
//
//        std::cout << std::endl;
//    }
//
//    for(int i = 0; i < batch_count; i++)
//    {
//        CHECK_HIP_ERROR(hipFree(x_1[i]));
//        CHECK_HIP_ERROR(hipFree(x_2[i]));
//    }
//    CHECK_HIP_ERROR(hipFree(dx_1));
//    CHECK_HIP_ERROR(hipFree(dx_2));
//}
//
//template <typename T, typename U = T>
//void testing_iamax_iamin_batched_bad_arg(const Arguments& arg)
//{
//    rocblas_int N = 100, incx = 1, batch_count = 5;
//
//    U                    h_alpha = U(1.0);
//    rocblas_local_handle handle;
//    size_t               size_x = N * size_t(incx);
//
//    // allocate memory on device
//    T** dx;
//    hipMalloc(&dx, batch_count * sizeof(T));
//    if(!dx)
//    {
//        CHECK_HIP_ERROR(hipErrorOutOfMemory);
//        return;
//    }
//
//    EXPECT_ROCBLAS_STATUS(
//        (rocblas_iamax_iamin_batched<T>)(handle, N, nullptr, dx, incx, batch_count, nullptr),
//        rocblas_status_invalid_pointer);
//    EXPECT_ROCBLAS_STATUS(
//        (rocblas_iamax_iamin_batched<T>)(handle, N, &h_alpha, nullptr, incx, batch_count, nullptr),
//        rocblas_status_invalid_pointer);
//
//    CHECK_HIP_ERROR(hipFree(dx));
//}
//
