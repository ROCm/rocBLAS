/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"



//!
//! @brief Kernel for all the versions (batched, strided batched) of axpy.
//!
template <typename T, typename A,typename X,typename Y>
__global__ void axpy_kernel(rocblas_int n,
			    A alpha_device_host,
			    X x,
			    rocblas_int incx,
			    ptrdiff_t offsetx,
			    rocblas_stride stridex,
			    Y y,
			    rocblas_int incy,
			    ptrdiff_t offsety,
			    rocblas_stride stridey)
{
  auto  alpha = load_scalar(alpha_device_host);
  //T  alpha = load_scalar(alpha_device_host);
  ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if(tid < n)
    {
      const T*  tx     = (const T*)load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
      T*ty     = (T*)load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);
      *ty += alpha * (*tx);
    }
}

#if 0
//!
//! @brief Kernel for all the versions (batched, strided batched) of axpy.
//!
template <typename A,typename X,typename Y>
__global__ void axpy_kernel(rocblas_int n,
			    A alpha_device_host,
			    X x,
			    rocblas_int incx,
			    ptrdiff_t offsetx,
			    rocblas_stride stridex,
			    Y y,
			    rocblas_int incy,
			    ptrdiff_t offsety,
			    rocblas_stride stridey)
{
  auto  alpha = load_scalar(alpha_device_host);
  ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if(tid < n)
    {
      auto   tx     = load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
      auto   ty     = load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);
      *ty += alpha * (*tx);
    }
}
#endif

#if 0
//!
//! @brief Kernel for all the versions (batched, strided batched) of axpy.
//!
template <typename X,typename Y>
__global__ void axpy_kernel(rocblas_int n,
			    const _Float16 * alpha_device_host,
			    X x,
			    rocblas_int incx,
			    ptrdiff_t offsetx,
			    rocblas_stride stridex,
			    Y y,
			    rocblas_int incy,
			    ptrdiff_t offsety,
			    rocblas_stride stridey)
{
  auto  alpha = load_scalar(alpha_device_host);
  ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if(tid < n)
    {
      // auto   tx     = load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
      //      auto   ty     = load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);

      const _Float16* tx     = (const _Float16*)load_ptr_batch(x, hipBlockIdx_y, offsetx + tid * incx, stridex);
      _Float16* ty     = ( _Float16*)load_ptr_batch(y, hipBlockIdx_y, offsety + tid * incy, stridey);
      
      *ty += alpha * (*tx);
    }
}

#endif


//!
//! @brief Optmized kernel for the groups of 8 half floating points.
//!
template <typename A, typename X,typename Y>
__global__ void haxpy_mlt_8_kernel(rocblas_int n_mlt_8, A alpha_device_host, X x, rocblas_stride stridex, Y y,rocblas_stride stridey )
{
  ptrdiff_t  t8id      = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  auto alpha_h2 = load_scalar(alpha_device_host);
    
  rocblas_half2 y0, y1, y2, y3;
  rocblas_half2 x0, x1, x2, x3;
  rocblas_half2 z0, z1, z2, z3;

  auto tid = t8id * 8;
  if(tid < n_mlt_8)
    {
      const rocblas_half8 * ax  = (const rocblas_half8 *)load_ptr_batch(x, hipBlockIdx_y, tid, stridex);
      rocblas_half8 * ay     	= (rocblas_half8 *)load_ptr_batch(y, hipBlockIdx_y, tid, stridey);
      
      y0[0] = (*ay)[0];
      y0[1] = (*ay)[1];
      y1[0] = (*ay)[2];
      y1[1] = (*ay)[3];
      y2[0] = (*ay)[4];
      y2[1] = (*ay)[5];
      y3[0] = (*ay)[6];
      y3[1] = (*ay)[7];

      x0[0] = (*ax)[0];
      x0[1] = (*ax)[1];
      x1[0] = (*ax)[2];
      x1[1] = (*ax)[3];
      x2[0] = (*ax)[4];
      x2[1] = (*ax)[5];
      x3[0] = (*ax)[6];
      x3[1] = (*ax)[7];

      z0 = rocblas_fmadd_half2(alpha_h2, x0, y0);
      z1 = rocblas_fmadd_half2(alpha_h2, x1, y1);
      z2 = rocblas_fmadd_half2(alpha_h2, x2, y2);
      z3 = rocblas_fmadd_half2(alpha_h2, x3, y3);

      (*ay)[0] = z0[0];
      (*ay)[1] = z0[1];
      (*ay)[2] = z1[0];
      (*ay)[3] = z1[1];
      (*ay)[4] = z2[0];
      (*ay)[5] = z2[1];
      (*ay)[6] = z3[0];
      (*ay)[7] = z3[1];
    }
}



//!
//! @brief Optmized kernel for the remaning part of 8 half floating points.
//!
template <typename A,typename X,typename Y>
__global__ void haxpy_mod_8_kernel(rocblas_int n_mod_8,
				   A alpha_device_host,
				   X x,
				   ptrdiff_t offsetx,
				   rocblas_stride stridex,
				   Y y,
				   ptrdiff_t offsety,
				   rocblas_stride stridey)
{
  auto  alpha = load_scalar(alpha_device_host);
  ptrdiff_t tid   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if(tid < n_mod_8)
    {
      const _Float16*   tx     = (const _Float16*)load_ptr_batch(x, hipBlockIdx_y, offsetx + tid, stridex);
      _Float16*   ty           = (_Float16*)load_ptr_batch(y, hipBlockIdx_y, offsety + tid, stridey);
      *ty += alpha * (*tx);      
    }
}



//!
//! @brief General template.
//!
template <    int NB,
	      typename A,
	      typename X,
	      typename Y >
static rocblas_status axpy_template(rocblas_handle      handle,
				    rocblas_int         n,
				    const A*            alpha,
				    X                   x,
				    rocblas_int         incx,
				    rocblas_stride      stridex,
				    Y                   y,
				    rocblas_int         incy,							      
				    rocblas_stride      stridey,
				    rocblas_int         batch_count)
{
  if(!handle)
    {
      return rocblas_status_invalid_handle;
    }
  
  RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
  if(!alpha)
    {
      return rocblas_status_invalid_pointer;
    }
  
  if(!x || !y)
    {
      return rocblas_status_invalid_pointer;
    }
  
  if(n <= 0 || batch_count == 0) // Quick return if possible. Not Argument error
    {
      return rocblas_status_success;
    }

  if (batch_count < 0)
    {
      return rocblas_status_invalid_size;      
    }

  ptrdiff_t offsetx = (incx < 0) ? ptrdiff_t(incx) * (1 - n) : 0;
  ptrdiff_t offsety = (incy < 0) ? ptrdiff_t(incy) * (1 - n) : 0;

  static constexpr bool is_rocblas_half = std::is_same<A, rocblas_half>::value;
  if (!is_rocblas_half || ( is_rocblas_half && (incx != 1 || incy != 1) ))
    {
      // Default calculation 
      dim3        blocks( (n - 1) / NB + 1 ,
			  batch_count);
      dim3        threads(NB);
      if (is_rocblas_half)
	{
	  //	  std::cout << "normal calculation "  << std::endl;
	  if(handle->pointer_mode == rocblas_pointer_mode_device)	    
	    {
	      //	  std::cout << "normal device "  << std::endl;
	  hipLaunchKernelGGL(axpy_kernel<_Float16>,
				 blocks,
				 threads,
				 0,
				 handle->rocblas_stream,
				 n,
				 (const _Float16*)alpha,
				 x,
				 incx,
				 offsetx,
				 stridex,
				 y,
				 incy,
				 offsety,
				 stridey);
	      
	    }
	  else
	    {
	      //	      std::cout << "normal host "  << std::endl;
	      hipLaunchKernelGGL(axpy_kernel<_Float16>,
				 blocks,
				 threads,
				 0,
				 handle->rocblas_stream,
				 n,
				 *(const _Float16*)alpha,
				 x,
				 incx,
				 stridex,
				 offsetx,
				 y,
				 incy,
				 offsety,
				 stridey);
	    }
	  
	}
      else
	{
	  if(handle->pointer_mode == rocblas_pointer_mode_device)
	    {
	      hipLaunchKernelGGL(axpy_kernel<A>,
				 blocks,
				 threads,
				 0,
				 handle->rocblas_stream,
				 n,
				 alpha,
				 x,
				 incx,
				 offsetx,
				 stridex,
				 y,
				 incy,
				 offsety,
				 stridey);
	    }
	  else if(*alpha) // alpha is on host
	    {
	      hipLaunchKernelGGL(axpy_kernel<A>,
				 blocks,
				 threads,
				 0,
				 handle->rocblas_stream,
				 n,
				 *alpha,
				 x,
				 incx,
				 offsetx,
				 stridex,
				 y,
				 incy,
				 offsety,
				 stridey);
	    }

	}
    }
  else
    {
#if 0      
      std::cout << "optimized calculation "  << std::endl;
#endif
      // rocblas_half8 load-store and rocblas_half2 arithmetic
      rocblas_int n_mod_8 = n & 7; // n mod 8
      rocblas_int n_mlt_8 = n & ~(rocblas_int)7; // multiple of 8
      int         blocks  = (n / 8 - 1) / NB + 1;
      dim3        grid(blocks, batch_count);
      dim3        threads(NB);	    
      if(handle->pointer_mode == rocblas_pointer_mode_device)
	{
	  hipLaunchKernelGGL(haxpy_mlt_8_kernel,
			     grid,
			     threads,
			     0,
			     handle->rocblas_stream,
			     n_mlt_8,
			     (const rocblas_half2*)alpha,			       
			     x,
			     stridex,			
			     y,
			     stridey);
	    
	  if(n_mod_8) 
	    {
	      //
	      // cleanup non-multiple of 8
	      //
	      hipLaunchKernelGGL(haxpy_mod_8_kernel,
				 dim3(1, batch_count),
				 n_mod_8,
				 0,
				 handle->rocblas_stream,
				 n_mod_8,
				 (const _Float16*)alpha,
				 x,
				 n_mlt_8,
				 stridex,
				 y,
				 n_mlt_8,
				 stridey);
	    }
	}
      else if(*(const _Float16*)alpha) // alpha is on host
	{
	  hipLaunchKernelGGL(haxpy_mlt_8_kernel,
			     grid,
			     threads,
			     0,
			     handle->rocblas_stream,
			     n_mlt_8,
			     load_scalar((const rocblas_half2*)alpha),
			     x,
			     stridex,
			     y,
			     stridey);

	  if(n_mod_8) 
	    {
	      //
	      // cleanup non-multiple of 8
	      //
	      hipLaunchKernelGGL(haxpy_mod_8_kernel,
				 dim3(1, batch_count),
				 n_mod_8,
				 0,
				 handle->rocblas_stream,
				 n_mod_8,
				 *(const _Float16*)alpha,
				 x,
				 n_mlt_8,
				 stridex,
				 y,
				 n_mlt_8,
				 stridey);
	    }
	}

    }

  return rocblas_status_success;
};



