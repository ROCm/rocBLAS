#ifndef REDUCTION_H_
#define REDUCTION_H_

#include "handle.h"
#include "rocblas.h"
#include "utility.h"
#include <type_traits>
#include <utility>

/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

// BLAS Level 1 includes routines and functions performing vector-vector
// operations. Most BLAS 1 routines are about reduction: compute the norm,
// calculate the dot production of two vectors, find the maximum/minimum index
// of the element of the vector. As you may observed, although the computation
// type is different, the core algorithm is the same: scan all element of the
// vector(s) and reduce to one single result.
//
// The reduction algorithm on GPU is called [parallel
// reduction](https://raw.githubusercontent.com/mateuszbuda/GPUExample/master/reduce3.png)
// which is adopted in rocBLAS. At the beginning, all the threads in the thread
// block participate. After each step of reduction (like a tree), the number of
// participating threads decrease by half. At the end of the parallel reduction,
// only one thread (usually thread 0) owns the result in its thread block.
//
// Classically, the BLAS 1 reduction needs more than one GPU kernel to finish,
// because the lack of global synchronization of thread blocks without exiting
// the kernel. The first kernels gather partial results, write into a temporary
// working buffer. The second kernel finishes the final reduction.
//
// For example, BLAS 1 routine i*amax is to find index of the maximum absolute
// value element of a vector. In this routine:
//
// Kernel 1: launch many thread block as needed. Each thread block works on a
// subset of the vector. Each thread block use the parallel reduction to find a
// local index with the maximum absolute value of the subset. There are
// number-of-the-thread-blocks local results.The results are written into a
// temporary working buffer. The working buffer has number-of-the-thread-blocks
// elements.
//
// Kernel 2: launch only one thread block which reads the temporary work buffer and
// reduces to final result still with the parallel reduction.
//
// As you may see, if there is a mechanism to synchronize all the thread blocks
// after local index is obtained in kernel 1 (without ending the kernel), then
// Kernel 2's computation can be merged into Kernel 1. One such mechanism is called
// atomic operation. However, atomic operation is new and is not used in rocBLAS
// yet. rocBLAS still use the classic standard parallel reduction right now.

// Recursively compute reduction
template <rocblas_int k, typename REDUCE, typename T>
struct rocblas_reduction_s
{
    __forceinline__ __device__ void operator()(rocblas_int tx, T* x)
    {
      // Reduce the lower half with the upper half
      if(tx < k)
	{
	  REDUCE{}(x[tx], x[tx + k]);
	}
      
      __syncthreads();
      
      // Recurse down with k / 2
      rocblas_reduction_s<k / 2, REDUCE, T>{}(tx, x);
    }
};

// leaf node for terminating recursion
template <typename REDUCE, typename T>
struct rocblas_reduction_s<0, REDUCE, T>
{
    __forceinline__ __device__ void operator()(rocblas_int tx, T* x) {}
};

/*! \brief general parallel reduction

    \details

    @param[in]
    n         rocblas_int. assume a power of 2
    @param[in]
    T         element type of vector x
    @param[in]
    REDUCE    reduction functor
    @param[in]
    tx        rocblas_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              x[0] store the final result.
    ********************************************************************/
template <rocblas_int NB, typename REDUCE, typename T>
__attribute__((flatten)) __device__ void rocblas_reduction(rocblas_int tx, T* x)
{
    static_assert(NB > 1 && !(NB & (NB - 1)), "NB must be a power of 2");
    __syncthreads();
    rocblas_reduction_s<NB / 2, REDUCE, T>{}(tx, x);
}





//
// ##############################################################################
//

template <rocblas_int half, rocblas_int P, typename REDUCE, typename T>
struct rocblas_reduction_batched_s
{
    __forceinline__ __device__ void operator()(rocblas_int threadIndex, T* x)
    {
      // Reduce the lower half with the upper half
      if(threadIndex < half)
	{
	  REDUCE{}(x[threadIndex], x[half + threadIndex]);
	}
      
      __syncthreads();
      
      // Recurse down with k / 2
      rocblas_reduction_batched_s<half / 2, P, REDUCE, T>{}(threadIndex, x);
    }
};

template <rocblas_int P, typename REDUCE, typename T>
  struct rocblas_reduction_batched_s<0, P, REDUCE, T>
{
    __forceinline__ __device__ void operator()(rocblas_int tx, T* x) {}
};

template <rocblas_int NB, rocblas_int P, typename REDUCE, typename T>
__attribute__((flatten)) __device__ void rocblas_reduction_batched(rocblas_int tx, T* x)
{
    static_assert(NB > 1 && !(NB & (NB - 1)), "NB must be a power of 2");
    __syncthreads();
    rocblas_reduction_batched_s<NB / 2, P, REDUCE, T>{}(tx, x);
}

//
// ##############################################################################
//



/*! \brief parallel reduction: sum

    \details

    @param[in]
    n         rocblas_int. assume a power of 2
    @param[in]
    tx        rocblas_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              x[0] store the final result.
    ********************************************************************/

struct rocblas_reduce_sum
{
    template <typename T>
    __forceinline__ __device__ void operator()(T& __restrict__ a, const T& __restrict__ b)
    {
      a += b;
    }
};

template <rocblas_int NB, typename T>
__attribute__((flatten)) __device__ void rocblas_sum_reduce(rocblas_int tx, T* x)
{
    rocblas_reduction<NB, rocblas_reduce_sum>(tx, x);
}
// end sum_reduce

// Identity finalizer
struct rocblas_finalize_identity
{
    template <typename T>
    __forceinline__ __host__ __device__ T&& operator()(T&& x)
    {
        return std::forward<T>(x); // Perfect identity, preserving valueness
    }
};

// Emulates value initialization T{}. Allows specialization for certain types.
template <typename T>
struct default_value
{
    __forceinline__ __host__ __device__ constexpr T operator()() const
    {
        return {};
    }
};

template <typename T>
struct traits_reduce
{
public: using item_t = T;
};

struct rocblas_reduce_sum_batched
{
    template <typename T>
    __forceinline__ __device__ void operator()(T& __restrict__ a, const T& __restrict__ b)
    {
      using item_t = typename traits_reduce<T>::item_t;
      item_t * ta =( item_t* )  &a;
      item_t * tb =( item_t * ) &b;
      static constexpr int n = sizeof(T) / sizeof(item_t);
      for (int i=0;i<n;++i)
	{
	  *ta += *tb;
	  ++ta;
	  ++tb;
	}
    }
};



#if 0

template <typename U>
struct traits
{
};

template <typename T>
struct traits<const T*>
{
  using t = const T*;
};

template <typename T>
struct traits<const T*const[]>
{
  using t = const T*const[];
};

template <typename T>
using const_t = const T;

  

template <typename U> typename traits<U>::t  get_ptr_batched(typename traits<U>::t u,
							     int i,
							     rocblas_int stridex)
{
  return u + i * stridex;
};
#endif


#if 0
template <typename U> typename traits<U>::t  get_ptr_batched(typename traits<U>::t const [] u,
							     int i,
							     rocblas_int stridex)
{
  return u + i * stridex;
};
#endif





// kennel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <rocblas_int NB,
          rocblas_int P,
          typename FETCH,
          typename REDUCE = rocblas_reduce_sum,
          typename Ti,
  typename To,typename U>

  __attribute__((amdgpu_flat_work_group_size(128, 1024)))
__global__ void
  rocblas_reduction_kernel_batched_part1(rocblas_int n,
					 U ux,
					 rocblas_int incx,
					 rocblas_int stridex,
					 To* workspace)
{

  //  ptrdiff_t     localThreadIndex  = hipThreadIdx_x + hipThreadDim_x * hipThreadIdx_y + hipThreadDim_x *hipThreadDim_y * hipThreadIdx_z;
  //  ptrdiff_t     globalThreadIndex = hipBlockIdx_x * hipBlockDim_x + localThreadIndex;
  ptrdiff_t     tx  = hipThreadIdx_x;
  ptrdiff_t     tid = hipBlockIdx_x * hipBlockDim_x + tx;

  //
  // batch_index 
  //
  
  //
  // 
  //
  
  //
  // Shared memory for each thread within a block.
  // Used to fetch and reduce.
  //
  __shared__ To tmp[NB];
  
  //
  // block_i, block_j
  //
  rocblas_int block_i = hipBlockIdx_x * hipBlockDim_x;
  rocblas_int block_j = hipBlockIdx_y * hipBlockDim_y;
  //
  //
  //
  rocblas_int batch_index = block_j;

  auto x = get_ptr_batch(ux, batch_index, stridex);
  
  // bound
  if(tid < n)
    {
      tmp[tx] = FETCH{}(x[tid * incx], tid);
    }
  else
    {
      tmp[tx] = default_value<To>{}(); // pad with default value
    }
  
  rocblas_reduction_batched<NB, P, REDUCE>(tx, tmp);

  if(tx == 0)
    {
      workspace[hipBlockIdx_x] = tmp[0];
    } 
}


  
// kennel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <rocblas_int NB,
          typename FETCH,
          typename REDUCE = rocblas_reduce_sum,
          typename Ti,
          typename To>
__attribute__((amdgpu_flat_work_group_size((NB < 128) ? NB : 128, (NB > 256) ? NB : 256)))
__global__ void
    rocblas_reduction_kernel_part1(rocblas_int n, const Ti* x, rocblas_int incx, To* workspace)
{
    ptrdiff_t     tx  = hipThreadIdx_x;
    ptrdiff_t     tid = hipBlockIdx_x * hipBlockDim_x + tx;
    __shared__ To tmp[NB];


//    __syncthreads();
//    if (tx==0)
//      {
//	for (int i=0;i<1024;++i)
//	  {
//	    rocblas_int* g = (rocblas_int*)(&tmp[i]);
//	    g[0] = 777;
//	  }
//      }
//    
//    __syncthreads();
    
    // bound
    if(tid < n)
      {
	// tmp[tx] = FETCH{}(x[tid * incx], tid);
	tmp[tx] = FETCH{}(x[tid * incx], tid);
	// tmp[tx] = FETCH{}( ((Ti)778) , tid);
      }
    else
      {
	//	rocblas_int* g = (rocblas_int*)(&tmp[tx]);
	//	g[0] = 777;
	tmp[tx] = default_value<To>{}(); // pad with default value
      }

    rocblas_reduction<NB, REDUCE>(tx, tmp);
    if(tx == 0)
      {
        workspace[hipBlockIdx_x] = tmp[0];
      } 
}

// kernel 2 gathers all the partial results in workspace and finishes the final reduction;
// number of threads (NB) loop blocks
template <rocblas_int NB,
          rocblas_int P,
          typename REDUCE,
          typename FINALIZE,
          typename To,
          typename Tr>
__attribute__((amdgpu_flat_work_group_size(128, 1024)))
__global__ void rocblas_reduction_kernel_batched_part2(rocblas_int nblocks, To* workspace, Tr* result)
{
    rocblas_int   threadIndex = hipThreadIdx_x;
    
    __shared__ To tmp[NB];

    if(threadIndex < nblocks)
    {
        tmp[threadIndex] = workspace[threadIndex];

        // bound, loop
        for(rocblas_int i = threadIndex + NB; i < nblocks; i += NB)
	  REDUCE{}(tmp[threadIndex], workspace[i]);
    }
    else
    { // pad with default value
        tmp[threadIndex] = default_value<To>{}();
    }

    if(nblocks < 32)
    {
        // no need parallel reduction
        __syncthreads();

        if(threadIndex == 0)
            for(rocblas_int i = 1; i < nblocks; i++)
                REDUCE{}(tmp[0], tmp[i]);
    }
    else
    {
        // parallel reduction
      rocblas_reduction_batched<NB, P, REDUCE>(threadIndex, tmp);
    }

    // Store result on device or in workspace
    if(threadIndex == 0)
        *result = Tr(FINALIZE{}(tmp[0]));
}



// kernel 2 gathers all the partial results in workspace and finishes the final reduction;
// number of threads (NB) loop blocks
template <rocblas_int NB,
          typename REDUCE   = rocblas_reduce_sum,
          typename FINALIZE = rocblas_finalize_identity,
          typename To,
          typename Tr>
__attribute__((amdgpu_flat_work_group_size((NB < 128) ? NB : 128, (NB > 256) ? NB : 256)))
__global__ void
    rocblas_reduction_kernel_part2(rocblas_int nblocks, To* workspace, Tr* result)
{
    rocblas_int   tx = hipThreadIdx_x;
    __shared__ To tmp[NB];

    if(tx < nblocks)
    {
        tmp[tx] = workspace[tx];

        // bound, loop
        for(rocblas_int i = tx + NB; i < nblocks; i += NB)
            REDUCE{}(tmp[tx], workspace[i]);
    }
    else
    { // pad with default value
        tmp[tx] = default_value<To>{}();
    }

    if(nblocks < 32)
    {
        // no need parallel reduction
        __syncthreads();

        if(tx == 0)
            for(rocblas_int i = 1; i < nblocks; i++)
                REDUCE{}(tmp[0], tmp[i]);
    }
    else
    {
        // parallel reduction
        rocblas_reduction<NB, REDUCE>(tx, tmp);
    }

    // Store result on device or in workspace
    if(tx == 0)
        *result = Tr(FINALIZE{}(tmp[0]));
}


  
// At least two kernels are needed to finish the reduction
// kernel 1 write partial result per thread block in workspace, blocks partial results
// kernel 2 gathers all the partial result in workspace and finishes the final reduction.
template <rocblas_int NB,
          typename FETCH,
          typename REDUCE   = rocblas_reduce_sum,
          typename FINALIZE = rocblas_finalize_identity,
          typename Ti,
          typename To,
          typename Tr>
rocblas_status rocblas_reduction_kernel(rocblas_handle __restrict__ handle,
                                        rocblas_int n,
                                        const Ti*   x,
                                        rocblas_int incx,
                                        Tr*         result,
                                        To*         workspace,
                                        rocblas_int blocks)
{
    
  hipLaunchKernelGGL((rocblas_reduction_kernel_part1<NB, FETCH, REDUCE>),
		     blocks,
		     NB,
		     0,
		     handle->rocblas_stream,
		     n,
		     x,
		     incx,
		     workspace);
    
  if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipLaunchKernelGGL((rocblas_reduction_kernel_part2<NB, REDUCE, FINALIZE>),
                           1,
                           NB,
                           0,
                           handle->rocblas_stream,
                           blocks,
                           workspace,
                           result);
    }
    else
    {
        // If in host pointer mode, workspace is converted to Tr* and the result is
        // placed there, and then copied from device to host. If To is a class type,
        // it must be a standard layout type and its first member must be of type Tr.
        static_assert(std::is_standard_layout<To>{}, "To must be a standard layout type");
        if(blocks > 1)
        {
            hipLaunchKernelGGL((rocblas_reduction_kernel_part2<NB, REDUCE, FINALIZE>),
                               1,
                               NB,
                               0,
                               handle->rocblas_stream,
                               blocks,
                               workspace,
                               (Tr*)workspace);
        }
        if(std::is_same<FINALIZE, rocblas_finalize_identity>{} || blocks > 1)
        {
	  RETURN_IF_HIP_ERROR(hipMemcpy(result, workspace, sizeof(Tr), hipMemcpyDeviceToHost));
        }
        else
        {
	  // If FINALIZE is not trivial and kernel part2 was not called, then
	  // workspace[0] needs to be finalized on host.
	  //	    printf("on gpu %d\n",sizeof(To));
	  To res;
	  RETURN_IF_HIP_ERROR(hipMemcpy(&res, workspace, sizeof(To), hipMemcpyDeviceToHost));
	  *result = FINALIZE{}(res);
        }
    }

    return rocblas_status_success;
}



// At least two kernels are needed to finish the reduction
// kernel 1 write partial result per thread block in workspace, blocks partial results
// kernel 2 gathers all the partial result in workspace and finishes the final reduction.
template <rocblas_int NB,
          rocblas_int P,
          typename FETCH,
          typename REDUCE   = rocblas_reduce_sum,
          typename FINALIZE = rocblas_finalize_identity,
          typename Ti,
          typename To,
  typename Tr,
  typename U
  >
rocblas_status rocblas_reduction_kernel_batched(rocblas_handle __restrict__ handle,
						rocblas_int n,
						U x,
						rocblas_int incx,
						Tr*         result,
						To*         workspace,
						rocblas_int blocks,
						rocblas_int batch_count)
{
  
  //
  // Note that this kernel expects the data
  //

  
  //
  // Define the partition of the problem.
  //

  //
  // n is the first dimension 
  // batch_count is the second dimension.
  //
  
  //
  // The workspace will contain the results of each block.
  //
  
  //
  // Launch the kernel who is responsible of:
  //  - each block each thread fetches its data
  //  - each block fetches its data
  //
  
  hipLaunchKernelGGL((rocblas_reduction_kernel_batched_part1<NB, P, FETCH, REDUCE>),
		     dim3(blocks, batch_count),
		     NB,
		     0,
		     handle->rocblas_stream,
		     n,
		     x,
		     incx,
		     workspace);


  
  if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
      hipLaunchKernelGGL((rocblas_reduction_kernel_batched_part2<NB, P, REDUCE, FINALIZE>),
			 dim3(1, batch_count),
			 NB,
			 0,
			 handle->rocblas_stream,
			 blocks,
			 workspace,
			 result);
    }
  else
    {
      // If in host pointer mode, workspace is converted to Tr* and the result is
      // placed there, and then copied from device to host. If To is a class type,
      // it must be a standard layout type and its first member must be of type Tr.
      static_assert(std::is_standard_layout<To>{}, "To must be a standard layout type");
      if(blocks > 1 || batch_count > 1)
        {
	  hipLaunchKernelGGL((rocblas_reduction_kernel_batched_part2<NB, P, REDUCE, FINALIZE>),
			     dim3(1, batch_count),
			     NB,
			     0,
			     handle->rocblas_stream,
			     blocks,
			     workspace,
			     (Tr*) (workspace + batch_count * blocks) );
        }
      if(std::is_same<FINALIZE, rocblas_finalize_identity>{} || blocks > 1 || batch_count > 1)
        {
	  RETURN_IF_HIP_ERROR(hipMemcpy(result, workspace + batch_count * blocks, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));
        }
      else
        {
	  // If FINALIZE is not trivial and kernel part2 was not called, then
	  // workspace[0] needs to be finalized on host.
	  //	    printf("on gpu %d\n",sizeof(To));
	  To res[batch_count];
	  RETURN_IF_HIP_ERROR(hipMemcpy(&res, workspace, sizeof(To) * batch_count, hipMemcpyDeviceToHost));
	  for (int batch_index = 0;batch_index < batch_count;++batch_index)
	    {	  
	      result[batch_index] = FINALIZE{}(res[batch_index]);
	    }
        }
    }

  return rocblas_status_success;
}



#endif
