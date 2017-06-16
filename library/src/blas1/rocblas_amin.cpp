/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
 
#include "status.h"
#include "definitions.h"
#include "device_template.h"
#include "fetch_template.h"
#include "rocblas_unique_ptr.hpp"

template<typename T1, typename T2, rocblas_int NB>
__global__ void
amin_kernel_part1(hipLaunchParm lp,
    rocblas_int n,
    const T1* x, rocblas_int incx,
    T2* workspace,
    rocblas_int* workspace_index)
{
    rocblas_int tx  = hipThreadIdx_x;
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    __shared__ T2 shared_tep[NB];
    __shared__ rocblas_int index[NB];

    //bound
    if ( tid < n ) {
        T2 real = fetch_real<T1, T2>(x[tid * incx]);
        T2 imag = fetch_imag<T1, T2>(x[tid * incx]);
        shared_tep[tx] =  fabs(real) + fabs(imag);
        index[tx] = tid;
    }
    else
    {   //pad with zero
        shared_tep[tx] =  0.0;
        index[tx] = -1;
    }

    rocblas_maxid_reduce<NB, T2>(tx, shared_tep, index);

    if(tx == 0) {
        workspace[hipBlockIdx_x] = shared_tep[0];
        workspace_index[hipBlockIdx_x] = index[0];
    }
}


template<typename T, rocblas_int NB, rocblas_int flag>
__global__ void
amin_kernel_part2(hipLaunchParm lp,
    rocblas_int n,
    T* workspace,
    rocblas_int* workspace_index,
    rocblas_int* result)
{

    rocblas_int tx  = hipThreadIdx_x;

    __shared__ T shared_tep[NB];
    __shared__ rocblas_int index[NB];

    shared_tep[tx] = 0.0;
    index[tx] = -1;

    //bound, loop
    for(rocblas_int i=tx; i<n; i+=NB){
        if( shared_tep[tx] == workspace[i] ){
            index[tx] = min(index[tx], workspace_index[i]); //if equal take the smaller index
        }
        else if (shared_tep[tx] < workspace[i]){ // if smaller, then take the bigger one
            shared_tep[tx] = workspace[i];
            index[tx] = workspace_index[i];
        }
    }

    __syncthreads();

    rocblas_maxid_reduce<NB, T>(tx, shared_tep, index);

    if(tx == 0){
        if(flag){
            //flag == 1, write the result on device memory
            *result = index[0]; //result[0] works, too
        }
        else{ //if flag == 0, cannot write to result which is in host memory, instead write to worksapce
            workspace_index[0] = index[0];
        }
    }
}


//HIP support up to 1024 threads/work itmes per thread block/work group
#define NB_X 1024

//assume workspace has already been allocated, recommened for repeated calling of amin product routine
template<typename T1, typename T2>
rocblas_status
rocblas_amin_template_workspace(rocblas_handle handle,
    rocblas_int n,
    const T1 *x, rocblas_int incx,
    rocblas_int *result,
    T2 *workspace,
    rocblas_int *workspace_index,
    rocblas_int lworkspace)
{

    rocblas_int blocks = (n-1)/ NB_X + 1;

    //At least two kernels are needed to finish the reduction
    //kennel 1 write partial result per thread block in workspace, number of partial result is blocks
    //kernel 2 gather all the partial result in workspace and finish the final reduction.

    if(lworkspace < blocks) {
        printf("size of workspace = %d is too small, allocate at least %d", lworkspace, blocks);
        return rocblas_status_not_implemented;
    }

    dim3 grid(blocks, 1, 1);
    dim3 threads(NB_X, 1, 1);

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    hipLaunchKernel(HIP_KERNEL_NAME(amin_kernel_part1<T1, T2, NB_X>), dim3(grid), dim3(threads), 0, rocblas_stream,
                                                                      n, x, incx, workspace, workspace_index);

    if( rocblas_pointer_to_mode(result) == rocblas_pointer_mode_device ){
        //the last argument 1 indicate the result is a device pointer, not memcpy is required
        hipLaunchKernel(HIP_KERNEL_NAME(amin_kernel_part2<T2, NB_X, 1>), dim3(1,1,1), dim3(threads), 0, rocblas_stream,
                                                                         blocks, workspace, workspace_index, result);
    }
    else{
        //the last argument 0 indicate the result is a host pointer
        // workspace[0] has a copy of the final result, if the result pointer is on host, a memory copy is required
        //printf("it is a host pointer\n");
        // only for blocks > 1, otherwise the final result is already reduced in workspace[0]
        if ( blocks > 1) hipLaunchKernel(HIP_KERNEL_NAME(amin_kernel_part2<T2, NB_X, 0>), dim3(1,1,1), dim3(threads), 0, rocblas_stream,
                                                                                          blocks, workspace, workspace_index, result);
        RETURN_IF_HIP_ERROR(hipMemcpy(result, workspace_index, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    }

    return rocblas_status_success;
}

/* ============================================================================================ */

/*! \brief BLAS Level 1 API

    \details
    amin finds the first index of the element of minimum magnitude of real vector x
         or the sum of magnitude of the real and imaginary parts of elements if x is a complex vector

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of y.
    @param[inout]
    result
              store the amin product. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.
    ********************************************************************/



//allocate workspace inside this API
template<typename T1, typename T2>
rocblas_status
rocblas_amin_template(rocblas_handle handle,
    rocblas_int n,
    const T1 *x, rocblas_int incx,
    rocblas_int *result)
{
    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( x == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( result == nullptr )
        return rocblas_status_invalid_pointer;

    /*
     * Quick return if possible.
     */

    if ( n <= 0 || incx <= 0){
        if( rocblas_pointer_to_mode(result) == rocblas_pointer_mode_device ){
            RETURN_IF_HIP_ERROR(hipMemset(result, 0, sizeof(T2)));
        }
        else{
            *result = 0.0;
        }
        return rocblas_status_success;
    }

    rocblas_int blocks = (n-1)/ NB_X + 1;

    rocblas_status status;

    auto workspace_managed = rocblas_unique_ptr{rocblas::device_malloc(sizeof(T2) * blocks),rocblas::device_free};
    T2 *workspace = (T2 *) workspace_managed.get();
    if(!workspace)
    {
        return rocblas_status_memory_error;
    }

    auto workspace_index_managed = rocblas_unique_ptr{rocblas::device_malloc(sizeof(rocblas_int) * blocks),rocblas::device_free};
    rocblas_int *workspace_index = (rocblas_int*) workspace_index_managed.get();
    if(!workspace_index)
    {
        return rocblas_status_memory_error;
    }

    status = rocblas_amin_template_workspace<T1, T2>(handle, n, x, incx, result, workspace, workspace_index, blocks);

    return status;
}




/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_samin(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    rocblas_int *result){

    return rocblas_amin_template<float, float>(handle, n, x, incx, result);
}


extern "C"
rocblas_status
rocblas_damin(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    rocblas_int *result){

    return rocblas_amin_template<double, double>(handle, n, x, incx, result);
}


extern "C"
rocblas_status
rocblas_scamin(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    rocblas_int *result){

    return rocblas_amin_template<rocblas_float_complex, float>(handle, n, x, incx, result);
}

extern "C"
rocblas_status
rocblas_dzamin(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    rocblas_int *result){

    return rocblas_amin_template<rocblas_double_complex, double>(handle, n, x, incx, result);
}
