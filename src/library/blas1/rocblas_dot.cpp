/* ************************************************************************
 * dotright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <hip_runtime.h>
#include "rocblas.h"
#include "../device_template.h"

template<typename T, rocblas_int NB>
__global__ void
dot_kernel_part1(hipLaunchParm lp,
    rocblas_int n,
    const T* x, rocblas_int incx,
    const T* y, rocblas_int incy,
    T* workspace)
{
    int tx  = hipThreadIdx_x;
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    __shared__ T shared_tep[NB];
    //bound
    if ( tid < n ) {
        shared_tep[tx] =  x[tid * incx] * y[tid * incy];
    }
    else
    {   //pad with zero
        shared_tep[tx] =  0.0;
    }

    rocblas_sum_reduce<NB, T>(tx, shared_tep);

    if(tx == 0) workspace[hipBlockIdx_x] = shared_tep[0];
}




template<typename T, rocblas_int NB, rocblas_int flag>
__global__ void
dot_kernel_part2(hipLaunchParm lp,
    rocblas_int n,
    T* workspace,
    T* result)
{

    int tx  = hipThreadIdx_x;

    __shared__ T shared_tep[NB];

    shared_tep[tx] = 0.0;

    //bound, loop
    for(int i=tx; i<n; i+=NB){
        shared_tep[i] += workspace[i];
    }

    if(n < 32){
        // no need parallel reduction
        if(tx == 0){
            for(int i=1;i<n;i++){
                shared_tep[0] += shared_tep[i];
            }
        }
    }
    else{
            //parallel reduction, TODO bug
            rocblas_sum_reduce<NB, T>(tx, shared_tep);
    }

    if(tx == 0){
        if(flag){
            //flag == 1, write to result of device memory
            *result = shared_tep[0]; //result[0] works, too
        }
        else{
            workspace[0] = shared_tep[0];
        }
    }
}


//HIP support up to 1024 threads/work itmes per thread block/work group
#define NB_X 1024

//assume workspace has already been allocated, recommened for repeated calling of dot product routine
template<typename T>
rocblas_status
rocblas_dot_template_workspace(rocblas_handle handle,
    rocblas_int n,
    const T *x, rocblas_int incx,
    const T *y, rocblas_int incy,
    T* result, T* workspace, rocblas_int lworkspace)
{

    if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( x == nullptr )
        return rocblas_status_invalid_size;
    else if ( incx < 0 )
        return rocblas_status_invalid_size;
    else if ( y == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( incy < 0 )
        return rocblas_status_invalid_size;


    /*
     * Quick return if possible.
     */

    if ( n == 0)
        return rocblas_status_success;

    int blocks = (n-1)/ NB_X + 1;

    //At least two kernels are needed to finish the reduction
    //kennel 1 write partial result per thread block in workspace, number of partial result is blocks
    //kernel 2 gather all the partial result in workspace and finish the final reduction. number of threads (NB_X) loop blocks

    if(lworkspace < blocks) {
        printf("size workspace = %d is too small, allocate at least %d", lworkspace, blocks);
        return rocblas_not_implemented;
    }

    dim3 grid(blocks, 1, 1);
    dim3 threads(NB_X, 1, 1);

    hipLaunchKernel(HIP_KERNEL_NAME(dot_kernel_part1<T, NB_X>), dim3(grid), dim3(threads), 0, 0 , n, x, incx, y, incy, workspace);

    if( rocblas_get_pointer_location(result) == rocblas_mem_location_device ){
        //the last argument 1 indicate the result is on device, not memcpy is required
        hipLaunchKernel(HIP_KERNEL_NAME(dot_kernel_part2<T, NB_X, 1>), dim3(1,1,1), dim3(threads), 0, 0, blocks, workspace, result);
        return rocblas_status_success;
    }
    else{
        //the last argument 0 indicate the result is on host
        // workspace[0] has a copy of the final result, if the result pointer is on host, a memory copy is required
        //printf("it is a host pointer\n");
        // only for blocks > 1, otherwise the final result is already reduced in workspace[0]
        if ( blocks > 1) hipLaunchKernel(HIP_KERNEL_NAME(dot_kernel_part2<T, NB_X, 0>), dim3(1,1,1), dim3(threads), 0, 0, blocks, workspace, result);
        return get_rocblas_status_for_hip_status(hipMemcpy(result, workspace, sizeof(T), hipMemcpyDeviceToHost));
    }
}

/* ============================================================================================ */

/*! \brief BLAS Level 1 API

    \details
    dot(u)  perform dot product of vector x and y

        result = x * y;

    dotc  perform dot product of complex vector x and complex y

        result = conjugate (x) * y;

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specif_templates_int
              specifies the increment for the elements of y.
    @param[inout]
    result
              store the dot product. either on the host CPU or device GPU.

    ********************************************************************/

//allocate workspace inside this API
template<typename T>
rocblas_status
rocblas_dot_template(rocblas_handle handle,
    rocblas_int n,
    const T *x, rocblas_int incx,
    const T *y, rocblas_int incy,
    T *result)
{
    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( x == nullptr )
        return rocblas_status_invalid_size;
    else if ( incx < 0 )
        return rocblas_status_invalid_size;
    else if ( y == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( incy < 0 )
        return rocblas_status_invalid_size;


    /*
     * Quick return if possible.
     */

    if ( n == 0)
        return rocblas_status_success;

    int blocks = (n-1)/ NB_X + 1;

    rocblas_status status;

    T *workspace;
    status = get_rocblas_status_for_hip_status(hipMalloc(&workspace, sizeof(T) * blocks));//potential error may rise here, blocking device operation
    if(status != rocblas_status_success) return status;

    rocblas_status status = rocblas_dot_template_workspace<T>(handle, n, x, incx, y, incy, result, workspace, blocks);
    if(status != rocblas_status_success){
        hipFree( workspace );
        return status;
    }

    status = get_rocblas_status_for_hip_status(hipFree(workspace));
    return status;
}



/* ============================================================================================ */

    /*
     * ===========================================================================
     *    template interface
     *    template specialization
     * ===========================================================================
     */


template<>
rocblas_status
rocblas_dot<float>(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    const float *y, rocblas_int incy,
    float *result){

    return rocblas_dot_template<float>(handle, n, x, incx, y, incy, result);
}



template<>
rocblas_status
rocblas_dot<double>(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    const double *y, rocblas_int incy,
    double *result){

    return rocblas_dot_template<double>(handle, n, x, incx, y, incy, result);
}

template<>
rocblas_status
rocblas_dot<rocblas_float_complex>(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    const rocblas_float_complex *y, rocblas_int incy,
    rocblas_float_complex *result){

    return rocblas_dot_template<rocblas_float_complex>(handle, n, x, incx, y, incy, result);
}

template<>
rocblas_status
rocblas_dot<rocblas_double_complex>(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    const rocblas_double_complex *y, rocblas_int incy,
    rocblas_double_complex *result){

    return rocblas_dot_template<rocblas_double_complex>(handle, n, x, incx, y, incy, result);
}




/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_sdot(rocblas_handle handle,
    rocblas_int n,
    const float *x, rocblas_int incx,
    const float *y, rocblas_int incy,
    float *result){

    return rocblas_dot<float>(handle, n, x, incx, y, incy, result);
}

extern "C"
rocblas_status
rocblas_ddot(rocblas_handle handle,
    rocblas_int n,
    const double *x, rocblas_int incx,
    const double *y, rocblas_int incy,
    double *result){

    return rocblas_dot<double>(handle, n, x, incx, y, incy, result);
}


extern "C"
rocblas_status
rocblas_cdotu(rocblas_handle handle,
    rocblas_int n,
    const rocblas_float_complex *x, rocblas_int incx,
    const rocblas_float_complex *y, rocblas_int incy,
    rocblas_float_complex *result){

    return rocblas_dot<rocblas_float_complex>(handle, n, x, incx, y, incy, result);
}


extern "C"
rocblas_status
rocblas_zdotu(rocblas_handle handle,
    rocblas_int n,
    const rocblas_double_complex *x, rocblas_int incx,
    const rocblas_double_complex *y, rocblas_int incy,
    rocblas_double_complex *result){

    return rocblas_dot<rocblas_double_complex>(handle, n, x, incx, y, incy, result);
}



/* ============================================================================================ */
