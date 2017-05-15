/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "definitions.h"
#include "rocblas-types.h"
#include "handle.h"
#include "rocblas-auxiliary.h"

    /* ============================================================================================ */

/*******************************************************************************
 * ! \brief  indicates whether the pointer is on the host or device.
 * currently HIP API can only recoginize the input ptr on deive or not
 *  can not recoginize it is on host or not
 ******************************************************************************/
rocblas_mem_location rocblas_get_pointer_location(void *ptr){
    hipPointerAttribute_t attribute;
    hipPointerGetAttributes(&attribute, ptr);
    if (ptr == attribute.devicePointer) {
        return rocblas_mem_location_device;
    } else {
        return rocblas_mem_location_host;
    }
}


/*******************************************************************************
 * ! \brief create rocblas handle called before any rocblas library routines
 ******************************************************************************/
extern "C"
rocblas_status rocblas_create_handle(rocblas_handle *handle){

    // if handle not valid
    if (handle == nullptr) {
        return rocblas_status_invalid_pointer;
    }

    // allocate on heap
    try {
      *handle = new _rocblas_handle();
    } catch (rocblas_status status) {
        return status;
    }

    return rocblas_status_success;
}


/*******************************************************************************
 *! \brief release rocblas handle, will implicitly synchronize host and device
 ******************************************************************************/
extern "C"
rocblas_status rocblas_destroy_handle(rocblas_handle handle){
    // call destructor
    try {
        delete handle;
    } catch (rocblas_status status) {
        return status;
    }
    return rocblas_status_success;
}


/*******************************************************************************
 *! \brief   set rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 *   stream_id must be created before this call
 ******************************************************************************/
extern "C"
rocblas_status
rocblas_set_stream(rocblas_handle handle, hipStream_t stream_id){
    return handle->set_stream( stream_id );
}


/*******************************************************************************
 *! \brief   get rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 ******************************************************************************/
extern "C"
rocblas_status
rocblas_get_stream(rocblas_handle handle, hipStream_t *stream_id){
    return handle->get_stream( stream_id );
}

/*******************************************************************************
 *! \brief  Vector copy for non unit stride vectors passed to function as
     void pointer + size of vector elements
 ******************************************************************************/
// arbitrarily assign max buffer size to 1Mb
#define MAX_BUFFER_BYTE_SIZE 1048576
#define NB_X 256

__global__ void
copy_void_ptr_kernel(hipLaunchParm lp,
    rocblas_int n, rocblas_int elem_size,
    const void *x, rocblas_int incx,
    void *y,  rocblas_int incy)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if ( tid < n ) {
        memcpy((void *)((size_t)y + (size_t)(tid * incy * elem_size)), 
               (void *)((size_t)x + (size_t)(tid * incx * elem_size)), elem_size);
    }
}

/*******************************************************************************
 *! \brief   copies vector x on host to vector y on device. Vector x has stride 
     incx. Vector y has stride incy.
 ******************************************************************************/
extern "C"
rocblas_status
rocblas_set_vector(rocblas_int n, rocblas_int elem_size, 
    const void *x_h, rocblas_int incx, 
    void *y_d, rocblas_int incy)
{
    if ( n == 0 ) 
        return rocblas_status_success;
    if ( n < 0 ) 
        return rocblas_status_invalid_size;
    if ( incx <= 0 ) 
        return rocblas_status_invalid_size;
    if ( incy <= 0 ) 
        return rocblas_status_invalid_size;
    if ( elem_size <= 0 ) 
        return rocblas_status_invalid_size;
    if ( x_h == nullptr )
        return rocblas_status_invalid_handle;
    if ( y_d == nullptr )
        return rocblas_status_invalid_handle;

    if ( incx == 1 && incy == 1)  // contiguous host vector -> contiguous device vector
    {
        PRINT_IF_HIP_ERROR(hipMemcpy(y_d, x_h, elem_size * n, hipMemcpyHostToDevice));
    }
    else                          // either non-contiguous host vector or non-contiguous device vector
    {
        int buffer_byte_size = elem_size * n < MAX_BUFFER_BYTE_SIZE ? elem_size * n : MAX_BUFFER_BYTE_SIZE;
        int n_elem = ((buffer_byte_size - 1 ) / elem_size) + 1; // number of elements in buffer
        int n_copy = ((n - 1) / n_elem) + 1;                    // number of times buffer is copied
        int blocks = (n_elem-1)/ NB_X + 1;
        dim3 grid( blocks, 1, 1 );
        dim3 threads( NB_X, 1, 1 );
        
        void *b_h;
        if (incx != 1)
        {
            b_h = malloc(buffer_byte_size);
            if (!b_h)
            {
                return rocblas_status_memory_error;
            }
        }
        
        void *b_d;
        hipStream_t rocblas_stream;
        if (incy != 1)
        {
            PRINT_IF_HIP_ERROR(hipMalloc(&b_d, buffer_byte_size));
            if (!b_d)
            {
                return rocblas_status_memory_error;
            }
            rocblas_handle handle;
            rocblas_create_handle(&handle);
            RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));
        }
        
        size_t x_h_byte_stride = (size_t) elem_size * (size_t) incx;
        size_t y_d_byte_stride = (size_t) elem_size * (size_t) incy;
        size_t b_h_byte_stride = (size_t) elem_size;

        for (int i_copy = 0; i_copy < n_copy; i_copy++)
        {
            int i_zero = i_copy * n_elem;
            int ib_max = n_elem < n - (n_elem * i_copy) ? n_elem : n - (n_elem * i_copy);
            void *y_d_copy = (void *)((size_t)y_d + ((size_t) i_zero * y_d_byte_stride));

            if((incx != 1) && (incy != 1))
            {
                // non-contiguous host vector -> host buffer
                for (int i_b = 0, i_x = i_zero; i_b < ib_max; i_b++, i_x++)
                {
                    memcpy((void *)((size_t)b_h + ((size_t) i_b * b_h_byte_stride)),
                           (void *)((size_t)x_h + ((size_t) i_x * x_h_byte_stride)), elem_size);
                }
                // host buffer -> device buffer 
                PRINT_IF_HIP_ERROR(hipMemcpy(b_d, b_h, elem_size * ib_max, hipMemcpyHostToDevice));
                // device buffer -> non-contiguous device vector
                hipLaunchKernel(HIP_KERNEL_NAME(copy_void_ptr_kernel), dim3(grid), dim3(threads), 0,
                    rocblas_stream, ib_max, elem_size, b_d, 1, y_d_copy, incy);
            }
            else if (incx == 1 && incy != 1)
            {
                // contiguous host vector -> device buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(b_d, (void *)((size_t)x_h + ((size_t) i_zero * x_h_byte_stride)),
                    elem_size * ib_max, hipMemcpyHostToDevice));
                // device buffer -> non-contiguous device vector
                hipLaunchKernel(HIP_KERNEL_NAME(copy_void_ptr_kernel), dim3(grid), dim3(threads), 0,
                    rocblas_stream, ib_max, elem_size, b_d, 1, y_d_copy, incy);
            }
            else if (incx != 1 && incy == 1)
            {
                // non-contiguous host vector -> host buffer
                for (int i_b = 0, i_x = i_zero; i_b < ib_max; i_b++, i_x++)
                {
                    memcpy((void *)((size_t)b_h + ((size_t) i_b * b_h_byte_stride)),
                           (void *)((size_t)x_h + ((size_t) i_x * x_h_byte_stride)), elem_size);
                }
                // host buffer -> contiguous device vector
                PRINT_IF_HIP_ERROR(hipMemcpy(y_d_copy, b_h, elem_size * ib_max, hipMemcpyHostToDevice));
            }
        }
        free(b_h);
        if (incy != 0)
        {
            hipFree(b_d);
        }
    }
    return rocblas_status_success;
}

/*******************************************************************************
 *! \brief   copies vector x on device to vector y on host. Vector x has stride 
     incx. Vector y has stride incy.
 ******************************************************************************/
extern "C"
rocblas_status
rocblas_get_vector(rocblas_int n, rocblas_int elem_size, 
    const void *x_d, rocblas_int incx, 
    void *y_h, rocblas_int incy)
{
    if ( n == 0 ) 
        return rocblas_status_success;
    if ( n < 0 ) 
        return rocblas_status_invalid_size;
    if ( incx <= 0 ) 
        return rocblas_status_invalid_size;
    if ( incy <= 0 ) 
        return rocblas_status_invalid_size;
    if ( elem_size <= 0 ) 
        return rocblas_status_invalid_size;
    if ( x_d == nullptr )
        return rocblas_status_invalid_handle;
    if ( y_h == nullptr )
        return rocblas_status_invalid_handle;

    if ( incx == 1 && incy == 1)           // congiguous device vector -> congiguous host vector
    {
        PRINT_IF_HIP_ERROR(hipMemcpy(y_h, x_d, elem_size * n, hipMemcpyDeviceToHost));
    }
    else                                   // either device or host vector is non-contiguous 
    {
        int buffer_byte_size = elem_size * n < MAX_BUFFER_BYTE_SIZE ? elem_size * n : MAX_BUFFER_BYTE_SIZE;
        int n_elem = ((buffer_byte_size - 1 ) / elem_size) + 1; // number elements in buffer
        int n_copy = ((n - 1) / n_elem) + 1;                    // number of times buffer is copied
        int blocks = (n_elem-1)/ NB_X + 1;
        dim3 grid( blocks, 1, 1 );
        dim3 threads( NB_X, 1, 1 );
        
        void *b_h;
        if (incy != 1)
        {
            b_h = malloc(buffer_byte_size);
            if (!b_h)
            {
                return rocblas_status_memory_error;
            }
        }
        
        void *b_d;
        hipStream_t rocblas_stream;
        if (incx != 1)
        {
            PRINT_IF_HIP_ERROR(hipMalloc(&b_d, buffer_byte_size));
            if (!b_d)
            {
                return rocblas_status_memory_error;
            }
            rocblas_handle handle;
            rocblas_create_handle(&handle);
            RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));
        }
        
        size_t x_d_byte_stride = (size_t) elem_size * (size_t) incx;
        size_t y_h_byte_stride = (size_t) elem_size * (size_t) incy;
        size_t b_h_byte_stride = (size_t) elem_size;

        for (int i_copy = 0; i_copy < n_copy; i_copy++)
        {
            int i_zero = i_copy * n_elem;
            int ib_max = n_elem < n - (n_elem * i_copy) ? n_elem : n - (n_elem * i_copy);

            void *x_d_copy = (void *)((size_t)x_d + ((size_t) i_zero * x_d_byte_stride));
                     
            if (incx !=1 && incy != 1)
            {
                // non-contiguous device vector -> device buffer
                hipLaunchKernel(HIP_KERNEL_NAME(copy_void_ptr_kernel), dim3(grid), dim3(threads), 0,
                    rocblas_stream, ib_max, elem_size, x_d_copy, incx, b_d, 1);
                // device buffer -> host buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(b_h, b_d, elem_size * ib_max, hipMemcpyDeviceToHost));
                // host buffer -> non-contiguous host vector
                for (int i_b = 0, i_y = i_zero; i_b < ib_max; i_b++, i_y++)
                {
                    memcpy((void *)((size_t)y_h + (size_t)(i_y * y_h_byte_stride)),
                           (void *)((size_t)b_h + (size_t)(i_b * b_h_byte_stride)), elem_size);
                }
            }
            else if (incx == 1 && incy != 1)
            {
                // congiguous device vector -> host buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(b_h, x_d_copy, elem_size * ib_max, hipMemcpyDeviceToHost));

                // host buffer -> non-contiguous host vector
                for (int i_b = 0, i_y = i_zero; i_b < ib_max; i_b++, i_y++)
                {
                    memcpy((void *)((size_t)y_h + (size_t)(i_y * y_h_byte_stride)),
                           (void *)((size_t)b_h + (size_t)(i_b * b_h_byte_stride)), elem_size);
                }
            }
            else if (incx != 1 && incy == 1)
            {
                // non-contiguous device vector -> device buffer
                hipLaunchKernel(HIP_KERNEL_NAME(copy_void_ptr_kernel), dim3(grid), dim3(threads), 0,
                    rocblas_stream, ib_max, elem_size, x_d_copy, incx, b_d, 1);
                // device buffer -> contiguous host vector
                PRINT_IF_HIP_ERROR(hipMemcpy((void *)((size_t)y_h + (size_t)(i_zero * y_h_byte_stride)),
                    b_d, elem_size * ib_max, hipMemcpyDeviceToHost));
            }
        }
        free(b_h);
        if (incx != 0) 
        {
            hipFree(b_d);
        }
    }
    return rocblas_status_success;
}
