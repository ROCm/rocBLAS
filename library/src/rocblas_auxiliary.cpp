/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas-auxiliary.h"
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>

/* ============================================================================================ */

/*******************************************************************************
 * ! \brief  indicates whether the pointer is on the host or device.
 * currently HIP API can only recoginize the input ptr on deive or not
 *  can not recoginize it is on host or not
 ******************************************************************************/
extern "C" rocblas_pointer_mode rocblas_pointer_to_mode(void* ptr)
{
    hipPointerAttribute_t attribute;
    hipPointerGetAttributes(&attribute, ptr);
    return ptr == attribute.devicePointer ? rocblas_pointer_mode_device : rocblas_pointer_mode_host;
}

/*******************************************************************************
 * ! \brief get pointer mode, can be host or device
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_pointer_mode(rocblas_handle        handle,
                                                   rocblas_pointer_mode* mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    *mode = handle->pointer_mode;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_get_pointer_mode", *mode);
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief set pointer mode to host or device
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_pointer_mode(rocblas_handle handle, rocblas_pointer_mode mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_set_pointer_mode", mode);
    handle->pointer_mode = mode;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief get atomics mode
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_atomics_mode(rocblas_handle        handle,
                                                   rocblas_atomics_mode* mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    *mode = handle->atomics_mode;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_get_atomics_mode", *mode);
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief set atomics mode
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_atomics_mode(rocblas_handle handle, rocblas_atomics_mode mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_set_atomics_mode", mode);
    handle->atomics_mode = mode;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief create rocblas handle called before any rocblas library routines
 ******************************************************************************/
extern "C" rocblas_status rocblas_create_handle(rocblas_handle* handle)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;

    // allocate on heap
    *handle = new _rocblas_handle;

    if((*handle)->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(*handle, "rocblas_create_handle");

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief release rocblas handle, will implicitly synchronize host and device
 ******************************************************************************/
extern "C" rocblas_status rocblas_destroy_handle(rocblas_handle handle)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_destroy_handle");
    // call destructor
    delete handle;

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   set rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 *   stream_id must be created before this call
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream_id)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_set_stream", stream_id);
    handle->rocblas_stream = stream_id;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   get rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t* stream_id)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!stream_id)
        return rocblas_status_invalid_pointer;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_get_stream", *stream_id);
    *stream_id = handle->rocblas_stream;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief  Non-unit stride vector copy on device. Vectors are void pointers
     with element size elem_size
 ******************************************************************************/
// arbitrarily assign max buffer size to 1Mb
constexpr size_t      VEC_BUFF_MAX_BYTES = 1048576;
constexpr rocblas_int NB_X               = 256;

__global__ void copy_void_ptr_vector_kernel(rocblas_int n,
                                            rocblas_int elem_size,
                                            const void* x,
                                            rocblas_int incx,
                                            void*       y,
                                            rocblas_int incy)
{
    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        memcpy(
            (char*)y + tid * incy * elem_size, (const char*)x + tid * incx * elem_size, elem_size);
    }
}

/* ============================================================================================ */
// TODO: Need to replace this with new device memory allocation system

// device_malloc wraps hipMalloc and provides same API as malloc
static void* device_malloc(size_t byte_size)
{
    void* pointer = nullptr;
    PRINT_IF_HIP_ERROR((hipMalloc)(&pointer, byte_size));
    return pointer;
}

// device_free wraps hipFree and provides same API as free
static void device_free(void* ptr)
{
    PRINT_IF_HIP_ERROR((hipFree)(ptr));
}

using rocblas_unique_ptr = std::unique_ptr<void, void (*)(void*)>;

/*******************************************************************************
 *! \brief   copies void* vector x with stride incx on host to void* vector
     y with stride incy on device. Vectors have n elements of size elem_size.
  TODO: Need to replace device memory allocation with new system
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_vector(rocblas_int n,
                                             rocblas_int elem_size,
                                             const void* x_h,
                                             rocblas_int incx,
                                             void*       y_d,
                                             rocblas_int incy)
try
{
    if(n == 0) // quick return
        return rocblas_status_success;
    if(n < 0 || incx <= 0 || incy <= 0 || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!x_h || !y_d)
        return rocblas_status_invalid_pointer;

    if(incx == 1 && incy == 1) // contiguous host vector -> contiguous device vector
    {
        PRINT_IF_HIP_ERROR(hipMemcpy(y_d, x_h, elem_size * n, hipMemcpyHostToDevice));
    }
    else // either non-contiguous host vector or non-contiguous device vector
    {
        size_t bytes_to_copy = static_cast<size_t>(elem_size) * static_cast<size_t>(n);
        size_t temp_byte_size
            = bytes_to_copy < VEC_BUFF_MAX_BYTES ? bytes_to_copy : VEC_BUFF_MAX_BYTES;
        int n_elem = temp_byte_size / elem_size; // number of elements in buffer
        int n_copy = ((n - 1) / n_elem) + 1; // number of times buffer is copied

        int  blocks = (n_elem - 1) / NB_X + 1; // parameters for device kernel
        dim3 grid(blocks);
        dim3 threads(NB_X);

        size_t x_h_byte_stride = (size_t)elem_size * incx;
        size_t y_d_byte_stride = (size_t)elem_size * incy;
        size_t t_h_byte_stride = (size_t)elem_size;

        for(int i_copy = 0; i_copy < n_copy; i_copy++)
        {
            int         i_start     = i_copy * n_elem;
            int         n_elem_max  = n - i_start < n_elem ? n - i_start : n_elem;
            int         contig_size = n_elem_max * elem_size;
            void*       y_d_start   = (char*)y_d + i_start * y_d_byte_stride;
            const void* x_h_start   = (const char*)x_h + i_start * x_h_byte_stride;

            if((incx != 1) && (incy != 1))
            {
                // used unique_ptr to avoid memory leak
                auto  t_h_managed = rocblas_unique_ptr{malloc(temp_byte_size), free};
                void* t_h         = t_h_managed.get();
                if(!t_h)
                    return rocblas_status_memory_error;
                auto  t_d_managed = rocblas_unique_ptr{device_malloc(temp_byte_size), device_free};
                void* t_d         = t_d_managed.get();
                if(!t_d)
                    return rocblas_status_memory_error;
                // non-contiguous host vector -> host buffer
                for(size_t i_b = 0, i_x = i_start; i_b < n_elem_max; i_b++, i_x++)
                {
                    memcpy((char*)t_h + i_b * t_h_byte_stride,
                           (const char*)x_h + i_x * x_h_byte_stride,
                           elem_size);
                }
                // host buffer -> device buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(t_d, t_h, contig_size, hipMemcpyHostToDevice));
                // device buffer -> non-contiguous device vector
                hipLaunchKernelGGL(copy_void_ptr_vector_kernel,
                                   grid,
                                   threads,
                                   0,
                                   0,
                                   n_elem_max,
                                   elem_size,
                                   t_d,
                                   1,
                                   y_d_start,
                                   incy);
            }
            else if(incx == 1 && incy != 1)
            {
                // used unique_ptr to avoid memory leak
                auto  t_d_managed = rocblas_unique_ptr{device_malloc(temp_byte_size), device_free};
                void* t_d         = t_d_managed.get();
                if(!t_d)
                    return rocblas_status_memory_error;
                // contiguous host vector -> device buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(t_d, x_h_start, contig_size, hipMemcpyHostToDevice));
                // device buffer -> non-contiguous device vector
                hipLaunchKernelGGL(copy_void_ptr_vector_kernel,
                                   grid,
                                   threads,
                                   0,
                                   0,
                                   n_elem_max,
                                   elem_size,
                                   t_d,
                                   1,
                                   y_d_start,
                                   incy);
            }
            else if(incx != 1 && incy == 1)
            {
                // used unique_ptr to avoid memory leak
                auto  t_h_managed = rocblas_unique_ptr{malloc(temp_byte_size), free};
                void* t_h         = t_h_managed.get();
                if(!t_h)
                    return rocblas_status_memory_error;
                // non-contiguous host vector -> host buffer
                for(size_t i_b = 0, i_x = i_start; i_b < n_elem_max; i_b++, i_x++)
                {
                    memcpy((char*)t_h + i_b * t_h_byte_stride,
                           (const char*)x_h + i_x * x_h_byte_stride,
                           elem_size);
                }
                // host buffer -> contiguous device vector
                PRINT_IF_HIP_ERROR(hipMemcpy(y_d_start, t_h, contig_size, hipMemcpyHostToDevice));
            }
        }
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   copies void* vector x with stride incx on device to void* vector
     y with stride incy on host. Vectors have n elements of size elem_size.
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_vector(rocblas_int n,
                                             rocblas_int elem_size,
                                             const void* x_d,
                                             rocblas_int incx,
                                             void*       y_h,
                                             rocblas_int incy)
try
{
    if(n == 0) // quick return
        return rocblas_status_success;
    if(n < 0 || incx <= 0 || incy <= 0 || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!x_d || !y_h)
        return rocblas_status_invalid_pointer;

    if(incx == 1 && incy == 1) // congiguous device vector -> congiguous host vector
    {
        PRINT_IF_HIP_ERROR(hipMemcpy(y_h, x_d, elem_size * n, hipMemcpyDeviceToHost));
    }
    else // either device or host vector is non-contiguous
    {
        size_t bytes_to_copy = static_cast<size_t>(elem_size) * static_cast<size_t>(n);
        size_t temp_byte_size
            = bytes_to_copy < VEC_BUFF_MAX_BYTES ? bytes_to_copy : VEC_BUFF_MAX_BYTES;
        int n_elem = temp_byte_size / elem_size; // number elements in buffer
        int n_copy = ((n - 1) / n_elem) + 1; // number of times buffer is copied

        int  blocks = (n_elem - 1) / NB_X + 1; // parameters for device kernel
        dim3 grid(blocks);
        dim3 threads(NB_X);

        size_t x_d_byte_stride = (size_t)elem_size * incx;
        size_t y_h_byte_stride = (size_t)elem_size * incy;
        size_t t_h_byte_stride = (size_t)elem_size;

        for(int i_copy = 0; i_copy < n_copy; i_copy++)
        {
            int i_start           = i_copy * n_elem;
            int n_elem_max        = n - (n_elem * i_copy) < n_elem ? n - (n_elem * i_copy) : n_elem;
            int contig_size       = elem_size * n_elem_max;
            const void* x_d_start = (const char*)x_d + i_start * x_d_byte_stride;
            void*       y_h_start = (char*)y_h + i_start * y_h_byte_stride;

            if(incx != 1 && incy != 1)
            {
                // used unique_ptr to avoid memory leak
                auto  t_h_managed = rocblas_unique_ptr{malloc(temp_byte_size), free};
                void* t_h         = t_h_managed.get();
                if(!t_h)
                    return rocblas_status_memory_error;
                auto  t_d_managed = rocblas_unique_ptr{device_malloc(temp_byte_size), device_free};
                void* t_d         = t_d_managed.get();
                if(!t_d)
                    return rocblas_status_memory_error;
                // non-contiguous device vector -> device buffer
                hipLaunchKernelGGL(copy_void_ptr_vector_kernel,
                                   grid,
                                   threads,
                                   0,
                                   0,
                                   n_elem_max,
                                   elem_size,
                                   x_d_start,
                                   incx,
                                   t_d,
                                   1);
                // device buffer -> host buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(t_h, t_d, contig_size, hipMemcpyDeviceToHost));
                // host buffer -> non-contiguous host vector
                for(size_t i_b = 0, i_y = i_start; i_b < n_elem_max; i_b++, i_y++)
                {
                    memcpy((char*)y_h + i_y * y_h_byte_stride,
                           (const char*)t_h + i_b * t_h_byte_stride,
                           elem_size);
                }
            }
            else if(incx == 1 && incy != 1)
            {
                // used unique_ptr to avoid memory leak
                auto  t_h_managed = rocblas_unique_ptr{malloc(temp_byte_size), free};
                void* t_h         = t_h_managed.get();
                if(!t_h)
                    return rocblas_status_memory_error;
                // congiguous device vector -> host buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(t_h, x_d_start, contig_size, hipMemcpyDeviceToHost));

                // host buffer -> non-contiguous host vector
                for(size_t i_b = 0, i_y = i_start; i_b < n_elem_max; i_b++, i_y++)
                {
                    memcpy((char*)y_h + i_y * y_h_byte_stride,
                           (const char*)t_h + i_b * t_h_byte_stride,
                           elem_size);
                }
            }
            else if(incx != 1 && incy == 1)
            {
                // used unique_ptr to avoid memory leak
                auto  t_d_managed = rocblas_unique_ptr{device_malloc(temp_byte_size), device_free};
                void* t_d         = t_d_managed.get();
                if(!t_d)
                    return rocblas_status_memory_error;
                // non-contiguous device vector -> device buffer
                hipLaunchKernelGGL(copy_void_ptr_vector_kernel,
                                   grid,
                                   threads,
                                   0,
                                   0,
                                   n_elem_max,
                                   elem_size,
                                   x_d_start,
                                   incx,
                                   t_d,
                                   1);
                // device buffer -> contiguous host vector
                PRINT_IF_HIP_ERROR(hipMemcpy(y_h_start, t_d, contig_size, hipMemcpyDeviceToHost));
            }
        }
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   copies void* vector x with stride incx on host to void* vector
     y with stride incy on device. Vectors have n elements of size elem_size.
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_vector_async(rocblas_int n,
                                                   rocblas_int elem_size,
                                                   const void* x_h,
                                                   rocblas_int incx,
                                                   void*       y_d,
                                                   rocblas_int incy,
                                                   hipStream_t stream)
try
{
    if(n == 0) // quick return
        return rocblas_status_success;
    if(n < 0 || incx <= 0 || incy <= 0 || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!x_h || !y_d)
        return rocblas_status_invalid_pointer;

    if(incx == 1 && incy == 1) // contiguous host vector -> contiguous device vector
    {
        PRINT_IF_HIP_ERROR(hipMemcpyAsync(y_d, x_h, elem_size * n, hipMemcpyHostToDevice, stream));
    }
    else // either non-contiguous host vector or non-contiguous device vector
    {
        // pretend data is 2D to compensate for non unit increments
        PRINT_IF_HIP_ERROR(hipMemcpy2DAsync(y_d,
                                            elem_size * incy,
                                            x_h,
                                            elem_size * incx,
                                            elem_size,
                                            n,
                                            hipMemcpyHostToDevice,
                                            stream));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   copies void* vector x with stride incx on device to void* vector
     y with stride incy on host. Vectors have n elements of size elem_size.
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_vector_async(rocblas_int n,
                                                   rocblas_int elem_size,
                                                   const void* x_d,
                                                   rocblas_int incx,
                                                   void*       y_h,
                                                   rocblas_int incy,
                                                   hipStream_t stream)
try
{
    if(n == 0) // quick return
        return rocblas_status_success;
    if(n < 0 || incx <= 0 || incy <= 0 || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!x_d || !y_h)
        return rocblas_status_invalid_pointer;

    if(incx == 1 && incy == 1) // congiguous device vector -> congiguous host vector
    {
        PRINT_IF_HIP_ERROR(hipMemcpyAsync(y_h, x_d, elem_size * n, hipMemcpyDeviceToHost, stream));
    }
    else // either device or host vector is non-contiguous
    {
        // pretend data is 2D to compensate for non unit increments
        PRINT_IF_HIP_ERROR(hipMemcpy2DAsync(y_h,
                                            elem_size * incy,
                                            x_d,
                                            elem_size * incx,
                                            elem_size,
                                            n,
                                            hipMemcpyDeviceToHost,
                                            stream));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief  Matrix copy on device. Matrices are void pointers with element
     size elem_size
 ******************************************************************************/

constexpr size_t      MAT_BUFF_MAX_BYTES = 1048576;
constexpr rocblas_int MATRIX_DIM_X       = 128;
constexpr rocblas_int MATRIX_DIM_Y       = 8;

__global__ void copy_void_ptr_matrix_kernel(rocblas_int rows,
                                            rocblas_int cols,
                                            size_t      elem_size,
                                            const void* a,
                                            rocblas_int lda,
                                            void*       b,
                                            rocblas_int ldb)
{
    rocblas_int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < rows && ty < cols)
        memcpy((char*)b + (tx + ldb * ty) * elem_size,
               (const char*)a + (tx + lda * ty) * elem_size,
               elem_size);
}

/*******************************************************************************
 *! \brief   copies void* matrix a_h with leading dimentsion lda on host to
     void* matrix b_d with leading dimension ldb on device. Matrices have
     size rows * cols with element size elem_size.
 ******************************************************************************/

extern "C" rocblas_status rocblas_set_matrix(rocblas_int rows,
                                             rocblas_int cols,
                                             rocblas_int elem_size,
                                             const void* a_h,
                                             rocblas_int lda,
                                             void*       b_d,
                                             rocblas_int ldb)
try
{
    if(rows == 0 || cols == 0) // quick return
        return rocblas_status_success;
    if(rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || rows > lda || rows > ldb || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!a_h || !b_d)
        return rocblas_status_invalid_pointer;

    // contiguous host matrix -> contiguous device matrix
    if(lda == rows && ldb == rows)
    {
        size_t bytes_to_copy = static_cast<size_t>(elem_size) * static_cast<size_t>(rows)
                               * static_cast<size_t>(cols);
        PRINT_IF_HIP_ERROR(hipMemcpy(b_d, a_h, bytes_to_copy, hipMemcpyHostToDevice));
    }
    // matrix colums too large to fit in temp buffer, copy matrix col by col
    else if(rows * elem_size > MAT_BUFF_MAX_BYTES)
    {
        for(size_t i = 0; i < cols; i++)
        {
            PRINT_IF_HIP_ERROR(hipMemcpy((char*)b_d + ldb * i * elem_size,
                                         (const char*)a_h + lda * i * elem_size,
                                         (size_t)elem_size * rows,
                                         hipMemcpyHostToDevice));
        }
    }
    // columns fit in temp buffer, pack columns in buffer, hipMemcpy host->device, unpack
    // columns
    else
    {
        size_t bytes_to_copy = static_cast<size_t>(elem_size) * static_cast<size_t>(rows)
                               * static_cast<size_t>(cols);
        size_t temp_byte_size
            = bytes_to_copy < MAT_BUFF_MAX_BYTES ? bytes_to_copy : MAT_BUFF_MAX_BYTES;
        int n_cols = temp_byte_size / (elem_size * rows); // number of columns in buffer
        int n_copy = ((cols - 1) / n_cols) + 1; // number of times buffer is copied

        rocblas_int blocksX = ((rows - 1) / MATRIX_DIM_X) + 1; // parameters for device kernel
        rocblas_int blocksY = ((n_cols - 1) / MATRIX_DIM_Y) + 1;
        dim3        grid(blocksX, blocksY);
        dim3        threads(MATRIX_DIM_X, MATRIX_DIM_Y);

        size_t lda_h_byte = (size_t)elem_size * lda;
        size_t ldb_d_byte = (size_t)elem_size * ldb;
        size_t ldt_h_byte = (size_t)elem_size * rows;

        for(int i_copy = 0; i_copy < n_copy; i_copy++)
        {
            size_t      i_start     = i_copy * n_cols;
            int         n_cols_max  = cols - i_start < n_cols ? cols - i_start : n_cols;
            int         contig_size = elem_size * rows * n_cols_max;
            void*       b_d_start   = (char*)b_d + i_start * ldb_d_byte;
            const void* a_h_start   = (const char*)a_h + i_start * lda_h_byte;

            if((lda != rows) && (ldb != rows))
            {
                // used unique_ptr to avoid memory leak
                auto  t_h_managed = rocblas_unique_ptr{malloc(temp_byte_size), free};
                void* t_h         = t_h_managed.get();
                if(!t_h)
                    return rocblas_status_memory_error;
                auto  t_d_managed = rocblas_unique_ptr{device_malloc(temp_byte_size), device_free};
                void* t_d         = t_d_managed.get();
                if(!t_d)
                    return rocblas_status_memory_error;
                // non-contiguous host matrix -> host buffer
                for(size_t i_t = 0, i_a = i_start; i_t < n_cols_max; i_t++, i_a++)
                {
                    memcpy((char*)t_h + i_t * ldt_h_byte,
                           (const char*)a_h + i_a * lda_h_byte,
                           ldt_h_byte);
                }
                // host buffer -> device buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(t_d, t_h, contig_size, hipMemcpyHostToDevice));
                // device buffer -> non-contiguous device matrix
                hipLaunchKernelGGL(copy_void_ptr_matrix_kernel,
                                   grid,
                                   threads,
                                   0,
                                   0,
                                   rows,
                                   n_cols_max,
                                   elem_size,
                                   t_d,
                                   rows,
                                   b_d_start,
                                   ldb);
            }
            else if(lda == rows && ldb != rows)
            {
                // used unique_ptr to avoid memory leak
                auto  t_d_managed = rocblas_unique_ptr{device_malloc(temp_byte_size), device_free};
                void* t_d         = t_d_managed.get();
                if(!t_d)
                    return rocblas_status_memory_error;
                // contiguous host matrix -> device buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(t_d, a_h_start, contig_size, hipMemcpyHostToDevice));
                // device buffer -> non-contiguous device matrix
                hipLaunchKernelGGL(copy_void_ptr_matrix_kernel,
                                   grid,
                                   threads,
                                   0,
                                   0,
                                   rows,
                                   n_cols_max,
                                   elem_size,
                                   t_d,
                                   rows,
                                   b_d_start,
                                   ldb);
            }
            else if(lda != rows && ldb == rows)
            {
                // used unique_ptr to avoid memory leak
                auto  t_h_managed = rocblas_unique_ptr{malloc(temp_byte_size), free};
                void* t_h         = t_h_managed.get();
                if(!t_h)
                    return rocblas_status_memory_error;
                // non-contiguous host matrix -> host buffer
                for(size_t i_t = 0, i_a = i_start; i_t < n_cols_max; i_t++, i_a++)
                {
                    memcpy((char*)t_h + i_t * ldt_h_byte,
                           (const char*)a_h + i_a * lda_h_byte,
                           ldt_h_byte);
                }
                // host buffer -> contiguous device matrix
                PRINT_IF_HIP_ERROR(hipMemcpy(b_d_start, t_h, contig_size, hipMemcpyHostToDevice));
            }
        }
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   copies void* matrix a_h with leading dimentsion lda on host to
     void* matrix b_d with leading dimension ldb on device. Matrices have
     size rows * cols with element size elem_size.
 ******************************************************************************/

extern "C" rocblas_status rocblas_get_matrix(rocblas_int rows,
                                             rocblas_int cols,
                                             rocblas_int elem_size,
                                             const void* a_d,
                                             rocblas_int lda,
                                             void*       b_h,
                                             rocblas_int ldb)
try
{
    if(rows == 0 || cols == 0) // quick return
        return rocblas_status_success;
    if(rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || rows > lda || rows > ldb || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!a_d || !b_h)
        return rocblas_status_invalid_pointer;

    // congiguous device matrix -> congiguous host matrix
    if(lda == rows && ldb == rows)
    {
        size_t bytes_to_copy = elem_size * static_cast<size_t>(rows) * cols;
        PRINT_IF_HIP_ERROR(hipMemcpy(b_h, a_d, bytes_to_copy, hipMemcpyDeviceToHost));
    }
    // columns too large for temp buffer, hipMemcpy column by column
    else if(rows * elem_size > MAT_BUFF_MAX_BYTES)
    {
        for(size_t i = 0; i < cols; i++)
        {
            PRINT_IF_HIP_ERROR(hipMemcpy((char*)b_h + i * ldb * elem_size,
                                         (const char*)a_d + i * lda * elem_size,
                                         elem_size * rows,
                                         hipMemcpyDeviceToHost));
        }
    }
    // columns fit in temp buffer, pack columns in buffer, hipMemcpy device->host, unpack
    // columns
    else
    {
        size_t bytes_to_copy = elem_size * static_cast<size_t>(rows) * cols;
        size_t temp_byte_size
            = bytes_to_copy < MAT_BUFF_MAX_BYTES ? bytes_to_copy : MAT_BUFF_MAX_BYTES;
        int n_cols = temp_byte_size / (elem_size * rows); // number of columns in buffer
        int n_copy = ((cols - 1) / n_cols) + 1; // number times buffer copied

        rocblas_int blocksX = ((rows - 1) / MATRIX_DIM_X) + 1; // parameters for device kernel
        rocblas_int blocksY = ((n_cols - 1) / MATRIX_DIM_Y) + 1;
        dim3        grid(blocksX, blocksY);
        dim3        threads(MATRIX_DIM_X, MATRIX_DIM_Y);

        size_t lda_d_byte = (size_t)elem_size * lda;
        size_t ldb_h_byte = (size_t)elem_size * ldb;
        size_t ldt_h_byte = (size_t)elem_size * rows;

        for(int i_copy = 0; i_copy < n_copy; i_copy++)
        {
            int         i_start     = i_copy * n_cols;
            int         n_cols_max  = cols - i_start < n_cols ? cols - i_start : n_cols;
            size_t      contig_size = elem_size * (size_t)rows * n_cols_max;
            const void* a_d_start   = (const char*)a_d + i_start * lda_d_byte;
            void*       b_h_start   = (char*)b_h + i_start * ldb_h_byte;
            if(lda != rows && ldb != rows)
            {
                // used unique_ptr to avoid memory leak
                auto  t_h_managed = rocblas_unique_ptr{malloc(temp_byte_size), free};
                void* t_h         = t_h_managed.get();
                if(!t_h)
                    return rocblas_status_memory_error;
                auto  t_d_managed = rocblas_unique_ptr{device_malloc(temp_byte_size), device_free};
                void* t_d         = t_d_managed.get();
                if(!t_d)
                    return rocblas_status_memory_error;
                // non-contiguous device matrix -> device buffer
                hipLaunchKernelGGL(copy_void_ptr_matrix_kernel,
                                   grid,
                                   threads,
                                   0,
                                   0,
                                   rows,
                                   n_cols_max,
                                   elem_size,
                                   a_d_start,
                                   lda,
                                   t_d,
                                   rows);
                // device buffer -> host buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(t_h, t_d, contig_size, hipMemcpyDeviceToHost));
                // host buffer -> non-contiguous host matrix
                for(size_t i_t = 0, i_b = i_start; i_t < n_cols_max; i_t++, i_b++)
                {
                    memcpy((char*)b_h + i_b * ldb_h_byte,
                           (const char*)t_h + i_t * ldt_h_byte,
                           ldt_h_byte);
                }
            }
            else if(lda == rows && ldb != rows)
            {
                // used unique_ptr to avoid memory leak
                auto  t_h_managed = rocblas_unique_ptr{malloc(temp_byte_size), free};
                void* t_h         = t_h_managed.get();
                if(!t_h)
                    return rocblas_status_memory_error;
                // congiguous device matrix -> host buffer
                PRINT_IF_HIP_ERROR(hipMemcpy(t_h, a_d_start, contig_size, hipMemcpyDeviceToHost));
                // host buffer -> non-contiguous host matrix
                for(size_t i_t = 0, i_b = i_start; i_t < n_cols_max; i_t++, i_b++)
                {
                    memcpy((char*)b_h + i_b * ldb_h_byte,
                           (const char*)t_h + i_t * ldt_h_byte,
                           ldt_h_byte);
                }
            }
            else if(lda != rows && ldb == rows)
            {
                // used unique_ptr to avoid memory leak
                auto  t_d_managed = rocblas_unique_ptr{device_malloc(temp_byte_size), device_free};
                void* t_d         = t_d_managed.get();
                if(!t_d)
                    return rocblas_status_memory_error;
                // non-contiguous device matrix -> device buffer
                hipLaunchKernelGGL(copy_void_ptr_matrix_kernel,
                                   grid,
                                   threads,
                                   0,
                                   0,
                                   rows,
                                   n_cols_max,
                                   elem_size,
                                   a_d_start,
                                   lda,
                                   t_d,
                                   rows);
                // device temp buffer -> contiguous host matrix
                PRINT_IF_HIP_ERROR(hipMemcpy(b_h_start, t_d, contig_size, hipMemcpyDeviceToHost));
            }
        }
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   copies void* matrix a_h with leading dimentsion lda on host to
     void* matrix b_d with leading dimension ldb on device. Matrices have
     size rows * cols with element size elem_size.
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_matrix_async(rocblas_int rows,
                                                   rocblas_int cols,
                                                   rocblas_int elem_size,
                                                   const void* a_h,
                                                   rocblas_int lda,
                                                   void*       b_d,
                                                   rocblas_int ldb,
                                                   hipStream_t stream)
try
{
    if(rows == 0 || cols == 0) // quick return
        return rocblas_status_success;
    if(rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || rows > lda || rows > ldb || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!a_h || !b_d)
        return rocblas_status_invalid_pointer;

    // contiguous host matrix -> contiguous device matrix
    if(lda == rows && ldb == rows)
    {
        size_t bytes_to_copy = size_t(elem_size) * rows * cols;
        PRINT_IF_HIP_ERROR(hipMemcpyAsync(b_d, a_h, bytes_to_copy, hipMemcpyHostToDevice, stream));
    }
    else
    {
        // width is column vector in matrix
        PRINT_IF_HIP_ERROR(hipMemcpy2DAsync(b_d,
                                            size_t(elem_size) * ldb,
                                            a_h,
                                            size_t(elem_size) * lda,
                                            size_t(elem_size) * rows,
                                            cols,
                                            hipMemcpyHostToDevice,
                                            stream));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   copies void* matrix a_h with leading dimentsion lda on host to
     void* matrix b_d with leading dimension ldb on device. Matrices have
     size rows * cols with element size elem_size.
 ******************************************************************************/

extern "C" rocblas_status rocblas_get_matrix_async(rocblas_int rows,
                                                   rocblas_int cols,
                                                   rocblas_int elem_size,
                                                   const void* a_d,
                                                   rocblas_int lda,
                                                   void*       b_h,
                                                   rocblas_int ldb,
                                                   hipStream_t stream)
try
{
    if(rows == 0 || cols == 0) // quick return
        return rocblas_status_success;
    if(rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || rows > lda || rows > ldb || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!a_d || !b_h)
        return rocblas_status_invalid_pointer;

    // contiguous host matrix -> contiguous device matrix
    if(lda == rows && ldb == rows)
    {
        size_t bytes_to_copy = size_t(elem_size) * rows * cols;
        PRINT_IF_HIP_ERROR(hipMemcpyAsync(b_h, a_d, bytes_to_copy, hipMemcpyDeviceToHost, stream));
    }
    else
    {
        // width is column vector in matrix
        PRINT_IF_HIP_ERROR(hipMemcpy2DAsync(b_h,
                                            size_t(elem_size) * ldb,
                                            a_d,
                                            size_t(elem_size) * lda,
                                            size_t(elem_size) * rows,
                                            cols,
                                            hipMemcpyDeviceToHost,
                                            stream));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

// Convert rocblas_status to string
extern "C" const char* rocblas_status_to_string(rocblas_status status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(rocblas_status_success);
        CASE(rocblas_status_invalid_handle);
        CASE(rocblas_status_not_implemented);
        CASE(rocblas_status_invalid_pointer);
        CASE(rocblas_status_invalid_size);
        CASE(rocblas_status_memory_error);
        CASE(rocblas_status_internal_error);
        CASE(rocblas_status_perf_degraded);
        CASE(rocblas_status_size_query_mismatch);
        CASE(rocblas_status_size_increased);
        CASE(rocblas_status_size_unchanged);
        CASE(rocblas_status_invalid_value);
        CASE(rocblas_status_continue);
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are missing
    // from our switch. If the value is not a valid rocblas_status, we return this string.
    return "<undefined rocblas_status value>";
}

/*******************************************************************************
 * Function to set start/stop event handlers (for internal use only)
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_start_stop_events(rocblas_handle handle,
                                                        hipEvent_t     startEvent,
                                                        hipEvent_t     stopEvent)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    handle->startEvent = startEvent;
    handle->stopEvent  = stopEvent;
    return rocblas_status_success;
}

/*******************************************************************************
 * GPU architecture-related functions
 ******************************************************************************/

// Emulate C++17 std::void_t
template <typename...>
using void_t = void;

// By default, use gcnArch converted to a string prepended by gfx
template <typename PROP, typename = void>
struct ArchName
{
    std::string operator()(const PROP& prop)
    {
        return "gfx" + std::to_string(prop.gcnArch);
    }
};

// If gcnArchName exists as a member, use it instead
template <typename PROP>
struct ArchName<PROP, void_t<decltype(PROP::gcnArchName)>>
{
    std::string operator()(const PROP& prop)
    {
        return prop.gcnArchName;
    }
};

// Get architecture name
std::string rocblas_get_arch_name()
{
    int deviceId;
    hipGetDevice(&deviceId);
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties(&deviceProperties, deviceId);
    return ArchName<hipDeviceProp_t>{}(deviceProperties);
}

// Whether Tensile supports ldc != ldd
// We parse the GPU architecture name, skipping any initial letters (e.g., "gfx")
// If there are not three or more characters after the initial letters, we assume false
// If there are more than 3 characters or any non-digits after the initial letters, we assume true
// Otherwise we assume true iff the value is greater than or equal to 906

bool tensile_supports_ldc_ne_ldd()
{
    std::string arch_name = rocblas_get_arch_name();
    const char* name      = arch_name.c_str();
    while(isalpha(*name))
        ++name;
    return name[0] && name[1] && name[2]
           && (name[3] || !isdigit(name[0]) || !isdigit(name[1]) || !isdigit(name[2])
               || atoi(name) >= 906);
}
