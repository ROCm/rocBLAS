/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#include "handle.h"
#include "logging.h"
#include "rocblas-auxiliary.h"
#include "rocblas-types.h"
#include "utility.h"
#include <cstdio>
#include <memory>

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
    if(ptr == attribute.devicePointer)
        return rocblas_pointer_mode_device;
    else
        return rocblas_pointer_mode_host;
}

/*******************************************************************************
 * ! \brief get pointer mode, can be host or device
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_pointer_mode(rocblas_handle        handle,
                                                   rocblas_pointer_mode* mode)
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_pointer;
    *mode = handle->pointer_mode;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_get_pointer_mode", *mode);
    return rocblas_status_success;
}

/*******************************************************************************
 * ! \brief set pointer mode to host or device
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_pointer_mode(rocblas_handle handle, rocblas_pointer_mode mode)
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_pointer;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_set_pointer_mode", mode);
    handle->pointer_mode = mode;
    return rocblas_status_success;
}

/*******************************************************************************
 * ! \brief create rocblas handle called before any rocblas library routines
 ******************************************************************************/
extern "C" rocblas_status rocblas_create_handle(rocblas_handle* handle)
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;

    // allocate on heap
    try
    {
        *handle = new _rocblas_handle();

        if((*handle)->layer_mode & rocblas_layer_mode_log_trace)
            log_trace(*handle, "rocblas_create_handle");
    }
    catch(...)
    {
        return rocblas_status_internal_error;
    }
    return rocblas_status_success;
}

/*******************************************************************************
 *! \brief release rocblas handle, will implicitly synchronize host and device
 ******************************************************************************/
extern "C" rocblas_status rocblas_destroy_handle(rocblas_handle handle)
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_destroy_handle");
    // call destructor
    try
    {
        delete handle;
    }
    catch(rocblas_status status)
    {
        return status;
    }
    return rocblas_status_success;
}

/*******************************************************************************
 *! \brief   set rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 *   stream_id must be created before this call
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream_id)
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_set_stream", stream_id);
    return handle->set_stream(stream_id);
}

/*******************************************************************************
 *! \brief   get rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t* stream_id)
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        log_trace(handle, "rocblas_get_stream", *stream_id);
    return handle->get_stream(stream_id);
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
    return rocblas_status_internal_error;
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
    return rocblas_status_internal_error;
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
    return rocblas_status_internal_error;
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
    return rocblas_status_internal_error;
}
