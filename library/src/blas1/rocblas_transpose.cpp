/* ************************************************************************
 * amaxright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
 
#include "status.h"
#include "definitions.h"
#include "device_template.h"
#include "fetch_template.h"
#include "rocblas_unique_ptr.hpp"

#include "transpose_device.h"



/* ============================================================================================ */

/*! \brief BLAS Extension API

    \details
    transpose matrix A of size (m by n) to matrix B (n by m) 

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    m         rocblas_int.
    @param[in]
    n         rocblas_int.
    @param[in]
    A     pointer storing A matrix on the GPU.
    @param[in]
    lda 
              rocblas_int
              specifies the leading dimension for the matrix A
    @param[inout]
    B    pointer storing B matrix on the GPU.
    @param[in]
    ldb
              rocblas_int
              specifies the leading dimension for the matrix B
    ********************************************************************/


template<typename T>
rocblas_status
rocblas_transpose_template(rocblas_handle handle, rocblas_int m, rocblas_int n, const T* A, rocblas_int lda, T* B, rocblas_int ldb, rocblas_int batch_count)
{
    
#define TRANSPOSE_DIM_X 64
#define TRANSPOSE_DIM_Y 16

    if ( nullptr == A )
        return rocblas_status_invalid_pointer;
    else if ( nullptr == B )
        return rocblas_status_invalid_pointer;
    else if( nullptr == handle )
        return rocblas_status_invalid_handle;

    if (m < 0)
        return rocblas_status_invalid_size;
    else if (n < 0)
        return rocblas_status_invalid_size;
    else if (lda < m )
        return rocblas_status_invalid_size;
    else if (ldb < n )
        return rocblas_status_invalid_size;

    if(m == 0 || n == 0 ) return rocblas_status_success;

    dim3 grid((m-1)/TRANSPOSE_DIM_X + 1, ( (n-1)/TRANSPOSE_DIM_X + 1 ) * batch_count, 1);
    dim3 threads(TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, 1);

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel<T, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y>), dim3(grid), dim3(threads), 0, rocblas_stream, m, n, A, B, lda, ldb);
 
    return rocblas_status_success;
 
#undef TRANSPOSE_DIM_X
#undef TRANSPOSE_DIM_Y

}

/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_stranspose(rocblas_handle handle, rocblas_int m, rocblas_int n, const float* A, rocblas_int lda, float* B, rocblas_int ldb)
{
    return rocblas_transpose_template<float>(handle, m, n, A, lda, B, ldb, 1);
}

extern "C"
rocblas_status
rocblas_dtranspose(rocblas_handle handle, rocblas_int m, rocblas_int n, const double* A, rocblas_int lda, double* B, rocblas_int ldb)
{
    return rocblas_transpose_template<double>(handle, m, n, A, lda, B, ldb, 1);
}


extern "C"
rocblas_status
rocblas_ctranspose(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_float_complex* A, rocblas_int lda, rocblas_float_complex* B, rocblas_int ldb)
{
    return rocblas_transpose_template<rocblas_float_complex>(handle, m, n, A, lda, B, ldb, 1);
}

extern "C"
rocblas_status
rocblas_ztranspose(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_double_complex* A, rocblas_int lda, rocblas_double_complex* B, rocblas_int ldb)
{
    //return rocblas_transpose_template<rocblas_double_complex>(handle, m, n, A, lda, B, ldb, 1);
}









/* ============================================================================================ */
