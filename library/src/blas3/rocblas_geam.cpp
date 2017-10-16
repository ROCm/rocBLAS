/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "geam_device.h"

// general cases for any transA, transB, alpha, beta, lda, ldb, ldc
template<typename T>
__global__ void
geam_kernel_host_pointer(hipLaunchParm lp,
    rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n,
    const T alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ B, rocblas_int ldb,
    const T beta,
    T *C, rocblas_int ldc)
{
    geam_device<T>(transA, transB, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename T>
__global__ void
geam_kernel_device_pointer(hipLaunchParm lp,
    rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n,
    const T *alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ B, rocblas_int ldb,
    const T *beta,
    T *C, rocblas_int ldc)
{
    geam_device<T>(transA, transB, m, n, *alpha, A, lda, B, ldb, *beta, C, ldc);
}

// special cases where: lda=ldb=ldc=m && transA==transB=none so matrices 
// are contiguous, there are no transposes, and therefore matrices
// can be treated as contiguous vectors
template<typename T>
__global__ void
geam_1D_kernel_host_pointer(hipLaunchParm lp,
    rocblas_int size,
    const T alpha,
    const T * __restrict__ A,
    const T * __restrict__ B,
    const T beta,
    T *C)
{
    geam_1D_device<T>(size, alpha, A, B, beta, C);
}

template<typename T>
__global__ void
geam_1D_kernel_device_pointer(hipLaunchParm lp,
    rocblas_int size,
    const T *alpha,
    const T * __restrict__ A,
    const T * __restrict__ B,
    const T *beta,
    T *C)
{
    geam_1D_device<T>(size, *alpha, A, B, *beta, C);
}

// special cases where: lda=ldb=ldc=m && transA==transB=none so matrices 
// are contiguous, there are no transposes, and therefore matrices
// can be treated as contiguous vectors.
// Also, alpha == 0  ||  beta == 0  so only one matrix contributes
template<typename T>
__global__ void
geam_1D_2matrix_kernel_host_pointer(hipLaunchParm lp,
    rocblas_int size,
    const T alpha,
    const T * __restrict__ A,
    T *C)
{
    geam_1D_2matrix_device<T>(size, alpha, A, C);
}

template<typename T>
__global__ void
geam_1D_2matrix_kernel_device_pointer(hipLaunchParm lp,
    rocblas_int size,
    const T *alpha,
    const T * __restrict__ A,
    T *C)
{
    geam_1D_2matrix_device<T>(size, *alpha, A, C);
}

// special cases where: alpha == 0 || beta == 0 so only one
// matrix contributes
template<typename T>
__global__ void
geam_2matrix_kernel_host_pointer(hipLaunchParm lp,
    rocblas_operation transA,
    rocblas_int m, rocblas_int n,
    const T alpha,
    const T * __restrict__ A, rocblas_int lda,
    T *C, rocblas_int ldc)
{
    geam_2matrix_device<T>(transA, m, n, alpha, A, lda, C, ldc);
}

template<typename T>
__global__ void
geam_2matrix_kernel_device_pointer(hipLaunchParm lp,
    rocblas_operation transA,
    rocblas_int m, rocblas_int n,
    const T *alpha,
    const T * __restrict__ A, rocblas_int lda,
    T *C, rocblas_int ldc)
{
    geam_2matrix_device<T>(transA, m, n, *alpha, A, lda, C, ldc);
}

// special cases where: A == C && lda == ldc && transA == none
// this is in place case C <- alpha*C + beta*B
template<typename T>
__global__ void
geam_inplace_kernel_host_pointer(hipLaunchParm lp,
    rocblas_operation transB, 
    rocblas_int m, rocblas_int n, 
    const T alpha, 
    const T * __restrict__ B, rocblas_int ldb, 
    const T beta, 
    T *C, rocblas_int ldc)
{
    geam_inplace_device(transB, m, n, alpha, B, ldb, beta, C, ldc);
}

template<typename T>
__global__ void
geam_inplace_kernel_device_pointer(hipLaunchParm lp,
    rocblas_operation transB, 
    rocblas_int m, rocblas_int n, 
    const T *alpha, 
    const T * __restrict__ B, rocblas_int ldb, 
    const T *beta, 
    T *C, rocblas_int ldc)
{
    geam_inplace_device(transB, m, n, *alpha, B, ldb, *beta, C, ldc);
}


/* ============================================================================================ */

    /*
     * ===========================================================================
     *    template interface
     *    template specialization
     *    call GEAM C interfaces (see geam.cpp in the same dir)
     * ===========================================================================
     */

/*! \brief BLAS Level 3 API

    \details
    xGEAM performs one of the matrix-matrix operations

        C = alpha*op( A ) + beta * op( B )

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by n matrix, op( B ) an m by n matrix and C an m by n matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    transA    rocblas_operation
              specifies the form of op( A )
    @param[in]
    transB    rocblas_operation
              specifies the form of op( B )
    @param[in]
    m         rocblas_int.
    @param[in]
    n         rocblas_int.
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    B         pointer storing matrix B on the GPU.
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.
    @param[in]
    beta      specifies the scalar beta.
    @param[in, out]
    C         pointer storing matrix C on the GPU.
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.

    ********************************************************************/

template<typename T>
rocblas_status 
rocblas_geam_template(rocblas_handle handle,
   rocblas_operation transA, rocblas_operation transB,
   rocblas_int m, rocblas_int n,
   const T *alpha,
   const T *A, rocblas_int lda,
   const T *B, rocblas_int ldb,
   const T *beta,
   T *C, rocblas_int ldc)
{
    int dim1_A, dim2_A, dim1_B, dim2_B;
    // quick return
    if (0 == m || 0 == n)
    {
        return rocblas_status_success;
    }

    if (transA == rocblas_operation_none)
    {
        dim1_A = m;
        dim2_A = n;
    }
    else
    {
        dim1_A = n;
        dim2_A = m;
    }

    if (transB == rocblas_operation_none)
    {
        dim1_B = m;
        dim2_B = n;
    }
    else
    {
        dim1_B = n;
        dim2_B = m;
    }

    if (m < 0 || n < 0 || lda < dim1_A || ldb < dim1_B || ldc < m)
    {
        return rocblas_status_invalid_size;
    }

    if (nullptr == A || nullptr == B || nullptr == C || nullptr == alpha || nullptr == beta) 
    {
        return rocblas_status_invalid_pointer;
    }

    if (nullptr == handle) 
    {
        return rocblas_status_invalid_handle;
    }

    if (((C == A) && ((lda != ldc) || (transA != rocblas_operation_none))) || 
        ((C == B) && ((ldb != ldc) || (transB != rocblas_operation_none))))
    {
        return rocblas_status_invalid_size;
    }

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    bool alpha_pointer_mode_host = (rocblas_pointer_to_mode((void*)alpha) == rocblas_pointer_mode_host);
    bool beta_pointer_mode_host = (rocblas_pointer_to_mode((void*)beta) == rocblas_pointer_mode_host);

    if ((alpha_pointer_mode_host && *alpha == 0) && (beta_pointer_mode_host && *beta  == 0))
    {
        // call hipMemset to set matrix C to zero because alpha and beta on host 
        // and alpha = beta = zero

        if(ldc == m)
        {
            // one call to hipMemset because matrix C is coniguous
            hipMemset(C, 0, sizeof(T) * m * n);     //  unit-test-covered
        }
        else
        {
            // n calls to hipMemset because matrix C is coniguous
            // note that matrix C is always normal (not transpose)
            for(int i = 0; i < n; i++)
            {
                hipMemset(&(C[i*ldc]), 0, sizeof(T) * m);    //  unit-test-covered
            }
        }
    }
    else if (C == A)
    {
        // C <- alpha * C + beta * B

        #define  GEAM_DIM_X 16
        #define  GEAM_DIM_Y 16
        rocblas_int blocksX = ((m-1) / GEAM_DIM_X) + 1;
        rocblas_int blocksY = ((n-1) / GEAM_DIM_Y) + 1;

        dim3 geam_grid( blocksX, blocksY, 1 );
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1 );

        if (alpha_pointer_mode_host)
        {
            T h_alpha_scalar = *alpha;
            T h_beta_scalar = *beta;
            hipLaunchKernel(geam_inplace_kernel_host_pointer<T>,
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                transB, m, n, h_alpha_scalar, B, ldb, h_beta_scalar, C, ldc);
        }
        else
        {
            hipLaunchKernel(geam_inplace_kernel_device_pointer<T>,
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                transB, m, n, alpha, B, ldb, beta, C, ldc);
        }
        #undef GEAM_DIM_X
        #undef GEAM_DIM_Y
    }
    else if (C == B)
    {
        // C <- alpha * A + beta * C

        #define  GEAM_DIM_X 16
        #define  GEAM_DIM_Y 16
        rocblas_int blocksX = ((m-1) / GEAM_DIM_X) + 1;
        rocblas_int blocksY = ((n-1) / GEAM_DIM_Y) + 1;

        dim3 geam_grid( blocksX, blocksY, 1 );
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1 );

        if (alpha_pointer_mode_host)
        {
            T h_alpha_scalar = *alpha;
            T h_beta_scalar = *beta;
            hipLaunchKernel(geam_inplace_kernel_host_pointer<T>,
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                transA, m, n, h_beta_scalar, A, lda, h_alpha_scalar, C, ldc);
        }
        else
        {
            hipLaunchKernel(geam_inplace_kernel_device_pointer<T>,
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                transA, m, n, beta, A, lda, alpha, C, ldc);
        }
        #undef GEAM_DIM_X
        #undef GEAM_DIM_Y
    }
    else if (beta_pointer_mode_host && (*beta == 0))
    {
        if ((m == lda) && (transA == rocblas_operation_none) &&
            (m == ldc))
        {
            // beta == 0
            // special case: A, C are processed as vectors because
            // A, C are contiguous, and A is normal (not transpose)

            #define  GEAM_DIM 256
            int size = m*n;
            int blocks = ((size-1) / GEAM_DIM) + 1;

            dim3 geam_grid(blocks, 1, 1);
            dim3 geam_threads(GEAM_DIM, 1, 1);

            hipStream_t rocblas_stream;
            RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

            T h_alpha_scalar = *alpha;                 // unit-test-covered
            T h_beta_scalar = *beta;
            hipLaunchKernel(geam_1D_2matrix_kernel_host_pointer<T>,
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                size, h_alpha_scalar, A, C);

            #undef GEAM_DIM
        }
        else
        {
            // beta == 0
            // general case for any transA, lda, ldc

            #define  GEAM_DIM_X 16
            #define  GEAM_DIM_Y 16
            rocblas_int blocksX = ((m-1) / GEAM_DIM_X) + 1;
            rocblas_int blocksY = ((n-1) / GEAM_DIM_Y) + 1;

            dim3 geam_grid( blocksX, blocksY, 1 );
            dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1 );

            T h_alpha_scalar = *alpha;
            hipLaunchKernel(geam_2matrix_kernel_host_pointer<T>,            // unit-test-covered
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                transA, m, n, h_alpha_scalar, A, lda, C, ldc);

            #undef GEAM_DIM_X
            #undef GEAM_DIM_Y
        }
    }
    else if (rocblas_pointer_mode_host && (*alpha == 0))
    {
        if ((m == ldb) && (transB == rocblas_operation_none) &&
            (m == ldc))
        {
            // alpha == 0
            // special case: B, C are processed as vectors because
            // B, C are contiguous, and B is normal (not transpose)

            #define  GEAM_DIM 256
            int size = m*n;
            int blocks = ((size-1) / GEAM_DIM) + 1;

            dim3 geam_grid(blocks, 1, 1);
            dim3 geam_threads(GEAM_DIM, 1, 1);

            hipStream_t rocblas_stream;
            RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

            T h_beta_scalar = *beta;
            hipLaunchKernel(geam_1D_2matrix_kernel_host_pointer<T>,   // unit-test-covered
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                size, h_beta_scalar, B, C);
            
            #undef GEAM_DIM
        }
        else
        {
            // alpha == 0
            // general case for any transB, ldb, ldc

            #define  GEAM_DIM_X 16
            #define  GEAM_DIM_Y 16
            rocblas_int blocksX = ((m-1) / GEAM_DIM_X) + 1;
            rocblas_int blocksY = ((n-1) / GEAM_DIM_Y) + 1;

            dim3 geam_grid( blocksX, blocksY, 1 );
            dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1 );

            T h_beta_scalar = *beta;
            hipLaunchKernel(geam_2matrix_kernel_host_pointer<T>,        // unit-test-covered
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                transB, m, n, h_beta_scalar, B, ldb, C, ldc);

            #undef GEAM_DIM_X
            #undef GEAM_DIM_Y
        }
    }
    else if ((m == lda) && (transA == rocblas_operation_none) &&
             (m == ldb) && (transB == rocblas_operation_none) &&
             (m == ldc))
    {
        // special case: A, B, C are processed as vectors because
        // A, B, C are contiguous, and A and B are normal (not transpose)

        #define  GEAM_DIM 256
        int size = m*n;
        int blocks = ((size-1) / GEAM_DIM) + 1;

        dim3 geam_grid(blocks, 1, 1);
        dim3 geam_threads(GEAM_DIM, 1, 1);

        hipStream_t rocblas_stream;
        RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

        if (alpha_pointer_mode_host)
        {
            T h_alpha_scalar = *alpha;
            T h_beta_scalar = *beta;
            hipLaunchKernel(geam_1D_kernel_host_pointer<T>,              // unit-test-covered
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                size, h_alpha_scalar, A, B, h_beta_scalar, C);
        }
        else
        {
            hipLaunchKernel(geam_1D_kernel_device_pointer<T>,            // unit-test-covered
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                size, alpha, A, B, beta, C);
        }
        #undef GEAM_DIM
    }
    else
    {
        // general case, any transA, transB, lda, ldb, ldc

        #define  GEAM_DIM_X 16
        #define  GEAM_DIM_Y 16
        rocblas_int blocksX = ((m-1) / GEAM_DIM_X) + 1;
        rocblas_int blocksY = ((n-1) / GEAM_DIM_Y) + 1;

        dim3 geam_grid( blocksX, blocksY, 1 );
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1 );

        if (alpha_pointer_mode_host)
        {
            T h_alpha_scalar = *alpha;
            T h_beta_scalar = *beta;
            hipLaunchKernel(geam_kernel_host_pointer<T>,                                     // unit-test-covered
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                transA, transB, m, n, h_alpha_scalar, A, lda, B, ldb, h_beta_scalar, C, ldc);
        }
        else
        {
            hipLaunchKernel(geam_kernel_device_pointer<T>,                                // unit-test-covered
                dim3(geam_grid), dim3(geam_threads), 0, rocblas_stream,
                transA, transB, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        #undef GEAM_DIM_X
        #undef GEAM_DIM_Y
    }

    return rocblas_status_success;
}

/* ============================================================================================ */
    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */

extern "C"
rocblas_status 
rocblas_sgeam(rocblas_handle handle,
   rocblas_operation transA, rocblas_operation transB,
   rocblas_int m, rocblas_int n,
   const float *alpha,
   const float *A, rocblas_int lda,
   const float *B, rocblas_int ldb,
   const float *beta,
   float *C, rocblas_int ldc)
{
    return rocblas_geam_template<float>(handle, transA, transB, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern "C"
rocblas_status 
rocblas_dgeam(rocblas_handle handle,
   rocblas_operation transA, rocblas_operation transB,
   rocblas_int m, rocblas_int n,
   const double *alpha,
   const double *A, rocblas_int lda,
   const double *B, rocblas_int ldb,
   const double *beta,
   double *C, rocblas_int ldc)
{
    return rocblas_geam_template<double>(handle, transA, transB, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
