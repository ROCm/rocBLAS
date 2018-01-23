/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "geam_device.h"
#include "handle.h"

// general cases for any transA, transB, alpha, beta, lda, ldb, ldc
template <typename T>
__global__ void geam_kernel_host_pointer(rocblas_operation transA,
                                         rocblas_operation transB,
                                         rocblas_int m,
                                         rocblas_int n,
                                         const T alpha,
                                         const T* __restrict__ A,
                                         rocblas_int lda,
                                         const T beta,
                                         const T* __restrict__ B,
                                         rocblas_int ldb,
                                         T* C,
                                         rocblas_int ldc)
{
    geam_device<T>(transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template <typename T>
__global__ void geam_kernel_device_pointer(rocblas_operation transA,
                                           rocblas_operation transB,
                                           rocblas_int m,
                                           rocblas_int n,
                                           const T* alpha,
                                           const T* __restrict__ A,
                                           rocblas_int lda,
                                           const T* beta,
                                           const T* __restrict__ B,
                                           rocblas_int ldb,
                                           T* C,
                                           rocblas_int ldc)
{
    geam_device<T>(transA, transB, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
}

// special cases where: lda=ldb=ldc=m && transA==transB=none so matrices
// are contiguous, there are no transposes, and therefore matrices
// can be treated as contiguous vectors
template <typename T>
__global__ void geam_1D_kernel_host_pointer(rocblas_int size,
                                            const T alpha,
                                            const T* __restrict__ A,
                                            const T beta,
                                            const T* __restrict__ B,
                                            T* C)
{
    geam_1D_device<T>(size, alpha, A, beta, B, C);
}

template <typename T>
__global__ void geam_1D_kernel_device_pointer(rocblas_int size,
                                              const T* alpha,
                                              const T* __restrict__ A,
                                              const T* beta,
                                              const T* __restrict__ B,
                                              T* C)
{
    geam_1D_device<T>(size, *alpha, A, *beta, B, C);
}

// special cases where: lda=ldb=ldc=m && transA==transB=none so matrices
// are contiguous, there are no transposes, and therefore matrices
// can be treated as contiguous vectors.
// Also, alpha == 0  ||  beta == 0  so only one matrix contributes
template <typename T>
__global__ void
geam_1D_2matrix_kernel_host_pointer(rocblas_int size, const T alpha, const T* __restrict__ A, T* C)
{
    geam_1D_2matrix_device<T>(size, alpha, A, C);
}

template <typename T>
__global__ void geam_1D_2matrix_kernel_device_pointer(rocblas_int size,
                                                      const T* alpha,
                                                      const T* __restrict__ A,
                                                      T* C)
{
    geam_1D_2matrix_device<T>(size, *alpha, A, C);
}

// special cases where: alpha == 0 || beta == 0 so only one
// matrix contributes
template <typename T>
__global__ void geam_2matrix_kernel_host_pointer(rocblas_operation transA,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 const T alpha,
                                                 const T* __restrict__ A,
                                                 rocblas_int lda,
                                                 T* C,
                                                 rocblas_int ldc)
{
    geam_2matrix_device<T>(transA, m, n, alpha, A, lda, C, ldc);
}

template <typename T>
__global__ void geam_2matrix_kernel_device_pointer(rocblas_operation transA,
                                                   rocblas_int m,
                                                   rocblas_int n,
                                                   const T* alpha,
                                                   const T* __restrict__ A,
                                                   rocblas_int lda,
                                                   T* C,
                                                   rocblas_int ldc)
{
    geam_2matrix_device<T>(transA, m, n, *alpha, A, lda, C, ldc);
}

// special cases where: A == C && lda == ldc && transA == none
// this is in place case C <- alpha*C + beta*B
template <typename T>
__global__ void geam_inplace_kernel_host_pointer(rocblas_operation transB,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 const T alpha,
                                                 const T beta,
                                                 const T* __restrict__ B,
                                                 rocblas_int ldb,
                                                 T* C,
                                                 rocblas_int ldc)
{
    geam_inplace_device(transB, m, n, alpha, beta, B, ldb, C, ldc);
}

template <typename T>
__global__ void geam_inplace_kernel_device_pointer(rocblas_operation transB,
                                                   rocblas_int m,
                                                   rocblas_int n,
                                                   const T* alpha,
                                                   const T* beta,
                                                   const T* __restrict__ B,
                                                   rocblas_int ldb,
                                                   T* C,
                                                   rocblas_int ldc)
{
    geam_inplace_device(transB, m, n, *alpha, *beta, B, ldb, C, ldc);
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
    beta      specifies the scalar beta.
    @param[in]
    B         pointer storing matrix B on the GPU.
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.
    @param[in, out]
    C         pointer storing matrix C on the GPU.
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.

    ********************************************************************/

template <typename T>
rocblas_status rocblas_geam_template(rocblas_handle handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int m,
                                     rocblas_int n,
                                     const T* alpha,
                                     const T* A,
                                     rocblas_int lda,
                                     const T* beta,
                                     const T* B,
                                     rocblas_int ldb,
                                     T* C,
                                     rocblas_int ldc)
{
    if(nullptr == handle)
    {
        return rocblas_status_invalid_handle;
    }

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_function(handle,
                     replaceX<T>("rocblas_Xgeam"),
                     transA,
                     transB,
                     m,
                     n,
                     *alpha,
                     (const void*&)A,
                     lda,
                     *beta,
                     (const void*&)B,
                     ldb,
                     (const void*&)C,
                     ldc);
    }
    else
    {

        log_function(handle,
                     replaceX<T>("rocblas_Xgeam"),
                     transA,
                     transB,
                     m,
                     n,
                     (const void*&)alpha,
                     (const void*&)A,
                     lda,
                     (const void*&)beta,
                     (const void*&)B,
                     ldb,
                     (const void*&)C,
                     ldc);
    }

    int dim1_A, dim2_A, dim1_B, dim2_B;
    // quick return
    if(0 == m || 0 == n)
    {
        return rocblas_status_success;
    }

    if(transA == rocblas_operation_none)
    {
        dim1_A = m;
        dim2_A = n;
    }
    else
    {
        dim1_A = n;
        dim2_A = m;
    }

    if(transB == rocblas_operation_none)
    {
        dim1_B = m;
        dim2_B = n;
    }
    else
    {
        dim1_B = n;
        dim2_B = m;
    }

    if(m < 0 || n < 0 || lda < dim1_A || ldb < dim1_B || ldc < m)
    {
        return rocblas_status_invalid_size;
    }

    if(nullptr == A || nullptr == B || nullptr == C || nullptr == alpha || nullptr == beta)
    {
        return rocblas_status_invalid_pointer;
    }

    if(((C == A) && ((lda != ldc) || (transA != rocblas_operation_none))) ||
       ((C == B) && ((ldb != ldc) || (transB != rocblas_operation_none))))
    {
        return rocblas_status_invalid_size;
    }

    hipStream_t rocblas_stream = handle->rocblas_stream;

    if((rocblas_pointer_mode_host == handle->pointer_mode) && (0 == *alpha) && (0 == *beta))
    {
        // call hipMemset to set matrix C to zero because alpha and beta on host
        // and alpha = beta = zero

        if(ldc == m)
        {
            // one call to hipMemset because matrix C is coniguous
            hipMemset(C, 0, sizeof(T) * m * n); //  unit-test-covered
        }
        else
        {
            // n calls to hipMemset because matrix C is coniguous
            // note that matrix C is always normal (not transpose)
            for(int i = 0; i < n; i++)
            {
                hipMemset(&(C[i * ldc]), 0, sizeof(T) * m); //  unit-test-covered
            }
        }
    }
    else if(C == A)
    {
// C <- alpha * C + beta * B

#define GEAM_DIM_X 16
#define GEAM_DIM_Y 16
        rocblas_int blocksX = ((m - 1) / GEAM_DIM_X) + 1;
        rocblas_int blocksY = ((n - 1) / GEAM_DIM_Y) + 1;

        dim3 geam_grid(blocksX, blocksY, 1);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1);

        if(rocblas_pointer_mode_host == handle->pointer_mode)
        {
            T h_alpha_scalar = *alpha;
            T h_beta_scalar  = *beta;
            hipLaunchKernelGGL(geam_inplace_kernel_host_pointer<T>,
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               transB,
                               m,
                               n,
                               h_alpha_scalar,
                               h_beta_scalar,
                               B,
                               ldb,
                               C,
                               ldc);
        }
        else
        {
            hipLaunchKernelGGL(geam_inplace_kernel_device_pointer<T>,
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               transB,
                               m,
                               n,
                               alpha,
                               beta,
                               B,
                               ldb,
                               C,
                               ldc);
        }
#undef GEAM_DIM_X
#undef GEAM_DIM_Y
    }
    else if(C == B)
    {
// C <- alpha * A + beta * C

#define GEAM_DIM_X 16
#define GEAM_DIM_Y 16
        rocblas_int blocksX = ((m - 1) / GEAM_DIM_X) + 1;
        rocblas_int blocksY = ((n - 1) / GEAM_DIM_Y) + 1;

        dim3 geam_grid(blocksX, blocksY, 1);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1);

        if(rocblas_pointer_mode_host == handle->pointer_mode)
        {
            T h_alpha_scalar = *alpha;
            T h_beta_scalar  = *beta;
            hipLaunchKernelGGL(geam_inplace_kernel_host_pointer<T>,
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               transA,
                               m,
                               n,
                               h_beta_scalar,
                               h_alpha_scalar,
                               A,
                               lda,
                               C,
                               ldc);
        }
        else
        {
            hipLaunchKernelGGL(geam_inplace_kernel_device_pointer<T>,
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               transA,
                               m,
                               n,
                               beta,
                               alpha,
                               A,
                               lda,
                               C,
                               ldc);
        }
#undef GEAM_DIM_X
#undef GEAM_DIM_Y
    }
    else if((rocblas_pointer_mode_host == handle->pointer_mode) && (0 == *beta))
    {
        if((m == lda) && (transA == rocblas_operation_none) && (m == ldc))
        {
// beta == 0
// special case: A, C are processed as vectors because
// A, C are contiguous, and A is normal (not transpose)

#define GEAM_DIM 256
            int size   = m * n;
            int blocks = ((size - 1) / GEAM_DIM) + 1;

            dim3 geam_grid(blocks, 1, 1);
            dim3 geam_threads(GEAM_DIM, 1, 1);

            hipStream_t rocblas_stream = handle->rocblas_stream;

            T h_alpha_scalar = *alpha; // unit-test-covered
            T h_beta_scalar  = *beta;
            hipLaunchKernelGGL(geam_1D_2matrix_kernel_host_pointer<T>,
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               size,
                               h_alpha_scalar,
                               A,
                               C);

#undef GEAM_DIM
        }
        else
        {
// beta == 0
// general case for any transA, lda, ldc

#define GEAM_DIM_X 16
#define GEAM_DIM_Y 16
            rocblas_int blocksX = ((m - 1) / GEAM_DIM_X) + 1;
            rocblas_int blocksY = ((n - 1) / GEAM_DIM_Y) + 1;

            dim3 geam_grid(blocksX, blocksY, 1);
            dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1);

            T h_alpha_scalar = *alpha;
            hipLaunchKernelGGL(geam_2matrix_kernel_host_pointer<T>, // unit-test-covered
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               transA,
                               m,
                               n,
                               h_alpha_scalar,
                               A,
                               lda,
                               C,
                               ldc);

#undef GEAM_DIM_X
#undef GEAM_DIM_Y
        }
    }
    else if((rocblas_pointer_mode_host == handle->pointer_mode) && (0 == *alpha))
    {
        if((m == ldb) && (transB == rocblas_operation_none) && (m == ldc))
        {
// alpha == 0
// special case: B, C are processed as vectors because
// B, C are contiguous, and B is normal (not transpose)

#define GEAM_DIM 256
            int size   = m * n;
            int blocks = ((size - 1) / GEAM_DIM) + 1;

            dim3 geam_grid(blocks, 1, 1);
            dim3 geam_threads(GEAM_DIM, 1, 1);

            hipStream_t rocblas_stream = handle->rocblas_stream;

            T h_beta_scalar = *beta;
            hipLaunchKernelGGL(geam_1D_2matrix_kernel_host_pointer<T>, // unit-test-covered
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               size,
                               h_beta_scalar,
                               B,
                               C);

#undef GEAM_DIM
        }
        else
        {
// alpha == 0
// general case for any transB, ldb, ldc

#define GEAM_DIM_X 16
#define GEAM_DIM_Y 16
            rocblas_int blocksX = ((m - 1) / GEAM_DIM_X) + 1;
            rocblas_int blocksY = ((n - 1) / GEAM_DIM_Y) + 1;

            dim3 geam_grid(blocksX, blocksY, 1);
            dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1);

            T h_beta_scalar = *beta;
            hipLaunchKernelGGL(geam_2matrix_kernel_host_pointer<T>, // unit-test-covered
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               transB,
                               m,
                               n,
                               h_beta_scalar,
                               B,
                               ldb,
                               C,
                               ldc);

#undef GEAM_DIM_X
#undef GEAM_DIM_Y
        }
    }
    else if((m == lda) && (transA == rocblas_operation_none) && (m == ldb) &&
            (transB == rocblas_operation_none) && (m == ldc))
    {
// special case: A, B, C are processed as vectors because
// A, B, C are contiguous, and A and B are normal (not transpose)

#define GEAM_DIM 256
        int size   = m * n;
        int blocks = ((size - 1) / GEAM_DIM) + 1;

        dim3 geam_grid(blocks, 1, 1);
        dim3 geam_threads(GEAM_DIM, 1, 1);

        hipStream_t rocblas_stream = handle->rocblas_stream;

        if(rocblas_pointer_mode_host == handle->pointer_mode)
        {
            T h_alpha_scalar = *alpha;
            T h_beta_scalar  = *beta;
            hipLaunchKernelGGL(geam_1D_kernel_host_pointer<T>, // unit-test-covered
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               size,
                               h_alpha_scalar,
                               A,
                               h_beta_scalar,
                               B,
                               C);
        }
        else
        {
            hipLaunchKernelGGL(geam_1D_kernel_device_pointer<T>, // unit-test-covered
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               size,
                               alpha,
                               A,
                               beta,
                               B,
                               C);
        }
#undef GEAM_DIM
    }
    else
    {
// general case, any transA, transB, lda, ldb, ldc

#define GEAM_DIM_X 16
#define GEAM_DIM_Y 16
        rocblas_int blocksX = ((m - 1) / GEAM_DIM_X) + 1;
        rocblas_int blocksY = ((n - 1) / GEAM_DIM_Y) + 1;

        dim3 geam_grid(blocksX, blocksY, 1);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y, 1);

        if(rocblas_pointer_mode_host == handle->pointer_mode)
        {
            T h_alpha_scalar = *alpha;
            T h_beta_scalar  = *beta;
            hipLaunchKernelGGL(geam_kernel_host_pointer<T>, // unit-test-covered
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               transA,
                               transB,
                               m,
                               n,
                               h_alpha_scalar,
                               A,
                               lda,
                               h_beta_scalar,
                               B,
                               ldb,
                               C,
                               ldc);
        }
        else
        {
            hipLaunchKernelGGL(geam_kernel_device_pointer<T>, // unit-test-covered
                               dim3(geam_grid),
                               dim3(geam_threads),
                               0,
                               rocblas_stream,
                               transA,
                               transB,
                               m,
                               n,
                               alpha,
                               A,
                               lda,
                               beta,
                               B,
                               ldb,
                               C,
                               ldc);
        }
#undef GEAM_DIM_X
#undef GEAM_DIM_Y
    }

    return rocblas_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocblas_sgeam(rocblas_handle handle,
                                        rocblas_operation transA,
                                        rocblas_operation transB,
                                        rocblas_int m,
                                        rocblas_int n,
                                        const float* alpha,
                                        const float* A,
                                        rocblas_int lda,
                                        const float* beta,
                                        const float* B,
                                        rocblas_int ldb,
                                        float* C,
                                        rocblas_int ldc)
{
    return rocblas_geam_template<float>(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

extern "C" rocblas_status rocblas_dgeam(rocblas_handle handle,
                                        rocblas_operation transA,
                                        rocblas_operation transB,
                                        rocblas_int m,
                                        rocblas_int n,
                                        const double* alpha,
                                        const double* A,
                                        rocblas_int lda,
                                        const double* beta,
                                        const double* B,
                                        rocblas_int ldb,
                                        double* C,
                                        rocblas_int ldc)
{
    return rocblas_geam_template<double>(
        handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
