/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "geam_device.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

namespace {

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
    geam_device(transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
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
    geam_device(transA, transB, m, n, *alpha, A, lda, *beta, B, ldb, C, ldc);
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
    geam_1D_device(size, alpha, A, beta, B, C);
}

template <typename T>
__global__ void geam_1D_kernel_device_pointer(rocblas_int size,
                                              const T* alpha,
                                              const T* __restrict__ A,
                                              const T* beta,
                                              const T* __restrict__ B,
                                              T* C)
{
    geam_1D_device(size, *alpha, A, *beta, B, C);
}

// special cases where: lda=ldb=ldc=m && transA==transB=none so matrices
// are contiguous, there are no transposes, and therefore matrices
// can be treated as contiguous vectors.
// Also, alpha == 0  ||  beta == 0  so only one matrix contributes
template <typename T>
__global__ void
geam_1D_2matrix_kernel_host_pointer(rocblas_int size, const T alpha, const T* __restrict__ A, T* C)
{
    geam_1D_2matrix_device(size, alpha, A, C);
}

template <typename T>
__global__ void geam_1D_2matrix_kernel_device_pointer(rocblas_int size,
                                                      const T* alpha,
                                                      const T* __restrict__ A,
                                                      T* C)
{
    geam_1D_2matrix_device(size, *alpha, A, C);
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
    geam_2matrix_device(transA, m, n, alpha, A, lda, C, ldc);
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
    geam_2matrix_device(transA, m, n, *alpha, A, lda, C, ldc);
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

template <typename>
constexpr char rocblas_geam_name[] = "unknown";
template <>
constexpr char rocblas_geam_name<float>[] = "rocblas_sgeam";
template <>
constexpr char rocblas_geam_name<double>[] = "rocblas_dgeam";

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
    if(!handle)
        return rocblas_status_invalid_handle;

    auto pointer_mode = handle->pointer_mode;
    auto layer_mode   = handle->layer_mode;

    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        auto transA_letter = rocblas_transpose_letter(transA);
        auto transB_letter = rocblas_transpose_letter(transB);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_geam_name<T>,
                          transA,
                          transB,
                          m,
                          n,
                          *alpha,
                          A,
                          lda,
                          *beta,
                          B,
                          ldb,
                          C,
                          ldc);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f geam -r",
                          rocblas_precision_string<T>,
                          "--transposeA",
                          transA_letter,
                          "--transposeB",
                          transB_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "--alpha",
                          *alpha,
                          "--lda",
                          lda,
                          "--beta",
                          *beta,
                          "--ldb",
                          ldb,
                          "--ldc",
                          ldc);
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_geam_name<T>,
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

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_geam_name<T>,
                        "transA",
                        transA_letter,
                        "transB",
                        transB_letter,
                        "M",
                        m,
                        "N",
                        n,
                        "lda",
                        lda,
                        "ldb",
                        ldb,
                        "ldc",
                        ldc);
    }

    // quick return
    if(!m || !n)
        return rocblas_status_success;

    if(m < 0 || n < 0 || ldc < m || lda < (transA == rocblas_operation_none ? m : n) ||
       ldb < (transB == rocblas_operation_none ? m : n))
        return rocblas_status_invalid_size;

    if(!A || !B || !C || !alpha || !beta)
        return rocblas_status_invalid_pointer;

    if((C == A && (lda != ldc || transA != rocblas_operation_none)) ||
       (C == B && (ldb != ldc || transB != rocblas_operation_none)))
        return rocblas_status_invalid_size;

    hipStream_t rocblas_stream = handle->rocblas_stream;

    if(pointer_mode == rocblas_pointer_mode_host && !*alpha && !*beta)
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
                hipMemset(&C[i * ldc], 0, sizeof(T) * m); //  unit-test-covered
        }
    }
    else if(C == A)
    {
        // C <- alpha * C + beta * B
        static constexpr int GEAM_DIM_X = 16;
        static constexpr int GEAM_DIM_Y = 16;
        rocblas_int blocksX             = (m - 1) / GEAM_DIM_X + 1;
        rocblas_int blocksY             = (n - 1) / GEAM_DIM_Y + 1;

        dim3 geam_grid(blocksX, blocksY);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            hipLaunchKernelGGL(geam_inplace_kernel_host_pointer<T>,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transB,
                               m,
                               n,
                               *alpha,
                               *beta,
                               B,
                               ldb,
                               C,
                               ldc);
        }
        else
        {
            hipLaunchKernelGGL(geam_inplace_kernel_device_pointer<T>,
                               geam_grid,
                               geam_threads,
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
    }
    else if(C == B)
    {
        // C <- alpha * A + beta * C
        static constexpr int GEAM_DIM_X = 16;
        static constexpr int GEAM_DIM_Y = 16;
        rocblas_int blocksX             = (m - 1) / GEAM_DIM_X + 1;
        rocblas_int blocksY             = (n - 1) / GEAM_DIM_Y + 1;

        dim3 geam_grid(blocksX, blocksY);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            hipLaunchKernelGGL(geam_inplace_kernel_host_pointer<T>,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transA,
                               m,
                               n,
                               *beta,
                               *alpha,
                               A,
                               lda,
                               C,
                               ldc);
        }
        else
        {
            hipLaunchKernelGGL(geam_inplace_kernel_device_pointer<T>,
                               geam_grid,
                               geam_threads,
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
    }
    else if(pointer_mode == rocblas_pointer_mode_host && !*beta)
    {
        if(m == lda && transA == rocblas_operation_none && m == ldc)
        {
            // beta == 0
            // special case: A, C are processed as vectors because
            // A, C are contiguous, and A is normal (not transpose)
            static constexpr int GEAM_DIM = 256;
            int size                      = m * n;
            int blocks                    = (size - 1) / GEAM_DIM + 1;

            dim3 geam_grid(blocks);
            dim3 geam_threads(GEAM_DIM);

            hipLaunchKernelGGL(geam_1D_2matrix_kernel_host_pointer<T>,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               size,
                               *alpha,
                               A,
                               C);
        }
        else
        {
            // beta == 0
            // general case for any transA, lda, ldc
            static constexpr int GEAM_DIM_X = 16;
            static constexpr int GEAM_DIM_Y = 16;
            rocblas_int blocksX             = (m - 1) / GEAM_DIM_X + 1;
            rocblas_int blocksY             = (n - 1) / GEAM_DIM_Y + 1;

            dim3 geam_grid(blocksX, blocksY);
            dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

            hipLaunchKernelGGL(geam_2matrix_kernel_host_pointer<T>,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transA,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               C,
                               ldc);
        }
    }
    else if(rocblas_pointer_mode_host == pointer_mode && !*alpha)
    {
        if(m == ldb && transB == rocblas_operation_none && m == ldc)
        {
            // alpha == 0
            // special case: B, C are processed as vectors because
            // B, C are contiguous, and B is normal (not transpose)
            static constexpr int GEAM_DIM = 256;
            int size                      = m * n;
            int blocks                    = (size - 1) / GEAM_DIM + 1;

            dim3 geam_grid(blocks);
            dim3 geam_threads(GEAM_DIM);

            hipLaunchKernelGGL(geam_1D_2matrix_kernel_host_pointer<T>,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               size,
                               *beta,
                               B,
                               C);
        }
        else
        {
            // alpha == 0
            // general case for any transB, ldb, ldc
            static constexpr int GEAM_DIM_X = 16;
            static constexpr int GEAM_DIM_Y = 16;

            rocblas_int blocksX = (m - 1) / GEAM_DIM_X + 1;
            rocblas_int blocksY = (n - 1) / GEAM_DIM_Y + 1;

            dim3 geam_grid(blocksX, blocksY);
            dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

            hipLaunchKernelGGL(geam_2matrix_kernel_host_pointer<T>,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transB,
                               m,
                               n,
                               *beta,
                               B,
                               ldb,
                               C,
                               ldc);
        }
    }
    else if(m == lda && transA == rocblas_operation_none && m == ldb &&
            transB == rocblas_operation_none && m == ldc)
    {
        // special case: A, B, C are processed as vectors because
        // A, B, C are contiguous, and A and B are normal (not transpose)
        static constexpr int GEAM_DIM = 256;
        int size                      = m * n;
        int blocks                    = (size - 1) / GEAM_DIM + 1;

        dim3 geam_grid(blocks);
        dim3 geam_threads(GEAM_DIM);

        if(rocblas_pointer_mode_host == pointer_mode)
        {
            hipLaunchKernelGGL(geam_1D_kernel_host_pointer<T>,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               size,
                               *alpha,
                               A,
                               *beta,
                               B,
                               C);
        }
        else
        {
            hipLaunchKernelGGL(geam_1D_kernel_device_pointer<T>,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               size,
                               alpha,
                               A,
                               beta,
                               B,
                               C);
        }
    }
    else
    {
        // general case, any transA, transB, lda, ldb, ldc
        static constexpr int GEAM_DIM_X = 16;
        static constexpr int GEAM_DIM_Y = 16;

        rocblas_int blocksX = (m - 1) / GEAM_DIM_X + 1;
        rocblas_int blocksY = (n - 1) / GEAM_DIM_Y + 1;

        dim3 geam_grid(blocksX, blocksY);
        dim3 geam_threads(GEAM_DIM_X, GEAM_DIM_Y);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            hipLaunchKernelGGL(geam_kernel_host_pointer<T>,
                               geam_grid,
                               geam_threads,
                               0,
                               rocblas_stream,
                               transA,
                               transB,
                               m,
                               n,
                               *alpha,
                               A,
                               lda,
                               *beta,
                               B,
                               ldb,
                               C,
                               ldc);
        }
        else
        {
            hipLaunchKernelGGL(geam_kernel_device_pointer<T>,
                               geam_grid,
                               geam_threads,
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
    }

    return rocblas_status_success;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_sgeam(rocblas_handle handle,
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

rocblas_status rocblas_dgeam(rocblas_handle handle,
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

} // extern "C"
