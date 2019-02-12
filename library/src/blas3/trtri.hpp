/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */

#pragma once
#ifndef _TRTRI_HPP_
#define _TRTRI_HPP_

#include <hip/hip_runtime.h>
#include "trtri_device.h"
#include "definitions.h"

/* ============================================================================================ */

/*
   when n <= IB
*/

template <typename T, rocblas_int NB>
__global__ void trtri_small_kernel(rocblas_fill uplo,
                                   rocblas_diagonal diag,
                                   rocblas_int n,
                                   const T* A,
                                   rocblas_int lda,
                                   T* invA,
                                   rocblas_int ldinvA)
{
    trtri_device<T, NB>(uplo, diag, n, A, lda, invA, ldinvA);
}

template <typename T, rocblas_int IB>
rocblas_status rocblas_trtri_small(rocblas_handle handle,
                                   rocblas_fill uplo,
                                   rocblas_diagonal diag,
                                   rocblas_int n,
                                   const T* A,
                                   rocblas_int lda,
                                   T* invA,
                                   rocblas_int ldinvA)
{

    if(n > IB)
    {
        printf("n is %d must be less than %d, will exit\n", n, IB);
        return rocblas_status_not_implemented;
    }

    dim3 grid(1);
    dim3 threads(IB);

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    hipLaunchKernelGGL((trtri_small_kernel<T, IB>),
                       grid,
                       threads,
                       0,
                       rocblas_stream,
                       uplo,
                       diag,
                       n,
                       A,
                       lda,
                       invA,
                       ldinvA);

    return rocblas_status_success;
}

/* ============================================================================================ */

/*
    Invert the IB by IB diagonal blocks of A of size n by n,
    and stores the results in part of invA
*/

template <typename T, rocblas_int IB>
__global__ void trtri_diagonal_kernel(rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      const T* A,
                                      rocblas_int lda,
                                      T* invA,
                                      rocblas_int ldinvA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix

    // each hip thread Block compute a inverse of a IB * IB diagonal block of A
    // notice the last digaonal block may be smaller than IB*IB

    const T* individual_A = A + hipBlockIdx_x * IB * lda + hipBlockIdx_x * IB;
    T* individual_invA    = invA + hipBlockIdx_x * IB * ldinvA + hipBlockIdx_x * IB;

    trtri_device<T, IB>(
        uplo, diag, min(IB, n - hipBlockIdx_x * IB), individual_A, lda, individual_invA, ldinvA);
}

/*
    suppose nn is the orginal matrix AA' size (denoted as n in TOP API)
    A, B, C, submatrices in AA in trtri
    this special gemm performs D  = -A*B*C after trtri

    if  lower,
        D = -A*B*C  ==>  invA21 = -invA22*A21*invA11,
        let m = (nn-IB), n = IB,

    if upper,
        D = -A*B*C  ==>  invA12 = -invA11*A12*invA22,
        let m = IB, n = (nn-IB),

    Then, either case,
        D is of m * n
        A is of m * m
        B if of m * n
        C is of n * n

   since m <= IB, n <= IB,
   create a shared memory space as the buffer to store the intermediate results of W=B*C, A*W

*/

template <typename T, rocblas_int IB>
__global__ void gemm_trsm_kernel(rocblas_int m,
                                 rocblas_int n,
                                 const T* A,
                                 rocblas_int lda,
                                 const T* B,
                                 rocblas_int ldb,
                                 const T* C,
                                 rocblas_int ldc,
                                 T* D,
                                 rocblas_int ldd)
{

    __shared__ T shared_tep[IB * IB];
    __shared__ T vec[IB];
    T reg[IB];

    rocblas_int tx = hipThreadIdx_x;

    // read B into registers, B is of m * n
    if(tx < m)
    {
        for(int col = 0; col < n; col++)
        {
            reg[col] = B[tx + col * ldb];
        }
    }

    // shared_tep = B * C; shared_tep is of m * n, C is of n * n
    for(int col = 0; col < n; col++)
    {
        // load C's column in vec
        if(tx < n)
            vec[tx] = C[col * ldc + tx];
        __syncthreads();

        T reg_tep = 0;
        // perform reduction
        for(int i = 0; i < n; i++)
        {
            reg_tep += reg[i] * vec[i];
        }

        if(tx < m)
        {
            shared_tep[tx + col * IB] = reg_tep;
        }
    }

    __syncthreads();

    // read A into registers A is of m * m
    if(tx < m)
    {
        for(int col = 0; col < m; col++)
        {
            reg[col] = A[tx + col * lda];
        }
    }

    // D = A * shared_tep; shared_tep is of m * n
    for(int col = 0; col < n; col++)
    {

        T reg_tep = 0;
        for(int i = 0; i < m; i++)
        {
            reg_tep += reg[i] * shared_tep[i + col * IB];
        }

        if(tx < m)
        {
            D[tx + col * ldd] = (-1) * reg_tep;
        }
    }
}

/*
   when  IB < n <= 2*IB

    If A is a lower triangular matrix, to compute the invA
    all of Aii, invAii are of size IB by IB

        [ A11   0  ] * [ invA11   0     ]    = [ I 0 ]
        [ A21  A22 ]   [ invA21  invA22 ]      [ 0 I ]

        A11*invA11 = I                 ->  invA11 =  A11^{-1}, by trtri directly
        A22*invA22 = I                 ->  invA22 =  A22^{-1}, by trtri directly
        A21*invA11 + invA22*invA21 = 0 ->  invA21 = -A22^{-1}*A21*invA11 = -invA22*A21*invA11, by
   gemm


    If A is a upper triangular matrix, to compute the invA
    all of Aii, invAii are of size IB by IB

        [ A11  A12  ] * [ invA11  invA12 ]    = [ I 0 ]
        [ 0    A22  ]   [   0     invA22 ]      [ 0 I ]

        A11*invA11 = I                 ->  invA11 =  A11^{-1}, by trtri directly
        A22*invA22 = I                 ->  invA22 =  A22^{-1}, by trtri directly
        A11*invA12 + A12*invA22    = 0 ->  invA12 =  -A11^{-1}*A12*invA22 = -invA11*A12*invA22, by
   gemm

*/

template <typename T, rocblas_int IB>
rocblas_status rocblas_trtri_large(rocblas_handle handle,
                                   rocblas_fill uplo,
                                   rocblas_diagonal diag,
                                   rocblas_int n,
                                   const T* A,
                                   rocblas_int lda,
                                   T* invA,
                                   rocblas_int ldinvA)
{

    if(n > 2 * IB)
    {
        printf("n is %d, n must be less than %d, will return\n", n, 2 * IB);
        return rocblas_status_not_implemented;
    }

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    dim3 grid_trtri(2);
    dim3 threads(IB);

    // first stage: invert IB * IB diagonal blocks of A and write the result of invA11 and invA22 in
    // invA
    hipLaunchKernelGGL((trtri_diagonal_kernel<T, IB>),
                       grid_trtri,
                       threads,
                       0,
                       rocblas_stream,
                       uplo,
                       diag,
                       n,
                       A,
                       lda,
                       invA,
                       ldinvA);

    if(n <= IB)
    {
        // if n is too small, no invA21 or invA12 exist, gemm is not required
        return rocblas_status_success;
    }

    // second stage: using a special gemm to compute invA21 (lower) or invA12 (upper)
    dim3 grid_gemm(1);

    rocblas_int m_gemm;
    rocblas_int n_gemm;
    T* A_gemm;
    const T* B_gemm;
    T* C_gemm;
    T* D_gemm;

    if(uplo == rocblas_fill_lower)
    {
        // perform D = -A*B*C  ==>  invA21 = -invA22*A21*invA11,
        m_gemm = (n - IB);
        n_gemm = IB;
        A_gemm = invA + IB + IB * ldinvA; // invA22
        B_gemm = A + IB;                  // A21
        C_gemm = invA;                    // invA11
        D_gemm = invA + IB;               // invA21
    }
    else
    {
        // perform D = -A*B*C  ==>  invA12 = -invA11*A12*invA22,
        m_gemm = IB;
        n_gemm = (n - IB);
        A_gemm = invA;                    // invA11
        B_gemm = A + lda * IB;            // A12
        C_gemm = invA + IB + IB * ldinvA; // invA22
        D_gemm = invA + IB * ldinvA;      // invA12
    }

    hipLaunchKernelGGL((gemm_trsm_kernel<T, IB>),
                       grid_gemm,
                       threads,
                       0,
                       rocblas_stream,
                       m_gemm,
                       n_gemm,
                       A_gemm,
                       ldinvA,
                       B_gemm,
                       lda,
                       C_gemm,
                       ldinvA,
                       D_gemm,
                       ldinvA);

    return rocblas_status_success;
}

/* ============================================================================================ */

/*! \brief BLAS Level 3 API

    \details
    trtri  compute the inverse of a matrix  A, namely, invA

        and write the result into invA;

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill.
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    diag      rocblas_diagonal.
              = 'rocblas_diagonal_non_unit', A is non-unit triangular;
              = 'rocblas_diagonal_unit', A is unit triangular;
    @param[in]
    n         rocblas_int.
              size of matrix A and invA
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[output]
    invA      pointer storing matrix invA on the GPU.
    @param[in]
    ldinvA    rocblas_int
              specifies the leading dimension of invA.

********************************************************************/
/* IB must be <= 64 in order to fit shared (local) memory */
template <typename T, rocblas_int IB>
rocblas_status rocblas_trtri_template(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      const T* A,
                                      rocblas_int lda,
                                      T* invA,
                                      rocblas_int ldinvA)
{
    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if(n < 0)
        return rocblas_status_invalid_size;
    else if(A == nullptr)
        return rocblas_status_invalid_pointer;
    else if(lda < n)
        return rocblas_status_invalid_size;
    else if(invA == nullptr)
        return rocblas_status_invalid_pointer;
    else if(ldinvA < n)
        return rocblas_status_invalid_size;

    if(n <= IB)
    {
        return rocblas_trtri_small<T, IB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
    }
    else if(n <= 2 * IB)
    {
        return rocblas_trtri_large<T, IB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
    }
    else
    {
        printf("n is %d, n must be less than %d, will return\n", n, 2 * IB);
        return rocblas_status_not_implemented;
    }
}

#endif // _TRTRI_HPP_
