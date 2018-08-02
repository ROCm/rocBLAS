/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef __TRTRI_TRSM_HPP__
#define __TRTRI_TRSM_HPP__

#include <hip/hip_runtime.h>

#include "definitions.h"
#include "status.h"
#include "trtri.hpp"
#include "gemm.hpp"
#include "rocblas_unique_ptr.hpp"

/*
    Invert the IB by IB diagonal blocks of A of size n by n, where n is divisible by IB
    and stores the results in part of invA of size NB by NB.
    currently IB = NB/2;
    flag indicate whether write into A or invA, invA: 1, A: 0


        [ IB    ]    NB = 2 * IB;
        [    IB ]

*/
template <typename T, rocblas_int NB>
__global__ void trtri_trsm_kernel(
    rocblas_fill uplo, rocblas_diagonal diag, rocblas_int n, const T* A, rocblas_int lda, T* invA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix

    // each hip thread Block compute a inverse of a IB * IB diagonal block of A

    T* individual_invA;
    individual_invA = invA + hipBlockIdx_x / 2 * NB * NB;
    // the odd thread block makes a shift
    if(hipBlockIdx_x % 2 == 1)
    {
        individual_invA += NB * (NB / 2) + (NB / 2);
    }

    trtri_device<T, NB / 2>(uplo,
                            diag,
                            (NB / 2),
                            A + hipBlockIdx_x * (NB / 2) * lda + hipBlockIdx_x * (NB / 2),
                            lda,
                            individual_invA,
                            NB);
}

/* ============================================================================================ */

/*! \brief BLAS Level 3 API

    \details

    This routine is a special routine only called by trsm, it is a private API.
    Internally, it calls batched trtri and batched gemm to
    compute the inverse of the diagonal blocks of a matrix  A. The result is in invA
    Each individual digaonal block invA is NB * NB
    The last individual diagonal block will be pad 0 if n is not divisible by NB

    Specifically, it first calls trtri to invert a  IB * IB digaonal in this NB * NB diagonal block
    Second, it finishs the diagonal block by calling batched GEMM


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
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[output]
    invA
              of dimension (NB, ceil(n/NB)*NB),
              On exit, contains inverses of the NB by NB diagonal blocks of A.

    ********************************************************************/

// assume invA has already been allocated, and leading dimension of invA is NB
// assume IB is exactly half of NB
template <typename T, rocblas_int NB>
rocblas_status rocblas_trtri_trsm_template(rocblas_handle handle,
                                           T* C_tmp,
                                           rocblas_fill uplo,
                                           rocblas_diagonal diag,
                                           rocblas_int n,
                                           const T* A,
                                           rocblas_int lda,
                                           T* invA)
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

    /*
     * Quick return if possible.
     */

    if(n == 0)
        return rocblas_status_success;

    rocblas_status status;

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    /* blocks is number of divisible NB*NB blocks, but 2 * blocks of IB*IB blocks.
       if n < NB. Then blocks = 0; the trtri_trsm and batched gemm are diabled */

    rocblas_int blocks = n / NB;

    if(blocks > 0)
    {

        rocblas_int IB = NB / 2;
        dim3 grid(blocks * 2, 1, 1);
        dim3 threads(IB, 1, 1);

        /*
           Algorithm:

            If A is a lower triangular matrix, to compute the invA
            all of Aii, invAii are of size IB by IB

                [ A11   0  ] * [ invA11   0     ]    = [ I 0 ]
                [ A21  A22 ]   [ invA21  invA22 ]      [ 0 I ]

                A11*invA11 = I                 ->  invA11 =  A11^{-1}, by trtri directly
                A22*invA22 = I                 ->  invA22 =  A22^{-1}, by trtri directly
                A21*invA11 +  A22*invA21 = 0 ->  invA21 = -A22^{-1}*A21*invA11 = -invA22*A21*invA11,
           by gemm


            If A is a upper triangular matrix, to compute the invA
            all of Aii, invAii are of size IB by IB

                [ A11  A12  ] * [ invA11  invA12 ]    = [ I 0 ]
                [ 0    A22  ]   [   0     invA22 ]      [ 0 I ]

                A11*invA11 = I                 ->  invA11 =  A11^{-1}, by trtri directly
                A22*invA22 = I                 ->  invA22 =  A22^{-1}, by trtri directly
                A11*invA12 + A12*invA22    = 0 ->  invA12 =  -A11^{-1}*A12*invA22 =
           -invA11*A12*invA22, by gemm

        */

        // invert IB * IB diagoanl blocks of A and write the result of invA11 and invA22 in invA

        hipLaunchKernelGGL((trtri_trsm_kernel<T, NB>),
                           dim3(grid),
                           dim3(threads),
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           (blocks)*NB,
                           A,
                           lda,
                           invA);

        T one          = 1;
        T zero         = 0;
        T negative_one = -1;

        rocblas_int stride_A    = NB * lda + NB;
        rocblas_int stride_invA = NB * NB;
        rocblas_int stride_C    = IB * IB;

        rocblas_int A12_A21_offset, invA11_offset, invA22_offset, invA21_invA12_offset;

        //                A21*invA11 + invA22*invA21 = 0 ->  invA21 = -A22^{-1}*A21*invA11 =
        //                -invA22*A21*invA11, by gemm

        if(uplo == rocblas_fill_lower)
        {
            A12_A21_offset       = IB;           // A21
            invA11_offset        = 0;            // invA11 in lower
            invA21_invA12_offset = IB;           // invA21
            invA22_offset        = IB * NB + IB; // invA22 in lower
        }
        else
        {
            A12_A21_offset       = IB * NB;      // A12
            invA11_offset        = NB * IB + IB; // invA22 in upper
            invA21_invA12_offset = IB * NB;      // invA12
            invA22_offset        = 0;            // A11 in upper
        }

#ifndef NDEBUG
        printf("first batched gemm\n");
#endif
        // first batched gemm compute C = A21*invA11 (lower) or C = A12*invA22 (upper)
        // distance between each invA11 or invA22 is stride_invA,  stride_A for each A21 or A12, C
        // of size IB * IB
        status = rocblas_gemm_strided_batched_template<T>(
            handle,
            rocblas_operation_none,
            rocblas_operation_none,
            IB,
            IB,
            IB,
            &one,
            (const T*)(A + ((uplo == rocblas_fill_lower) ? IB : IB * lda)),
            lda,
            stride_A,
            (const T*)(invA + ((uplo == rocblas_fill_lower) ? 0 : IB * NB + IB)),
            NB,
            stride_invA,
            &zero,
            (T*)C_tmp,
            IB,
            stride_C,
            blocks);

#ifndef NDEBUG
        printf("second batched gemm\n");
#endif
        // second batched gemm compute  invA21 = -invA22 * C (lower) or invA12 = -invA11*C (upper)
        // distance between each invA21 or invA12 is stride_invA,
        status = rocblas_gemm_strided_batched_template<T>(
            handle,
            rocblas_operation_none,
            rocblas_operation_none,
            IB,
            IB,
            IB,
            &negative_one,
            (const T*)(invA + ((uplo == rocblas_fill_lower) ? IB * NB + IB : 0)),
            NB,
            stride_invA,
            (const T*)C_tmp,
            IB,
            stride_C,
            &zero,
            (invA + ((uplo == rocblas_fill_lower) ? IB : IB * NB)),
            NB,
            stride_invA,
            blocks);

    } // end if

    // the last digaonal block is handled seperately if n is not divisible by NB, or if there is
    // only one block
    if(n % NB != 0 || blocks == 0)
    {
        status = rocblas_trtri_template<T, NB / 2>(handle,
                                                   uplo,
                                                   diag,
                                                   n - blocks * NB,
                                                   A + blocks * NB * lda + blocks * NB,
                                                   lda,
                                                   invA + blocks * NB * NB,
                                                   NB);
    }

    return rocblas_status_success;
}

#endif // __TRTRI_TRSM_HPP__
