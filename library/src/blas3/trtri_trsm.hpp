/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef __TRTRI_TRSM_HPP__
#define __TRTRI_TRSM_HPP__

#include "gemm.hpp"
#include "handle.h"
#include "rocblas_trtri.hpp"

static constexpr rocblas_int ROCBLAS_TRTRI_NB = 16;

/*
    Invert the IB by IB diagonal blocks of A of size n by n, where n is divisible by IB
    and stores the results in part of invA of size NB by NB.
    currently IB = NB/2;
    flag indicate whether write into A or invA, invA: 1, A: 0


        [ IB    ]    NB = 2 * IB;
        [    IB ]

*/
template <rocblas_int NB, rocblas_int IB, rocblas_int IBD, typename T, typename U, typename V>
__global__ void trtri_trsm_kernel(rocblas_fill     uplo,
                                  rocblas_diagonal diag,
                                  U                A,
                                  rocblas_int      offset_A,
                                  rocblas_int      lda,
                                  rocblas_stride   stride_A,
                                  V                invA,
                                  rocblas_int      offset_invA,
                                  rocblas_stride   stride_invA)
{
    // get the individual matrix which is processed by device function
    // device function only see one matrix

    // each hip thread Block compute a inverse of a IB * IB diagonal block of A
    rocblas_int offA    = (2 * hipBlockIdx_x) * (IB * lda + IB) + offset_A;
    rocblas_int offinvA = ((2 * hipBlockIdx_x) / IBD) * (NB * NB)
                          + ((2 * hipBlockIdx_x) % IBD) * (IB * NB + IB) + offset_invA;
    const T* a_i    = load_ptr_batch(A, hipBlockIdx_y, offA, stride_A);
    T*       invA_i = load_ptr_batch(invA, hipBlockIdx_y, offinvA, stride_invA);

    custom_trtri_device<IB>(uplo, diag, IB, a_i, lda, invA_i, NB);
}

/* ============================================================================================ */

/*! \brief BLAS Level 3 API

    \details

    This routine is a special routine only called by trsm, it is a private API.
    Internally, it calls batched trtri and batched gemm to compute the inverse
    of the diagonal blocks of a matrix A. The result is in invA. Each individual
    diagonal block invA is NB * NB The last individual diagonal block will be
    padded with 0s if n is not divisible by NB.

    Specifically, it first calls trtri to invert an IB * IB digaonal in this
    NB * NB diagonal block. Second, it finishes the diagonal block by calling
    batched GEMM.

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
template <rocblas_int NB, bool BATCHED, typename T, typename U, typename V>
rocblas_status rocblas_trtri_trsm_template(rocblas_handle   handle,
                                           V                C_tmp,
                                           rocblas_fill     uplo,
                                           rocblas_diagonal diag,
                                           rocblas_int      n,
                                           U                A,
                                           rocblas_int      offset_Ain,
                                           rocblas_int      lda,
                                           rocblas_stride   stride_A,
                                           V                invA,
                                           rocblas_int      offset_invAin,
                                           rocblas_stride   stride_invA,
                                           rocblas_int      batch_count)
{
    // Quick return if possible.
    if(!n)
        return rocblas_status_success;

    rocblas_status status;

    /* sub_blocks is number of divisible NB*NB sub_blocks, but 2 * sub_blocks of IB*IB sub_blocks.
       if n < NB. Then sub_blocks = 0; the trtri_trsm and batched gemm are diabled */

    rocblas_int sub_blocks = n / NB;

    if(sub_blocks)
    {
        constexpr rocblas_int IBD = 8;
        constexpr rocblas_int IB  = NB / IBD;
        dim3                  grid(sub_blocks * IBD / 2, batch_count);
        dim3                  threads(IB * IB);

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

        // invert IB * IB diagonal blocks of A and write the result of invA11 and invA22 in invA

        hipLaunchKernelGGL((trtri_trsm_kernel<NB, IB, IBD, T>),
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           uplo,
                           diag,
                           A,
                           offset_Ain,
                           lda,
                           stride_A,
                           invA,
                           offset_invAin,
                           stride_invA);

        size_t sub_blockSize        = 128;
        size_t tri_elements_to_zero = num_non_tri_elements(NB) * sub_blocks;
        size_t num_sub_blocks       = (tri_elements_to_zero + sub_blockSize - 1) / sub_blockSize;
        hipLaunchKernelGGL(rocblas_trtri_fill<T>,
                           dim3(num_sub_blocks, batch_count),
                           dim3(sub_blockSize),
                           0,
                           handle->rocblas_stream,
                           handle,
                           uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                           NB,
                           num_non_tri_elements(NB),
                           NB,
                           NB * NB,
                           invA,
                           offset_invAin,
                           stride_invA,
                           sub_blocks);

        constexpr rocblas_int JB              = IB * 4;
        rocblas_int           sub_stride_A    = NB * lda + NB;
        rocblas_int           sub_stride_invA = NB * NB;
        rocblas_int           sub_stride_C    = JB * JB;
        rocblas_int           offset_A = (uplo == rocblas_fill_lower ? IB * 2 : IB * 2 * lda);
        rocblas_int offset_invA1       = (uplo == rocblas_fill_lower ? 0 : IB * 2 * NB + IB * 2);
        rocblas_int offset_invA2       = (uplo == rocblas_fill_lower ? IB * 2 * NB + IB * 2 : 0);
        rocblas_int offset_invA3       = (uplo == rocblas_fill_lower ? IB * 2 : IB * 2 * NB);

        trtri_gemm_block<BATCHED, !BATCHED, T>(handle,
                                               IB * 2,
                                               IB * 2,
                                               (U)A,
                                               lda,
                                               stride_A,
                                               sub_stride_A,
                                               (U)invA,
                                               (U)invA,
                                               (V)invA,
                                               NB,
                                               stride_invA,
                                               sub_stride_invA,
                                               (V)C_tmp,
                                               JB,
                                               0,
                                               sub_stride_C,
                                               batch_count,
                                               sub_blocks,
                                               offset_Ain + offset_A,
                                               offset_invAin + offset_invA1,
                                               offset_invAin + offset_invA2,
                                               offset_invAin + offset_invA3,
                                               0);

        offset_A     = (uplo == rocblas_fill_lower ? IB * 4 * lda + IB * 6 : IB * 6 * lda + IB * 4);
        offset_invA1 = (uplo == rocblas_fill_lower ? IB * 4 * NB + IB * 4 : IB * 6 * NB + IB * 6);
        offset_invA2 = (uplo == rocblas_fill_lower ? IB * 6 * NB + IB * 6 : IB * 4 * NB + IB * 4);
        offset_invA3 = (uplo == rocblas_fill_lower ? IB * 4 * NB + IB * 6 : IB * 6 * NB + IB * 4);

        trtri_gemm_block<BATCHED, !BATCHED, T>(handle,
                                               IB * 2,
                                               IB * 2,
                                               (U)A,
                                               lda,
                                               stride_A,
                                               sub_stride_A,
                                               (U)invA,
                                               (U)invA,
                                               (V)invA,
                                               NB,
                                               stride_invA,
                                               sub_stride_invA,
                                               (V)C_tmp,
                                               JB,
                                               0,
                                               sub_stride_C,
                                               batch_count,
                                               sub_blocks,
                                               offset_Ain + offset_A,
                                               offset_invAin + offset_invA1,
                                               offset_invAin + offset_invA2,
                                               offset_invAin + offset_invA3,
                                               0);

        offset_A     = (uplo == rocblas_fill_lower ? JB : JB * lda);
        offset_invA1 = (uplo == rocblas_fill_lower ? 0 : JB * NB + JB);
        offset_invA2 = (uplo == rocblas_fill_lower ? JB * NB + JB : 0);
        offset_invA3 = (uplo == rocblas_fill_lower ? JB : JB * NB);

        trtri_gemm_block<BATCHED, !BATCHED, T>(handle,
                                               JB,
                                               JB,
                                               (U)A,
                                               lda,
                                               stride_A,
                                               sub_stride_A,
                                               (U)invA,
                                               (U)invA,
                                               (V)invA,
                                               NB,
                                               stride_invA,
                                               sub_stride_invA,
                                               (V)C_tmp,
                                               JB,
                                               0,
                                               sub_stride_C,
                                               batch_count,
                                               sub_blocks,
                                               offset_Ain + offset_A,
                                               offset_invAin + offset_invA1,
                                               offset_invAin + offset_invA2,
                                               offset_invAin + offset_invA3,
                                               0);

    } // end if

    // the last digaonal block is handled separately if n is not divisible by NB
    rocblas_int rem = n - sub_blocks * NB;
    if(rem)
    {
        size_t sub_blockSize        = 128;
        size_t tri_elements_to_zero = num_non_tri_elements(rem);
        size_t num_sub_blocks       = (tri_elements_to_zero + sub_blockSize - 1) / sub_blockSize;

        hipLaunchKernelGGL(rocblas_trtri_fill<T>,
                           dim3(num_sub_blocks, batch_count),
                           dim3(sub_blockSize),
                           0,
                           handle->rocblas_stream,
                           handle,
                           uplo == rocblas_fill_lower ? rocblas_fill_upper : rocblas_fill_lower,
                           rem,
                           num_non_tri_elements(rem),
                           NB,
                           0,
                           invA,
                           sub_blocks * NB * NB + offset_invAin,
                           stride_invA,
                           1);

        status = rocblas_trtri_template<ROCBLAS_TRTRI_NB, BATCHED, !BATCHED, T>(
            handle,
            uplo,
            diag,
            rem,
            A,
            sub_blocks * NB * lda + sub_blocks * NB + offset_Ain,
            lda,
            stride_A,
            0,
            invA,
            sub_blocks * NB * NB + offset_invAin,
            NB,
            stride_invA,
            0,
            batch_count,
            1,
            C_tmp);

        if(status != rocblas_status_success)
            return status;
    }

    return rocblas_status_success;
}

#endif // __TRTRI_TRSM_HPP__
