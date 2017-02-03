/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "gemm.hpp"
#include "trtri_trsm.hpp"

#define A(ii, jj) (A + (ii) + (jj)*lda)
#define B(ii, jj) (B + (ii) + (jj)*ldb)
#define X(ii, jj) (X + (ii) + (jj)*ldb)
#define invA(ii) (invA + (ii)*BLOCK)



/* ===============left==================================================== */

template<typename T, rocblas_int  BLOCK>
rocblas_status rocblas_trsm_left(rocblas_handle handle,
    rocblas_fill uplo,
    rocblas_operation transA,
    rocblas_int m, rocblas_int n,
    const T* alpha,
    const T* A, rocblas_int lda,
    T*       B, rocblas_int ldb,
    const T* invA, T* X)
{

    const T negtive_one = -1.0;
    const T one     = 1.0;
    const T zero    = 0.0;

    rocblas_int i, jb;

    //transB is always non-transpose
    rocblas_operation transB = rocblas_operation_none;

    if (transA == transB) {
        if (uplo == rocblas_fill_lower) {
            // left, lower no-transpose
            jb = min(BLOCK, m);
            #ifndef NDEBUG
            printf("1st gemm size %d, %d, %d, %d, %d, %d \n", jb, n, jb, BLOCK, ldb, ldb);
            #endif 
            rocblas_gemm_template<T>(handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero, X, ldb);
            if (BLOCK < m) {
                #ifndef NDEBUG
                printf("2en gemm size %d, %d, %d, %d, %d, %d \n", m-BLOCK, n, BLOCK, lda, ldb, ldb);
                #endif
                rocblas_gemm_template<T>(handle, transA, transB, m-BLOCK, n, BLOCK, &negtive_one, A(BLOCK,0), lda, X, ldb, alpha, B(BLOCK,0), ldb);

                // remaining blocks
                for( i=BLOCK; i < m; i += BLOCK ) {
                    jb = min(m-i, BLOCK);
                    #ifndef NDEBUG
                    printf("3rd gemm size %d, %d, %d, %d, %d, %d \n", jb, n, jb, BLOCK, ldb, ldb);
                    #endif
                    rocblas_gemm_template<T>(handle, transA, transB, jb, n, jb, &one, invA(i), BLOCK, B(i,0), ldb, &zero, X(i,0), ldb);
                    if (i+BLOCK >= m)// this condition is not necessary at all and can be changed as if (i+BLOCK<m)
                        break;
                    #ifndef NDEBUG
                    printf("4th gemm size %d, %d, %d, %d, %d, %d \n", m-i-BLOCK, n, BLOCK, lda, ldb, ldb);
                    #endif
                    rocblas_gemm_template<T>(handle, transA, transB, m-i-BLOCK, n, BLOCK, &negtive_one, A(i+BLOCK,i), lda, X(i,0), ldb, &one, B(i+BLOCK,0), ldb);
                }
            }

#if 0
            for( i=0; i < m; i += BLOCK ) {
                jb = min(m-i, BLOCK);
                T *tmp = (i == 0) ? alpha : one;
                rocblas_gemm_template<T>(handle, transA, transB, jb, n, jb, tmp, invA(i), BLOCK, B(i,0), ldb, &zero, X(i,0), ldb);
                if(i + BLOCK < m){
                    rocblas_gemm_template<T>(handle, transA, transB, m-i-BLOCK, n, BLOCK, &negtive_one, A(i+BLOCK,i), lda, X(i,0), ldb, tmp, B(i+BLOCK,0), ldb);
                }
            }

#endif
        }
        else {
            // left, upper no-transpose
            jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
            i = m-jb; 

            //if m=n=35=lda=ldb, BLOCK =32, then jb = 3, i = 32; {3, 35, 3, 32, 35, 35} 
            rocblas_gemm_template<T>(handle, transA, transB, jb, n, jb, alpha, invA(i), BLOCK, B(i,0), ldb, &zero, X(i,0), ldb);
            if (i-BLOCK >= 0) {

                rocblas_gemm_template<T>(handle, transA, transB, i, n, jb, &negtive_one, A(0,i), lda, X(i,0), ldb, alpha, B, ldb);

                // remaining blocks
                for( i=m-jb-BLOCK; i >= 0; i -= BLOCK ) {
                    //{32, 35, 32, 32, 35, 35}
                    rocblas_gemm_template<T>(handle, transA, transB, BLOCK, n, BLOCK, &one, invA(i), BLOCK, B(i,0), ldb, &zero, X(i,0), ldb);
                    if (i-BLOCK < 0)
                        break;
                    rocblas_gemm_template<T>(handle, transA, transB, i, n, BLOCK, &negtive_one, A(0,i), lda, X(i,0), ldb, &one, B, ldb);
                }
            }
        }
    }
    else {  // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
        if (uplo == rocblas_fill_lower) {
            // left, lower transpose
            jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
            i = m-jb;
            rocblas_gemm_template<T>(handle, transA, transB, jb, n, jb, alpha, invA(i), BLOCK, B(i,0), ldb, &zero, X(i,0), ldb);
            if (i-BLOCK >= 0) {
                rocblas_gemm_template<T>(handle, transA, transB, i, n, jb, &negtive_one, A(i,0), lda, X(i,0), ldb, alpha, B, ldb);

                // remaining blocks
                for( i=m-jb-BLOCK; i >= 0; i -= BLOCK ) {
                    rocblas_gemm_template<T>(handle, transA, transB, BLOCK, n, BLOCK, &one, invA(i), BLOCK, B(i,0), ldb, &zero, X(i,0), ldb);
                    if (i-BLOCK < 0)
                        break;
                    rocblas_gemm_template<T>(handle, transA, transB, i, n, BLOCK, &negtive_one, A(i,0), lda, X(i,0), ldb, &one, B, ldb);
                }
            }
        }
        else {
            // left, upper transpose
            jb = min(BLOCK, m);
            rocblas_gemm_template<T>(handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero, X, ldb);
            if (BLOCK < m) {
                rocblas_gemm_template<T>(handle, transA, transB, m-BLOCK, n, BLOCK, &negtive_one, A(0,BLOCK), lda, X, ldb, alpha, B(BLOCK,0), ldb);

                // remaining blocks
                for( i=BLOCK; i < m; i += BLOCK ) {
                    jb = min(m-i, BLOCK);
                    rocblas_gemm_template<T>(handle, transA, transB, jb, n, jb, &one, invA(i), BLOCK, B(i,0), ldb, &zero, X(i,0), ldb);
                    if (i+BLOCK >= m)
                        break;
                    rocblas_gemm_template<T>(handle, transA, transB, m-i-BLOCK, n, BLOCK, &negtive_one, A(i,i+BLOCK), lda, X(i,0), ldb, &one, B(i+BLOCK,0), ldb);
                }
            }
        }
    }//transpose

    return rocblas_status_success;

}

/* ===============right==================================================== */


template<typename T, rocblas_int  BLOCK>
rocblas_status rocblas_trsm_right(rocblas_handle handle,
    rocblas_fill uplo,
    rocblas_operation transA,
    rocblas_int m, rocblas_int n,
    const T* alpha,
    const T* A, rocblas_int lda,
    T*       B, rocblas_int ldb,
    const T* invA, T* X)
{

    const T negtive_one = -1.0;
    const T one     = 1.0;
    const T zero    = 0.0;

    rocblas_int i, jb;

    //transB is always non-transpose
    rocblas_operation transB = rocblas_operation_none;

    if (transA == transB) {
        if (uplo == rocblas_fill_lower) {
            // right, lower no-transpose
            jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
            i = n-jb;
            rocblas_gemm_template<T>(handle, transB, transA, m, jb, jb, alpha, B(0,i), ldb, invA(i), BLOCK, &zero, X(0,i), ldb);
            if (i-BLOCK >= 0) {
                rocblas_gemm_template<T>(handle, transB, transA, m, i, jb, &negtive_one, X(0,i), ldb, A(i,0), lda, alpha, B, ldb);

                // remaining blocks
                for( i=n-jb-BLOCK; i >= 0; i -= BLOCK ) {
                    rocblas_gemm_template<T>(handle, transB, transA, m, BLOCK, BLOCK, &one, B(0,i), ldb, invA(i), BLOCK, &zero, X(0,i), ldb);
                    if (i-BLOCK < 0)
                        break;
                    rocblas_gemm_template<T>(handle, transB, transA, m, i, BLOCK, &negtive_one, X(0,i), ldb, A(i,0), lda, &one, B, ldb);
                }
            }
        }
        else {
            // right, upper no-transpose
            jb = min(BLOCK, n);
            rocblas_gemm_template<T>(handle, transB, transA, m, jb, jb, alpha, B, ldb, invA, BLOCK, &zero, X, ldb);
            if (BLOCK < n) {
                rocblas_gemm_template<T>(handle, transB, transA, m, n-BLOCK, BLOCK, &negtive_one, X, ldb, A(0,BLOCK), lda, alpha, B(0,BLOCK), ldb);

                // remaining blocks
                for( i=BLOCK; i < n; i += BLOCK ) {
                    jb = min(BLOCK, n-i);
                    rocblas_gemm_template<T>(handle, transB, transA, m, jb, jb, &one, B(0,i), ldb, invA(i), BLOCK, &zero, X(0,i), ldb);
                    if (i+BLOCK >= n)
                        break;
                    rocblas_gemm_template<T>(handle, transB, transA, m, n-i-BLOCK, BLOCK, &negtive_one, X(0,i), ldb, A(i,i+BLOCK), lda, &one, B(0,i+BLOCK), ldb);
                }
            }
        }
    }
    else { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
        if (uplo == rocblas_fill_lower) {
            // right, lower transpose
            jb = min(BLOCK, n);
            rocblas_gemm_template<T>(handle, transB, transA, m, jb, jb, alpha, B, ldb, invA, BLOCK, &zero, X, ldb);
            if (BLOCK < n) {
                rocblas_gemm_template<T>(handle, transB, transA, m, n-BLOCK, BLOCK, &negtive_one, X, ldb, A(BLOCK,0), lda, alpha, B(0,BLOCK), ldb);

                // remaining blocks
                for( i=BLOCK; i < n; i += BLOCK ) {
                    jb = min(BLOCK, n-i);
                    rocblas_gemm_template<T>(handle, transB, transA, m, jb, jb, &one, B(0,i), ldb, invA(i), BLOCK, &zero, X(0,i), ldb);
                    if (i+BLOCK >= n)
                        break;
                    rocblas_gemm_template<T>(handle, transB, transA, m, n-i-BLOCK, BLOCK, &negtive_one, X(0,i), ldb, A(BLOCK+i,i), lda, &one, B(0,i+BLOCK), ldb);
                }
            }
        }
        else {
            // right, upper transpose
            jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
            i = n-jb;
            rocblas_gemm_template<T>(handle, transB, transA, m, jb, jb, alpha, B(0,i), ldb, invA(i), BLOCK, &zero, X(0,i), ldb);
            if (i-BLOCK >= 0) {
                rocblas_gemm_template<T>(handle, transB, transA, m, i, jb, &negtive_one, X(0,i), ldb, A(0,i), lda, alpha, B, ldb);

                // remaining blocks
                for( i=n-jb-BLOCK; i >= 0; i -= BLOCK ) {
                    rocblas_gemm_template<T>(handle, transB, transA, m, BLOCK, BLOCK, &one, B(0,i), ldb, invA(i), BLOCK, &zero, X(0,i), ldb);
                    if (i-BLOCK < 0)
                        break;
                    rocblas_gemm_template<T>(handle, transB, transA, m, i, BLOCK, &negtive_one, X(0,i), ldb, A(0,i), lda, &one, B, ldb);
                }
            }
        }
    }// tranpsose

    return rocblas_status_success;

}

/* ============================================================================================ */

/*! \brief BLAS Level 3 API

    \details

    trsm solves

        op(A)*X = alpha*B or  X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices,
    A is triangular matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    The matrix X is overwritten on B.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.

    @param[in]
    side    rocblas_side.
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.

    @param[in]
    uplo    rocblas_fill.
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA  rocblas_operation.
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.

    @param[in]
    diag    rocblas_diagonal.
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m       rocblas_int.
            m specifies the number of rows of B. m >= 0.

    @param[in]
    n       rocblas_int.
            n specifies the number of columns of B. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            &zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    A       pointer storing matrix A on the GPU.
            of dimension ( lda, k ), where k is m
            when  rocblas_side_left  and
            is  n  when  rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     rocblas_int.
            lda specifies the first dimension of A.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).

    @param[in,output]
    B       pointer storing matrix B on the GPU.

    @param[in]
    ldb    rocblas_int.
           ldb specifies the first dimension of B. ldb >= max( 1, m ).

    ********************************************************************/

template<typename T, rocblas_int BLOCK>
rocblas_status rocblas_trsm_template(rocblas_handle handle,
    rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag,
    rocblas_int m, rocblas_int n,
    const T* alpha,
    T* A, rocblas_int lda,
    T* B, rocblas_int ldb)
{
    //A is of size lda*k
    rocblas_int k = (side == rocblas_side_left ? m : n);

    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if ( m < 0 )
        return rocblas_status_invalid_size;
    else if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( alpha == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( A == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( lda < k )
        return rocblas_status_invalid_size;
    else if ( B == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( ldb < m )
        return rocblas_status_invalid_size;

    // quick return if possible.
    if (m == 0 || n == 0)
        return rocblas_status_success;

    T* invA;
    T* X;
    //invA is of size BLOCK*k, BLOCK is the blocking size
    PRINT_IF_HIP_ERROR(hipMalloc( &invA, BLOCK*k*sizeof(T) ));
    //X is the same size of B
    PRINT_IF_HIP_ERROR(hipMalloc( &X, ldb*n * sizeof(T) ));

    //intialize invA and X to be &zero
    PRINT_IF_HIP_ERROR(hipMemset(invA, 0, BLOCK*k*sizeof(T)));
    //potential bug, may use hipMemcpy B to X
    PRINT_IF_HIP_ERROR(hipMemset(X, 0, ldb*n*sizeof(T)));

    //batched trtri invert diagonal part (BLOCK*BLOCK) of A into invA
    rocblas_status status = rocblas_trtri_trsm_template<T, BLOCK>(handle, uplo, diag,
                                    k, A, lda,
                                    invA);


    if (side == rocblas_side_left) {
        status = rocblas_trsm_left<T, BLOCK>(handle, uplo, transA, m, n, alpha, A, lda, B, ldb, invA, X);
    }
    else {  // side == rocblas_side_right
        status = rocblas_trsm_right<T, BLOCK>(handle, uplo, transA, m, n, alpha, A, lda, B, ldb, invA, X);
    }

    #ifndef NDEBUG
    printf("copy x to b\n");
    #endif
    PRINT_IF_HIP_ERROR(hipMemcpy(B, X, ldb*n*sizeof(T), hipMemcpyDeviceToDevice));//TODO: optimized it with copy kernel
    PRINT_IF_HIP_ERROR(hipFree(invA));
    PRINT_IF_HIP_ERROR(hipFree(X));

    return status;
}


/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_strsm(rocblas_handle handle,
    rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag,
    rocblas_int m, rocblas_int n,
    const float* alpha,
    float* A, rocblas_int lda,
    float* B, rocblas_int ldb){

    //shared memory usuage is (192/2)^2 * sizeof(float) = 36K. LDS is 64K per CU. Theoretically u can use all 64K, but in practice no.
    return rocblas_trsm_template<float, 128>(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}


extern "C"
rocblas_status
rocblas_dtrsm(rocblas_handle handle,
    rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag,
    rocblas_int m, rocblas_int n,
    const double* alpha,
    double* A, rocblas_int lda,
    double* B, rocblas_int ldb){

    //shared memory usuage is (128/2)^2 * sizeof(float) = 32K. LDS is 64K per CU. Theoretically u can use all 64K, but in practice no.
    return rocblas_trsm_template<double, DTRSM_BLOCK>(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}
