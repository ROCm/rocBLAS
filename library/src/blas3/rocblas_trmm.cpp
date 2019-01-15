/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "definitions.h"
#include "logging.h"
#include "utility.h"

namespace {

// do not use fma which is 50% slower than regular fmaf
#define fmaf(a, b, c) (a) * (b) + (c)

#define M6x6                                       \
    rA[0][0] = lA[offA + 0];                       \
    rA[0][1] = lA[offA + 16];                      \
    rA[0][2] = lA[offA + 32];                      \
    rA[0][3] = lA[offA + 48];                      \
    rA[0][4] = lA[offA + 64];                      \
    rA[0][5] = lA[offA + 80];                      \
    rB[0][0] = lB[offB + 0];                       \
    rB[0][1] = lB[offB + 16];                      \
    rB[0][2] = lB[offB + 32];                      \
    rB[0][3] = lB[offB + 48];                      \
    rB[0][4] = lB[offB + 64];                      \
    rB[0][5] = lB[offB + 80];                      \
    offA += 97;                                    \
    offB += 97;                                    \
    rC[0][0] = fmaf(rA[0][0], rB[0][0], rC[0][0]); \
    rC[1][0] = fmaf(rA[0][1], rB[0][0], rC[1][0]); \
    rC[2][0] = fmaf(rA[0][2], rB[0][0], rC[2][0]); \
    rC[3][0] = fmaf(rA[0][3], rB[0][0], rC[3][0]); \
    rC[4][0] = fmaf(rA[0][4], rB[0][0], rC[4][0]); \
    rC[5][0] = fmaf(rA[0][5], rB[0][0], rC[5][0]); \
    rC[0][1] = fmaf(rA[0][0], rB[0][1], rC[0][1]); \
    rC[1][1] = fmaf(rA[0][1], rB[0][1], rC[1][1]); \
    rC[2][1] = fmaf(rA[0][2], rB[0][1], rC[2][1]); \
    rC[3][1] = fmaf(rA[0][3], rB[0][1], rC[3][1]); \
    rC[4][1] = fmaf(rA[0][4], rB[0][1], rC[4][1]); \
    rC[5][1] = fmaf(rA[0][5], rB[0][1], rC[5][1]); \
    rC[0][2] = fmaf(rA[0][0], rB[0][2], rC[0][2]); \
    rC[1][2] = fmaf(rA[0][1], rB[0][2], rC[1][2]); \
    rC[2][2] = fmaf(rA[0][2], rB[0][2], rC[2][2]); \
    rC[3][2] = fmaf(rA[0][3], rB[0][2], rC[3][2]); \
    rC[4][2] = fmaf(rA[0][4], rB[0][2], rC[4][2]); \
    rC[5][2] = fmaf(rA[0][5], rB[0][2], rC[5][2]); \
    rC[0][3] = fmaf(rA[0][0], rB[0][3], rC[0][3]); \
    rC[1][3] = fmaf(rA[0][1], rB[0][3], rC[1][3]); \
    rC[2][3] = fmaf(rA[0][2], rB[0][3], rC[2][3]); \
    rC[3][3] = fmaf(rA[0][3], rB[0][3], rC[3][3]); \
    rC[4][3] = fmaf(rA[0][4], rB[0][3], rC[4][3]); \
    rC[5][3] = fmaf(rA[0][5], rB[0][3], rC[5][3]); \
    rC[0][4] = fmaf(rA[0][0], rB[0][4], rC[0][4]); \
    rC[1][4] = fmaf(rA[0][1], rB[0][4], rC[1][4]); \
    rC[2][4] = fmaf(rA[0][2], rB[0][4], rC[2][4]); \
    rC[3][4] = fmaf(rA[0][3], rB[0][4], rC[3][4]); \
    rC[4][4] = fmaf(rA[0][4], rB[0][4], rC[4][4]); \
    rC[5][4] = fmaf(rA[0][5], rB[0][4], rC[5][4]); \
    rC[0][5] = fmaf(rA[0][0], rB[0][5], rC[0][5]); \
    rC[1][5] = fmaf(rA[0][1], rB[0][5], rC[1][5]); \
    rC[2][5] = fmaf(rA[0][2], rB[0][5], rC[2][5]); \
    rC[3][5] = fmaf(rA[0][3], rB[0][5], rC[3][5]); \
    rC[4][5] = fmaf(rA[0][4], rB[0][5], rC[4][5]); \
    rC[5][5] = fmaf(rA[0][5], rB[0][5], rC[5][5]);

//__threadfence_block(); \ does not compile

template <typename T>
__global__ void trmm_left_lower_nontrans_MX096_NX096_KX16(rocblas_fill uplo,
                                                          rocblas_operation transA,
                                                          rocblas_diagonal diag,
                                                          rocblas_int M,
                                                          rocblas_int N,
                                                          const T* alpha,
                                                          const T* A,
                                                          rocblas_int lda,
                                                          const T* B,
                                                          rocblas_int ldb,
                                                          T* C,
                                                          rocblas_int ldc)
{
    T rC[6][6] = {{(T)0}};
    T rA[1][6];
    T rB[1][6];

    __shared__ T lA[1552];
    __shared__ T lB[1552];

    T *plA, *plB;

    uint gidx = hipBlockIdx_x;
    uint gidy = hipBlockIdx_y;
    uint idx  = hipThreadIdx_x; // get_local_id(0);
    uint idy  = hipThreadIdx_y; // get_local_id(1);

    A += gidx * 96 + idx + idy * lda;
    B += gidy * 96 * ldb + idx + idy * ldb;

    uint block_k = K >> 4;
    do
    {
        plA = lA + idy * 97 + idx;
        plB = lB + idx * 97 + idy;

        plB[0]  = B[0];
        plB[16] = B[16 * ldb];
        plB[32] = B[32 * ldb];
        plB[48] = B[48 * ldb];
        plB[64] = B[64 * ldb];
        plB[80] = B[80 * ldb];

        plA[0]  = A[0 + 0 * lda];
        plA[16] = A[16 + 0 * lda];
        plA[32] = A[32 + 0 * lda];
        plA[48] = A[48 + 0 * lda];
        plA[64] = A[64 + 0 * lda];
        plA[80] = A[80 + 0 * lda];

        __syncthreads();

        uint offA = idx;
        uint offB = idy;

        M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6 M6x6

            A += lda << 4;
        B += 16;
    } while(--block_k > 0);

    C += gidx * 96 + idx;
    C += gidy * 96 * ldc;
    C += idy * ldc;

    C[0 * ldc]  = alpha * rC[0][0];
    C[16 * ldc] = alpha * rC[0][1];
    C[32 * ldc] = alpha * rC[0][2];
    C[48 * ldc] = alpha * rC[0][3];
    C[64 * ldc] = alpha * rC[0][4];
    C[80 * ldc] = alpha * rC[0][5];
    C += 16;
    C[0 * ldc]  = alpha * rC[1][0];
    C[16 * ldc] = alpha * rC[1][1];
    C[32 * ldc] = alpha * rC[1][2];
    C[48 * ldc] = alpha * rC[1][3];
    C[64 * ldc] = alpha * rC[1][4];
    C[80 * ldc] = alpha * rC[1][5];
    C += 16;
    C[0 * ldc]  = alpha * rC[2][0];
    C[16 * ldc] = alpha * rC[2][1];
    C[32 * ldc] = alpha * rC[2][2];
    C[48 * ldc] = alpha * rC[2][3];
    C[64 * ldc] = alpha * rC[2][4];
    C[80 * ldc] = alpha * rC[2][5];
    C += 16;
    C[0 * ldc]  = alpha * rC[3][0];
    C[16 * ldc] = alpha * rC[3][1];
    C[32 * ldc] = alpha * rC[3][2];
    C[48 * ldc] = alpha * rC[3][3];
    C[64 * ldc] = alpha * rC[3][4];
    C[80 * ldc] = alpha * rC[3][5];
    C += 16;
    C[0 * ldc]  = alpha * rC[4][0];
    C[16 * ldc] = alpha * rC[4][1];
    C[32 * ldc] = alpha * rC[4][2];
    C[48 * ldc] = alpha * rC[4][3];
    C[64 * ldc] = alpha * rC[4][4];
    C[80 * ldc] = alpha * rC[4][5];
    C += 16;
    C[0 * ldc]  = alpha * rC[5][0];
    C[16 * ldc] = alpha * rC[5][1];
    C[32 * ldc] = alpha * rC[5][2];
    C[48 * ldc] = alpha * rC[5][3];
    C[64 * ldc] = alpha * rC[5][4];
    C[80 * ldc] = alpha * rC[5][5];
}

template <typename>
constexpr char rocblas_trmm_name[] = "unknown";
template <>
constexpr char rocblas_trmm_name<float>[] = "rocblas_strmm";
template <>
constexpr char rocblas_trmm_name<double>[] = "rocblas_dtrmm";

/*! \brief BLAS Level 3 API

    \details

    trmm solves

    C := alpha*op( A )*B,   or   C := alpha*B*op( A )

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.

    @param[in]
    side    rocblas_side.
            rocblas_side_left:       C := alpha*op( A )*B.
            rocblas_side_right:      C := alpha*B*op( A ).

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
            rocblas_diagonal_unit:      A is assumed to be unit triangular.
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
            zero then A is not referenced and B need not be set before
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

    @param[in]
    B       pointer storing matrix B on the GPU.

    @param[in]
    ldb    rocblas_int.
           ldb specifies the first dimension of B. ldb >= max( 1, m ).

    @param[in,output]
    C       pointer storing matrix C on the GPU.

    @param[in]
    ldc    rocblas_int.
           ldb specifies the first dimension of C. ldc >= max( 1, m ).

    ********************************************************************/

template <typename T>
rocblas_status rocblas_trmm(rocblas_handle handle,
                            rocblas_side side,
                            rocblas_fill uplo,
                            rocblas_operation transA,
                            rocblas_diagonal diag,
                            rocblas_int M,
                            rocblas_int N,
                            const T* alpha,
                            const T* A,
                            rocblas_int lda,
                            const T* B,
                            rocblas_int ldb,
                            T* C,
                            rocblas_int ldc)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!alpha)
        return rocblas_status_invalid_pointer;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & rocblas_layer_mode_log_trace)
    {
        if(handle->pointer_mode == rocblas_pointer_mode_host)
            log_trace(handle,
                      rocblas_trmm_name<T>,
                      side,
                      uplo,
                      transA,
                      diag,
                      M,
                      N,
                      *alpha,
                      A,
                      lda,
                      B,
                      ldb,
                      C,
                      ldc);
        else
            log_trace(handle,
                      rocblas_trmm_name<T>,
                      side,
                      uplo,
                      transA,
                      diag,
                      M,
                      N,
                      alpha,
                      A,
                      lda,
                      B,
                      ldb,
                      C,
                      ldc);
    }

    if(layer_mode & rocblas_layer_mode_log_profile)
        log_profile(handle,
                    rocblas_trmm_name<T>,
                    "side",
                    rocblas_side_letter(side),
                    "uplo",
                    rocblas_uplo_letter(uplo),
                    "transA",
                    rocblas_transpose_letter(transA),
                    "diag",
                    rocblas_diag_letter(diag),
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

    rocblas_int A_row = side == rocblas_side_left ? M : N;

    if(M < 0 || N < 0)
        return rocblas_status_invalid_size;
    if(!A)
        return rocblas_status_invalid_pointer;
    if(lda < A_row)
        return rocblas_status_invalid_size;
    if(!B)
        return rocblas_status_invalid_pointer;
    if(ldb < M)
        return rocblas_status_invalid_size;
    if(!C)
        return rocblas_status_invalid_pointer;
    if(ldc < M)
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible.
     */

    if(!M || !N)
        return rocblas_status_success;

    if(transA == rocblas_operation_transpose)
        return rocblas_status_not_implemented;

    static constexpr int NB = 16;
    rocblas_int blocks_x    = (M - 1) / (NB * 6) + 1;
    rocblas_int blocks_y    = (N - 1) / (NB * 6) + 1;

    dim3 grid(blocks_x, blocks_y);
    dim3 threads(NB, NB);

    hipStream_t rocblas_stream;
    RETURN_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &rocblas_stream));

    hipLaunchKernelGGL(trmm_Col_NN_B1_MX096_NX096_KX16,
                       grid,
                       threads,
                       0,
                       rocblas_stream,
                       M,
                       N,
                       K,
                       *alpha,
                       A,
                       lda,
                       B,
                       ldb,
                       C,
                       ldc);

    return rocblas_status_success;
}

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strmm(rocblas_handle handle,
                             rocblas_side side,
                             rocblas_fill uplo,
                             rocblas_operation transA,
                             rocblas_diagonal diag,
                             rocblas_int M,
                             rocblas_int N,
                             const float* alpha,
                             const float* A,
                             rocblas_int lda,
                             const float* B,
                             rocblas_int ldb,
                             float* C,
                             rocblas_int ldc)
{
    return rocblas_trmm(handle, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb, C, ldc);
}

rocblas_status rocblas_dtrmm(rocblas_handle handle,
                             rocblas_side side,
                             rocblas_fill uplo,
                             rocblas_operation transA,
                             rocblas_diagonal diag,
                             rocblas_int M,
                             rocblas_int N,
                             const double* alpha,
                             const double* A,
                             rocblas_int lda,
                             const double* B,
                             rocblas_int ldb,
                             double* C,
                             rocblas_int ldc)
{
    return rocblas_trmm(handle, side, uplo, transA, diag, M, N, alpha, A, lda, B, ldb, C, ldc);
}

} // extern "C"

/* ============================================================================================ */
