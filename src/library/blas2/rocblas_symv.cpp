/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <hip_runtime.h>
#include "rocblas.h"

#define NB_X         64
#define NB_Y          4
#define bank_shift   33
#define quarter_NB_X 16
#define half_NB_X    32
#define rocblas_S_CONJ( a )  (a)


/*******************************************************************************
    Lower case, compute block multiply, work = A*x, for any size n:

           [ (A11*x1)   (A21^H*x2)          (A31^H*x3)                 ]   [ A11  A21^H  A31^H ]   [ x1 ]
    work = [   ---      (A21*x1 + A22*x2)   (A32^H*x3)                 ] = [ A21  A22    A32^H ] * [ x2 ]
           [   ---        ---               (A31*x1 + A32*x2 + A33*x3) ]   [ A31  A32    A33   ]   [ x3 ]

    Uses a 64x4 thread block.
    For     diagonal tiles, covers a 64x64 tile using three 32x32 tiles (plus one gets transposed).
    For off-diagonal tiles, covers a 64x64 tile using four  64x16 tiles.
    In both cases, each thread multiplies 4 elements.

    For rows past the bottom of the matrix, the A pointer is adjusted to be the
    last valid row of A, which multiple threads will read.
    Extra rows are ignored when saving results to work.
    Columns past the right edge are explicitly ignored when loading.
    x values past the bottom are set to 'L', thus, extra columns are 'L'ed
    when multiplying.

    Previously:
           [ (A11*x1)       ---                                          ]
    work = [ (A21^H*x2)   (A21*x1 + A22*x2)     ---                      ]
           [ (A31^H*x3)   (A32^H*x3)          (A31*x1 + A32*x2 + A33*x3) ]
    which doesn't work as well because that has dimension blocks*NB by blocks,
    where blocks*NB >= n, and it can be that blocks*NB > lda, so it won't fit in
    lda*blocks space. This is why it used to need lworkspace = lda*(blocks + 1).
    ********************************************************************/
template<typename T>
__global__ void
symv_kernel_L(hipLaunchParm lp,
    rocblas_int n,
    T const * __restrict__ A, rocblas_int lda,
    T const * __restrict__ x, rocblas_int incx,
    T       * __restrict__ work)
{

    // treats sA as 16x64 block
    #define sA16(i_, j_) (sA[(i_)][(j_)])  // i.e., sA[ (i_)*(NB_X+3) + (j_) ]

    // treats sA as 32x32 block
    #define sA32(i_, j_) (sA[0][(i_) + bank_shift*(j_)])

    // 64x4 thread block
    const int tx  = hipThreadIdx_x;
    const int ty  = hipThreadIdx_y;
    const int blk = hipBlockIdx_x;
    const int blk_ind = NB_X * blk;
    const int td  = NB_X * ty + tx;

    // 32x8 thread block
    const int tx2 = td % half_NB_X;
    const int ty2 = td / half_NB_X;

    // If this blk has fewer than NB_X rows, partial is the number of valid rows,
    // so tx = 0, ..., partial-1 are valid rows, and tx >= partial are invalid.
    // Else, partial == 0.
    const int partial = (blk == hipGridDim_x - 1 ? (n % NB_X) : 0);

    T psum, psum_t;
    T total = 0.0;

    // sA is used as a 32x32 block, sA32(i,j),
    // and as a 16x64 block, sA16(i,j), in different parts of the code.
    // sA must be at least half_NB_X*bank_shift = 32x33 = 1056;
    // quarter_NB_X*(NB_X + 2) = 16*(64 + 2) = 1056
    __shared__ T sA [quarter_NB_X][NB_X + 3]; /* Why +3? seems it only needs +2. Does +3 reduce bank conflicts? */
    __shared__ T sx_blk[NB_X];  // for x[ blk ]
    __shared__ T sx_jj [NB_X];  // for x[ jj ], which cycles over all blocks left of diag

    T rA[4];
    T psums_t[4];

    // --------------------
    // load 64x1 block x(blk_ind + 0:63) into sx_blk
    x += (blk_ind + tx)*incx;  // x is x(blk_ind + tx)
    if ( ty == 0 ) {
        if ( partial == 0 || tx < partial ) {
            sx_blk[tx] = x[0];
        }
        else {
            sx_blk[tx] = 0.0;
        }
    }

    // --------------------
    // move to block row
    work += blk*lda;     // work is work(0, blk)

    A += blk_ind;        // A is A(blk_ind, 0)
    A += ty2*lda + tx2;  // A is A(blk_ind + tx2, ty2)

    // move to 32x32 diag block
    A += blk_ind*lda;    // A is A(blk_ind + tx2, blk_ind + ty2)

    // load 32x32 diag block A(blk_ind + 0:31, blk_ind + 0:31) into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if ( partial ) {
        if ( tx2 >= partial ) {
            A = A - tx2 + (partial - 1);  // A is A(blk_ind + partial-1, blk_ind + ty2), the bottom-most valid row
        }
        #pragma unroll
        for (int j=0; j < half_NB_X; j += 8) {
            if ( ty2+j < partial ) {
                sA32(tx2, ty2 + j) = A[j*lda];
            }
            else {
                sA32(tx2, ty2 + j) = 0.0;
            }
        }
        if ( tx2 >= partial ) {
            A = A + tx2 - (partial - 1);  // A is A(blk_ind + tx2, blk_ind + ty2)
        }
    }
    else {
        #pragma unroll
        for (int j=0; j < half_NB_X; j += 8) {
            sA32(tx2, ty2 + j) = A[j*lda];
        }
    }
    __syncthreads();

    // symmetrize 32x32 diag block, copying lower to upper triangle,
    // as four 32x8 sections in parallel:
    // columns 0,4,8,12,16,20,24,28; then 1,5,...,29; then 2,6,...,30, then 3,7,...,31
    #pragma unroll
    for (int j=ty2*4; j < ty2*4 + 4; j++) {
        if ( j < tx2 ) {
            sA32(j, tx2) = rocblas_S_CONJ( sA32(tx2, j) );
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    // each thread does partial row sA(tx2, ty2*4 : ty2*4 + 3)
    psum = 0.0;
    #pragma unroll
    for (int j=0; j < 4; j++) {
        psum += sA32(tx2, ty2*4 + j) * sx_blk[ty2*4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if ( ty2 == 0 ) {
        total = sA32(0, tx2) + sA32(1, tx2)
              + sA32(2, tx2) + sA32(3, tx2)
              + sA32(4, tx2) + sA32(5, tx2)
              + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to next 32x32 diag block, then repeat steps from first diag block
    A += half_NB_X + half_NB_X*lda;  // A is A(blk_ind + NB/2 + tx2, blk_ind + NB/2 + ty2)

    // load 32x32 diag block A[block + 0:31, block + 0:31] into sA
    if ( partial ) {
        if ( tx2 + half_NB_X >= partial ) {
            A = A - (tx2 + half_NB_X) + (partial - 1);
        }
        #pragma unroll
        for (int j=0; j < half_NB_X; j += 8) {
            if ( ty2+j + half_NB_X < partial ) {
                sA32(tx2, ty2 + j) = A[j*lda];
            }
            else {
                sA32(tx2, ty2 + j) = 0.0;
            }
        }
        if ( tx2 + half_NB_X >= partial ) {
            A = A + (tx2 + half_NB_X) - (partial - 1);
        }
    }
    else {
        #pragma unroll
        for (int j=0; j < half_NB_X; j += 8) {
            sA32(tx2, ty2 + j) = A[j*lda];
        }
    }
    __syncthreads();

    // symmetrize 32x32 diag block, copying lower to upper triangle
    #pragma unroll
    for (int j=ty2*4; j < ty2*4 + 4; j++) {
        if ( j < tx2 ) {
            sA32(j, tx2) = rocblas_S_CONJ( sA32(tx2, j) );
        }
    }
    __syncthreads();

    // multiply 32x32 diag block * x
    psum = 0.0;
    #pragma unroll
    for (int j=0; j < 4; j++) {
        psum += sA32(tx2, ty2*4 + j) * sx_blk[half_NB_X + ty2*4 + j];
    }
    __syncthreads();

    // store partial row sums
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if ( ty2 == 1 ) {
        total = sA32(0, tx2) + sA32(1, tx2)
              + sA32(2, tx2) + sA32(3, tx2)
              + sA32(4, tx2) + sA32(5, tx2)
              + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to off-diag 32x32 block
    A -= half_NB_X*lda;  // A is A(blk_ind + NB/2 + tx2, blk_ind + ty2)

    // load 32x32 block of A into sA,
    // as four 32x8 sections one after another:
    // columns 0:7, then 8:15, then 16:23, then 24:31
    if ( partial ) {
        if ( tx2 + half_NB_X >= partial ) {
            A = A - (tx2 + half_NB_X) + (partial - 1);
        }
        #pragma unroll
        for (int j=0; j < half_NB_X; j += 8) {
            if ( ty2+j < partial ) {
                sA32(tx2, ty2 + j) = A[j*lda];
            }
            else {
                sA32(tx2, ty2 + j) = 0.0;
            }
        }
        if ( tx2 + half_NB_X >= partial ) {
            A = A + (tx2 + half_NB_X) - (partial - 1);
        }
    }
    else {
        #pragma unroll
        for (int j=0; j < half_NB_X; j += 8) {
            sA32(tx2, ty2 + j) = A[j*lda];
        }
    }
    __syncthreads();

    // multiply 32x32 block (below diag)
    psum = 0.0;
    #pragma unroll
    for (int j=0; j < 4; j++) {
        psum += sA32(tx2, ty2 + j*8) * sx_blk[j*8 + ty2];
    }
    //__syncthreads();  // no sync needed here

    // multiply transposed 32x32 block (above diag)
    psum_t = 0.0;
    #pragma unroll
    for (int j=0; j < 4; j++) {
        psum_t += rocblas_S_CONJ( sA32(ty2*4 + j, tx2) ) * sx_blk[half_NB_X + ty2*4 + j];
    }
    __syncthreads();

    // store partial sums for non-transposed 32x32 block
    sA32(ty2, tx2) = psum;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
    if ( ty2 == 1 ) {
        total = total
              + sA32(0, tx2) + sA32(1, tx2)
              + sA32(2, tx2) + sA32(3, tx2)
              + sA32(4, tx2) + sA32(5, tx2)
              + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // store partial sums for transposed 32x32 block
    sA32(ty2, tx2) = psum_t;
    __syncthreads();

    // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
    if ( ty2 == 0 ) {
        total = total
              + sA32(0, tx2) + sA32(1, tx2)
              + sA32(2, tx2) + sA32(3, tx2)
              + sA32(4, tx2) + sA32(5, tx2)
              + sA32(6, tx2) + sA32(7, tx2);
    }
    __syncthreads();

    // --------------------
    // move to leftmost 64x64 block in block row, and
    // switch thread offset from (tx2,ty2) 32x8 block to (tx,ty) 64x4 block
    A -= half_NB_X;      // A is A(blk_ind + tx2, blk_ind + ty2)
    A -= blk_ind*lda;    // A is A(blk_ind + tx2,           ty2)
    A -= ty2*lda + tx2;  // A is A(blk_ind, 0)
    A += 4*ty*lda + tx;  // A is A(blk_ind + tx, 4*ty)

    if ( partial && tx >= partial ) {
        A = A - tx + (partial - 1);  // A is A(blk_ind + partial-1, 4*ty), the bottom-most valid row
    }

    x -= blk_ind*incx;  // x is x(tx)

    // 16x16 thread block
    const int tx4 = td % quarter_NB_X;
    const int ty4 = td / quarter_NB_X;

    // cycle over blocks jj left of diagonal, in block row blk
    for (int jj=0; jj < blk; ++jj) {
        // load 64x1 block x(jj_ind + 0:63) into sx_jj
        // since this block is left of diagonal, x must have all NB rows
        if ( ty == 0 ) {
            sx_jj[tx] = x[jj*NB_X*incx];
        }
        __syncthreads();

        for (int k=0; k < 4; k++) {
            // load 64x16 block of A into rA, 4 elements per thread,
            // as four 64x4 sections in parallel:
            // columns 0,4,8,12; then 1,5,9,13; then 2,6,10,14; then 3,7,11,15
            // since this block is left of diagonal, it has all NB columns,
            // and block of x must have all NB rows.
            #pragma unroll
            for (int j=0; j < 4; j++) {
                rA[j] = A[j*lda];
            }

            // 1) multiply 64x16 block A_{blk,jj} * x_jj
            //    each thread does partial row rA(tx + 16*k, ty*4 + 16*k : ty*4 + 3 + 16*k)
            // 2) multiply transposed 16x64 block A_{blk,jj}^H * x_blk,
            //    storing each product Aji*xi to sA(j,i)
            #pragma unroll
            for (int j=0; j < 4; j++) {
                total += rA[j] * sx_jj[quarter_NB_X*k + ty*4 + j];  // y_blk = A_{blk,jj}   * x_jj
                sA16(ty*4 + j, tx) = rocblas_S_CONJ( rA[j] ) * sx_blk[tx];  // y_jj  = A_{blk,jj}^H * x_blk
            }
            __syncthreads();

            // do partial row sums for transposed 16x64 result
            // use 16x16 thread grid (tx4, ty4) instead of 64x4 (tx, ty)
            // sum sixteen 16x4 sections in parallel:
            // columns 0,4,8,...,60; then 1,5,...,61; then 2,6,...,62; then 3,7,...,63
            psum_t = 0.0;
            #pragma unroll
            for (int j=0; j < 4; j++) {
                psum_t += sA16(tx4, ty4*4 + j);
            }
            __syncthreads();

            // store partial row sums of transposed result, y_jj (locally)
            psums_t[k] = psum_t;

            // move right to next 64x16 block
            A += lda * quarter_NB_X;  // A is A(blk_ind + tx#, jj*NB_x + (k+1)*NB_X/4 + 4*ty), # tx or partial
        }
        // already at next 64x64 block
        // A is A(blk_ind + tx#, (jj+1)*NB_x + 4*ty), # tx or partial

        // store partial row sums of transposed result, y_jj
        #pragma unroll
        for (int k=0; k < 4; k++) {
            sA16(tx4, ty4 + quarter_NB_X*k) = psums_t[k];
        }
        __syncthreads();

        // sum up partial row sums of transposed result, y_jj, and store final total to workspace
        // thread (tx4,ty4) where ty4 < 4 sums row tx4 + ty4*16
        // since this is the transposed block above the diagonal, it must have all NB rows
        if ( ty4 < 4 ) {
            int ty4_nb4 = ty4*quarter_NB_X;
            psum_t = sA16(tx4,  0 + ty4_nb4) + sA16(tx4,  1 + ty4_nb4)
                   + sA16(tx4,  2 + ty4_nb4) + sA16(tx4,  3 + ty4_nb4)
                   + sA16(tx4,  4 + ty4_nb4) + sA16(tx4,  5 + ty4_nb4)
                   + sA16(tx4,  6 + ty4_nb4) + sA16(tx4,  7 + ty4_nb4)
                   + sA16(tx4,  8 + ty4_nb4) + sA16(tx4,  9 + ty4_nb4)
                   + sA16(tx4, 10 + ty4_nb4) + sA16(tx4, 11 + ty4_nb4)
                   + sA16(tx4, 12 + ty4_nb4) + sA16(tx4, 13 + ty4_nb4)
                   + sA16(tx4, 14 + ty4_nb4) + sA16(tx4, 15 + ty4_nb4);
            work[jj*NB_X + tx4 + ty4_nb4] = psum_t;  // store at work( jj*NB_X + tx4 + ty4*16, blk )
        }
        __syncthreads();
    }

    // store row sums
    sA16(ty, tx) = total;
    __syncthreads();

    // sum up final total, y_blk, for row tx
    if ( ty == 0 && (partial == 0 || tx < partial) ) {
        total = sA16(0, tx)
              + sA16(1, tx)
              + sA16(2, tx)
              + sA16(3, tx);
        work[blk*NB_X + tx] = total;  // store at work( blk*NB_X + tx, blk )
    }

}
// end symv_kernel_L


/**************************************************************
    Lower case, sum up final results
    Each block sums one block row; each thread sums one row.

    On input (for 3 blocks):
           [ (A11*x1)   (A21^H*x2)          (A31^H*x3)                 ]
    work = [   ---      (A21*x1 + A22*x2)   (A32^H*x3)                 ]
           [   ---        ---               (A31*x1 + A32*x2 + A33*x3) ]

    On output:
              [ (A11*x1) + (A21^H*x2) + (A31^H*x3) ]
    y = alpha*[ (A21*x1 + A22*x2)     + (A32^H*x3) ] + beta*y
              [ (A21*x1 + A22*x2 + A33*x3)         ]
    ********************************************************************/

template<typename T>
__global__ void
symv_kernel_L_sum_host_pointer(hipLaunchParm lp,
    rocblas_int n,
    T alpha,
    rocblas_int lda,
    T beta,
    T       * __restrict__ y, rocblas_int incy,
    T const * __restrict__ work )
{
    int tx  = hipThreadIdx_x;
    int blk = hipBlockIdx_x;
    int blk_ind = blk * NB_X;
    int ind     = blk_ind + tx;
    int blocks  = hipGridDim_x;

    // Don't write outside [0, ..., n)
    if ( ind < n ) {
        work += ind + blk*lda;
        T Ax = 0.0;
        for (int j = blk; j < blocks; ++j) {
            Ax += work[0];
            work += lda;
        }
        y[ind * incy] = (beta)*y[ind * incy] + (alpha)*Ax;
    }
}


template<typename T>
__global__ void
symv_kernel_L_sum_device_pointer(hipLaunchParm lp,
    rocblas_int n,
    const T *alpha,
    rocblas_int lda,
    const T *beta,
    T       * __restrict__ y, rocblas_int incy,
    T const * __restrict__ work )
{
    int tx  = hipThreadIdx_x;
    int blk = hipBlockIdx_x;
    int blk_ind = blk * NB_X;
    int ind     = blk_ind + tx;
    int blocks  = hipGridDim_x;

    // Don't write outside [0, ..., n)
    if ( ind < n ) {
        work += ind + blk*lda;
        T Ax = 0.0;
        for (int j = blk; j < blocks; ++j) {
            Ax += work[0];
            work += lda;
        }
        y[ind * incy] = (*beta)*y[ind * incy] + (*alpha)*Ax;
    }
}


/*! \brief BLAS Level 2 API

    \details
    SYMV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n symmetric matrix.

    @param[in]
    handle    rocblas_handle.
              handle to the ablas library context queue.
    @param[in]
    uplo      rocblas_uplo.
              specifies whether the upper or lower
    @param[in]
    n         rocblas_int.
    @param[in]
    alpha
              specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    x         pointer storing vector x on the GPU.
    @param[in]
    incx      specifies the increment for the elements of x.
    @param[in]
    beta      specifies the scalar beta.
    @param[out]
    y         pointer storing vector y on the GPU.
    @param[in]
    incy      rocblas_int
              specifies the increment for the elements of y.

    ********************************************************************/

template<typename T>
rocblas_status
rocblas_symv_template_workspace(rocblas_handle handle,
    rocblas_uplo uplo, rocblas_int n,
    const T *alpha,
    const T *A, rocblas_int lda,
    const T *x, rocblas_int incx,
    const T *beta,
    T * y, rocblas_int incy,
    T * workspace, rocblas_int lworkspace)
{

    if ( uplo != rocblas_lower )
        return rocblas_not_implemented; //only lower is implemented right now
    if ( n < 0 )
        return rocblas_invalid_dim;
    else if ( A == NULL )
        return rocblas_invalid_matA;
    else if ( lda < n )
        return rocblas_invalid_leadDimA;
    else if ( x == NULL )
        return rocblas_invalid_vecX;
    else if ( incx < 0 )
        return rocblas_invalid_incx;
    else if ( y == NULL )
        return rocblas_invalid_vecY;
    else if ( incy < 0 )
        return rocblas_invalid_incy;

    /*
     * Quick return if possible. Not Argument error
     */

    if ( n == 0 )
        return rocblas_success;


    rocblas_int blocks = (n-1)/NB_X + 1;

    dim3 grid( blocks, 1, 1 );
    dim3 threads( NB_X, NB_Y, 1 );
    dim3 threads_sum( NB_X, 1, 1 );

    if(lworkspace < blocks*lda) {
        printf("size workspace = %d is too small, allocate at least %d", lworkspace, blocks*lda);
        return rocblas_not_implemented;
    }

    if ( uplo != rocblas_lower  ) {
        return rocblas_not_implemented;
    }
    else {
        hipLaunchKernel(HIP_KERNEL_NAME(symv_kernel_L<T>), dim3(grid), dim3(threads), 0, 0 , n, A, lda, x, incx, workspace);

        if( rocblas_get_pointer_type((void*)alpha) == DEVICE_POINTER &&   rocblas_get_pointer_type((void*)beta) == DEVICE_POINTER ){
            hipLaunchKernel(HIP_KERNEL_NAME(symv_kernel_L_sum_device_pointer<T>), dim3(grid), dim3(threads_sum), 0, 0 ,
                                            n, alpha, lda, beta, y, incy, workspace);
        }
        else{
            T h_alpha_scalar = *alpha; T h_beta_scalar = *beta;
            hipLaunchKernel(HIP_KERNEL_NAME(symv_kernel_L_sum_host_pointer<T>), dim3(grid), dim3(threads_sum), 0, 0 ,
                                            n, h_alpha_scalar, lda, h_beta_scalar, y, incy, workspace);
        }
    }
    return rocblas_success;
}


// end rocblas_symv_work



template<typename T>
rocblas_status
rocblas_symv_template(rocblas_handle handle,
             rocblas_uplo uplo,
             rocblas_int n,
             const T *alpha,
             const T *A, rocblas_int lda,
             const T *x, rocblas_int incx,
             const T *beta,
             T *y, rocblas_int incy)
{

    if ( uplo != rocblas_lower )
        return rocblas_not_implemented; //only lower is implemented right now
    if ( n < 0 )
        return rocblas_invalid_dim;
    else if ( A == NULL )
        return rocblas_invalid_matA;
    else if ( lda < n )
        return rocblas_invalid_leadDimA;
    else if ( x == NULL )
        return rocblas_invalid_vecX;
    else if ( incx < 0 )
        return rocblas_invalid_incx;
    else if ( y == NULL )
        return rocblas_invalid_vecY;
    else if ( incy < 0 )
        return rocblas_invalid_incy;

    /*
     * Quick return if possible. Not Argument error
     */

    if ( n == 0 )
        return rocblas_success;

    rocblas_int default_device;
    //save the current device
    //CHECK_ERROR(hipGetDevice(&default_device));
    //set the devcie to the one associated with the handle
    //CHECK_ERROR(hipSetDevice(handle.device_id));// this operation set the deafult device is destructive

    T * workspace;
    rocblas_int blocks = (n - 1)/ NB_X + 1;
    rocblas_int lworkspace  = lda*blocks;
    CHECK_ERROR(hipMalloc( &workspace, lworkspace * sizeof(T)));

    rocblas_status status = rocblas_symv_template_workspace<T>(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy,
                          workspace, lworkspace);

    CHECK_ERROR(hipFree( workspace ));
    //reset device to default one
    //CHECK_ERROR(hipSetDevice(default_device));

    return status;
}
// end rocblas_symv


/* ============================================================================================ */

    /*
     * ===========================================================================
     *    template interface
     *    template specialization
     * ===========================================================================
     */


template<>
rocblas_status
rocblas_symv<float>(rocblas_handle handle,
             rocblas_uplo uplo,
             rocblas_int n,
             const float *alpha,
             const float *A, rocblas_int lda,
             const float *x, rocblas_int incx,
             const float *beta,
             float *y, rocblas_int incy){

    return rocblas_symv_template<float>(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template<>
rocblas_status
rocblas_symv<double>(rocblas_handle handle,
             rocblas_uplo uplo,
             rocblas_int n,
             const double *alpha,
             const double *A, rocblas_int lda,
             const double *x, rocblas_int incx,
             const double *beta,
             double *y, rocblas_int incy){

    return rocblas_symv_template<double>(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}
