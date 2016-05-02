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


template<typename T, int BLK_X, int DIM_X, int DIM_Y>
__global__ void
symv_kernel_L(hipLaunchParm lp,
    rocblas_int n,
    T const * __restrict__ A, rocblas_int lda,
    T const * __restrict__ x, rocblas_int incx,
    T       * __restrict__ work)
{

    int tx  = hipThreadIdx_x;
    int ty  = hipThreadIdx_y;
    int tid = ty * DIM_X + tx;

    __shared__ sA[(DIM_X+1][DIM_Y] = 0.0;
    __shared__ T sx[BLK_X];

    T reg_A = 0.0;

    //off diagonal
    int A_x = hipBlockIdx_x * hipBlockDim_x + tx;
    if (A_x >= n ) A_x = 0;

    if(A_x < n){
        sx[tx] = x[A_x];
    }
    else{
        sx[tx] = 0;
    }

    __syncthreads();

    for(int step=0; step< hipBlockIdx_x * hipBlockDim_x ; step+=BLK_X){
        //fetch a DIM_X * DIM_Y data tile
        for(int k=0;k<BLK_X;k+= DIM_Y){
            int A_y = step + k;
            reg_A  =  A[A+x + lda * A_y];

            // calculate row, natural reduction
            reg_row +=  reg_A * x[A_y];
            //calculate columns and write into workspace
            sA[tx][ty]    +=  reg_A * sx[A_x];
            __syncthreads();
        }

        int tx_ = tid / DIM_Y;

        if(tx >= k && t)
            reg_column += sA[tx_][i];
        }

    }


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
