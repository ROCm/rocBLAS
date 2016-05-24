/* ************************************************************************
 * trtriright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <hip_runtime.h>
#include "rocblas.h"
#include "status.h"
#include "definitions.h"

template<typename T, rocblas_int NB, rocblas_int flag>
__global__ void
trtri_kernel_lower(hipLaunchParm lp,
    rocblas_diagonal diag,
    rocblas_int n,
    T *A, rocblas_int lda,
    T *invA)
{
    int tx  = hipThreadIdx_x;

    __shared__ T sA[NB * NB];

    //read matrix A into shared memory, only need to read lower part
    //its inverse will overwrite the shared memory
    if (tx < n){
        //compute only diagonal element
        for(int i=0;i<=tx;i++){
            sA[tx + i * n] = A[tx + i * lda];
        }
    }

    //__syncthreads(); // since NB < 64, this synch can be avoided

    //invert the diagonal element
    if (tx < n){
        //compute only diagonal element
        if (diag == rocblas_diagonal_unit){
            sA[tx + tx * n] = 1;
        }
        else{
            if(sA[tx + tx * n] == 0){ // notice this does not apply for complex
                sA[tx + tx * n] = 1; // means the matrix is singular
            }
            else{
                sA[tx + tx * n] = 1/sA[tx + tx * n];
            }
        }
    }

    // solve the inverse of A column by column, each inverse(A)' column will overwrite sA'column which store A
    // this operation is safe
    for(int col=0; col<n; col++){

        T reg = 0;
        //use the diagonal one to update current column
        if(tx > col) reg += sA[tx + col * n] * sA[col + col * n];

        //__syncthreads(); // since NB < 64, this synch can be avoided

        // in each column, it solves step, each step solve an inverse(A)[step][col]
        for(int step=col+1;step<n;step++){

            // only tx == step solve off-diagonal
            if(tx == step) {
                //solve the step row, off-diagonal elements, notice sA[tx][tx] is already inversed, so multiply
                sA[tx + col * n] = (0 - reg) * sA[tx + tx * n];
            }

            //__syncthreads(); // since NB < 64, this synch can be avoided

            //tx > step  update with (tx = step)'s result
            if(tx > step){
                reg +=  sA[tx + step * n] * sA[step + col * n];
            }
            //__syncthreads(); // since NB < 64, this synch can be avoided
        }
        //__syncthreads();
    }

#if 1
    //if flag 0, then A will be overwritten by sA, invA won't be touched
    if(flag == 0){
        if (tx < n){
            //compute only diagonal element
            for(int i=0;i<=tx;i++){
                A[tx + i * lda] = sA[tx + i * n];
            }
        }
    }
    else{  //else invA will be overwritten by sA
        if (tx < n){
            //compute only diagonal element
            for(int i=0;i<=tx;i++){
                invA[tx + i * lda] = sA[tx + i * n];
            }
        }
    }
#endif
}



//HIP support up to 1024 threads/work itmes per thread block/work group
#define NB_X 32

//assume invA has already been allocated, recommened for repeated calling of trtri product routine
template<typename T>
rocblas_status
rocblas_trtri_template_workspace(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    T *A, rocblas_int lda,
    T *invA)
{
    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( A == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( lda < n )
        return rocblas_status_invalid_size;
    else if ( invA == nullptr )
        return rocblas_status_invalid_pointer;

    /*
     * Quick return if possible.
     */

    if ( n == 0)
        return rocblas_status_success;

    if(n > NB_X ){
        printf("n is %d, n must be less than %d, will return\n", n, NB_X);
        return rocblas_status_not_implemented;
    }

    dim3 grid(1, 1, 1);
    dim3 threads(NB_X, 1, 1);

    if(uplo == rocblas_fill_upper){
        return rocblas_status_not_implemented;
    }
    else{
        hipLaunchKernel(HIP_KERNEL_NAME(trtri_kernel_lower<T, NB_X, 1>), dim3(grid), dim3(threads), 0, 0 , diag, n, A, lda, invA);
    }

    return rocblas_status_success;

}

/* ============================================================================================ */

/*! \brief BLAS Level 3 API

    \details
    trtri  compute the inverse of a matrix  A

        inv(A);

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill.
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
    @param[in]
    diag      rocblas_diagonal.
              = 'rocblas_diagonal_non_unit', A is non-unit triangular;
              = 'rocblas_diagonal_unit', A is unit triangular;
    @param[in]
    n         rocblas_int.
    @param[in,output]
    A         pointer storing matrix A on the GPU.
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[output]
    invA         pointer storing the inverse matrix A on the GPU.

    ********************************************************************/

//allocate invA inside this API
template<typename T>
rocblas_status
rocblas_trtri_template(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    T *A, rocblas_int lda)
{

    if(handle == nullptr)
        return rocblas_status_invalid_handle;
    else if ( uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if ( n < 0 )
        return rocblas_status_invalid_size;
    else if ( A == nullptr )
        return rocblas_status_invalid_pointer;
    else if ( lda < n )
        return rocblas_status_invalid_size;

    /*
     * Quick return if possible.
     */

    if ( n == 0)
        return rocblas_status_success;

    if(n > NB_X ){
        printf("n is %d must be less than %d, will exit\n", n, NB_X);
        return rocblas_status_not_implemented;
    }

    dim3 grid(1, 1, 1);
    dim3 threads(NB_X, 1, 1);


    if(uplo == rocblas_fill_upper){
        return rocblas_status_not_implemented;
    }
    else{
        hipLaunchKernel(HIP_KERNEL_NAME(trtri_kernel_lower<T, NB_X, 0>), dim3(grid), dim3(threads), 0, 0 , diag, n, A, lda, nullptr);
    }

    return rocblas_status_success;
}



/* ============================================================================================ */

    /*
     * ===========================================================================
     *    template interface
     *    template specialization
     * ===========================================================================
     */


template<>
rocblas_status
rocblas_trtri<float>(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    float *A, rocblas_int lda){

    return rocblas_trtri_template<float>(handle, uplo, diag, n, A, lda);
}




/* ============================================================================================ */

    /*
     * ===========================================================================
     *    C wrapper
     * ===========================================================================
     */


extern "C"
rocblas_status
rocblas_strtri(rocblas_handle handle,
    rocblas_fill uplo, rocblas_diagonal diag,
    rocblas_int n,
    float *A, rocblas_int lda){

    return rocblas_trtri<float>(handle, uplo, diag, n, A, lda);
}


/* ============================================================================================ */
