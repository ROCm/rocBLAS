
    /*
     * ===========================================================================
     *    This file provide common device function for gemv routines
     * ===========================================================================
     */

/* ============================================================================================ */


#include "../blas1/device_template.h"


#if 0
template<typename T, const rocblas_int DIM_X, const rocblas_int DIM_Y>
static __device__ void
gemvn_device(
    rocblas_int M, rocblas_int N,
    T alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ X, rocblas_int incx,
    T beta,
    T       * Y, rocblas_int incy)

{
    // M always denotes length of Y and N denotes length of X in the kernel
    double4 a0;
    double4 x0;
    double4 y0;
    y0 = 0;

    __shared__ double4 localRes[64][1];
    uint coordA = hipBlockIdx_x * 32 + (hipThreadIdx_x % 8) * 4;
    uint k0 = (hipThreadIdx_x / 8) * 4;

    if (coordA < M && k0 < N) {
        T* Ag = A;
        T* Xg = X;

        uint Ntail = N % 4;
        N -= Ntail;

        uint k = k0;
        for (; k < N; k += 32) {
            const uint xk = k / 2;
            x0.s01 = Xg.d2v[xk + 0];
            x0.s23 = Xg.d2v[xk + 1];
            /* -- Tiles multiplier -- */
            const uint2 ay = ((uint2)(0, 1) + (uint)(coordA >> 1)) % (uint)(lda >> 1);
            const uint4 ak = {mad24(k, (lda >> 1), 0u), mad24(k + 1, (lda >> 1), 0u), mad24(k + 2, (lda >> 1), 0u),
            		mad24(k + 3, (lda >> 1), 0u)};


            a0.s01 = Ag.d2v[ay.s0 + ak.s0]; // put
            a0.s23 = Ag.d2v[ay.s1 + ak.s0];

            y0 = mad(a0, x0.s0, y0);

            a0.s01 = Ag.d2v[ay.s0 + ak.s1];
            a0.s23 = Ag.d2v[ay.s1 + ak.s1];

            y0 = mad(a0, x0.s1, y0);

            a0.s01 = Ag.d2v[ay.s0 + ak.s2];
            a0.s23 = Ag.d2v[ay.s1 + ak.s2];

            y0 = mad(a0, x0.s2, y0);

            a0.s01 = Ag.d2v[ay.s0 + ak.s3];
            a0.s23 = Ag.d2v[ay.s1 + ak.s3];

            y0 = mad(a0, x0.s3, y0);
            /* ---------------------- */
        }
        N += Ntail;
        if (k < N) {
            x0.s0 = X[k + 0 < N ? k : 0];
            x0.s1 = X[k + 1 < N ? k + 1 : 0];
            x0.s2 = X[k + 2 < N ? k + 2 : 0];
            x0.s3 = X[k + 3 < N ? k + 3 : 0];
            x0.s0 = k + 0 < N ? x0.s0 : 0;
            x0.s1 = k + 1 < N ? x0.s1 : 0;
            x0.s2 = k + 2 < N ? x0.s2 : 0;
            x0.s3 = k + 3 < N ? x0.s3 : 0;
            /* -- Tiles multiplier -- */
            const uint2 ay = ((uint2)(0, 1) + (uint)(coordA >> 1)) % (uint)(lda >> 1);
            const uint4 ak = {mad24(k % N, (lda >> 1), 0u), mad24((k + 1) % N, (lda >> 1), 0u), mad24((k + 2) % N, (lda >> 1), 0u),
            		mad24((k + 3) % N, (lda >> 1), 0u)};

            a0.s01 = Ag.d2v[ay.s0 + ak.s0];
            a0.s23 = Ag.d2v[ay.s1 + ak.s0];


            y0 = mad(a0, x0.s0, y0);

            a0.s01 = Ag.d2v[ay.s0 + ak.s1];
            a0.s23 = Ag.d2v[ay.s1 + ak.s1];


            y0 = mad(a0, x0.s1, y0);

            a0.s01 = Ag.d2v[ay.s0 + ak.s2];
            a0.s23 = Ag.d2v[ay.s1 + ak.s2];


            y0 = mad(a0, x0.s2, y0);

            a0.s01 = Ag.d2v[ay.s0 + ak.s3];
            a0.s23 = Ag.d2v[ay.s1 + ak.s3];


            y0 = mad(a0, x0.s3, y0);
            /* ---------------------- */
        }
    }
    localRes[get_local_id(0)][0] = y0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < 8 && coordA < M && k0 < N) {
        for (uint i = 1; i < 8; i++) {
            y0 += localRes[get_local_id(0) + i*8][0];
        }
        Y += coordA;
        double4 r0;
        GPtr uC;
        uC.f = Y;
        r0.s0 = Y[coordA + 0 >= M ? 0 : 0];
        r0.s1 = Y[coordA + 1 >= M ? 0 : 1];
        r0.s2 = Y[coordA + 2 >= M ? 0 : 2];
        r0.s3 = Y[coordA + 3 >= M ? 0 : 3];
        r0 = beta * r0 + alpha * y0;
        Y[coordA + 3 >= M ? 0 : 3] = r0.s3;
        Y[coordA + 2 >= M ? 0 : 2] = r0.s2;
        Y[coordA + 1 >= M ? 0 : 1] = r0.s1;
        Y[coordA + 0 >= M ? 0 : 0] = r0.s0;
    }
}



#endif





template<typename T, const rocblas_int DIM_X, const rocblas_int DIM_Y>
static __device__ void
gemvn_device(
    rocblas_int m, rocblas_int n,
    T alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ x, rocblas_int incx,
    T beta,
    T       * y, rocblas_int incy)
{
    if (m <= 0 || n <= 0) return;

    rocblas_int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;

    if (DIM_X * DIM_Y != num_threads) return; // need to launch exactly the same number of threads as template parameters indicate

    rocblas_int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind;

    __shared__ T sdata[DIM_X * 4 * DIM_Y];

    T res_A[4]; //micor tile is 4 * 4
    T res_x[4];

    res_A[0] = res_x[0] = 0.0;
    res_A[1] = res_x[0] = 0.0;
    res_A[2] = res_x[0] = 0.0;
    res_A[3] = res_x[0] = 0.0;

    ind = hipBlockIdx_x*DIM_X*4 + tx ;

    rocblas_int n_tail = n % (4 * DIM_Y);
    rocblas_int col = ty * 4;

    for (col=ty*4; col < (n - n_tail); col += 4 * DIM_Y)
    {

        res_x[0] = x[(col+0)*incx];
        res_x[1] = x[(col+1)*incx];
        res_x[2] = x[(col+2)*incx];
        res_x[3] = x[(col+3)*incx];

        if(ind < m){
            res_A[0] += A[ind + (col+0)*lda] * res_x[0];
            res_A[0] += A[ind + (col+1)*lda] * res_x[1];
            res_A[0] += A[ind + (col+2)*lda] * res_x[2];
            res_A[0] += A[ind + (col+3)*lda] * res_x[3];
        }

        if(ind + DIM_X < m){
            res_A[1] += A[ind + DIM_X + (col+0)*lda] * res_x[0];
            res_A[1] += A[ind + DIM_X + (col+1)*lda] * res_x[1];
            res_A[1] += A[ind + DIM_X + (col+2)*lda] * res_x[2];
            res_A[1] += A[ind + DIM_X + (col+3)*lda] * res_x[3];
        }

        if(ind + 2 * DIM_X < m){
            res_A[2] += A[ind + 2 * DIM_X + (col+0)*lda] * res_x[0];
            res_A[2] += A[ind + 2 * DIM_X + (col+1)*lda] * res_x[1];
            res_A[2] += A[ind + 2 * DIM_X + (col+2)*lda] * res_x[2];
            res_A[2] += A[ind + 2 * DIM_X + (col+3)*lda] * res_x[3];
        }

        if(ind + 3 * DIM_X < m){
            res_A[3] += A[ind + 3 * DIM_X + (col+0)*lda] * res_x[0];
            res_A[3] += A[ind + 3 * DIM_X + (col+1)*lda] * res_x[1];
            res_A[3] += A[ind + 3 * DIM_X + (col+2)*lda] * res_x[2];
            res_A[3] += A[ind + 3 * DIM_X + (col+3)*lda] * res_x[3];
        }
    }

    // if n  is not multiple of (DIM_Y * 4)
    if(n_tail > 0)
    {
        res_x[0] = (col < n) ? x[(col+0)*incx] : 0 ;
        res_x[1] = (col + 1 < n) ? x[(col+1)*incx] : 0;
        res_x[2] = (col + 2 < n) ? x[(col+2)*incx] : 0;
        res_x[3] = (col + 3 < n) ? x[(col+3)*incx] : 0;

        if(ind < m){
            res_A[0] += A[ind + (col+0)*lda*(col+0 < n)] * res_x[0];
            res_A[0] += A[ind + (col+1)*lda*(col+1 < n)] * res_x[1];
            res_A[0] += A[ind + (col+2)*lda*(col+2 < n)] * res_x[2];
            res_A[0] += A[ind + (col+3)*lda*(col+3 < n)] * res_x[3];
        }


        if(ind + DIM_X < m){
            res_A[1] += A[ind + DIM_X + (col+0)*lda*(col+0 < n)] * res_x[0];
            res_A[1] += A[ind + DIM_X + (col+1)*lda*(col+1 < n)] * res_x[1];
            res_A[1] += A[ind + DIM_X + (col+2)*lda*(col+2 < n)] * res_x[2];
            res_A[1] += A[ind + DIM_X + (col+3)*lda*(col+3 < n)] * res_x[3];
        }

        if(ind + 2 * DIM_X < m){
            res_A[2] += A[ind + 2 * DIM_X + (col+0)*lda*(col+0 < n)] * res_x[0];
            res_A[2] += A[ind + 2 * DIM_X + (col+1)*lda*(col+1 < n)] * res_x[1];
            res_A[2] += A[ind + 2 * DIM_X + (col+2)*lda*(col+2 < n)] * res_x[2];
            res_A[2] += A[ind + 2 * DIM_X + (col+3)*lda*(col+3 < n)] * res_x[3];
        }

        if(ind + 3 * DIM_X < m){
            res_A[3] += A[ind + 3 * DIM_X + (col+0)*lda*(col+0 < n)] * res_x[0];
            res_A[3] += A[ind + 3 * DIM_X + (col+1)*lda*(col+1 < n)] * res_x[1];
            res_A[3] += A[ind + 3 * DIM_X + (col+2)*lda*(col+2 < n)] * res_x[2];
            res_A[3] += A[ind + 3 * DIM_X + (col+3)*lda*(col+3 < n)] * res_x[3];
        }
    }

    sdata[ tx + ty * DIM_X * 4] = res_A[0];
    sdata[ tx + DIM_X + ty * DIM_X * 4] = res_A[1];
    sdata[ tx + 2 * DIM_X + ty * DIM_X * 4] = res_A[2];
    sdata[ tx + 3 * DIM_X + ty * DIM_X * 4] = res_A[3];

    __syncthreads();


    ind = hipBlockIdx_x*DIM_X*4 + thread_id ;
    if(thread_id < DIM_X * 4)
    {
        for (rocblas_int i=1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * 4 * i];
        }

        if(ind < m)
        {
            y[ind*incy] = alpha*sdata[thread_id] + beta*y[ind*incy];
        }
    }

}



template<typename T, const rocblas_int NB_X>
static __device__ void
gemvc_device(
    rocblas_int m, rocblas_int n,
    T alpha,
    const T * __restrict__ A, rocblas_int lda,
    const T * __restrict__ x, rocblas_int incx,
    T beta,
    T       * y, rocblas_int incy)
{
    if (m <= 0 || n <= 0) return;

    rocblas_int tx = hipThreadIdx_x;
    if (tx < m) A += tx;

    rocblas_int col = hipBlockIdx_x;
    A += col * lda;

    T res;
    res = 0.0;

    __shared__ T sdata[NB_X];

    // partial sums
    rocblas_int m_full = (m / NB_X) * NB_X;

    for (rocblas_int i=0; i < m_full; i += NB_X) {
        res += (A[i]) * x[(tx + i)*incx];
    }

    if ( tx + m_full < m ) {
        res += (A[m_full]) * x[(tx + m_full)*incx];
    }

    sdata[tx] = res;

    // tree reduction of partial sums,
    if ( NB_X > 16)
    {
        rocblas_sum_reduce< NB_X >( tx, sdata);
    }
    else
    {
        __syncthreads();

        if (tx == 0)
        {
            for (rocblas_int i=1; i < m && i < NB_X; i++)
            {
                sdata[0] += sdata[i];
            }
        }
        __syncthreads();
    }

    if ( tx == 0) {
        y[col*incy] = alpha*sdata[0] + beta*y[col*incy];
    }

}
