/*
 * ===========================================================================
 *    This file provide common device function for gemv routines
 * ===========================================================================
 */

/* ============================================================================================ */

#pragma once
#ifndef _TRSV_DEVICE_H_
#define _TRSV_DEVICE_H_

#include "../blas1/device_template.h"

template <typename T, const rocblas_int BLOCK, const rocblas_int threads>
__device__ void tocache(const T* A, rocblas_int lda, rocblas_int col, T* cache)
{
//        //what index in to cache[]
//        if(diag == rocblas_diagonal_unit)
//            if() // if indexing diag
//                cache[] = 1.0f; //how does this work with diff precisions
//            else
//                cache []  = A[i*lda+hipThreadIdx_x]  ;
//        else if(hipThreadIdx_x) // lower or higher cache?

    //            cache[] = A[i*lda+hipThreadIdx_x];

//    int tid= hipBlockDim_x * hipThreadIdx_y + hipThreadIdx_x;
//    int skip =  hipThreadIdx_x / BLOCK;
//    int index = tid - (hipBlockDim_x*hipBlockDim_y - threads);
//    if(index<BLOCK*BOCK)
//        cache[index] = A[index+skip*lda]; //   think about index if cannot make even columns

    int tid= hipBlockDim_x * hipThreadIdx_y + hipThreadIdx_x;
    int index = tid - (hipBlockDim_x*hipBlockDim_y - threads);

    if(index<BLOCK)
        for(int i=index*BLOCK; i<index*BLOCK+BLOCK; i++)
            cache[i-BLOCK*col-(lda-BLOCK)*index] = A[i];
}

template <typename T, const rocblas_int BLOCK, const rocblas_int threads>
__device__ void cacherect(const T* A, rocblas_int lda, rocblas_int col, T* cache)
{
    int tid= hipBlockDim_x * hipThreadIdx_y + hipThreadIdx_x;
    int index = tid - (hipBlockDim_x*hipBlockDim_y - threads);

    if(index<lda-BLOCK*(col+1))
        for(int i=index*BLOCK; i<index*BLOCK+BLOCK; i++)
            cache[i-BLOCK*col-(lda-BLOCK)*index] = A[i];
}

template <typename T, const rocblas_int BLOCK>
__device__ void dblkSolve(const T *A , rocblas_int lda , T &val)
{

    volatile T __shared__ xs;
//    int num_threads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;

//    if(DIM_X * DIM_Y != num_threads)
//        return; // need to launch exactly the same number of threads as template parameters indicate

//    int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;

    // threads are all configurated locally
//    int tx = thread_id % DIM_X;
//    int ty = thread_id / DIM_X;

#pragma unroll 32
    for(int i =0; i<BLOCK; i++)
    {
        if( hipThreadIdx_x == i)
            xs = val;
        if(hipThreadIdx_x>i)
            val -= A[i*lda+hipThreadIdx_x] * xs;
    }
}

// BLOCK = 32
// example 128 x 128 matrix
//NT_X=hipThreadIdx_x=m-BLOCK = 96
//NT_Y=hipThreadIdx_y=4or8,say(autotune).
template <typename T, const rocblas_int BLOCK, const rocblas_int NT_X, const rocblas_int NT_Y>
__global__ void trsv_device(rocblas_int m,
                             const T* __restrict__ A,
                             rocblas_int lda,
                             T* __restrict__ xglobal,
                             rocblas_int incx)
{
//    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
//    T *smem = reinterpret_cast<T *>(my_smem);

#if defined(__HIP_PLATFORM_HCC__)
    HIP_DYNAMIC_SHARED(T, smem)
#else
    HIP_DYNAMIC_SHARED(__align__(sizeof(T)) unsigned char, my_sdata)
    T* smem = reinterpret_cast<T*>(my_sdata);
#endif
//    T __shared__ xshared_actual[m], rect [(m-BLOCK)*BLOCK], cache_even [BLOCK*BLOCK], cache_odd[BLOCK*BLOCK];
    T* xshared_actual = smem;
    T* rect = &smem[m];
    T* cache_even = &smem[m+(m-BLOCK)*BLOCK];
    T* cache_odd = &smem[m+(m-BLOCK)*BLOCK + BLOCK*BLOCK];

    int tid = NT_X*hipThreadIdx_y+hipThreadIdx_x;

    //Precache x and first diagonal block

    if(tid<m)
        xshared_actual[tid]= xglobal[tid]; //should be xshared_actual?
    tocache<T, BLOCK,NT_X*NT_Y>(A, lda, 0, cache_even);  // all 96*NT_Y threads launch this
    __syncthreads();

//    /*Main loop*/
    T* xshared=xshared_actual;
    for(int ii=0;ii<m;ii+=BLOCK) // ii is A multiple of BLOCK
    {
//        Preload entries for rectangular block during stalls of dblkSolve
        if(hipThreadIdx_y!=0)
            cacherect<T, BLOCK, NT_X*(NT_Y-1)>(A+BLOCK*lda,lda,ii/BLOCK,rect); // rect length should get smaller as ii increases
//        Solve diagonal block
        if(hipThreadIdx_y==0&&hipThreadIdx_x<BLOCK)
        {
            T val= xshared[hipThreadIdx_x]; 
            if(ii%(2*BLOCK)==0)
                dblkSolve<T,BLOCK>(cache_even,BLOCK,val);
            else
                dblkSolve<T,BLOCK>(cache_odd,BLOCK,val);
            xshared[hipThreadIdx_x]=val;
        }
        else if(ii+BLOCK<m) // load next diag block
            if(ii%(2*BLOCK)==0)
                tocache<T, BLOCK,NT_X*NT_Y-BLOCK>(&A[BLOCK*lda],lda,ii/BLOCK,cache_odd); // BLOCK threads being used for dblkSolve 64 x NT_Y
            else
                tocache<T, BLOCK,NT_X*NT_Y-BLOCK>(&A[BLOCK*lda],lda,ii/BLOCK,cache_even);
        __syncthreads();

        //Apply rectangular block

        if(hipThreadIdx_y==0&&hipThreadIdx_x<m-BLOCK)
            for(int j=BLOCK;j+ii<m;j+=m-BLOCK)
            {
                T val=0;
                if(ii+j+hipThreadIdx_x<m)
                {
                    for(int i=0;i<BLOCK;i++)
                            val+=rect[i*(m-BLOCK)+j-BLOCK+hipThreadIdx_x]*xshared[i];
                    xshared[j+hipThreadIdx_x]-=val;
                }
            }
        __syncthreads();
        A+=BLOCK*lda; //why did they have + 1
        xshared+=BLOCK;
    }

    //    /*Store x back to global memory*/
        if(tid<m)
            xglobal[tid]=xshared_actual[tid];
}

#define STRSV_BLOCK 128
#define DTRSV_BLOCK 128

 #endif // _TRSV_DEVICE_H_
