/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TRANSPOSE_DEVICE_H
#define TRANSPOSE_DEVICE_H

/*
   transpose input of size m * n (up to DIM_X * DIM_X) to output of size n * m
   input, output are in device memory 
   shared memory of size DIM_X*DIM_Y is allocated internally as working space
   
   Assume DIM_X by DIM_Y threads are reading & wrting a tile size DIM_X * DIM_X
   DIM_X is divisible by DIM_Y
*/

template<typename T, int DIM_X, int DIM_Y>
__device__ void
transpose_tile_device(const T* input, T* output, int m, int n, int input_lda, int output_lda )
{

    __shared__ T shared_A[DIM_X][DIM_X+1];//avoid bank conflicts

    int tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    int tx1 = tid % DIM_X;
    int ty1 = tid / DIM_X;
    
    #pragma unroll
    for(int i=0; i<n; i+=DIM_Y)
    {
        if( tx1 < m && (ty1 + i) < n)
        {
            shared_A[ty1+i][tx1] = input[tx1 + (ty1 + i) * input_lda];   // the transpose taking place here
        }
    }
    __syncthreads();// ? 

    for(int i=0; i<m; i+=DIM_Y)
    {
        //reconfigure the threads 
        if( tx1 < n && (ty1 + i)< m)
        {
            output[tx1 + (i + ty1) * output_lda] = shared_A[tx1][ty1+i];
        }
    }

}

/*
   transpose input of size m * n to output of size n * m
   input, output are in device memory 

   2D grid and 2D thread block 
   the grid is orgnized into m * n matrix 
   
   Assume DIM_X by DIM_Y threads are transposing each tile DIM_X * DIM_X
*/



template<typename T, int DIM_X, int DIM_Y>
__global__ void
transpose_kernel( hipLaunchParm lp, int m, int n, const T* input, T* output, int input_lda, int output_lda )
{

    input += hipBlockIdx_x * DIM_X + hipBlockIdx_y * DIM_X * input_lda;   
    output += hipBlockIdx_x * DIM_X * output_lda + hipBlockIdx_y * DIM_X;
    
    int mm = min(m - hipBlockIdx_x * DIM_X, DIM_X); // the corner case along m 
    int nn = min(n - hipBlockIdx_y * DIM_X, DIM_X); // the corner case along n 

    transpose_tile_device<T, DIM_X, DIM_Y>(input, output, mm, nn, input_lda, output_lda );
}


#endif //TRANSPOSE_DEVICE_H







