/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

// BLAS Level 1 includes routines and functions performing vector-vector
// operations. Most BLAS 1 routines are about reduction: compute the norm,
// calculate the dot production of two vectors, find the maximum/minimum index
// of the element of the vector. As you may observed, although the computation
// type is different, the core algorithm is the same: scan all element of the
// vector(s) and reduce to one single result. 
// 
// The reduction algorithm on GPU is called [parallel
// reduction](https://raw.githubusercontent.com/mateuszbuda/GPUExample/master/reduce3.png)
// which is adopted in rocBLAS. At the beginning, all the threads in the thread
// block participate. After each step of reduction (like a tree), the number of
// participating threads decrease by half. At the end of the parallel reduction,
// only one thread (usually thread 0) owns the result in its thread
// block. 
// 
// Classically, the BLAS 1 reduction needs more than one GPU kernel to finish,
// because the lack of global synchronization of thread blocks without exiting
// the kernel. The first kernels gather partial results, write into a temporary
// working buffer. The second kernel finish the final reduction. 
// 
// For example, BLAS 1 routine iXamax is to find index of the maximum absolute
// value element of a vector. In this routine:
// 
// Kernel 1: launch many thread block as needed. Each thread block works on a
// subset of the vector. Each thread block use the parallel reduction to find a
// local index with the maximum absolute value of the subset. There are
// number-of-the-thread-blocks local results.The results are written into a
// temporary working buffer. The working buffer has number-of-the-thread-blocks
// elements.
//    
// Kernel 2: launch only one thread block which reads the temporary work buffer and
// reduces to final result still with the parallel reduction. 
// 
// As you may see, if there is a mechanism to synchronize all the thread blocks
// after local index is obtained in kernel 1 (without ending the kernel), then
// Kernel 2's computation can be merged into Kernel 1. One such mechanism is called
// atomic operation. However, atomic operation is new and is not used in rocBLAS
// yet. rocBLAS still use the classic standard parallel reduction right now.  

/*! \brief parallel reduction: sum

    \details

    @param[in]
    n         rocblas_int. assume n <= 1024 and a mutiple of 2;
    @param[in]
    tx        rocblas_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              x[0] store the final result.
    ********************************************************************/
template <rocblas_int n, typename T>
__device__ void rocblas_sum_reduce(rocblas_int tx, T* x)
{
    // clang-format off
    __syncthreads();
    if(n > 512) { if(tx < 512 && tx + 512 < n) { x[tx] += x[tx + 512]; } __syncthreads(); }
    if(n > 256) { if(tx < 256 && tx + 256 < n) { x[tx] += x[tx + 256]; } __syncthreads(); }
    if(n > 128) { if(tx < 128 && tx + 128 < n) { x[tx] += x[tx + 128]; } __syncthreads(); } 
    if(n >  64) { if(tx <  64 && tx +  64 < n) { x[tx] += x[tx +  64]; } __syncthreads(); }
    if(n >  32) { if(tx <  32 && tx +  32 < n) { x[tx] += x[tx +  32]; } __syncthreads(); }
    if(n >  16) { if(tx <  16 && tx +  16 < n) { x[tx] += x[tx +  16]; } __syncthreads(); }
    if(n >   8) { if(tx <   8 && tx +   8 < n) { x[tx] += x[tx +   8]; } __syncthreads(); }
    if(n >   4) { if(tx <   4 && tx +   4 < n) { x[tx] += x[tx +   4]; } __syncthreads(); }
    if(n >   2) { if(tx <   2 && tx +   2 < n) { x[tx] += x[tx +   2]; } __syncthreads(); }
    if(n >   1) { if(tx <   1 && tx +   1 < n) { x[tx] += x[tx +   1]; } __syncthreads(); }
    // clang-format on
}
// end sum_reduce

/*! \brief parallel reduction: min

    \details

    @param[in]
    n         rocblas_int. assume n <= 1024 and a mutiple of 2;
    @param[in]
    tx        rocblas_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              x[0] store the final result.
    ********************************************************************/

template <rocblas_int n, typename T>
__device__ void rocblas_min_reduce(rocblas_int tx, T* x)
{
    // clang-format off
    __syncthreads();
    if(n > 512) { if(tx < 512 && tx + 512 < n) { x[tx] = min(x[tx], x[tx + 512]); } __syncthreads(); }
    if(n > 256) { if(tx < 256 && tx + 256 < n) { x[tx] = min(x[tx], x[tx + 256]); } __syncthreads(); }
    if(n > 128) { if(tx < 128 && tx + 128 < n) { x[tx] = min(x[tx], x[tx + 128]); } __syncthreads(); }
    if(n >  64) { if(tx <  64 && tx +  64 < n) { x[tx] = min(x[tx], x[tx +  64]); } __syncthreads(); }
    if(n >  32) { if(tx <  32 && tx +  32 < n) { x[tx] = min(x[tx], x[tx +  32]); } __syncthreads(); }
    if(n >  16) { if(tx <  16 && tx +  16 < n) { x[tx] = min(x[tx], x[tx +  16]); } __syncthreads(); }
    if(n >   8) { if(tx <   8 && tx +   8 < n) { x[tx] = min(x[tx], x[tx +   8]); } __syncthreads(); }
    if(n >   4) { if(tx <   4 && tx +   4 < n) { x[tx] = min(x[tx], x[tx +   4]); } __syncthreads(); }
    if(n >   2) { if(tx <   2 && tx +   2 < n) { x[tx] = min(x[tx], x[tx +   2]); } __syncthreads(); }
    if(n >   1) { if(tx <   1 && tx +   1 < n) { x[tx] = min(x[tx], x[tx +   1]); } __syncthreads(); }
    // clang-format on
}
// end min_reduce

/*! \brief parallel reduction: max

    \details

    @param[in]
    n         rocblas_int. assume n <= 1024 and a mutiple of 2;
    @param[in]
    tx        rocblas_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              x[0] store the final result.
    ********************************************************************/
template <rocblas_int n, typename T>
__device__ void rocblas_max_reduce(rocblas_int tx, T* x)
{
    // clang-format off
    __syncthreads();
    if(n > 512) { if(tx < 512 && tx + 512 < n) { x[tx] = max(x[tx], x[tx + 512]); } __syncthreads(); }
    if(n > 256) { if(tx < 256 && tx + 256 < n) { x[tx] = max(x[tx], x[tx + 256]); } __syncthreads(); }
    if(n > 128) { if(tx < 128 && tx + 128 < n) { x[tx] = max(x[tx], x[tx + 128]); } __syncthreads(); }
    if(n >  64) { if(tx <  64 && tx +  64 < n) { x[tx] = max(x[tx], x[tx +  64]); } __syncthreads(); }
    if(n >  32) { if(tx <  32 && tx +  32 < n) { x[tx] = max(x[tx], x[tx +  32]); } __syncthreads(); }
    if(n >  16) { if(tx <  16 && tx +  16 < n) { x[tx] = max(x[tx], x[tx +  16]); } __syncthreads(); }
    if(n >   8) { if(tx <   8 && tx +   8 < n) { x[tx] = max(x[tx], x[tx +   8]); } __syncthreads(); }
    if(n >   4) { if(tx <   4 && tx +   4 < n) { x[tx] = max(x[tx], x[tx +   4]); } __syncthreads(); }
    if(n >   2) { if(tx <   2 && tx +   2 < n) { x[tx] = max(x[tx], x[tx +   2]); } __syncthreads(); }
    if(n >   1) { if(tx <   1 && tx +   1 < n) { x[tx] = max(x[tx], x[tx +   1]); } __syncthreads(); }
    // clang-format on
}
// end max_reduce

/*! \brief parallel reduction: minid

    \details

    finds the first index of the minimum element of vector x

    @param[in]
    n         rocblas_int. assume n <= 1024 and a mutiple of 2;
    @param[in]
    tx        rocblas_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              on exit, x will be overwritten
    @param[inout]
    index     pointer storing vector index of x on the GPU.
              usually index is stored in shared memory;
              on exit, index will be overwritten
              index[0] stores the final result

    ********************************************************************/
template <rocblas_int n, typename T>
__device__ void rocblas_minid_reduce(rocblas_int tx, T* x, rocblas_int* index)
{
    __syncthreads();
    if(n > 512)
    {
        if(tx < 512 && tx + 512 < n)
        {
            if(x[tx] == x[tx + 512])
            {
                index[tx] = min(index[tx], index[tx + 512]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 512])
            {
                index[tx] = index[tx + 512];
                x[tx]     = x[tx + 512];
            }
        }
        __syncthreads();
    }

    if(n > 256)
    {
        if(tx < 256 && tx + 256 < n)
        {
            if(x[tx] == x[tx + 256])
            {
                index[tx] = min(index[tx], index[tx + 256]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 256])
            {
                index[tx] = index[tx + 256];
                x[tx]     = x[tx + 256];
            }
        }
        __syncthreads();
    }

    if(n > 128)
    {
        if(tx < 128 && tx + 128 < n)
        {
            if(x[tx] == x[tx + 128])
            {
                index[tx] = min(index[tx], index[tx + 128]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 128])
            {
                index[tx] = index[tx + 128];
                x[tx]     = x[tx + 128];
            }
        }
        __syncthreads();
    }

    if(n > 64)
    {
        if(tx < 64 && tx + 64 < n)
        {
            if(x[tx] == x[tx + 64])
            {
                index[tx] = min(index[tx], index[tx + 64]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 64])
            {
                index[tx] = index[tx + 64];
                x[tx]     = x[tx + 64];
            }
        }
        __syncthreads();
    }

    if(n > 32)
    {
        if(tx < 32 && tx + 32 < n)
        {
            if(x[tx] == x[tx + 32])
            {
                index[tx] = min(index[tx], index[tx + 32]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 32])
            {
                index[tx] = index[tx + 32];
                x[tx]     = x[tx + 32];
            }
        }
        __syncthreads();
    }

    if(n > 16)
    {
        if(tx < 16 && tx + 16 < n)
        {
            if(x[tx] == x[tx + 16])
            {
                index[tx] = min(index[tx], index[tx + 16]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 16])
            {
                index[tx] = index[tx + 16];
                x[tx]     = x[tx + 16];
            }
        }
        __syncthreads();
    }

    if(n > 8)
    {
        if(tx < 8 && tx + 8 < n)
        {
            if(x[tx] == x[tx + 8])
            {
                index[tx] = min(index[tx], index[tx + 8]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 8])
            {
                index[tx] = index[tx + 8];
                x[tx]     = x[tx + 8];
            }
        }
        __syncthreads();
    }

    if(n > 4)
    {
        if(tx < 4 && tx + 4 < n)
        {
            if(x[tx] == x[tx + 4])
            {
                index[tx] = min(index[tx], index[tx + 4]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 4])
            {
                index[tx] = index[tx + 4];
                x[tx]     = x[tx + 4];
            }
        }
        __syncthreads();
    }

    if(n > 2)
    {
        if(tx < 2 && tx + 2 < n)
        {
            if(x[tx] == x[tx + 2])
            {
                index[tx] = min(index[tx], index[tx + 2]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 2])
            {
                index[tx] = index[tx + 2];
                x[tx]     = x[tx + 2];
            }
        }
        __syncthreads();
    }

    if(n > 1)
    {
        if(tx < 1 && tx + 1 < n)
        {
            if(x[tx] == x[tx + 1])
            {
                index[tx] = min(index[tx], index[tx + 1]);
            } // if equal take the smaller index
            else if(x[tx] > x[tx + 1])
            {
                index[tx] = index[tx + 1];
                x[tx]     = x[tx + 1];
            }
        }
        __syncthreads();
    }
}
// end minid_reduce

/*! \brief parallel reduction: maxid

    \details

    finds the first index of the maximum element of vector x

    @param[in]
    n         rocblas_int. assume n <= 1024 and a mutiple of 2;
    @param[in]
    tx        rocblas_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              on exit, x will be overwritten
    @param[inout]
    index     pointer storing vector index of x on the GPU.
              usually index is stored in shared memory;
              on exit, index will be overwritten
              index[0] stores the final result

    ********************************************************************/
template <rocblas_int n, typename T>
__device__ void rocblas_maxid_reduce(rocblas_int tx, T* x, rocblas_int* index)
{
    __syncthreads();
    if(n > 512)
    {
        if(tx < 512 && tx + 512 < n)
        {
            if(x[tx] == x[tx + 512])
            {
                index[tx] = min(index[tx], index[tx + 512]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 512])
            {
                index[tx] = index[tx + 512];
                x[tx]     = x[tx + 512];
            }
        }
        __syncthreads();
    }

    if(n > 256)
    {
        if(tx < 256 && tx + 256 < n)
        {
            if(x[tx] == x[tx + 256])
            {
                index[tx] = min(index[tx], index[tx + 256]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 256])
            {
                index[tx] = index[tx + 256];
                x[tx]     = x[tx + 256];
            }
        }
        __syncthreads();
    }

    if(n > 128)
    {
        if(tx < 128 && tx + 128 < n)
        {
            if(x[tx] == x[tx + 128])
            {
                index[tx] = min(index[tx], index[tx + 128]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 128])
            {
                index[tx] = index[tx + 128];
                x[tx]     = x[tx + 128];
            }
        }
        __syncthreads();
    }

    if(n > 64)
    {
        if(tx < 64 && tx + 64 < n)
        {
            if(x[tx] == x[tx + 64])
            {
                index[tx] = min(index[tx], index[tx + 64]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 64])
            {
                index[tx] = index[tx + 64];
                x[tx]     = x[tx + 64];
            }
        }
        __syncthreads();
    }

    if(n > 32)
    {
        if(tx < 32 && tx + 32 < n)
        {
            if(x[tx] == x[tx + 32])
            {
                index[tx] = min(index[tx], index[tx + 32]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 32])
            {
                index[tx] = index[tx + 32];
                x[tx]     = x[tx + 32];
            }
        }
        __syncthreads();
    }

    if(n > 16)
    {
        if(tx < 16 && tx + 16 < n)
        {
            if(x[tx] == x[tx + 16])
            {
                index[tx] = min(index[tx], index[tx + 16]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 16])
            {
                index[tx] = index[tx + 16];
                x[tx]     = x[tx + 16];
            }
        }
        __syncthreads();
    }

    if(n > 8)
    {
        if(tx < 8 && tx + 8 < n)
        {
            if(x[tx] == x[tx + 8])
            {
                index[tx] = min(index[tx], index[tx + 8]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 8])
            {
                index[tx] = index[tx + 8];
                x[tx]     = x[tx + 8];
            }
        }
        __syncthreads();
    }

    if(n > 4)
    {
        if(tx < 4 && tx + 4 < n)
        {
            if(x[tx] == x[tx + 4])
            {
                index[tx] = min(index[tx], index[tx + 4]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 4])
            {
                index[tx] = index[tx + 4];
                x[tx]     = x[tx + 4];
            }
        }
        __syncthreads();
    }

    if(n > 2)
    {
        if(tx < 2 && tx + 2 < n)
        {
            if(x[tx] == x[tx + 2])
            {
                index[tx] = min(index[tx], index[tx + 2]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 2])
            {
                index[tx] = index[tx + 2];
                x[tx]     = x[tx + 2];
            }
        }
        __syncthreads();
    }

    if(n > 1)
    {
        if(tx < 1 && tx + 1 < n)
        {
            if(x[tx] == x[tx + 1])
            {
                index[tx] = min(index[tx], index[tx + 1]);
            } // if equal take the smaller index
            else if(x[tx] < x[tx + 1])
            {
                index[tx] = index[tx + 1];
                x[tx]     = x[tx + 1];
            }
        }
        __syncthreads();
    }
}
// end maxid_reduce
