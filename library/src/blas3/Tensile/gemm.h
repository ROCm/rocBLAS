#include "Tensile.h"
#include "rocblas-types.h"

/*******************************************************************************
 * Infer Batch Strides
 ******************************************************************************/
inline void infer_batch_strides(rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                rocblas_int       m,
                                rocblas_int       n,
                                rocblas_int       k,
                                rocblas_int       ld_a,
                                rocblas_int*      stride_a,
                                rocblas_int       ld_b,
                                rocblas_int*      stride_b,
                                rocblas_int       ld_c,
                                rocblas_int*      stride_c)
{

    rocblas_int num_cols_c = n;
    rocblas_int num_rows_c = m;
    rocblas_int num_cols_a = (trans_a == rocblas_operation_none ? k : m);
    rocblas_int num_rows_a = (trans_a == rocblas_operation_none ? m : k);
    rocblas_int num_cols_b = (trans_b == rocblas_operation_none ? n : k);
    rocblas_int num_rows_b = (trans_b == rocblas_operation_none ? k : n);

    *stride_a = ld_a * num_cols_a;
    *stride_b = ld_b * num_cols_b;
    *stride_c = ld_c * num_cols_c;

} // infer batched strides

/*******************************************************************************
 * Validate Arguments
 ******************************************************************************/
inline rocblas_status validateArgs(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const void*       alpha,
                                   const void*       a,
                                   rocblas_int       ld_a,
                                   rocblas_int       stride_a,
                                   const void*       b,
                                   rocblas_int       ld_b,
                                   rocblas_int       stride_b,
                                   const void*       beta,
                                   void*             c,
                                   rocblas_int       ld_c,
                                   rocblas_int       stride_c,
                                   rocblas_int       batch_count)
{

    // quick return 0 is valid in BLAS
    if(m == 0 || n == 0 || k == 0 || batch_count == 0)
    {
        return rocblas_status_success;
    }

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
    {
        return rocblas_status_invalid_size;
    }

    // handle must be valid
    if(handle == nullptr)
    {
        return rocblas_status_invalid_handle;
    }

    // pointers must be valid
    if(c == nullptr || a == nullptr || b == nullptr || alpha == nullptr || beta == nullptr)
    {
        return rocblas_status_invalid_pointer;
    }

    rocblas_int num_cols_c = n;
    rocblas_int num_rows_c = m;
    rocblas_int num_cols_a = (trans_a == rocblas_operation_none) ? k : m;
    rocblas_int num_rows_a = (trans_a == rocblas_operation_none) ? m : k;
    rocblas_int num_cols_b = (trans_b == rocblas_operation_none) ? n : k;
    rocblas_int num_rows_b = (trans_b == rocblas_operation_none) ? k : n;

    // leading dimensions must be valid
    if(num_rows_a > ld_a || num_rows_b > ld_b || num_rows_c > ld_c)
    {
        return rocblas_status_invalid_size;
    }

    return rocblas_status_success;
} // validate parameters

/*******************************************************************************
 * temporary solution: complex gemm helper
 * we will use tensile for final solution
 ******************************************************************************/
template <typename DT, typename ST>
__global__ void kernel_copy_real_part_of_complex_matrix(DT*               dst,
                                                        const rocblas_int ld_dst,
                                                        const rocblas_int str_dst,
                                                        const ST*         src,
                                                        const rocblas_int ld_src,
                                                        const rocblas_int str_src,
                                                        const rocblas_int m,
                                                        const rocblas_int n)
{
    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    dst = dst + hipBlockIdx_z * str_dst;
    src = src + hipBlockIdx_z * str_src;

    if(tx < m && ty < n)
        dst[tx + ld_dst * ty] = src[tx + ld_src * ty].x;
}

template <typename DT, typename ST>
__global__ void kernel_copy_imag_part_of_complex_matrix(DT*               dst,
                                                        const rocblas_int ld_dst,
                                                        const rocblas_int str_dst,
                                                        const ST*         src,
                                                        const rocblas_int ld_src,
                                                        const rocblas_int str_src,
                                                        const rocblas_int m,
                                                        const rocblas_int n)
{
    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    dst = dst + hipBlockIdx_z * str_dst;
    src = src + hipBlockIdx_z * str_src;

    if(tx < m && ty < n)
        dst[tx + ld_dst * ty] = src[tx + ld_src * ty].y;
}

template <typename DT, typename ST>
void copy_part_of_complex_matrix(const hipStream_t rocblas_stream,
                                 const bool        real,
                                 DT*               dst,
                                 const rocblas_int ld_dst,
                                 const rocblas_int str_dst,
                                 const ST*         src,
                                 const rocblas_int ld_src,
                                 const rocblas_int str_src,
                                 const rocblas_int m,
                                 const rocblas_int n,
                                 const rocblas_int b)
{
    rocblas_int blocksX = ((m - 1) / 128) + 1; // parameters for device kernel
    rocblas_int blocksY = ((n - 1) / 8) + 1;
    rocblas_int blocksZ = b;
    dim3        grid(blocksX, blocksY, blocksZ);
    dim3        threads(128, 8, 1);

    if(real)
    {
        hipLaunchKernelGGL(kernel_copy_real_part_of_complex_matrix,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           dst,
                           ld_dst,
                           str_dst,
                           src,
                           ld_src,
                           str_src,
                           m,
                           n);
    }
    else
    {
        hipLaunchKernelGGL(kernel_copy_imag_part_of_complex_matrix,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           dst,
                           ld_dst,
                           str_dst,
                           src,
                           ld_src,
                           str_src,
                           m,
                           n);
    }
}

template <typename T1, typename T2>
__device__ void kernel_complex_addition(const T1*         data1_r,
                                        const T1*         data1_i,
                                        const rocblas_int ld1,
                                        const rocblas_int str1,
                                        const T2          alpha,
                                        const T2*         data2,
                                        const rocblas_int ld2,
                                        const rocblas_int str2,
                                        const T2          beta,
                                        T2*               data3,
                                        const rocblas_int ld3,
                                        const rocblas_int str3,
                                        const rocblas_int m,
                                        const rocblas_int n)
{
    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    data1_r = data1_r + hipBlockIdx_z * str1;
    data1_i = data1_i + hipBlockIdx_z * str1;
    data2   = data2 + hipBlockIdx_z * str2;
    data3   = data3 + hipBlockIdx_z * str3;

    if(tx < m && ty < n)
    {
        T1 d1_r = data1_r[tx + ld1 * ty];
        T1 d1_i = data1_i[tx + ld1 * ty];
        T2 d2   = data2[tx + ld2 * ty];

        T2 res;
        res.x = (alpha.x * d1_r - alpha.y * d1_i) + (beta.x * d2.x - beta.y * d2.y);
        res.y = (alpha.x * d1_i + alpha.y * d1_r) + (beta.x * d2.y + beta.y * d2.x);

        data3[tx + ld3 * ty] = res;
    }
}

template <typename T1, typename T2>
__global__ void kernel_complex_addition_h(const T1*         data1_r,
                                          const T1*         data1_i,
                                          const rocblas_int ld1,
                                          const rocblas_int str1,
                                          const T2          alpha,
                                          const T2*         data2,
                                          const rocblas_int ld2,
                                          const rocblas_int str2,
                                          const T2          beta,
                                          T2*               data3,
                                          const rocblas_int ld3,
                                          const rocblas_int str3,
                                          const rocblas_int m,
                                          const rocblas_int n)
{
    kernel_complex_addition(
        data1_r, data1_i, ld1, str1, alpha, data2, ld2, str2, beta, data3, ld3, str3, m, n);
}

template <typename T1, typename T2>
__global__ void kernel_complex_addition_d(const T1*         data1_r,
                                          const T1*         data1_i,
                                          const rocblas_int ld1,
                                          const rocblas_int str1,
                                          const T2*         alpha,
                                          const T2*         data2,
                                          const rocblas_int ld2,
                                          const rocblas_int str2,
                                          const T2*         beta,
                                          T2*               data3,
                                          const rocblas_int ld3,
                                          const rocblas_int str3,
                                          const rocblas_int m,
                                          const rocblas_int n)
{
    kernel_complex_addition(
        data1_r, data1_i, ld1, str1, *alpha, data2, ld2, str2, *beta, data3, ld3, str3, m, n);
}

// T1 is float, T2 is complex
template <typename T1, typename T2>
void complex_addition(const hipStream_t          rocblas_stream,
                      const rocblas_pointer_mode pointer_mode,
                      const T1*                  data1_r,
                      const T1*                  data1_i,
                      const rocblas_int          ld1,
                      const rocblas_int          str1,
                      const T2*                  alpha,
                      const T2*                  data2,
                      const rocblas_int          ld2,
                      const rocblas_int          str2,
                      const T2*                  beta,
                      T2*                        data3,
                      const rocblas_int          ld3,
                      const rocblas_int          str3,
                      const rocblas_int          m,
                      const rocblas_int          n,
                      const rocblas_int          b)
{
    rocblas_int blocksX = ((m - 1) / 128) + 1; // parameters for device kernel
    rocblas_int blocksY = ((n - 1) / 8) + 1;
    rocblas_int blocksZ = b;

    dim3 grid(blocksX, blocksY, blocksZ);
    dim3 threads(128, 8, 1);

    if(rocblas_pointer_mode_host == pointer_mode)
    {
        hipLaunchKernelGGL(kernel_complex_addition_h,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           data1_r,
                           data1_i,
                           ld1,
                           str1,
                           *alpha,
                           data2,
                           ld2,
                           str2,
                           *beta,
                           data3,
                           ld3,
                           str3,
                           m,
                           n);
    }
    else
    {
        hipLaunchKernelGGL(kernel_complex_addition_d,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           data1_r,
                           data1_i,
                           ld1,
                           str1,
                           alpha,
                           data2,
                           ld2,
                           str2,
                           beta,
                           data3,
                           ld3,
                           str3,
                           m,
                           n);
    }
}
