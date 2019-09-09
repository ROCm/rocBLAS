/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
 #include "rocblas_gemm_ex_strided_batched.hpp"




extern "C" rocblas_status rocblas_gemm_strided_batched_ex(rocblas_handle    handle,
                                                          rocblas_operation trans_a,
                                                          rocblas_operation trans_b,
                                                          rocblas_int       m,
                                                          rocblas_int       n,
                                                          rocblas_int       k,
                                                          const void*       alpha,
                                                          const void*       a,
                                                          rocblas_datatype  a_type,
                                                          rocblas_int       lda,
                                                          rocblas_long      stride_a,
                                                          const void*       b,
                                                          rocblas_datatype  b_type,
                                                          rocblas_int       ldb,
                                                          rocblas_long      stride_b,
                                                          const void*       beta,
                                                          const void*       c,
                                                          rocblas_datatype  c_type,
                                                          rocblas_int       ldc,
                                                          rocblas_long      stride_c,
                                                          void*             d,
                                                          rocblas_datatype  d_type,
                                                          rocblas_int       ldd,
                                                          rocblas_long      stride_d,
                                                          rocblas_int       batch_count,
                                                          rocblas_datatype  compute_type,
                                                          rocblas_gemm_algo algo,
                                                          int32_t           solution_index,
                                                          uint32_t          flags)
{
    return rocblas_gemm_strided_batched_ex_impl(handle, trans_a, trans_b, m, n, k, alpha, a, a_type, lda, stride_a, b, b_type,
                                                ldb, stride_b, beta, c, c_type, ldc, stride_c, d, d_type, ldd, stride_d,
                                                batch_count, compute_type, algo, solution_index, flags);
}