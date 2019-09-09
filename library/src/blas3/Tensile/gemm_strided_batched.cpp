/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "../gemm_strided_batched.hpp"

extern "C" {

/*******************************************************************************
 * Strided_Batched GEMM APIs
 ******************************************************************************/

rocblas_status rocblas_hgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const rocblas_half *alpha,
                                             const rocblas_half *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const rocblas_half *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const rocblas_half *beta,
                                             rocblas_half *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<rocblas_half>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_sgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const float *alpha,
                                             const float *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const float *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const float *beta,
                                             float *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<float>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_dgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const double *alpha,
                                             const double *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const double *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const double *beta,
                                             double *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<double>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_cgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const rocblas_float_complex *alpha,
                                             const rocblas_float_complex *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const rocblas_float_complex *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const rocblas_float_complex *beta,
                                             rocblas_float_complex *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<rocblas_float_complex>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_zgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const rocblas_double_complex *alpha,
                                             const rocblas_double_complex *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const rocblas_double_complex *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const rocblas_double_complex *beta,
                                             rocblas_double_complex *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<rocblas_double_complex>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

/*******************************************************************************
 * Strided Batched GEMM Kernel name APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm_strided_batched_kernel_name(rocblas_handle handle,
                                                         rocblas_operation trans_a,
                                                         rocblas_operation trans_b,
                                                         rocblas_int m,
                                                         rocblas_int n,
                                                         rocblas_int k,
                                                         const rocblas_half *alpha,
                                                         const rocblas_half *A,
                                                         rocblas_int ld_a,
                                                         rocblas_int stride_a,
                                                         const rocblas_half *B,
                                                         rocblas_int ld_b,
                                                         rocblas_int stride_b,
                                                         const rocblas_half *beta,
                                                         rocblas_half *C,
                                                         rocblas_int ld_c,
                                                         rocblas_int stride_c,
                                                         rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_kernel_name_impl<rocblas_half>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_sgemm_strided_batched_kernel_name(rocblas_handle handle,
                                                         rocblas_operation trans_a,
                                                         rocblas_operation trans_b,
                                                         rocblas_int m,
                                                         rocblas_int n,
                                                         rocblas_int k,
                                                         const float *alpha,
                                                         const float *A,
                                                         rocblas_int ld_a,
                                                         rocblas_int stride_a,
                                                         const float *B,
                                                         rocblas_int ld_b,
                                                         rocblas_int stride_b,
                                                         const float *beta,
                                                         float *C,
                                                         rocblas_int ld_c,
                                                         rocblas_int stride_c,
                                                         rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_kernel_name_impl<float>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_dgemm_strided_batched_kernel_name(rocblas_handle handle,
                                                         rocblas_operation trans_a,
                                                         rocblas_operation trans_b,
                                                         rocblas_int m,
                                                         rocblas_int n,
                                                         rocblas_int k,
                                                         const double *alpha,
                                                         const double *A,
                                                         rocblas_int ld_a,
                                                         rocblas_int stride_a,
                                                         const double *B,
                                                         rocblas_int ld_b,
                                                         rocblas_int stride_b,
                                                         const double *beta,
                                                         double *C,
                                                         rocblas_int ld_c,
                                                         rocblas_int stride_c,
                                                         rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_kernel_name_impl<double>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}



}