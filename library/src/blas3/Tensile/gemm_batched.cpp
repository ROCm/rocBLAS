/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
 #include "../gemm_batched.hpp"


/*******************************************************************************
 * Batched GEMM APIs
 ******************************************************************************/

extern "C" {
rocblas_status rocblas_hgemm_batched(rocblas_handle            handle,
                                     rocblas_operation         trans_a,
                                     rocblas_operation         trans_b,
                                     rocblas_int               m,
                                     rocblas_int               n,
                                     rocblas_int               k,
                                     const rocblas_half*       alpha,
                                     const rocblas_half* const A[],
                                     rocblas_int               ld_a,
                                     const rocblas_half* const B[],
                                     rocblas_int               ld_b,
                                     const rocblas_half*       beta,
                                     rocblas_half* const       C[],
                                     rocblas_int               ld_c,
                                     rocblas_int               b_c)
{
    return rocblas_gemm_batched_impl<rocblas_half>(handle, trans_a, trans_b, m, n, k, alpha, A, 0, ld_a, B, 0, ld_b, beta, C, 0, ld_c, b_c);
}

rocblas_status rocblas_sgemm_batched(rocblas_handle    handle,
                                     rocblas_operation  trans_a,
                                     rocblas_operation  trans_b,
                                     rocblas_int        m,
                                     rocblas_int        n,
                                     rocblas_int        k,
                                     const float*       alpha,
                                     const float* const A[],
                                     rocblas_int        ld_a,
                                     const float* const B[],
                                     rocblas_int        ld_b,
                                     const float*       beta,
                                     float* const       C[],
                                     rocblas_int        ld_c,
                                     rocblas_int        b_c)
{
    return rocblas_gemm_batched_impl<float>(handle, trans_a, trans_b, m, n, k, alpha, A, 0, ld_a, B, 0, ld_b, beta, C, 0, ld_c, b_c);
}

rocblas_status rocblas_dgemm_batched(rocblas_handle     handle,
                                     rocblas_operation   trans_a,
                                     rocblas_operation   trans_b,
                                     rocblas_int         m,
                                     rocblas_int         n,
                                     rocblas_int         k,
                                     const double*       alpha,
                                     const double* const A[],
                                     rocblas_int         ld_a,
                                     const double* const B[],
                                     rocblas_int         ld_b,
                                     const double*       beta,
                                     double* const       C[],
                                     rocblas_int         ld_c,
                                     rocblas_int         b_c)
{
    return rocblas_gemm_batched_impl<double>(handle, trans_a, trans_b, m, n, k, alpha, A, 0, ld_a, B, 0, ld_b, beta, C, 0, ld_c, b_c);
}

rocblas_status rocblas_cgemm_batched(rocblas_handle                    handle,
                                     rocblas_operation                  trans_a,
                                     rocblas_operation                  trans_b,
                                     rocblas_int                        m,
                                     rocblas_int                        n,
                                     rocblas_int                        k,
                                     const rocblas_float_complex*       alpha,
                                     const rocblas_float_complex* const A[],
                                     rocblas_int                        ld_a,
                                     const rocblas_float_complex* const B[],
                                     rocblas_int                        ld_b,
                                     const rocblas_float_complex*       beta,
                                     rocblas_float_complex* const       C[],
                                     rocblas_int                        ld_c,
                                     rocblas_int                        b_c)
{
    return rocblas_gemm_batched_impl<rocblas_float_complex>(handle, trans_a, trans_b, m, n, k, alpha, A, 0, ld_a, B, 0, ld_b, beta, C, 0, ld_c, b_c);
}

rocblas_status rocblas_zgemm_batched(rocblas_handle                     handle,
                                     rocblas_operation                   trans_a,
                                     rocblas_operation                   trans_b,
                                     rocblas_int                         m,
                                     rocblas_int                         n,
                                     rocblas_int                         k,
                                     const rocblas_double_complex*       alpha,
                                     const rocblas_double_complex* const A[],
                                     rocblas_int                         ld_a,
                                     const rocblas_double_complex* const B[],
                                     rocblas_int                         ld_b,
                                     const rocblas_double_complex*       beta,
                                     rocblas_double_complex* const       C[],
                                     rocblas_int                         ld_c,
                                     rocblas_int                         b_c)
{
    return rocblas_gemm_batched_impl<rocblas_double_complex>(handle, trans_a, trans_b, m, n, k, alpha, A, 0, ld_a, B, 0, ld_b, beta, C, 0, ld_c, b_c);
}


/*******************************************************************************
 * Batched GEMM Kernel name APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm_batched_kernel_name(rocblas_handle handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_int k,
                                                 const rocblas_half *alpha,
                                                 const rocblas_half *A[],
                                                 rocblas_int ld_a,
                                                 const rocblas_half *B[],
                                                 rocblas_int ld_b,
                                                 const rocblas_half *beta,
                                                 rocblas_half *C[],
                                                 rocblas_int ld_c,
                                                 rocblas_int b_c)
{
    return rocblas_gemm_batched_kernel_name_impl<rocblas_half>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a,
        B, ld_b,
        beta,
        C, ld_c, b_c);
}

rocblas_status rocblas_sgemm_batched_kernel_name(rocblas_handle handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_int k,
                                                 const float *alpha,
                                                 const float *A[],
                                                 rocblas_int ld_a,
                                                 const float *B[],
                                                 rocblas_int ld_b,
                                                 const float *beta,
                                                 float *C[],
                                                 rocblas_int ld_c,
                                                 rocblas_int b_c)
{
    return rocblas_gemm_batched_kernel_name_impl<float>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a,
        B, ld_b,
        beta,
        C, ld_c, b_c);
}

rocblas_status rocblas_dgemm_batched_kernel_name(rocblas_handle handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_int k,
                                                 const double *alpha,
                                                 const double *A[],
                                                 rocblas_int ld_a,
                                                 const double *B[],
                                                 rocblas_int ld_b,
                                                 const double *beta,
                                                 double *C[],
                                                 rocblas_int ld_c,
                                                 rocblas_int b_c)
{
    return rocblas_gemm_batched_kernel_name_impl<double>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a,
        B, ld_b,
        beta,
        C, ld_c, b_c);
}



}