/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_trsm_batched.hpp"

// Shared memory usuage is (128/2)^2 * sizeof(float) = 32K. LDS is 64K per CU. Theoretically
// you can use all 64K, but in practice no.
constexpr rocblas_int STRSM_BLOCK = 128;
constexpr rocblas_int DTRSM_BLOCK = 128;


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsm_batched(rocblas_handle     handle,
                                     rocblas_side       side,
                                     rocblas_fill       uplo,
                                     rocblas_operation  transA,
                                     rocblas_diagonal   diag,
                                     rocblas_int        m,
                                     rocblas_int        n,
                                     const float*       alpha,
                                     const float* const A[],
                                     rocblas_int        lda,
                                     float*             B[],
                                     rocblas_int        ldb,
                                     rocblas_int        batch_count)
{
    return rocblas_trsm_batched_ex_impl<STRSM_BLOCK>(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

rocblas_status rocblas_dtrsm_batched(rocblas_handle      handle,
                                     rocblas_side        side,
                                     rocblas_fill        uplo,
                                     rocblas_operation   transA,
                                     rocblas_diagonal    diag,
                                     rocblas_int         m,
                                     rocblas_int         n,
                                     const double*       alpha,
                                     const double* const A[],
                                     rocblas_int         lda,
                                     double*             B[],
                                     rocblas_int         ldb,
                                     rocblas_int         batch_count)
{
    return rocblas_trsm_batched_ex_impl<DTRSM_BLOCK>(
        handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);
}

rocblas_status rocblas_trsm_batched_ex(rocblas_handle    handle,
                                       rocblas_side      side,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_diagonal  diag,
                                       rocblas_int       m,
                                       rocblas_int       n,
                                       const void*       alpha,
                                       const void* const A,
                                       rocblas_int       lda,
                                       void*             B,
                                       rocblas_int       ldb,
                                       rocblas_int       batch_count,
                                       const void*       invA,
                                       rocblas_int       invA_size,
                                       rocblas_datatype  compute_type)
{
    switch(compute_type)
    {
    case rocblas_datatype_f64_r:
        return rocblas_trsm_batched_ex_impl<DTRSM_BLOCK>(handle,
                                                         side,
                                                         uplo,
                                                         transA,
                                                         diag,
                                                         m,
                                                         n,
                                                         static_cast<const double*>(alpha),
                                                         static_cast<const double*const *>(A),
                                                         lda,
                                                         static_cast<double**>(B),
                                                         ldb,
                                                         batch_count,
                                                         static_cast<const double**>(invA),
                                                         invA_size);

    case rocblas_datatype_f32_r:
        return rocblas_trsm_batched_ex_impl<STRSM_BLOCK>(handle,
                                                         side,
                                                         uplo,
                                                         transA,
                                                         diag,
                                                         m,
                                                         n,
                                                         static_cast<const float*>(alpha),
                                                         static_cast<const float* const*>(A),
                                                         lda,
                                                         static_cast<float**>(B),
                                                         ldb,
                                                         batch_count,
                                                         static_cast<const float**>(invA),
                                                         invA_size);

    default:
        return rocblas_status_not_implemented;
    }
}

} // extern "C"
