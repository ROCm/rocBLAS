/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_matrix.hpp"
#include "handle.hpp"

#ifdef BUILD_WITH_TENSILE

#include "tensile_host.hpp"

#include "../../blas_ex/rocblas_gemm_ex.hpp"

/*******************************************************************************
 * Tensile Function call
 ******************************************************************************/
template <typename T>
inline rocblas_status call_tensile(rocblas_handle    handle,
                                   const T*          alpha,
                                   const T*          beta,
                                   const T* const*   batchA,
                                   const T* const*   batchB,
                                   T* const*         batchC,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       ld_c,
                                   rocblas_stride    stride_c,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ld_a,
                                   rocblas_stride    stride_a,
                                   rocblas_stride    offset_a,
                                   rocblas_int       ld_b,
                                   rocblas_stride    stride_b,
                                   rocblas_stride    offset_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   rocblas_int       batch_count = 1)
{
#if 0
    // if tensile supports we can remove special case handling here
    if(k == 0 || (alpha && !*alpha))
    {
        // !beta early return and beta always on host here so can dereference
        return rocblas_gemm_ex_scale_template(handle,
                                              m,
                                              n,
                                              *beta,
                                              batchC,
                                              offset_c,
                                              ld_c,
                                              stride_c,
                                              batchC,
                                              offset_c,
                                              ld_c,
                                              stride_c,
                                              batch_count);
    }
#endif

    RocblasContractionProblem<T> problem{handle,   trans_a,
                                         trans_b,  m,
                                         n,        k,
                                         alpha,    nullptr,
                                         batchA,   ld_a,
                                         stride_a, offset_a,
                                         nullptr,  batchB,
                                         ld_b,     stride_b,
                                         offset_b, beta,
                                         nullptr,  batchC,
                                         ld_c,     stride_c,
                                         offset_c, batch_count,
                                         false,    rocblas_gemm_flags_none};

    return runContractionProblem(problem);
}

template <typename T>
inline rocblas_status call_tensile(rocblas_handle    handle,
                                   const T*          alpha,
                                   const T*          beta,
                                   const T*          A,
                                   const T*          B,
                                   T*                C,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       ld_c,
                                   rocblas_stride    stride_c,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ld_a,
                                   rocblas_stride    stride_a,
                                   rocblas_stride    offset_a,
                                   rocblas_int       ld_b,
                                   rocblas_stride    stride_b,
                                   rocblas_stride    offset_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   rocblas_int       batch_count = 1)
{
#if 0
    // if tensile supports we can remove special case handling here
    if(k == 0 || (alpha && !*alpha))
    {
        // !beta early return and beta always on host here so can dereference
        return rocblas_gemm_ex_scale_template(handle,
                                              m,
                                              n,
                                              *beta,
                                              C,
                                              offset_c,
                                              ld_c,
                                              stride_c,
                                              C,
                                              offset_c,
                                              ld_c,
                                              stride_c,
                                              batch_count);
    }
#endif

    RocblasContractionProblem<T> problem{handle,   trans_a,
                                         trans_b,  m,
                                         n,        k,
                                         alpha,    A,
                                         nullptr,  ld_a,
                                         stride_a, offset_a,
                                         B,        nullptr,
                                         ld_b,     stride_b,
                                         offset_b, beta,
                                         C,        nullptr,
                                         ld_c,     stride_c,
                                         offset_c, batch_count,
                                         true,     rocblas_gemm_flags_none};

    return runContractionProblem(problem);
}

#endif // BUILD_WITH_TENSILE
