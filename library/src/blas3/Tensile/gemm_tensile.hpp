/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "check_numerics_matrix.hpp"
#include "handle.hpp"

#ifdef BUILD_WITH_TENSILE

#include "tensile_host.hpp"

/*******************************************************************************
 * Tensile Function call
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
inline rocblas_status rocblas_call_tensile(rocblas_handle     handle,
                                           const Tc*          alpha,
                                           const Tc*          beta,
                                           const Ti* const*   batchA,
                                           const Ti* const*   batchB,
                                           const To* const*   batchC,
                                           To* const*         batchD,
                                           rocblas_operation  trans_a,
                                           rocblas_operation  trans_b,
                                           rocblas_int        ld_d,
                                           rocblas_stride     stride_d,
                                           rocblas_stride     offset_d,
                                           rocblas_int        ld_c,
                                           rocblas_stride     stride_c,
                                           rocblas_stride     offset_c,
                                           rocblas_int        ld_a,
                                           rocblas_stride     stride_a,
                                           rocblas_stride     offset_a,
                                           rocblas_int        ld_b,
                                           rocblas_stride     stride_b,
                                           rocblas_stride     offset_b,
                                           rocblas_int        m,
                                           rocblas_int        n,
                                           rocblas_int        k,
                                           rocblas_int        batch_count = 1,
                                           rocblas_gemm_algo  algo = rocblas_gemm_algo_standard,
                                           int32_t            solution_index = 0,
                                           rocblas_gemm_flags flags = rocblas_gemm_flags_none)
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
                                              batchD,
                                              offset_d,
                                              ld_d,
                                              stride_d,
                                              batch_count);
    }
#endif

    RocblasContractionProblem<Ti, To, Tc> problem{
        handle,   trans_a, trans_b,  m,        n,           k,        alpha,    nullptr,
        batchA,   ld_a,    stride_a, offset_a, nullptr,     batchB,   ld_b,     stride_b,
        offset_b, beta,    nullptr,  batchC,   ld_c,        stride_c, offset_c, nullptr,
        batchD,   ld_d,    stride_d, offset_d, batch_count, false,    flags};

    return runContractionProblem(problem, algo, solution_index);
}

template <typename Ti, typename To, typename Tc>
inline rocblas_status rocblas_call_tensile(rocblas_handle     handle,
                                           const Tc*          alpha,
                                           const Tc*          beta,
                                           const Ti*          A,
                                           const Ti*          B,
                                           const To*          C,
                                           To*                D,
                                           rocblas_operation  trans_a,
                                           rocblas_operation  trans_b,
                                           rocblas_int        ld_d,
                                           rocblas_stride     stride_d,
                                           rocblas_stride     offset_d,
                                           rocblas_int        ld_c,
                                           rocblas_stride     stride_c,
                                           rocblas_stride     offset_c,
                                           rocblas_int        ld_a,
                                           rocblas_stride     stride_a,
                                           rocblas_stride     offset_a,
                                           rocblas_int        ld_b,
                                           rocblas_stride     stride_b,
                                           rocblas_stride     offset_b,
                                           rocblas_int        m,
                                           rocblas_int        n,
                                           rocblas_int        k,
                                           rocblas_int        batch_count = 1,
                                           rocblas_gemm_algo  algo = rocblas_gemm_algo_standard,
                                           int32_t            solution_index = 0,
                                           rocblas_gemm_flags flags = rocblas_gemm_flags_none)
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
                                              D,
                                              offset_d,
                                              ld_d,
                                              stride_d,
                                              batch_count);
    }
#endif

    // pre apply offsets for non-batched and strided
    RocblasContractionProblem<Ti, To, Tc> problem{handle,
                                                  trans_a,
                                                  trans_b,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  A + offset_a,
                                                  nullptr,
                                                  ld_a,
                                                  stride_a,
                                                  0 /* offset_a */,
                                                  B + offset_b,
                                                  nullptr,
                                                  ld_b,
                                                  stride_b,
                                                  0 /* offset_b */,
                                                  beta,
                                                  C + offset_c,
                                                  nullptr,
                                                  ld_c,
                                                  stride_c,
                                                  0 /* offset_c */,
                                                  D + offset_d,
                                                  nullptr,
                                                  ld_d,
                                                  stride_d,
                                                  0 /* offset_d */,
                                                  batch_count,
                                                  true,
                                                  flags};

    return runContractionProblem(problem, algo, solution_index);
}

#endif // BUILD_WITH_TENSILE
