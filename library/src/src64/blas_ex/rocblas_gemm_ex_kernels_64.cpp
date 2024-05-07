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

#include "handle.hpp"
#include "int64_helpers.hpp"

#include "rocblas_gemm_ex_64.hpp"

#include "blas_ex/rocblas_gemm_ex.hpp" // int32 API called

template <bool BATCHED>
rocblas_status rocblas_gemm_ex_template_64(rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           int64_t           m_64,
                                           int64_t           n_64,
                                           int64_t           k_64,
                                           const void*       alpha,
                                           const void*       A,
                                           rocblas_datatype  a_type,
                                           rocblas_stride    offsetAin,
                                           int64_t           lda_64,
                                           rocblas_stride    stride_a,
                                           const void*       B,
                                           rocblas_datatype  b_type,
                                           rocblas_stride    offsetBin,
                                           int64_t           ldb_64,
                                           rocblas_stride    stride_b,
                                           const void*       beta,
                                           const void*       C,
                                           rocblas_datatype  c_type,
                                           rocblas_stride    offsetCin,
                                           int64_t           ldc_64,
                                           rocblas_stride    stride_c,
                                           void*             D,
                                           rocblas_datatype  d_type,
                                           rocblas_stride    offsetDin,
                                           int64_t           ldd_64,
                                           rocblas_stride    stride_d,
                                           int64_t           batch_count_64,
                                           rocblas_datatype  compute_type,
                                           rocblas_gemm_algo algo,
                                           int32_t           solution_index,
                                           uint32_t          flags)
{
    bool dims_32bit = m_64 <= c_i32_max && n_64 <= c_i32_max && k_64 <= c_i32_max
                      && lda_64 <= c_i32_max && ldb_64 <= c_i32_max && ldc_64 <= c_i32_max
                      && ldd_64 <= c_i32_max;

    if(!dims_32bit)
    {
        // TODO: not yet supported, will support in another PR soon
        return rocblas_status_invalid_size;
    }

    // if all dims are 32-bit, can use regular gemm_ex
    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        rocblas_stride offsetA = offsetAin;
        rocblas_stride offsetB = offsetBin;
        rocblas_stride offsetC = offsetCin;
        rocblas_stride offsetD = offsetDin;
        auto           Aptr    = A;
        auto           Bptr    = B;
        auto           Cptr    = C;
        auto           Dptr    = D;

        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        if constexpr(BATCHED)
        {
            // avoiding typecasting
            Aptr = (void**)Aptr + b_base;
            Bptr = (void**)Bptr + b_base;
            Cptr = (void**)Cptr + b_base;
            Dptr = (void**)Dptr + b_base;
        }
        else if(batch_count > 1)
        {
            offsetA += b_base * stride_a;
            offsetB += b_base * stride_b;
            offsetC += b_base * stride_c;
            offsetD += b_base * stride_d;
        }

        rocblas_status status = rocblas_gemm_ex_template<BATCHED>(handle,
                                                                  trans_a,
                                                                  trans_b,
                                                                  m_64,
                                                                  n_64,
                                                                  k_64,
                                                                  alpha,
                                                                  Aptr,
                                                                  a_type,
                                                                  offsetA,
                                                                  lda_64,
                                                                  stride_a,
                                                                  Bptr,
                                                                  b_type,
                                                                  offsetB,
                                                                  ldb_64,
                                                                  stride_b,
                                                                  beta,
                                                                  Cptr,
                                                                  c_type,
                                                                  offsetC,
                                                                  ldc_64,
                                                                  stride_c,
                                                                  Dptr,
                                                                  d_type,
                                                                  offsetD,
                                                                  ldd_64,
                                                                  stride_d,
                                                                  batch_count,
                                                                  compute_type,
                                                                  algo,
                                                                  solution_index,
                                                                  flags);

        if(status != rocblas_status_success)
            return status;
    }

    return rocblas_status_success;
}

#ifdef INSTANTIATE_GEMM_EX_64_TEMPLATE
#error INSTANTIATE_GEMM_EX_64_TEMPLATE already defined
#endif

#define INSTANTIATE_GEMM_EX_64_TEMPLATE(BATCHED_)                                                   \
    template rocblas_status rocblas_gemm_ex_template_64<BATCHED_>(rocblas_handle    handle,         \
                                                                  rocblas_operation trans_a,        \
                                                                  rocblas_operation trans_b,        \
                                                                  int64_t           m,              \
                                                                  int64_t           n,              \
                                                                  int64_t           k,              \
                                                                  const void*       alpha,          \
                                                                  const void*       a,              \
                                                                  rocblas_datatype  a_type,         \
                                                                  rocblas_stride    offsetAin,      \
                                                                  int64_t           lda,            \
                                                                  rocblas_stride    stride_a,       \
                                                                  const void*       b,              \
                                                                  rocblas_datatype  b_type,         \
                                                                  rocblas_stride    offsetBin,      \
                                                                  int64_t           ldb,            \
                                                                  rocblas_stride    stride_b,       \
                                                                  const void*       beta,           \
                                                                  const void*       c,              \
                                                                  rocblas_datatype  c_type,         \
                                                                  rocblas_stride    offsetCin,      \
                                                                  int64_t           ldc,            \
                                                                  rocblas_stride    stride_c,       \
                                                                  void*             d,              \
                                                                  rocblas_datatype  d_type,         \
                                                                  rocblas_stride    offsetDin,      \
                                                                  int64_t           ldd,            \
                                                                  rocblas_stride    stride_d,       \
                                                                  int64_t           batch_count,    \
                                                                  rocblas_datatype  compute_type,   \
                                                                  rocblas_gemm_algo algo,           \
                                                                  int32_t           solution_index, \
                                                                  uint32_t          flags);

INSTANTIATE_GEMM_EX_64_TEMPLATE(true)
INSTANTIATE_GEMM_EX_64_TEMPLATE(false)
