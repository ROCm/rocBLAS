/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "Tensile/gemm.hpp"
#include "definitions.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_syr2k_her2k.hpp"
#include "rocblas_syrkx.hpp"

#define OFFSET_A(i1) offset_a + i1* rocblas_stride(a_s1)
#define OFFSET_B(i1) offset_b + i1* rocblas_stride(b_s1)
#define OFFSET_C(i1, i2) offset_c + i1* rocblas_stride(c_s1) + i2* rocblas_stride(c_s2)

template <int  MIN_NB,
          bool BATCHED,
          bool HERK,
          typename T,
          typename TScal,
          typename TPtr,
          typename TConstPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrkx_herkx_template(rocblas_handle    handle,
                                          rocblas_fill      uplo,
                                          rocblas_operation trans,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          TScal*            alpha,
                                          TConstPtr*        da,
                                          rocblas_stride    offset_a,
                                          rocblas_int       lda,
                                          rocblas_stride    stride_a,
                                          TConstPtr*        db,
                                          rocblas_stride    offset_b,
                                          rocblas_int       ldb,
                                          rocblas_stride    stride_b,
                                          TScal*            beta,
                                          TPtr*             dc,
                                          rocblas_stride    offset_c,
                                          rocblas_int       ldc,
                                          rocblas_stride    stride_c,
                                          rocblas_int       batch_count)
{
    static constexpr bool TWOK = false;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*beta == 1)
        {
            if(!k || !*alpha)
                return rocblas_status_success;
        }
        // all early return success now handled so
        // pointers must be valid
        bool ab_calc_invalid = !alpha || (*alpha != 0 && (!da || !db));
        if(!dc || (k && ab_calc_invalid))
            return rocblas_status_invalid_pointer;
    }
    else
    {
        return rocblas_status_internal_error; // always pushed host_mode prevalidation
    }

    return rocblas_internal_syr2k_her2k_template<MIN_NB, BATCHED, TWOK, HERK>(handle,
                                                                              uplo,
                                                                              trans,
                                                                              n,
                                                                              k,
                                                                              alpha,
                                                                              da,
                                                                              offset_a,
                                                                              lda,
                                                                              stride_a,
                                                                              db,
                                                                              offset_b,
                                                                              ldb,
                                                                              stride_b,
                                                                              beta,
                                                                              dc,
                                                                              offset_c,
                                                                              ldc,
                                                                              stride_c,
                                                                              batch_count);
}
#undef OFFSET_A
#undef OFFSET_B
#undef OFFSET_C

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syrkx*.cpp

// clang-format off
#ifdef INSTANTIATE_SYRKX_HERKX_TEMPLATE
#error INSTANTIATE_SYRKX_HERKX_TEMPLATE already defined
#endif

#define INSTANTIATE_SYRKX_HERKX_TEMPLATE(MIN_NB_, BATCHED_, HERK_, T_, TScal_, TPtr_, TConstPtr_)        \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_syrkx_herkx_template    \
                                   <MIN_NB_, BATCHED_, HERK_, T_, TScal_, TPtr_, TConstPtr_>       \
                                   (rocblas_handle    handle,                               \
                                    rocblas_fill      uplo,                                 \
                                    rocblas_operation trans,                                \
                                    rocblas_int       n,                                    \
                                    rocblas_int       k,                                    \
                                    TScal_ *          alpha,                                \
                                    TConstPtr_ *      da,                                   \
                                    rocblas_stride    offset_a,                             \
                                    rocblas_int       lda,                                  \
                                    rocblas_stride    stride_a,                             \
                                    TConstPtr_ *      db,                                   \
                                    rocblas_stride    offset_b,                             \
                                    rocblas_int       ldb,                                  \
                                    rocblas_stride    stride_b,                             \
                                    TScal_ *          beta,                                 \
                                    TPtr_ *           dc,                                   \
                                    rocblas_stride    offset_c,                             \
                                    rocblas_int       ldc,                                  \
                                    rocblas_stride    stride_c,                             \
                                    rocblas_int       batch_count);

// instantiate for rocblas_Xsyrkx and rocblas_Xsyrkx_strided_batched
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_SSYRKX_NB, false, false, float,  float const,  float,  float const)
// INSTANTIATE_SYRKX_HERKX_TEMPLATE(16, false, false, double, double const, double, double const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_DCZSYRKX_NB, false, false, double, double const, double, double const)
// INSTANTIATE_SYRKX_HERKX_TEMPLATE( 8, false, false,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex,  rocblas_float_complex const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_DCZSYRKX_NB, false, false,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex,  rocblas_float_complex const)
// INSTANTIATE_SYRKX_HERKX_TEMPLATE( 8, false, false, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex, rocblas_double_complex const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_DCZSYRKX_NB, false, false, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex, rocblas_double_complex const)

// instantiate for rocblas_Xsyrk, double/double complex precisions already covered
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_SDZSYRK_NB, false, false, float,  float const,  float,  float const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_CSYRK_NB, false, false,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex,  rocblas_float_complex const)

// instantiate for rocblas_Xherkx and rocblas_Xherkx_strided_batched
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_HERKX_NB, false, true,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex,  rocblas_float_complex const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_HERKX_NB, false, true, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex, rocblas_double_complex const)

// instantiate for rocblas_Xherk
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_CHERK_NB, false, true,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex,  rocblas_float_complex const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_ZHERK_NB, false, true, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex, rocblas_double_complex const)

// instantiate for rocblas_Xsyrk(x)_batched
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_SDSYRKX_BATCHED_NB,  true, false,  float,  float const,  float* const,  float const* const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_SDSYRKX_BATCHED_NB,  true, false, double, double const, double* const, double const* const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_CZSYRKX_BATCHED_NB,  true, false,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex* const,  rocblas_float_complex const* const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_CZSYRKX_BATCHED_NB,  true, false, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex* const, rocblas_double_complex const* const)

// instantiate for rocblas_Xherk(x)_batched
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_HERKX_BATCHED_NB,  true, true,  rocblas_float_complex,  rocblas_float_complex const,  rocblas_float_complex* const,  rocblas_float_complex const* const)
INSTANTIATE_SYRKX_HERKX_TEMPLATE(ROCBLAS_HERKX_BATCHED_NB,  true, true, rocblas_double_complex, rocblas_double_complex const, rocblas_double_complex* const, rocblas_double_complex const* const)

#undef INSTANTIATE_SYRKX_HERKX_TEMPLATE
// clang-format on
