/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocblas_block_sizes.h"
#include "rocblas_syrk_herk.hpp"
#include "rocblas_syrkx_herkx.hpp"

/**
  * T is base type, i.e. float, double, rocblas_float_complex, or rocblas_double_complex
  * TScal is base type of scalars, for HERM == false, TScal == T, for HERM == true, TScal == real_t<T>
  * TConstPtr is either: const T* OR const T* const*
  * TPtr      is either:       T* OR       T* const*
  */
template <rocblas_int NB,
          bool        BATCHED,
          bool        HERM,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_syrk_herk_template(rocblas_handle    handle,
                                                   rocblas_fill      uplo,
                                                   rocblas_operation trans_a,
                                                   rocblas_int       n,
                                                   rocblas_int       k,
                                                   const TScal*      alpha,
                                                   TConstPtr         A,
                                                   rocblas_stride    offset_a,
                                                   rocblas_int       lda,
                                                   rocblas_stride    stride_A,
                                                   const TScal*      beta,
                                                   TPtr              C,
                                                   rocblas_stride    offset_c,
                                                   rocblas_int       ldc,
                                                   rocblas_stride    stride_C,
                                                   rocblas_int       batch_count)
{
    // quick returns handled in rocblas_internal_syr2k_her2k_template
    constexpr bool TWOK = false;
    return rocblas_internal_syr2k_her2k_template<rocblas_int, NB, BATCHED, TWOK, HERM, T>(
        handle,
        uplo,
        trans_a,
        n,
        k,
        alpha,
        A,
        offset_a,
        lda,
        stride_A,
        A,
        offset_a,
        lda,
        stride_A,
        beta,
        C,
        offset_c,
        ldc,
        stride_C,
        batch_count);
}

#define ROCBLAS_INTERNAL_SYRK_HERK_PARAMS                                                   \
    handle, uplo, trans_a, n, k, alpha, A, offset_a, lda, stride_A, beta, C, offset_c, ldc, \
        stride_C, batch_count

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation trans_a,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const T*          alpha,
                                   const T*          A,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_A,
                                   const T*          beta,
                                   T*                C,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_C,
                                   rocblas_int       batch_count)
{
    constexpr bool BATCHED = false;
    constexpr bool HERM    = false;
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDZSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDZSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_CSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDZSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_batched_template(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation trans_a,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const T*          alpha,
                                           const T* const*   A,
                                           rocblas_stride    offset_a,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_A,
                                           const T*          beta,
                                           T* const*         C,
                                           rocblas_stride    offset_c,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_C,
                                           rocblas_int       batch_count)
{
    constexpr bool BATCHED = true;
    constexpr bool HERM    = false;
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_CZSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_CZSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation trans_a,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const real_t<T>*  alpha,
                                   const T*          A,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_A,
                                   const real_t<T>*  beta,
                                   T*                C,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_C,
                                   rocblas_int       batch_count)
{
    constexpr bool BATCHED = false;
    constexpr bool HERM    = true;
    if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_CHERK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_ZHERK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_batched_template(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation trans_a,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const real_t<T>*  alpha,
                                           const T* const*   A,
                                           rocblas_stride    offset_a,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_A,
                                           const real_t<T>*  beta,
                                           T* const*         C,
                                           rocblas_stride    offset_c,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_C,
                                           rocblas_int       batch_count)
{
    constexpr bool BATCHED = true;
    constexpr bool HERM    = true;
    if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_HERK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_HERK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

#undef ROCBLAS_INTERNAL_SYRK_HERK_PARAMS

template <bool HERM, typename TConstPtr, typename TPtr>
rocblas_status rocblas_herk_syrk_check_numerics(const char*       function_name,
                                                rocblas_handle    handle,
                                                rocblas_fill      uplo,
                                                rocblas_operation trans,
                                                int64_t           n_64,
                                                int64_t           k_64,
                                                TConstPtr         A,
                                                int64_t           lda_64,
                                                rocblas_stride    stride_A,
                                                TPtr              C,
                                                int64_t           ldc_64,
                                                rocblas_stride    stride_C,
                                                int64_t           batch_count_64,
                                                const int         check_numerics,
                                                bool              is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;
    if(is_input)
    {
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              n_64,
                                                              k_64,
                                                              A,
                                                              0,
                                                              lda_64,
                                                              stride_A,
                                                              batch_count_64,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    check_numerics_status = rocblas_internal_check_numerics_matrix_template(
        function_name,
        handle,
        rocblas_operation_none,
        uplo,
        HERM ? rocblas_client_hermitian_matrix : rocblas_client_symmetric_matrix,
        n_64,
        n_64,
        C,
        0,
        ldc_64,
        stride_C,
        batch_count_64,
        check_numerics,
        is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syrk*.cpp or herk*.cpp

#ifdef INSTANTIATE_SYRK_TEMPLATE
#error INSTANTIATE_SYRK_TEMPLATE already defined
#endif

#define INSTANTIATE_SYRK_TEMPLATE(T_)                                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_syrk_template<T_>( \
        rocblas_handle    handle,                                                                \
        rocblas_fill      uplo,                                                                  \
        rocblas_operation trans_a,                                                               \
        rocblas_int       n,                                                                     \
        rocblas_int       k,                                                                     \
        const T_*         alpha,                                                                 \
        const T_*         A,                                                                     \
        rocblas_stride    offset_a,                                                              \
        rocblas_int       lda,                                                                   \
        rocblas_stride    stride_A,                                                              \
        const T_*         beta,                                                                  \
        T_*               C,                                                                     \
        rocblas_stride    offset_c,                                                              \
        rocblas_int       ldc,                                                                   \
        rocblas_stride    stride_C,                                                              \
        rocblas_int       batch_count);

INSTANTIATE_SYRK_TEMPLATE(float)
INSTANTIATE_SYRK_TEMPLATE(double)
INSTANTIATE_SYRK_TEMPLATE(rocblas_float_complex)
INSTANTIATE_SYRK_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_SYRK_TEMPLATE

#ifdef INSTANTIATE_SYRK_BATCHED_TEMPLATE
#error INSTANTIATE_SYRK_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_SYRK_BATCHED_TEMPLATE(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                   \
        rocblas_internal_syrk_batched_template<T_>(rocblas_handle    handle,   \
                                                   rocblas_fill      uplo,     \
                                                   rocblas_operation trans_a,  \
                                                   rocblas_int       n,        \
                                                   rocblas_int       k,        \
                                                   const T_*         alpha,    \
                                                   const T_* const*  A,        \
                                                   rocblas_stride    offset_a, \
                                                   rocblas_int       lda,      \
                                                   rocblas_stride    stride_A, \
                                                   const T_*         beta,     \
                                                   T_* const*        C,        \
                                                   rocblas_stride    offset_c, \
                                                   rocblas_int       ldc,      \
                                                   rocblas_stride    stride_C, \
                                                   rocblas_int       batch_count);

INSTANTIATE_SYRK_BATCHED_TEMPLATE(float)
INSTANTIATE_SYRK_BATCHED_TEMPLATE(double)
INSTANTIATE_SYRK_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_SYRK_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_SYRK_BATCHED_TEMPLATE

#ifdef INSTANTIATE_HERK_TEMPLATE
#error INSTANTIATE_HERK_TEMPLATE already defined
#endif

#define INSTANTIATE_HERK_TEMPLATE(T_)                                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_herk_template<T_>( \
        rocblas_handle    handle,                                                                \
        rocblas_fill      uplo,                                                                  \
        rocblas_operation trans_a,                                                               \
        rocblas_int       n,                                                                     \
        rocblas_int       k,                                                                     \
        const real_t<T_>* alpha,                                                                 \
        const T_*         A,                                                                     \
        rocblas_stride    offset_a,                                                              \
        rocblas_int       lda,                                                                   \
        rocblas_stride    stride_A,                                                              \
        const real_t<T_>* beta,                                                                  \
        T_*               C,                                                                     \
        rocblas_stride    offset_c,                                                              \
        rocblas_int       ldc,                                                                   \
        rocblas_stride    stride_C,                                                              \
        rocblas_int       batch_count);

INSTANTIATE_HERK_TEMPLATE(rocblas_float_complex)
INSTANTIATE_HERK_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_HERK_TEMPLATE

#ifdef INSTANTIATE_HERK_BATCHED_TEMPLATE
#error INSTANTIATE_HERK_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_HERK_BATCHED_TEMPLATE(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                   \
        rocblas_internal_herk_batched_template<T_>(rocblas_handle    handle,   \
                                                   rocblas_fill      uplo,     \
                                                   rocblas_operation trans_a,  \
                                                   rocblas_int       n,        \
                                                   rocblas_int       k,        \
                                                   const real_t<T_>* alpha,    \
                                                   const T_* const*  A,        \
                                                   rocblas_stride    offset_a, \
                                                   rocblas_int       lda,      \
                                                   rocblas_stride    stride_A, \
                                                   const real_t<T_>* beta,     \
                                                   T_* const*        C,        \
                                                   rocblas_stride    offset_c, \
                                                   rocblas_int       ldc,      \
                                                   rocblas_stride    stride_C, \
                                                   rocblas_int       batch_count);

INSTANTIATE_HERK_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_HERK_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_HERK_BATCHED_TEMPLATE

#ifdef INSTANTIATE_HERK_SYRK_NUMERICS
#error INSTANTIATE_HERK_SYRK_NUMERICS already defined
#endif

#define INSTANTIATE_HERK_SYRK_NUMERICS(HERM_, TConstPtr_, TPtr_)                        \
    template rocblas_status rocblas_herk_syrk_check_numerics<HERM_, TConstPtr_, TPtr_>( \
        const char*       function_name,                                                \
        rocblas_handle    handle,                                                       \
        rocblas_fill      uplo,                                                         \
        rocblas_operation trans,                                                        \
        int64_t           n_64,                                                         \
        int64_t           k_64,                                                         \
        TConstPtr_        A,                                                            \
        int64_t           lda_64,                                                       \
        rocblas_stride    stride_A,                                                     \
        TPtr_             C,                                                            \
        int64_t           ldc_64,                                                       \
        rocblas_stride    stride_C,                                                     \
        int64_t           batch_count_64,                                               \
        const int         check_numerics,                                               \
        bool              is_input);

// instantiate for rocblas_Xherk_Xsyrk and rocblas_Xherk_Xsyrk_strided_batched
INSTANTIATE_HERK_SYRK_NUMERICS(false, float const*, float*)
INSTANTIATE_HERK_SYRK_NUMERICS(false, double const*, double*)
INSTANTIATE_HERK_SYRK_NUMERICS(false, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HERK_SYRK_NUMERICS(true, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HERK_SYRK_NUMERICS(false, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_HERK_SYRK_NUMERICS(true, rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xherk_Xsyrk_batched
INSTANTIATE_HERK_SYRK_NUMERICS(false, float const* const*, float* const*)
INSTANTIATE_HERK_SYRK_NUMERICS(false, double const* const*, double* const*)
INSTANTIATE_HERK_SYRK_NUMERICS(false,
                               rocblas_float_complex const* const*,
                               rocblas_float_complex* const*)
INSTANTIATE_HERK_SYRK_NUMERICS(true,
                               rocblas_float_complex const* const*,
                               rocblas_float_complex* const*)
INSTANTIATE_HERK_SYRK_NUMERICS(false,
                               rocblas_double_complex const* const*,
                               rocblas_double_complex* const*)
INSTANTIATE_HERK_SYRK_NUMERICS(true,
                               rocblas_double_complex const* const*,
                               rocblas_double_complex* const*)

#undef INSTANTIATE_HERK_SYRK_NUMERICS
