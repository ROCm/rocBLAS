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

#include "handle.hpp"
#include "herk_scale_device.hpp"
#include "rocblas_syrk_herk.hpp"
#include "rocblas_syrkx.hpp"

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <rocblas_int NB,
          bool        BATCHED,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation trans_a,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   TScal             alpha,
                                   TConstPtr         AP,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   TScal             beta,
                                   TPtr              CP,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    T alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    // we can just call syrkx here
    constexpr bool HERM = false;
    return rocblas_internal_syrkx_herkx_template<NB, BATCHED, HERM, T>(handle,
                                                                       uplo,
                                                                       trans_a,
                                                                       n,
                                                                       k,
                                                                       alpha,
                                                                       AP,
                                                                       offset_a,
                                                                       lda,
                                                                       stride_a,
                                                                       AP,
                                                                       offset_a,
                                                                       lda,
                                                                       stride_a,
                                                                       beta,
                                                                       CP,
                                                                       offset_c,
                                                                       ldc,
                                                                       stride_c,
                                                                       batch_count);
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <rocblas_int NB,
          bool        BATCHED,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation trans_a,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   TScal             alpha,
                                   TConstPtr         AP,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   TScal             beta,
                                   TPtr              CP,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    real_t<T> alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    if((k == 0 || !*alpha) && *beta == 1)
        return rocblas_status_success;

    // we can just call herkx here
    const rocblas_complex_num<real_t<T>> alpha_comp = {*alpha, 0};
    const rocblas_complex_num<real_t<T>> beta_comp  = {*beta, 0};
    constexpr bool                       HERM       = true;
    return rocblas_internal_syrkx_herkx_template<NB, BATCHED, HERM, T>(handle,
                                                                       uplo,
                                                                       trans_a,
                                                                       n,
                                                                       k,
                                                                       &alpha_comp,
                                                                       AP,
                                                                       offset_a,
                                                                       lda,
                                                                       stride_a,
                                                                       AP,
                                                                       offset_a,
                                                                       lda,
                                                                       stride_a,
                                                                       &beta_comp,
                                                                       CP,
                                                                       offset_c,
                                                                       ldc,
                                                                       stride_c,
                                                                       batch_count);
}
template <bool HERM, typename TConstPtr, typename TPtr>
rocblas_status rocblas_herk_syrk_check_numerics(const char*       function_name,
                                                rocblas_handle    handle,
                                                rocblas_fill      uplo,
                                                rocblas_operation trans,
                                                rocblas_int       n,
                                                rocblas_int       k,
                                                TConstPtr         A,
                                                rocblas_int       lda,
                                                rocblas_stride    stride_a,
                                                TPtr              C,
                                                rocblas_int       ldc,
                                                rocblas_stride    stride_c,
                                                rocblas_int       batch_count,
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
                                                              n,
                                                              k,
                                                              A,
                                                              0,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
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
        n,
        n,
        C,
        0,
        ldc,
        stride_c,
        batch_count,
        check_numerics,
        is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syrk*.cpp or herk*.cpp

// clang-format off
#undef INSTANTIATE_SYRK_HERK_KERNEL

#ifdef INSTANTIATE_SYRK_TEMPLATE
#error INSTANTIATE_SYRK_TEMPLATE already defined
#endif

#define INSTANTIATE_SYRK_TEMPLATE(NB_, BATCHED_, T_, TScal_, TConstPtr_, TPtr_)                             \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                 \
rocblas_internal_syrk_template<NB_, BATCHED_, T_, TScal_, TConstPtr_, TPtr_>(rocblas_handle    handle,      \
                                                          rocblas_fill      uplo,        \
                                                          rocblas_operation trans_a,      \
                                                          rocblas_int       n,           \
                                                          rocblas_int       k,           \
                                                          TScal_             alpha,      \
                                                          TConstPtr_         AP,         \
                                                          rocblas_stride    offset_a,    \
                                                          rocblas_int       lda,         \
                                                          rocblas_stride    stride_a,    \
                                                          TScal_             beta,       \
                                                          TPtr_              CP,         \
                                                          rocblas_stride    offset_c,    \
                                                          rocblas_int       ldc,         \
                                                          rocblas_stride    stride_c,    \
                                                          rocblas_int       batch_count);

INSTANTIATE_SYRK_TEMPLATE( 16, false, float, float const*, float const*,  float*)
INSTANTIATE_SYRK_TEMPLATE(32, false, double, double const*, double const*, double*)
INSTANTIATE_SYRK_TEMPLATE( 32, false, rocblas_float_complex, rocblas_float_complex const*,  rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_SYRK_TEMPLATE(32, false, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_SYRK_TEMPLATE( 16, true, float, float const*,  float const* const*,  float* const*)
INSTANTIATE_SYRK_TEMPLATE(16, true, double, double const*, double const* const*, double* const*)
INSTANTIATE_SYRK_TEMPLATE( 8, true, rocblas_float_complex, rocblas_float_complex const*,  rocblas_float_complex const* const*,  rocblas_float_complex* const*)
INSTANTIATE_SYRK_TEMPLATE(8, true, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_SYRK_TEMPLATE

#ifdef INSTANTIATE_HERK_TEMPLATE
#error INSTANTIATE_HERK_TEMPLATE already defined
#endif

#define INSTANTIATE_HERK_TEMPLATE(NB_, BATCHED_, T_, Tscal_, TConstPtr_, TPtr_)                             \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                 \
rocblas_internal_herk_template<NB_, BATCHED_, T_, Tscal_, TConstPtr_, TPtr_>(rocblas_handle    handle,      \
                                                          rocblas_fill      uplo,        \
                                                          rocblas_operation trans_a,      \
                                                          rocblas_int       n,           \
                                                          rocblas_int       k,           \
                                                          Tscal_            alpha,       \
                                                          TConstPtr_        AP,          \
                                                          rocblas_stride    offset_a,    \
                                                          rocblas_int       lda,         \
                                                          rocblas_stride    stride_a,    \
                                                          Tscal_            beta,        \
                                                          TPtr_             CP,          \
                                                          rocblas_stride    offset_c,    \
                                                          rocblas_int       ldc,         \
                                                          rocblas_stride    stride_c,    \
                                                          rocblas_int       batch_count);

INSTANTIATE_HERK_TEMPLATE(32, false, rocblas_float_complex, float const*, rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_HERK_TEMPLATE(8, true, rocblas_float_complex, float const*, rocblas_float_complex const* const*,  rocblas_float_complex* const*)
INSTANTIATE_HERK_TEMPLATE(32, false, rocblas_double_complex, double const*, rocblas_double_complex const*,  rocblas_double_complex*)
INSTANTIATE_HERK_TEMPLATE(8, true, rocblas_double_complex, double const*, rocblas_double_complex const* const*,  rocblas_double_complex* const*)

#undef INSTANTIATE_HERK_TEMPLATE

#ifdef INSTANTIATE_HERK_SYRK_NUMERICS
#error INSTANTIATE_HERK_SYRK_NUMERICS already defined
#endif

#define INSTANTIATE_HERK_SYRK_NUMERICS(HERM_, TConstPtr_, TPtr_)                        \
template rocblas_status rocblas_herk_syrk_check_numerics                                \
                                  <HERM_, TConstPtr_, TPtr_>                            \
                                  (const char*       function_name,                     \
                                   rocblas_handle handle,                               \
                                   rocblas_fill   uplo,                                 \
                                   rocblas_operation trans,                             \
                                   rocblas_int    n,                                    \
                                   rocblas_int    k,                                    \
                                   TConstPtr_     A,                                    \
                                   rocblas_int    lda,                                  \
                                   rocblas_stride stride_a,                             \
                                   TPtr_          C,                                    \
                                   rocblas_int    ldc,                                  \
                                   rocblas_stride stride_c,                             \
                                   rocblas_int    batch_count,                          \
                                   const int      check_numerics,                       \
                                   bool           is_input);

// instantiate for rocblas_Xherk_Xsyrk and rocblas_Xherk_Xsyrk_strided_batched
INSTANTIATE_HERK_SYRK_NUMERICS(false, float const*, float*)
INSTANTIATE_HERK_SYRK_NUMERICS(false, double const*, double*)
INSTANTIATE_HERK_SYRK_NUMERICS(false,  rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HERK_SYRK_NUMERICS( true,  rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HERK_SYRK_NUMERICS(false, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_HERK_SYRK_NUMERICS( true, rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xherk_Xsyrk_batched
INSTANTIATE_HERK_SYRK_NUMERICS(false, float const* const*, float* const*)
INSTANTIATE_HERK_SYRK_NUMERICS(false, double const* const*, double* const*)
INSTANTIATE_HERK_SYRK_NUMERICS(false,  rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HERK_SYRK_NUMERICS( true,  rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HERK_SYRK_NUMERICS(false, rocblas_double_complex const* const*, rocblas_double_complex* const*)
INSTANTIATE_HERK_SYRK_NUMERICS( true, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_HERK_SYRK_NUMERICS
// clang-format on
