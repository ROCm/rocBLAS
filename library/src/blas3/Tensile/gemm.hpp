/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstring> // std::memcpy for graph capture use cases

#ifdef BUILD_WITH_TENSILE
#include "gemm_tensile.hpp"
#else
#include "gemm_source.hpp"
#endif

#include "check_numerics_matrix.hpp"
#include "handle.hpp"

/*********************************************************************************
 * Right now Tensile requires alpha and beta to be passed by value on host.      *
 * If in device pointer mode, copy alpha and beta to host.                       *
 * If k == 0, we set alpha = 0 instead of copying from device.                   *
 * TODO: Make this asynchronous, putting synchronization closer to Tensile call. *
 *********************************************************************************/
template <typename Ta, typename Tac, typename Tb, typename Tbc>
rocblas_status copy_alpha_beta_to_host_if_on_device(rocblas_handle handle,
                                                    const Ta*&     alpha,
                                                    const Tb*&     beta,
                                                    Tac&           alpha_h,
                                                    Tbc&           beta_h,
                                                    rocblas_int    k)
{
    if(handle->is_stream_in_capture_mode() && handle->skip_alpha_beta_memcpy())
        return rocblas_status_success;

    handle->alpha_beta_memcpy_completed();

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        if(alpha)
        {
            if(k == 0)
                alpha_h = 0;
            else
                RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(Tac), hipMemcpyDeviceToHost));
            alpha = &alpha_h;
        }
        if(beta)
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(&beta_h, beta, sizeof(Tbc), hipMemcpyDeviceToHost));
            beta = &beta_h;
        }
    }
    else if(handle->is_stream_in_capture_mode()
            && handle->pointer_mode == rocblas_pointer_mode_host)
    {

        if(alpha)
        {
            auto alpha_mem = handle->host_malloc(sizeof(Tac));
            std::memcpy(alpha_mem, alpha, sizeof(Tac));
            alpha = (Ta*)alpha_mem;
        }
        if(beta)
        {
            auto beta_mem = handle->host_malloc(sizeof(Tbc));
            std::memcpy(beta_mem, beta, sizeof(Tbc));
            beta = (Tb*)beta_mem;
        }
    }
    return rocblas_status_success;
}

/*******************************************************************************
 * Validate Arguments
 ******************************************************************************/
template <typename T>
inline rocblas_status validateArgs(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const T*          alpha,
                                   const void*       a,
                                   rocblas_int       lda,
                                   const void*       b,
                                   rocblas_int       ldb,
                                   const T*          beta,
                                   const void*       c,
                                   rocblas_int       ldc,
                                   rocblas_int       batch_count = 1)
{
    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    if(trans_a != rocblas_operation_none && trans_a != rocblas_operation_transpose
       && trans_a != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if(trans_b != rocblas_operation_none && trans_b != rocblas_operation_transpose
       && trans_b != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    rocblas_int num_rows_a = trans_a == rocblas_operation_none ? m : k;
    rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;
    rocblas_int num_rows_c = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc)
        return rocblas_status_invalid_size;

    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*beta == 1)
        {
            if(!k)
                return rocblas_status_success;

            if(!alpha)
                return rocblas_status_invalid_pointer;

            if(!*alpha)
                return rocblas_status_success;
        }
        // all early return success now handled so
        // pointers must be valid
        bool ab_calc_invalid = !alpha || (*alpha != 0 && (!a || !b));
        if(!c || (k && ab_calc_invalid))
            return rocblas_status_invalid_pointer;
    }
    else
    {
        return rocblas_status_internal_error; // always pushed host_mode prevalidation
    }

    return rocblas_status_continue;
}

/*
 * ===========================================================================
 *    template interface
 * ===========================================================================
 */
template <bool BATCHED, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemm_template(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const TScal*      alpha,
                                   const TConstPtr*  A,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   const TConstPtr*  B,
                                   rocblas_stride    offset_b,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   const TScal*      beta,
                                   TPtr*             C,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count)
{
    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    TScal alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

#ifdef BUILD_WITH_TENSILE
    if(BATCHED)
    {
        return call_tensile(handle,
                            alpha,
                            beta,
                            A,
                            B,
                            C,
                            trans_a,
                            trans_b,
                            ldc,
                            stride_c,
                            offset_c,
                            lda,
                            stride_a,
                            offset_a,
                            ldb,
                            stride_b,
                            offset_b,
                            m,
                            n,
                            k,
                            batch_count);
    }
    else
    {
        return call_tensile(handle,
                            alpha,
                            beta,
                            A + offset_a,
                            B + offset_b,
                            C + offset_c,
                            trans_a,
                            trans_b,
                            ldc,
                            stride_c,
                            0,
                            lda,
                            stride_a,
                            0,
                            ldb,
                            stride_b,
                            0,
                            m,
                            n,
                            k,
                            batch_count);
    }
#else // BUILD_WITH_TENSILE
    hipStream_t rocblas_stream = handle->get_stream();

    if(k == 0 || (alpha && *alpha == 0))
    {
        return rocblas_gemm_scale_template(
            m, n, *beta, C, offset_c, ldc, stride_c, batch_count, rocblas_stream);
    }

    gemm_source_solution<BATCHED>(trans_a,
                                  trans_b,
                                  m,
                                  n,
                                  k,
                                  *alpha,
                                  A,
                                  lda,
                                  stride_a,
                                  offset_a,
                                  B,
                                  ldb,
                                  stride_b,
                                  offset_b,
                                  *beta,
                                  C,
                                  ldc,
                                  stride_c,
                                  offset_c,
                                  batch_count,
                                  rocblas_stream);
    return rocblas_status_success;
#endif // BUILD_WITH_TENSILE
}

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_gemm_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           TConstPtr         A,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_a,
                                           TConstPtr         B,
                                           rocblas_int       ldb,
                                           rocblas_stride    stride_b,
                                           TPtr              C,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_c,
                                           rocblas_int       batch_count,
                                           const int         check_numerics,
                                           bool              is_input)
{

    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          trans_a,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
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

    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          trans_b,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          k,
                                                          n,
                                                          B,
                                                          0,
                                                          ldb,
                                                          stride_b,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
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
