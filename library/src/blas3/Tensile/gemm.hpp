/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "check_numerics_matrix.hpp"
#include "handle.hpp"

/*********************************************************************************
 * Right now Tensile requires alpha and beta to be passed by value on host.      *
 * If in device pointer mode, copy alpha and beta to host.                       *
 * If k == 0, we set alpha = 0 instead of copying from device.                   *
 * TODO: Make this asynchronous, putting synchronization closer to Tensile call. *
 *********************************************************************************/
template <typename T, typename Tc>
rocblas_status copy_alpha_beta_to_host_if_on_device(
    rocblas_handle handle, const T*& alpha, const T*& beta, Tc& alpha_h, Tc& beta_h, rocblas_int k)
{
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        if(alpha)
        {
            if(k == 0)
                alpha_h = 0;
            else
                RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(Tc), hipMemcpyDeviceToHost));
            alpha = &alpha_h;
        }
        if(beta)
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(&beta_h, beta, sizeof(Tc), hipMemcpyDeviceToHost));
            beta = &beta_h;
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

    if(handle->pointer_mode == rocblas_pointer_mode_host && *beta == 1)
    {
        if(!k)
            return rocblas_status_success;

        if(!alpha)
            return rocblas_status_invalid_pointer;

        if(!*alpha)
            return rocblas_status_success;
    }

    // pointers must be valid
    if((k && (!a || !b || !alpha)) || !c)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/*
 * ===========================================================================
 *    template interface
 * ===========================================================================
 */
template <bool BATCHED, typename TScal, typename TConstPtr, typename TPtr, typename TLd>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemm_template(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const TScal*      alpha,
                                   const TConstPtr*  A,
                                   TLd               offset_a,
                                   TLd               lda,
                                   rocblas_stride    stride_a,
                                   const TConstPtr*  B,
                                   TLd               offset_b,
                                   TLd               ldb,
                                   rocblas_stride    stride_b,
                                   const TScal*      beta,
                                   TPtr*             C,
                                   TLd               offset_c,
                                   TLd               ldc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count)
{
    // Early exit. Note: k==0 is not an early exit, since C still needs to be multiplied by beta.
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    TScal alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

    // When beta == 1 and either k == 0 or alpha == 0, the operation is a no-op
    if(*beta == 1 && (k == 0 || *alpha == 0))
        return rocblas_status_success;

#ifdef BUILD_WITH_TENSILE
    return call_tensile(handle,
                        alpha,
                        beta,
                        A,
                        B,
                        C,
                        trans_a,
                        trans_b,
                        rocblas_int(ldc),
                        stride_c,
                        rocblas_int(offset_c),
                        rocblas_int(lda),
                        stride_a,
                        rocblas_int(offset_a),
                        rocblas_int(ldb),
                        stride_b,
                        rocblas_int(offset_b),
                        m,
                        n,
                        k,
                        batch_count);
#else // BUILD_WITH_TENSILE
    hipStream_t rocblas_stream = handle->get_stream();
    gemm_source_solution(trans_a,
                         trans_b,
                         m,
                         n,
                         k,
                         *alpha,
                         A,
                         rocblas_int(lda),
                         stride_a,
                         rocblas_int(offset_a),
                         B,
                         rocblas_int(ldb),
                         stride_b,
                         rocblas_int(offset_b),
                         *beta,
                         C,
                         rocblas_int(ldc),
                         stride_c,
                         rocblas_int(offset_c),
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
        = rocblas_internal_check_numerics_ge_matrix_template(function_name,
                                                             handle,
                                                             trans_a,
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

    check_numerics_status = rocblas_internal_check_numerics_ge_matrix_template(function_name,
                                                                               handle,
                                                                               trans_b,
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
        = rocblas_internal_check_numerics_ge_matrix_template(function_name,
                                                             handle,
                                                             rocblas_operation_none,
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
