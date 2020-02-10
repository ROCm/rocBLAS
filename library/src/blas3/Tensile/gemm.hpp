/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _GEMM_HOST_HPP_
#define _GEMM_HOST_HPP_

#include "handle.h"
#include "tensile_host.hpp"

/*******************************************************************************
 * Tensile Function call
 ******************************************************************************/
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
                                   rocblas_int       ld_a,
                                   rocblas_stride    stride_a,
                                   rocblas_int       ld_b,
                                   rocblas_stride    stride_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   rocblas_int       batch_count = 1)

{
    RocblasContractionProblem<T> problem{handle,
                                         trans_a,
                                         trans_b,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         A,
                                         ld_a,
                                         stride_a,
                                         B,
                                         ld_b,
                                         stride_b,
                                         beta,
                                         C,
                                         ld_c,
                                         stride_c,
                                         batch_count};

    return handle->host->runContractionProblem(problem);
}

/*******************************************************************************
 * Validate Arguments
 ******************************************************************************/
inline rocblas_status validateArgs(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const void*       alpha,
                                   const void*       a,
                                   rocblas_int       ld_a,
                                   const void*       b,
                                   rocblas_int       ld_b,
                                   const void*       beta,
                                   const void*       c,
                                   rocblas_int       ld_c,
                                   rocblas_int       batch_count = 1)
{
    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    // pointers must be valid
    if(!c || !alpha || !beta || ((!a || !b) && k != 0))
        return rocblas_status_invalid_pointer;

    rocblas_int num_rows_a = trans_a == rocblas_operation_none ? m : k;
    rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;
    rocblas_int num_rows_c = m;

    // leading dimensions must be valid
    if(num_rows_a > ld_a || num_rows_b > ld_b || num_rows_c > ld_c)
        return rocblas_status_invalid_size;

    return rocblas_status_success;
}

/*
 * ===========================================================================
 *    template interface
 * ===========================================================================
 */

template <bool BATCHED, bool STRIDED, typename T, typename U, typename V>
rocblas_status rocblas_gemm_template(rocblas_handle    handle,
                                     rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       k,
                                     const T*          alpha,
                                     const U*          A,
                                     rocblas_int       offset_a,
                                     rocblas_int       ld_a,
                                     rocblas_stride    stride_a,
                                     const U*          B,
                                     rocblas_int       offset_b,
                                     rocblas_int       ld_b,
                                     rocblas_stride    stride_b,
                                     const T*          beta,
                                     V*                C,
                                     rocblas_int       offset_c,
                                     rocblas_int       ld_c,
                                     rocblas_stride    stride_c,
                                     rocblas_int       batch_count)
{
    // Early exit. Note: k==0 is not an early exit, since C still needs to be multiplied by beta.
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    T alpha_h, beta_h;

    // Right now Tensile requires alpha and beta to be passed by value on host.
    // If in device pointer mode, copy alpha and beta to host.
    // TODO: Make this asynchronous, putting synchronization in closer to Tensile call.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(hipMemcpy(&beta_h, beta, sizeof(T), hipMemcpyDeviceToHost));
        alpha = &alpha_h;
        beta  = &beta_h;
    }

    // When beta == 1 and either k == 0 or alpha == 0, the operation is a no-op
    if(*beta == 1 && (k == 0 || *alpha == 0))
        return rocblas_status_success;

    rocblas_status status = rocblas_status_success;
    if(BATCHED)
    {
        // We cannot do this with a device array, so array of pointers must be on host for now

        // Host arrays of device pointers.
        T* hostA[batch_count];
        T* hostB[batch_count];
        T* hostC[batch_count];

        RETURN_IF_HIP_ERROR(hipMemcpy(hostA, A, sizeof(hostA), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(hipMemcpy(hostB, B, sizeof(hostB), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(hipMemcpy(hostC, C, sizeof(hostC), hipMemcpyDeviceToHost));

        for(rocblas_int b = 0; b < batch_count; b++)
        {
            status = call_tensile(handle,
                                  alpha,
                                  beta,
                                  hostA[b] + offset_a,
                                  hostB[b] + offset_b,
                                  hostC[b] + offset_c,
                                  trans_a,
                                  trans_b,
                                  ld_c,
                                  stride_c,
                                  ld_a,
                                  stride_a,
                                  ld_b,
                                  stride_b,
                                  m,
                                  n,
                                  k);

            if(status != rocblas_status_success)
                break;
        }
    }
    else
    {
        // If STRIDED == false, compute the strides from the sizes of the arrays
        // so that they are interpreted as consecutive matrices in memory
        if(!STRIDED)
        {
            stride_a = ld_a * (trans_a == rocblas_operation_none ? k : m);
            stride_b = ld_b * (trans_b == rocblas_operation_none ? n : k);
            stride_c = ld_c * n;
        }

        // The (T*) casts are to prevent template deduction errors when BATCHED==true and the A, B, C
        // pointers are pointers to arrays of pointers. constexpr if(BATCHED) above could avoid this.
        status = call_tensile(handle,
                              alpha,
                              beta,
                              (T*)A + offset_a,
                              (T*)B + offset_b,
                              (T*)C + offset_c,
                              trans_a,
                              trans_b,
                              ld_c,
                              stride_c,
                              ld_a,
                              stride_a,
                              ld_b,
                              stride_b,
                              m,
                              n,
                              k,
                              batch_count);
    }

    return status;
}

#endif // _GEMM_HOST_HPP_
