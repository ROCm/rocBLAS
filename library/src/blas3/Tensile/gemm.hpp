/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _GEMM_HOST_HPP_
#define _GEMM_HOST_HPP_

#include "handle.h"

#ifdef USE_TENSILE_HOST

#include "tensile_host.hpp"

#else // USE_TENSILE_HOST

/*******************************************************************************
 * Helper enumeration over different transpose combinations
 ******************************************************************************/
typedef enum
{
    // First letter refers to A, second letter refers to B
    NN,
    NT,
    TN,
    TT,
    NC,
    CN,
    TC,
    CT,
    CC,
} transpose_mode;

constexpr transpose_mode GetTransposeMode(rocblas_operation trans_a, rocblas_operation trans_b)
{
    if(trans_a == rocblas_operation_none)
    {
        if(trans_b == rocblas_operation_none)
            return NN;
        if(trans_b == rocblas_operation_conjugate_transpose)
            return NC;
        return NT;
    }
    else if(trans_a == rocblas_operation_conjugate_transpose)
    {
        if(trans_b == rocblas_operation_none)
            return CN;
        if(trans_b == rocblas_operation_conjugate_transpose)
            return CC;
        return CT;
    }
    else
    {
        if(trans_b == rocblas_operation_none)
            return TN;
        if(trans_b == rocblas_operation_conjugate_transpose)
            return TC;
        return TT;
    }
}

#include "Tensile.h"

/*******************************************************************************
 * Tensile Helper Function call
 ******************************************************************************/
template <typename T>
rocblas_status tensile_helper(const T&          alpha_h,
                              const T&          beta_h,
                              const T*          A,
                              const T*          B,
                              T*                C,
                              rocblas_operation trans_a,
                              rocblas_operation trans_b,
                              rocblas_stride    strideC1,
                              rocblas_stride    strideC2,
                              rocblas_stride    strideA1,
                              rocblas_stride    strideA2,
                              rocblas_stride    strideB1,
                              rocblas_stride    strideB2,
                              rocblas_int       sizeI,
                              rocblas_int       sizeJ,
                              rocblas_int       sizeK,
                              rocblas_int       sizeL,
                              rocblas_handle    handle);

#define TENSILE_ARGS(T)                                                                        \
    (T*)C, (const T*)C, (const T*)A, (const T*)B, *((const T*)&alpha_h), *((const T*)&beta_h), \
        strideC1, strideC2, strideC1, strideC2, strideA1, strideA2, strideB1, strideB2, sizeI, \
        sizeJ, sizeK, sizeL, handle->rocblas_stream, 0, nullptr, nullptr

template <>
inline rocblas_status tensile_helper(const rocblas_half& alpha_h,
                                     const rocblas_half& beta_h,
                                     const rocblas_half* A,
                                     const rocblas_half* B,
                                     rocblas_half*       C,
                                     rocblas_operation   trans_a,
                                     rocblas_operation   trans_b,
                                     rocblas_stride      strideC1,
                                     rocblas_stride      strideC2,
                                     rocblas_stride      strideA1,
                                     rocblas_stride      strideA2,
                                     rocblas_stride      strideB1,
                                     rocblas_stride      strideB2,
                                     rocblas_int         sizeI,
                                     rocblas_int         sizeJ,
                                     rocblas_int         sizeK,
                                     rocblas_int         sizeL,
                                     rocblas_handle      handle)
{
    hipError_t status = hipErrorInvalidValue;

    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        status = tensile_Cijk_Ailk_Bljk_HB(TENSILE_ARGS(rocblas_half));
        break;
    case NT:
    case NC:
        status = tensile_Cijk_Ailk_Bjlk_HB(TENSILE_ARGS(rocblas_half));
        break;
    case TN:
    case CN:
        status = tensile_Cijk_Alik_Bljk_HB(TENSILE_ARGS(rocblas_half));
        break;
    case TT:
    case TC:
    case CT:
    case CC:
        status = tensile_Cijk_Alik_Bjlk_HB(TENSILE_ARGS(rocblas_half));
        break;
    }

    return get_rocblas_status_for_hip_status(status);
}

template <>
inline rocblas_status tensile_helper(const float&      alpha_h,
                                     const float&      beta_h,
                                     const float*      A,
                                     const float*      B,
                                     float*            C,
                                     rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     rocblas_stride    strideC1,
                                     rocblas_stride    strideC2,
                                     rocblas_stride    strideA1,
                                     rocblas_stride    strideA2,
                                     rocblas_stride    strideB1,
                                     rocblas_stride    strideB2,
                                     rocblas_int       sizeI,
                                     rocblas_int       sizeJ,
                                     rocblas_int       sizeK,
                                     rocblas_int       sizeL,
                                     rocblas_handle    handle)
{
    hipError_t status = hipErrorInvalidValue;

    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        status = tensile_Cijk_Ailk_Bljk_SB(TENSILE_ARGS(float));
        break;
    case NT:
    case NC:
        status = tensile_Cijk_Ailk_Bjlk_SB(TENSILE_ARGS(float));
        break;
    case TN:
    case CN:
        status = tensile_Cijk_Alik_Bljk_SB(TENSILE_ARGS(float));
        break;
    case TT:
    case TC:
    case CT:
    case CC:
        status = tensile_Cijk_Alik_Bjlk_SB(TENSILE_ARGS(float));
        break;
    }

    return get_rocblas_status_for_hip_status(status);
}

template <>
inline rocblas_status tensile_helper(const double&     alpha_h,
                                     const double&     beta_h,
                                     const double*     A,
                                     const double*     B,
                                     double*           C,
                                     rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     rocblas_stride    strideC1,
                                     rocblas_stride    strideC2,
                                     rocblas_stride    strideA1,
                                     rocblas_stride    strideA2,
                                     rocblas_stride    strideB1,
                                     rocblas_stride    strideB2,
                                     rocblas_int       sizeI,
                                     rocblas_int       sizeJ,
                                     rocblas_int       sizeK,
                                     rocblas_int       sizeL,
                                     rocblas_handle    handle)
{
    hipError_t status = hipErrorInvalidValue;

    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        status = tensile_Cijk_Ailk_Bljk_DB(TENSILE_ARGS(double));
        break;
    case NT:
    case NC:
        status = tensile_Cijk_Ailk_Bjlk_DB(TENSILE_ARGS(double));
        break;
    case TN:
    case CN:
        status = tensile_Cijk_Alik_Bljk_DB(TENSILE_ARGS(double));
        break;
    case TT:
    case TC:
    case CT:
    case CC:
        status = tensile_Cijk_Alik_Bjlk_DB(TENSILE_ARGS(double));
        break;
    }

    return get_rocblas_status_for_hip_status(status);
}

template <>
inline rocblas_status tensile_helper(const rocblas_float_complex& alpha_h,
                                     const rocblas_float_complex& beta_h,
                                     const rocblas_float_complex* A,
                                     const rocblas_float_complex* B,
                                     rocblas_float_complex*       C,
                                     rocblas_operation            trans_a,
                                     rocblas_operation            trans_b,
                                     rocblas_stride               strideC1,
                                     rocblas_stride               strideC2,
                                     rocblas_stride               strideA1,
                                     rocblas_stride               strideA2,
                                     rocblas_stride               strideB1,
                                     rocblas_stride               strideB2,
                                     rocblas_int                  sizeI,
                                     rocblas_int                  sizeJ,
                                     rocblas_int                  sizeK,
                                     rocblas_int                  sizeL,
                                     rocblas_handle               handle)
{
    static_assert(std::is_standard_layout<TensileComplexFloat>{},
                  "TensileComplexFloat is not a standard layout type, and thus is "
                  "incompatible with C.");

    static_assert(std::is_trivial<TensileComplexFloat>{},
                  "TensileComplexFloat is not a trivial type, and thus is "
                  "incompatible with C.");

    static_assert(sizeof(rocblas_float_complex) == sizeof(TensileComplexFloat),
                  "TensileComplexFloat does not match rocblas_float_complex");

    hipError_t status = hipErrorInvalidValue;

    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        status = tensile_Cijk_Ailk_Bljk_CB(TENSILE_ARGS(TensileComplexFloat));
        break;
    case NT:
        status = tensile_Cijk_Ailk_Bjlk_CB(TENSILE_ARGS(TensileComplexFloat));
        break;
    case TN:
        status = tensile_Cijk_Alik_Bljk_CB(TENSILE_ARGS(TensileComplexFloat));
        break;
    case TT:
        status = tensile_Cijk_Alik_Bjlk_CB(TENSILE_ARGS(TensileComplexFloat));
        break;
    case NC:
        status = tensile_Cijk_Ailk_BjlkC_CB(TENSILE_ARGS(TensileComplexFloat));
        break;
    case CN:
        status = tensile_Cijk_AlikC_Bljk_CB(TENSILE_ARGS(TensileComplexFloat));
        break;
    case TC:
        status = tensile_Cijk_Alik_BjlkC_CB(TENSILE_ARGS(TensileComplexFloat));
        break;
    case CT:
        status = tensile_Cijk_AlikC_Bjlk_CB(TENSILE_ARGS(TensileComplexFloat));
        break;
    case CC:
        status = tensile_Cijk_AlikC_BjlkC_CB(TENSILE_ARGS(TensileComplexFloat));
        break;
    }

    return get_rocblas_status_for_hip_status(status);
}

template <>
inline rocblas_status tensile_helper(const rocblas_double_complex& alpha_h,
                                     const rocblas_double_complex& beta_h,
                                     const rocblas_double_complex* A,
                                     const rocblas_double_complex* B,
                                     rocblas_double_complex*       C,
                                     rocblas_operation             trans_a,
                                     rocblas_operation             trans_b,
                                     rocblas_stride                strideC1,
                                     rocblas_stride                strideC2,
                                     rocblas_stride                strideA1,
                                     rocblas_stride                strideA2,
                                     rocblas_stride                strideB1,
                                     rocblas_stride                strideB2,
                                     rocblas_int                   sizeI,
                                     rocblas_int                   sizeJ,
                                     rocblas_int                   sizeK,
                                     rocblas_int                   sizeL,
                                     rocblas_handle                handle)
{
    static_assert(std::is_standard_layout<TensileComplexDouble>{},
                  "TensileComplexDouble is not a standard layout type, and thus is "
                  "incompatible with C.");

    static_assert(std::is_trivial<TensileComplexDouble>{},
                  "TensileComplexDouble is not a trivial type, and thus is "
                  "incompatible with C.");

    static_assert(sizeof(rocblas_double_complex) == sizeof(TensileComplexDouble),
                  "TensileComplexDouble does not match rocblas_double_complex");

    hipError_t status = hipErrorInvalidValue;

    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        status = tensile_Cijk_Ailk_Bljk_ZB(TENSILE_ARGS(TensileComplexDouble));
        break;
    case NT:
        status = tensile_Cijk_Ailk_Bjlk_ZB(TENSILE_ARGS(TensileComplexDouble));
        break;
    case TN:
        status = tensile_Cijk_Alik_Bljk_ZB(TENSILE_ARGS(TensileComplexDouble));
        break;
    case TT:
        status = tensile_Cijk_Alik_Bjlk_ZB(TENSILE_ARGS(TensileComplexDouble));
        break;
    case NC:
        status = tensile_Cijk_Ailk_BjlkC_ZB(TENSILE_ARGS(TensileComplexDouble));
        break;
    case CN:
        status = tensile_Cijk_AlikC_Bljk_ZB(TENSILE_ARGS(TensileComplexDouble));
        break;
    case TC:
        status = tensile_Cijk_Alik_BjlkC_ZB(TENSILE_ARGS(TensileComplexDouble));
        break;
    case CT:
        status = tensile_Cijk_AlikC_Bjlk_ZB(TENSILE_ARGS(TensileComplexDouble));
        break;
    case CC:
        status = tensile_Cijk_AlikC_BjlkC_ZB(TENSILE_ARGS(TensileComplexDouble));
        break;
    }

    return get_rocblas_status_for_hip_status(status);
}
#undef TENSILE_ARGS

#endif // USE_TENSILE_HOST

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

#ifdef USE_TENSILE_HOST

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

    return runContractionProblem(problem);

#else // USE_TENSILE_HOST

    return tensile_helper(*alpha,
                          *beta,
                          A,
                          B,
                          C,
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
                          batch_count,
                          k,
                          handle);

#endif // USE_TENSILE_HOST
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
                                   rocblas_int       ld_a,
                                   const void*       b,
                                   rocblas_int       ld_b,
                                   const T*          beta,
                                   const void*       c,
                                   rocblas_int       ld_c,
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
    if(num_rows_a > ld_a || num_rows_b > ld_b || num_rows_c > ld_c)
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

template <bool BATCHED, typename T, typename U, typename V>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_gemm_template(rocblas_handle    handle,
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
    RETURN_IF_ROCBLAS_ERROR(
        copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

    // When beta == 1 and either k == 0 or alpha == 0, the operation is a no-op
    if(*beta == 1 && (k == 0 || *alpha == 0))
        return rocblas_status_success;

    rocblas_status status = rocblas_status_success;

    // TODO: Use C++17 constexpr if
    if(BATCHED)
    {
        // We cannot do this with a device array, so array of pointers must be on host for now

        // Host arrays of device pointers.
        auto hostA = std::make_unique<T*[]>(batch_count);
        auto hostB = std::make_unique<T*[]>(batch_count);
        auto hostC = std::make_unique<T*[]>(batch_count);

        RETURN_IF_HIP_ERROR(
            hipMemcpy(&hostA[0], A, sizeof(T*) * batch_count, hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&hostB[0], B, sizeof(T*) * batch_count, hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&hostC[0], C, sizeof(T*) * batch_count, hipMemcpyDeviceToHost));

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
