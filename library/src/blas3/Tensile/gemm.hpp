/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _GEMM_HOST_HPP_
#define _GEMM_HOST_HPP_
#include "Tensile.h"
#include "handle.h"
#include "rocblas-types.h"
#include "rocblas.h"
#include "utility.h"
#include <sys/time.h>

/*******************************************************************************
 * Helper enumeration over different transpose combinations
 ******************************************************************************/
typedef enum transpose_mode_
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

/*******************************************************************************
 * Tensile Helper Funcation call
 ******************************************************************************/
template <typename T>
hipError_t tensile_helper(T&                alpha_h,
                          T&                beta_h,
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

#define TENSILE_ARGS(T)                                                                            \
    (T*)C, (const T*)C, (const T*)A, (const T*)B, *((T*)&alpha_h), *((T*)&beta_h), strideC1,       \
        strideC2, strideC1, strideC2, strideA1, strideA2, strideB1, strideB2, sizeI, sizeJ, sizeK, \
        sizeL, handle->rocblas_stream, 0, nullptr, nullptr

template <>
hipError_t tensile_helper(rocblas_half&       alpha_h,
                          rocblas_half&       beta_h,
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
        status = tensile_Cijk_Ailk_Bljk_HB(TENSILE_ARGS(_Float16));
        break;
    case NT:
    case NC:
        status = tensile_Cijk_Ailk_Bjlk_HB(TENSILE_ARGS(_Float16));
        break;
    case TN:
    case CN:
        status = tensile_Cijk_Alik_Bljk_HB(TENSILE_ARGS(_Float16));
        break;
    case TT:
    case TC:
    case CT:
    case CC:
        status = tensile_Cijk_Alik_Bjlk_HB(TENSILE_ARGS(_Float16));
        break;
    }

    return status;
}

template <>
hipError_t tensile_helper(float&            alpha_h,
                          float&            beta_h,
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

    return status;
}

template <>
hipError_t tensile_helper(double&           alpha_h,
                          double&           beta_h,
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

    return status;
}

template <>
hipError_t tensile_helper(rocblas_float_complex&       alpha_h,
                          rocblas_float_complex&       beta_h,
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

    return status;
}

template <>
hipError_t tensile_helper(rocblas_double_complex&       alpha_h,
                          rocblas_double_complex&       beta_h,
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

    return status;
}
#undef TENSILE_ARGS

/*******************************************************************************
 * Tensile Function call
 ******************************************************************************/
template <typename T>
hipError_t call_tensile(const T*          alpha,
                        rocblas_stride    stride_alpha,
                        const T*          beta,
                        rocblas_stride    stride_beta,
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
                        rocblas_handle    handle)
{

    // Collect alpha / beta (either from host or device).
    // Tensile doesn't support arrays of scalars for now, so we must handle
    // this case before we enter call_tensile and only pass a single scalar
    T alpha_h;
    T beta_h;
    if(rocblas_pointer_mode_host == handle->pointer_mode)
    {
        alpha_h = *alpha;
        beta_h  = *beta;
    }
    else
    {
        hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost);
        hipMemcpy(&beta_h, beta, sizeof(T), hipMemcpyDeviceToHost);
    }

    hipError_t status = tensile_helper(alpha_h,
                                       beta_h,
                                       A,
                                       B,
                                       C,
                                       trans_a,
                                       trans_b,
                                       strideC1,
                                       strideC2,
                                       strideA1,
                                       strideA2,
                                       strideB1,
                                       strideB2,
                                       sizeI,
                                       sizeJ,
                                       sizeK,
                                       sizeL,
                                       handle);

#ifndef NDEBUG
    std::cout << "Return Status: " << status << std::endl;
#endif

    return status;
}

template <typename T>
hipError_t call_tensile(const T*          alpha,
                        rocblas_stride    stride_alpha,
                        const T*          beta,
                        rocblas_stride    stride_beta,
                        const T* const    A[],
                        const T* const    B[],
                        T* const          C[],
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
    return hipErrorUnknown;
}

/*******************************************************************************
 * Infer Batch Strides
 ******************************************************************************/
inline void infer_batch_strides(rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                rocblas_int       m,
                                rocblas_int       n,
                                rocblas_int       k,
                                rocblas_int       ld_a,
                                rocblas_stride*   stride_a,
                                rocblas_int       ld_b,
                                rocblas_stride*   stride_b,
                                rocblas_int       ld_c,
                                rocblas_stride*   stride_c)
{

    rocblas_int num_cols_c = n;
    rocblas_int num_rows_c = m;
    rocblas_int num_cols_a = (trans_a == rocblas_operation_none ? k : m);
    rocblas_int num_rows_a = (trans_a == rocblas_operation_none ? m : k);
    rocblas_int num_cols_b = (trans_b == rocblas_operation_none ? n : k);
    rocblas_int num_rows_b = (trans_b == rocblas_operation_none ? k : n);

    *stride_a = ld_a * num_cols_a;
    *stride_b = ld_b * num_cols_b;
    *stride_c = ld_c * num_cols_c;

} // infer batched strides

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
                                   rocblas_stride    stride_a,
                                   const void*       b,
                                   rocblas_int       ld_b,
                                   rocblas_stride    stride_b,
                                   const void*       beta,
                                   void*             c,
                                   rocblas_int       ld_c,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count)
{
    // quick return 0 is valid in BLAS
    if(!m || !n || !k || !batch_count)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    // pointers must be valid
    if(!c || !a || !b || !alpha || !beta)
        return rocblas_status_invalid_pointer;

    rocblas_int num_cols_c = n;
    rocblas_int num_rows_c = m;
    rocblas_int num_cols_a = trans_a == rocblas_operation_none ? k : m;
    rocblas_int num_rows_a = trans_a == rocblas_operation_none ? m : k;
    rocblas_int num_cols_b = trans_b == rocblas_operation_none ? n : k;
    rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;

    // leading dimensions must be valid
    if(num_rows_a > ld_a || num_rows_b > ld_b || num_rows_c > ld_c)
        return rocblas_status_invalid_size;

    return rocblas_status_success;
} // validate parameters

template <typename T>
inline rocblas_status validateArgs(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const T*          alpha,
                                   const T* const    a[],
                                   rocblas_int       ld_a,
                                   const T* const    b[],
                                   rocblas_int       ld_b,
                                   const T*          beta,
                                   T* const          c[],
                                   rocblas_int       ld_c,
                                   rocblas_int       batch_count)
{
    // quick return 0 is valid in BLAS
    if(!m || !n || !k || !batch_count)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    // pointers must be valid
    if(!c || !a || !b || !alpha || !beta)
        return rocblas_status_invalid_pointer;

    // for(int i = 0; i < batch_count; i++)
    //     if(!a[i] || !b[i] || !c[i])
    //         return rocblas_status_invalid_pointer;

    rocblas_int num_cols_c = n;
    rocblas_int num_rows_c = m;
    rocblas_int num_cols_a = trans_a == rocblas_operation_none ? k : m;
    rocblas_int num_rows_a = trans_a == rocblas_operation_none ? m : k;
    rocblas_int num_cols_b = trans_b == rocblas_operation_none ? n : k;
    rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;

    // leading dimensions must be valid
    if(num_rows_a > ld_a || num_rows_b > ld_b || num_rows_c > ld_c)
        return rocblas_status_invalid_size;

    return rocblas_status_success;
} // validate parameters

/*******************************************************************************
 * Tensile Solution Name (debug only)
 ******************************************************************************/
template <typename T>
const char* tensileGetSolutionName(rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       strideC1,
                                   rocblas_int       strideC2,
                                   rocblas_int       strideA1,
                                   rocblas_int       strideA2,
                                   rocblas_int       strideB1,
                                   rocblas_int       strideB2,
                                   rocblas_int       sizeI,
                                   rocblas_int       sizeJ,
                                   rocblas_int       sizeK,
                                   rocblas_int       sizeL)
{
    return "";
};

// This macro condenses all the identical arguments to the various
// tensileGetSolutionName function calls for consistency / brevity
#define TENSILE_ARG_NAMES                                                                         \
    strideC1, strideC2, strideC1, strideC2, strideA1, strideA2, strideB1, strideB2, sizeI, sizeJ, \
        sizeK, sizeL

template <>
const char* tensileGetSolutionName<rocblas_half>(rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int       strideC1,
                                                 rocblas_int       strideC2,
                                                 rocblas_int       strideA1,
                                                 rocblas_int       strideA2,
                                                 rocblas_int       strideB1,
                                                 rocblas_int       strideB2,
                                                 rocblas_int       sizeI,
                                                 rocblas_int       sizeJ,
                                                 rocblas_int       sizeK,
                                                 rocblas_int       sizeL)
{
    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        return tensileGetSolutionName_Cijk_Ailk_Bljk_HB(TENSILE_ARG_NAMES);
    case NT:
    case NC:
        return tensileGetSolutionName_Cijk_Ailk_Bjlk_HB(TENSILE_ARG_NAMES);
    case TN:
    case CN:
        return tensileGetSolutionName_Cijk_Alik_Bljk_HB(TENSILE_ARG_NAMES);
    case TT:
    case TC:
    case CT:
    case CC:
        return tensileGetSolutionName_Cijk_Alik_Bjlk_HB(TENSILE_ARG_NAMES);
    }
}

template <>
const char* tensileGetSolutionName<float>(rocblas_operation trans_a,
                                          rocblas_operation trans_b,
                                          rocblas_int       strideC1,
                                          rocblas_int       strideC2,
                                          rocblas_int       strideA1,
                                          rocblas_int       strideA2,
                                          rocblas_int       strideB1,
                                          rocblas_int       strideB2,
                                          rocblas_int       sizeI,
                                          rocblas_int       sizeJ,
                                          rocblas_int       sizeK,
                                          rocblas_int       sizeL)
{
    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        return tensileGetSolutionName_Cijk_Ailk_Bljk_SB(TENSILE_ARG_NAMES);
    case NT:
    case NC:
        return tensileGetSolutionName_Cijk_Ailk_Bjlk_SB(TENSILE_ARG_NAMES);
    case TN:
    case CN:
        return tensileGetSolutionName_Cijk_Alik_Bljk_SB(TENSILE_ARG_NAMES);
    case TT:
    case TC:
    case CT:
    case CC:
        return tensileGetSolutionName_Cijk_Alik_Bjlk_SB(TENSILE_ARG_NAMES);
    }
}

template <>
const char* tensileGetSolutionName<double>(rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           rocblas_int       strideC1,
                                           rocblas_int       strideC2,
                                           rocblas_int       strideA1,
                                           rocblas_int       strideA2,
                                           rocblas_int       strideB1,
                                           rocblas_int       strideB2,
                                           rocblas_int       sizeI,
                                           rocblas_int       sizeJ,
                                           rocblas_int       sizeK,
                                           rocblas_int       sizeL)
{
    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        return tensileGetSolutionName_Cijk_Ailk_Bljk_DB(TENSILE_ARG_NAMES);
    case NT:
    case NC:
        return tensileGetSolutionName_Cijk_Ailk_Bjlk_DB(TENSILE_ARG_NAMES);
    case TN:
    case CN:
        return tensileGetSolutionName_Cijk_Alik_Bljk_DB(TENSILE_ARG_NAMES);
    case TT:
    case TC:
    case CT:
    case CC:
        return tensileGetSolutionName_Cijk_Alik_Bjlk_DB(TENSILE_ARG_NAMES);
    }
}

template <>
const char* tensileGetSolutionName<rocblas_float_complex>(rocblas_operation trans_a,
                                                          rocblas_operation trans_b,
                                                          rocblas_int       strideC1,
                                                          rocblas_int       strideC2,
                                                          rocblas_int       strideA1,
                                                          rocblas_int       strideA2,
                                                          rocblas_int       strideB1,
                                                          rocblas_int       strideB2,
                                                          rocblas_int       sizeI,
                                                          rocblas_int       sizeJ,
                                                          rocblas_int       sizeK,
                                                          rocblas_int       sizeL)
{
    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        return tensileGetSolutionName_Cijk_Ailk_Bljk_CB(TENSILE_ARG_NAMES);
    case NT:
        return tensileGetSolutionName_Cijk_Ailk_Bjlk_CB(TENSILE_ARG_NAMES);
    case TN:
        return tensileGetSolutionName_Cijk_Alik_Bljk_CB(TENSILE_ARG_NAMES);
    case TT:
        return tensileGetSolutionName_Cijk_Alik_Bjlk_CB(TENSILE_ARG_NAMES);
    case NC:
        return tensileGetSolutionName_Cijk_Ailk_BjlkC_CB(TENSILE_ARG_NAMES);
    case CN:
        return tensileGetSolutionName_Cijk_AlikC_Bljk_CB(TENSILE_ARG_NAMES);
    case TC:
        return tensileGetSolutionName_Cijk_Alik_BjlkC_CB(TENSILE_ARG_NAMES);
    case CT:
        return tensileGetSolutionName_Cijk_AlikC_Bjlk_CB(TENSILE_ARG_NAMES);
    case CC:
        return tensileGetSolutionName_Cijk_AlikC_BjlkC_CB(TENSILE_ARG_NAMES);
    }
}

template <>
const char* tensileGetSolutionName<rocblas_double_complex>(rocblas_operation trans_a,
                                                           rocblas_operation trans_b,
                                                           rocblas_int       strideC1,
                                                           rocblas_int       strideC2,
                                                           rocblas_int       strideA1,
                                                           rocblas_int       strideA2,
                                                           rocblas_int       strideB1,
                                                           rocblas_int       strideB2,
                                                           rocblas_int       sizeI,
                                                           rocblas_int       sizeJ,
                                                           rocblas_int       sizeK,
                                                           rocblas_int       sizeL)
{
    switch(GetTransposeMode(trans_a, trans_b))
    {
    case NN:
        return tensileGetSolutionName_Cijk_Ailk_Bljk_ZB(TENSILE_ARG_NAMES);
    case NT:
        return tensileGetSolutionName_Cijk_Ailk_Bjlk_ZB(TENSILE_ARG_NAMES);
    case TN:
        return tensileGetSolutionName_Cijk_Alik_Bljk_ZB(TENSILE_ARG_NAMES);
    case TT:
        return tensileGetSolutionName_Cijk_Alik_Bjlk_ZB(TENSILE_ARG_NAMES);
    case NC:
        return tensileGetSolutionName_Cijk_Ailk_BjlkC_ZB(TENSILE_ARG_NAMES);
    case CN:
        return tensileGetSolutionName_Cijk_AlikC_Bljk_ZB(TENSILE_ARG_NAMES);
    case TC:
        return tensileGetSolutionName_Cijk_Alik_BjlkC_ZB(TENSILE_ARG_NAMES);
    case CT:
        return tensileGetSolutionName_Cijk_AlikC_Bjlk_ZB(TENSILE_ARG_NAMES);
    case CC:
        return tensileGetSolutionName_Cijk_AlikC_BjlkC_ZB(TENSILE_ARG_NAMES);
    }
}

#undef TENSILE_ARG_NAMES

/* ============================================================================================ */

/*
 * ===========================================================================
 *    template interface
 *    template specialization
 *    call GEMM C interfaces (see gemm.cpp in the same dir)
 * ===========================================================================
 */

/* ============================================================================================ */

template <bool BATCHED, bool STRIDED, typename T, typename U, typename V>
inline rocblas_status rocblas_gemm_template(rocblas_handle    handle,
                                            rocblas_operation trans_a,
                                            rocblas_operation trans_b,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const T*          alpha,
                                            rocblas_stride    stride_alpha,
                                            const U           A,
                                            rocblas_int       offset_a,
                                            rocblas_int       ld_a,
                                            rocblas_stride    stride_a,
                                            const U           B,
                                            rocblas_int       offset_b,
                                            rocblas_int       ld_b,
                                            rocblas_stride    stride_b,
                                            const T*          beta,
                                            rocblas_stride    stride_beta,
                                            V                 C,
                                            rocblas_int       offset_c,
                                            rocblas_int       ld_c,
                                            rocblas_stride    stride_c,
                                            rocblas_int       b_c)
{
    if(!STRIDED)
    {
        infer_batch_strides(
            trans_a, trans_b, m, n, k, ld_a, &stride_a, ld_b, &stride_b, ld_c, &stride_c);
    }

    if(m == 0 || n == 0 || k == 0 || b_c == 0)
    {
        return rocblas_status_success;
    }

    unsigned int strideC1 = unsigned(ld_c);
    unsigned int strideC2 = unsigned(stride_c);
    unsigned int strideA1 = unsigned(ld_a);
    unsigned int strideA2 = unsigned(stride_a);
    unsigned int strideB1 = unsigned(ld_b);
    unsigned int strideB2 = unsigned(stride_b);
    unsigned int sizeI    = unsigned(m);
    unsigned int sizeJ    = unsigned(n);
    unsigned int sizeK    = unsigned(b_c);
    unsigned int sizeL    = unsigned(k);

    hipError_t status;
    if(BATCHED)
    {
        sizeK = 1;

        size_t sizecopy = b_c * sizeof(T*);
        // Host arrays of device pointers.
        T* hostA[b_c];
        T* hostB[b_c];
        T* hostC[b_c];

        hipError_t errA = hipMemcpy(hostA, A, sizecopy, hipMemcpyDeviceToHost);
        hipError_t errB = hipMemcpy(hostB, B, sizecopy, hipMemcpyDeviceToHost);
        hipError_t errC = hipMemcpy(hostC, C, sizecopy, hipMemcpyDeviceToHost);

        if(get_rocblas_status_for_hip_status(errA) != rocblas_status_success)
            return get_rocblas_status_for_hip_status(errA);
        else if(get_rocblas_status_for_hip_status(errB) != rocblas_status_success)
            return get_rocblas_status_for_hip_status(errB);
        else if(get_rocblas_status_for_hip_status(errC) != rocblas_status_success)
            return get_rocblas_status_for_hip_status(errC);

        for(int b = 0; b < b_c; b++)
        {
            // We cannot do this with a device array, so array of pointers
            // must be on host for now
            status = call_tensile<T>(alpha + b * stride_alpha,
                                     0,
                                     beta + b * stride_beta,
                                     0,
                                     hostA[b] + offset_a,
                                     hostB[b] + offset_b,
                                     hostC[b] + offset_c,
                                     trans_a,
                                     trans_b,
                                     strideC1,
                                     strideC2,
                                     strideA1,
                                     strideA2,
                                     strideB1,
                                     strideB2,
                                     sizeI,
                                     sizeJ,
                                     sizeK,
                                     sizeL,
                                     handle);

            if(get_rocblas_status_for_hip_status(status) != rocblas_status_success)
                break;
        }
    }
    else
    {
        if(stride_alpha > 0 || stride_beta > 0)
        {
            // We can do this, but performance will be degraded,
            // we will have to use a loop to go over batches rather
            // than using tensile, since tensile does not support
            // different scalars as of now.

            sizeK = 1;
            for(int b = 0; b < b_c; b++)
            {
                status = call_tensile<T>(alpha + b * stride_alpha,
                                         0,
                                         beta + b * stride_beta,
                                         0,
                                         A + b * stride_a + offset_a,
                                         B + b * stride_b + offset_b,
                                         C + b * stride_c + offset_c,
                                         trans_a,
                                         trans_b,
                                         strideC1,
                                         strideC2,
                                         strideA1,
                                         strideA2,
                                         strideB1,
                                         strideB2,
                                         sizeI,
                                         sizeJ,
                                         sizeK,
                                         sizeL,
                                         handle);
            }
        }
        else
        {
            status = call_tensile<T>(alpha,
                                     stride_alpha, // == 0, otherwise not supported
                                     beta,
                                     stride_beta, // see ^
                                     A + offset_a,
                                     B + offset_b,
                                     C + offset_c,
                                     trans_a,
                                     trans_b,
                                     strideC1,
                                     strideC2,
                                     strideA1,
                                     strideA2,
                                     strideB1,
                                     strideB2,
                                     sizeI,
                                     sizeJ,
                                     sizeK,
                                     sizeL,
                                     handle);
        }
    }

    return get_rocblas_status_for_hip_status(status);
}

/* ============================================================================================ */
template <bool BATCHED, typename T>
inline void rocblas_gemm_kernel_name_template(rocblas_operation trans_a,
                                              rocblas_operation trans_b,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              rocblas_int       ld_a,
                                              rocblas_stride    stride_a,
                                              rocblas_int       ld_b,
                                              rocblas_stride    stride_b,
                                              rocblas_int       ld_c,
                                              rocblas_stride    stride_c,
                                              rocblas_int       b_c)
{
    unsigned int strideC1 = unsigned(ld_c);
    unsigned int strideC2 = unsigned(stride_c);
    unsigned int strideA1 = unsigned(ld_a);
    unsigned int strideA2 = unsigned(stride_a);
    unsigned int strideB1 = unsigned(ld_b);
    unsigned int strideB2 = unsigned(stride_b);
    unsigned int sizeI    = unsigned(m);
    unsigned int sizeJ    = unsigned(n);
    unsigned int sizeK    = unsigned(b_c);
    unsigned int sizeL    = unsigned(k);

    std::cout << "gemm kernel Name: ";
    if(BATCHED)
    {
        std::cout << "batched kernels have not yet been implemented" << std::endl;
        return;
    }

    const char* solution_name = tensileGetSolutionName<T>(trans_a,
                                                          trans_b,
                                                          strideC1,
                                                          strideC2,
                                                          strideA1,
                                                          strideA2,
                                                          strideB1,
                                                          strideB2,
                                                          sizeI,
                                                          sizeJ,
                                                          sizeK,
                                                          sizeL);

    std::cout << solution_name << std::endl;
}

#endif // _GEMM_HOST_HPP_
