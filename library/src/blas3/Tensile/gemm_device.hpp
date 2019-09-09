/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef __GEMM_DEVICE_HPP__
#define __GEMM_DEVICE_HPP__

#include "utility.h"
#include "Tensile.h"
#include "handle.h"
#include "logging.h"
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
                          rocblas_int       strideC1,
                          rocblas_int       strideC2,
                          rocblas_int       strideA1,
                          rocblas_int       strideA2,
                          rocblas_int       strideB1,
                          rocblas_int       strideB2,
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
                          rocblas_int         strideC1,
                          rocblas_int         strideC2,
                          rocblas_int         strideA1,
                          rocblas_int         strideA2,
                          rocblas_int         strideB1,
                          rocblas_int         strideB2,
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
                          rocblas_int       strideC1,
                          rocblas_int       strideC2,
                          rocblas_int       strideA1,
                          rocblas_int       strideA2,
                          rocblas_int       strideB1,
                          rocblas_int       strideB2,
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
                          rocblas_int       strideC1,
                          rocblas_int       strideC2,
                          rocblas_int       strideA1,
                          rocblas_int       strideA2,
                          rocblas_int       strideB1,
                          rocblas_int       strideB2,
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
                          rocblas_int                  strideC1,
                          rocblas_int                  strideC2,
                          rocblas_int                  strideA1,
                          rocblas_int                  strideA2,
                          rocblas_int                  strideB1,
                          rocblas_int                  strideB2,
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
                          rocblas_int                   strideC1,
                          rocblas_int                   strideC2,
                          rocblas_int                   strideA1,
                          rocblas_int                   strideA2,
                          rocblas_int                   strideB1,
                          rocblas_int                   strideB2,
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
                        const T*          beta,
                        const T*          A,
                        const T*          B,
                        T*                C,
                        rocblas_operation trans_a,
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
                        rocblas_int       sizeL,
                        rocblas_handle    handle)
{
#ifndef NDEBUG
    std::cout << "Solution Name: "
              << tensileGetSolutionName<T>(trans_a,
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
                                           sizeL)
              << std::endl;
#endif

    // Collect alpha / beta (either from host or device)
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

#endif