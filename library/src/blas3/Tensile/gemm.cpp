/**************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 ************************************************************************** */
#include "gemm.h"
#include "Tensile.h"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"
#include <sys/time.h>

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

template <typename>
static constexpr char rocblas_gemm_name[] = "unknown";
template <>
static constexpr char rocblas_gemm_name<rocblas_half>[] = "rocblas_hgemm";
template <>
static constexpr char rocblas_gemm_name<float>[] = "rocblas_sgemm";
template <>
static constexpr char rocblas_gemm_name<double>[] = "rocblas_dgemm";
template <>
static constexpr char rocblas_gemm_name<rocblas_float_complex>[] = "rocblas_cgemm";
template <>
static constexpr char rocblas_gemm_name<rocblas_double_complex>[] = "rocblas_zgemm";

/*******************************************************************************
 * GEMM implementation
 ******************************************************************************/
template <typename T>
rocblas_status rocblas_gemm_impl(rocblas_handle    handle,
                                 rocblas_operation trans_a,
                                 rocblas_operation trans_b,
                                 rocblas_int       m,
                                 rocblas_int       n,
                                 rocblas_int       k,
                                 const T*          alpha,
                                 const T*          A,
                                 rocblas_int       ld_a,
                                 const T*          B,
                                 rocblas_int       ld_b,
                                 const T*          beta,
                                 T*                C,
                                 rocblas_int       ld_c)
{
    // clang-format off
    // Perform logging
    if(!handle)
        return rocblas_status_invalid_handle;
    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    if(!alpha || !beta)
        return rocblas_status_invalid_pointer;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        auto trans_a_letter = rocblas_transpose_letter(trans_a);
        auto trans_b_letter = rocblas_transpose_letter(trans_b);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemm_name<T>,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          *alpha,
                          A,
                          ld_a,
                          B,
                          ld_b,
                          *beta,
                          C,
                          ld_c);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                std::stringstream alphass;
                alphass << "--alpha " << std::real(*alpha);
                if (std::imag(*alpha) != 0)
                    alphass << " --alphai " << std::imag(*alpha);

                std::stringstream betass;
                betass << "--beta " << std::real(*beta);
                if (std::imag(*beta) != 0)
                    betass << " --betai " << std::imag(*beta);

                log_bench(handle,
                          "./rocblas-bench -f gemm -r",
                          rocblas_precision_string<T>,
                          "--transposeA",
                          trans_a_letter,
                          "--transposeB",
                          trans_b_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "-k",
                          k,
                          alphass.str(),
                          "--lda",
                          ld_a,
                          "--ldb",
                          ld_b,
                          betass.str(),
                          "--ldc",
                          ld_c);
            }
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemm_name<T>,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          alpha,
                          A,
                          ld_a,
                          B,
                          ld_b,
                          beta,
                          C,
                          ld_c);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_gemm_name<T>,
                        "transA",
                        trans_a_letter,
                        "transB",
                        trans_b_letter,
                        "M",
                        m,
                        "N",
                        n,
                        "K",
                        k,
                        "lda",
                        ld_a,
                        "ldb",
                        ld_b,
                        "ldc",
                        ld_c);
    }

    rocblas_int b_c = 1;
    if(m == 0 || n == 0 || k == 0 || b_c == 0)
    {
        return rocblas_status_success;
    }

    rocblas_int stride_a;
    rocblas_int stride_b;
    rocblas_int stride_c;
    infer_batch_strides(trans_a, trans_b, m, n, k, ld_a,
                        &stride_a, ld_b, &stride_b, ld_c, &stride_c);

    rocblas_status validArgs = validateArgs(handle, trans_a, trans_b,
                                            m, n, k, alpha,
                                            A, ld_a, stride_a,
                                            B, ld_b, stride_b, beta,
                                            C, ld_c, stride_c, b_c);

    if(validArgs != rocblas_status_success)
        return validArgs;

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

    hipError_t status = call_tensile<T>(alpha, beta, A, B, C,
                                        trans_a, trans_b,
                                        strideC1, strideC2,
                                        strideA1, strideA2,
                                        strideB1, strideB2,
                                        sizeI, sizeJ, sizeK, sizeL,
                                        handle);
    // clang-format on

    return get_rocblas_status_for_hip_status(status);
}

template <typename>
static constexpr char rocblas_gemm_strided_batched_name[] = "unknown";
template <>
static constexpr char rocblas_gemm_strided_batched_name<rocblas_half>[]
    = "rocblas_hgemm_strided_batched";
template <>
static constexpr char rocblas_gemm_strided_batched_name<float>[] = "rocblas_sgemm_strided_batched";
template <>
static constexpr char rocblas_gemm_strided_batched_name<double>[] = "rocblas_dgemm_strided_batched";
template <>
static constexpr char rocblas_gemm_strided_batched_name<rocblas_float_complex>[]
    = "rocblas_cgemm_strided_batched";
template <>
static constexpr char rocblas_gemm_strided_batched_name<rocblas_double_complex>[]
    = "rocblas_zgemm_strided_batched";

/*******************************************************************************
 * Strided / Batched GEMM implementation
 ******************************************************************************/
template <typename T>
rocblas_status rocblas_gemm_strided_batched_impl(rocblas_handle    handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int       m,
                                                 rocblas_int       n,
                                                 rocblas_int       k,
                                                 const T*          alpha,
                                                 const T*          A,
                                                 rocblas_int       ld_a,
                                                 rocblas_int       stride_a,
                                                 const T*          B,
                                                 rocblas_int       ld_b,
                                                 rocblas_int       stride_b,
                                                 const T*          beta,
                                                 T*                C,
                                                 rocblas_int       ld_c,
                                                 rocblas_int       stride_c,
                                                 rocblas_int       b_c)

{
    // clang-format off
    if(!handle)
        return rocblas_status_invalid_handle;
    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    auto layer_mode = handle->layer_mode;

    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        auto trans_a_letter = rocblas_transpose_letter(trans_a);
        auto trans_b_letter = rocblas_transpose_letter(trans_b);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemm_strided_batched_name<T>,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          *alpha,
                          A,
                          ld_a,
                          stride_a,
                          B,
                          ld_b,
                          stride_b,
                          *beta,
                          C,
                          ld_c,
                          stride_c,
                          b_c);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                std::stringstream alphass;
                alphass << "--alpha " << std::real(*alpha);
                if (std::imag(*alpha) != 0)
                    alphass << " --alphai " << std::imag(*alpha);

                std::stringstream betass;
                betass << "--beta " << std::real(*beta);
                if (std::imag(*beta) != 0)
                    betass << " --betai " << std::imag(*beta);

                log_bench(handle,
                          "./rocblas-bench -f gemm_strided_batched -r",
                          rocblas_precision_string<T>,
                          "--transposeA",
                          trans_a_letter,
                          "--transposeB",
                          trans_b_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "-k",
                          k,
                          alphass.str(),
                          "--lda",
                          ld_a,
                          "--stride_a",
                          stride_a,
                          "--ldb",
                          ld_b,
                          "--stride_b",
                          stride_b,
                          betass.str(),
                          "--ldc",
                          ld_c,
                          "--stride_c",
                          stride_c,
                          "--batch",
                          b_c);
            }
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
            {
                log_trace(handle,
                          rocblas_gemm_strided_batched_name<T>,
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
                          b_c);
            }
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            log_profile(handle,
                        rocblas_gemm_strided_batched_name<T>,
                        "transA",
                        trans_a_letter,
                        "transB",
                        trans_b_letter,
                        "M",
                        m,
                        "N",
                        n,
                        "K",
                        k,
                        "lda",
                        ld_a,
                        "stride_a",
                        stride_a,
                        "ldb",
                        ld_b,
                        "stride_b",
                        stride_b,
                        "ldc",
                        ld_c,
                        "stride_c",
                        stride_c,
                        "batch_count",
                        b_c);
        }
    }

    if(m == 0 || n == 0 || k == 0 || b_c == 0)
    {
        return rocblas_status_success;
    }

    rocblas_status validArgs = validateArgs(handle, trans_a, trans_b,
                                            m, n, k, alpha,
                                            A, ld_a, stride_a,
                                            B, ld_b, stride_b, beta,
                                            C, ld_c, stride_c, b_c);

    if(validArgs != rocblas_status_success)
        return validArgs;

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

    hipError_t status = call_tensile<T>(alpha, beta, A, B, C,
                                        trans_a, trans_b,
                                        strideC1, strideC2,
                                        strideA1, strideA2,
                                        strideB1, strideB2,
                                        sizeI, sizeJ, sizeK, sizeL,
                                        handle);
    return get_rocblas_status_for_hip_status(status);

    // clang-format on
}

/*******************************************************************************
 * Batched / Strided GEMM Kernel name implementation
 ******************************************************************************/
template <typename T>
rocblas_status rocblas_gemm_kernel_name_impl(rocblas_handle    handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int       m,
                                             rocblas_int       n,
                                             rocblas_int       k,
                                             const T*          alpha,
                                             const T*          A,
                                             rocblas_int       ld_a,
                                             rocblas_int       stride_a,
                                             const T*          B,
                                             rocblas_int       ld_b,
                                             rocblas_int       stride_b,
                                             const T*          beta,
                                             T*                C,
                                             rocblas_int       ld_c,
                                             rocblas_int       stride_c,
                                             rocblas_int       b_c)
{
    // clang-format off
    if(!handle)
        return rocblas_status_invalid_handle;
    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    auto layer_mode = handle->layer_mode;

    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        auto trans_a_letter = rocblas_transpose_letter(trans_a);
        auto trans_b_letter = rocblas_transpose_letter(trans_b);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemm_strided_batched_name<T>,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          k,
                          *alpha,
                          A,
                          ld_a,
                          stride_a,
                          B,
                          ld_b,
                          stride_b,
                          *beta,
                          C,
                          ld_c,
                          stride_c,
                          b_c);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f gemm_strided_batched -r",
                          rocblas_precision_string<T>,
                          "--transposeA",
                          trans_a_letter,
                          "--transposeB",
                          trans_b_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "-k",
                          k,
                          "--alpha",
                          *alpha,
                          "--lda",
                          ld_a,
                          "--bsa",
                          stride_a,
                          "--ldb",
                          ld_b,
                          "--bsb",
                          stride_b,
                          "--beta",
                          *beta,
                          "--ldc",
                          ld_c,
                          "--bsc",
                          stride_c,
                          "--batch",
                          b_c);
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_gemm_strided_batched_name<T>,
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
                          b_c);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_gemm_strided_batched_name<T>,
                        "transA",
                        trans_a_letter,
                        "transB",
                        trans_b_letter,
                        "M",
                        m,
                        "N",
                        n,
                        "K",
                        k,
                        "lda",
                        ld_a,
                        "stride_a",
                        stride_a,
                        "ldb",
                        ld_b,
                        "stride_b",
                        stride_b,
                        "ldc",
                        ld_c,
                        "stride_c",
                        stride_c,
                        "batch_count",
                        b_c);
    }

    rocblas_status validArgs = validateArgs(handle, trans_a, trans_b,
                                            m, n, k, alpha,
                                            A, ld_a, stride_a,
                                            B, ld_b, stride_b, beta,
                                            C, ld_c, stride_c, b_c);

    if(validArgs != rocblas_status_success)
        return validArgs;

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


    const char* solution_name = tensileGetSolutionName<T>(trans_a, trans_b,
                                                          strideC1, strideC2,
                                                          strideA1, strideA2,
                                                          strideB1, strideB2,
                                                          sizeI, sizeJ, sizeK, sizeL);

    std::cout << solution_name << std::endl;

    return validArgs;
}

/*******************************************************************************
 * GEMM APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const rocblas_half *alpha,
                             const rocblas_half *A,
                             rocblas_int ld_a,
                             const rocblas_half *B,
                             rocblas_int ld_b,
                             const rocblas_half *beta,
                             rocblas_half *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<rocblas_half>(handle, trans_a, trans_b,
                                           m, n, k, alpha, A, ld_a,
                                           B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_sgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const float *alpha,
                             const float *A,
                             rocblas_int ld_a,
                             const float *B,
                             rocblas_int ld_b,
                             const float *beta,
                             float *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<float>(handle, trans_a, trans_b,
                                    m, n, k, alpha, A, ld_a,
                                    B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_dgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const double *alpha,
                             const double *A,
                             rocblas_int ld_a,
                             const double *B,
                             rocblas_int ld_b,
                             const double *beta,
                             double *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<double>(handle, trans_a, trans_b,
                                     m, n, k, alpha, A, ld_a,
                                     B, ld_b, beta, C, ld_c);
}

rocblas_status rocblas_cgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const rocblas_float_complex *alpha,
                             const rocblas_float_complex *A,
                             rocblas_int ld_a,
                             const rocblas_float_complex *B,
                             rocblas_int ld_b,
                             const rocblas_float_complex *beta,
                             rocblas_float_complex *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<rocblas_float_complex>(handle, trans_a, trans_b,
                                                    m, n, k, alpha, A, ld_a,
                                                    B, ld_b, beta, C, ld_c);
}


rocblas_status rocblas_zgemm(rocblas_handle handle,
                             rocblas_operation trans_a,
                             rocblas_operation trans_b,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             const rocblas_double_complex *alpha,
                             const rocblas_double_complex *A,
                             rocblas_int ld_a,
                             const rocblas_double_complex *B,
                             rocblas_int ld_b,
                             const rocblas_double_complex *beta,
                             rocblas_double_complex *C,
                             rocblas_int ld_c)
{
    return rocblas_gemm_impl<rocblas_double_complex>(handle, trans_a, trans_b,
                                                    m, n, k, alpha, A, ld_a,
                                                    B, ld_b, beta, C, ld_c);
}

/*******************************************************************************
 * Batched / Strided GEMM APIs
 ******************************************************************************/

rocblas_status rocblas_hgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const rocblas_half *alpha,
                                             const rocblas_half *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const rocblas_half *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const rocblas_half *beta,
                                             rocblas_half *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<rocblas_half>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_sgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const float *alpha,
                                             const float *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const float *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const float *beta,
                                             float *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<float>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_dgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const double *alpha,
                                             const double *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const double *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const double *beta,
                                             double *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<double>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_cgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const rocblas_float_complex *alpha,
                                             const rocblas_float_complex *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const rocblas_float_complex *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const rocblas_float_complex *beta,
                                             rocblas_float_complex *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<rocblas_float_complex>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}

rocblas_status rocblas_zgemm_strided_batched(rocblas_handle handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             const rocblas_double_complex *alpha,
                                             const rocblas_double_complex *A,
                                             rocblas_int ld_a,
                                             rocblas_int stride_a,
                                             const rocblas_double_complex *B,
                                             rocblas_int ld_b,
                                             rocblas_int stride_b,
                                             const rocblas_double_complex *beta,
                                             rocblas_double_complex *C,
                                             rocblas_int ld_c,
                                             rocblas_int stride_c,
                                             rocblas_int b_c)
{
    return rocblas_gemm_strided_batched_impl<rocblas_double_complex>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
}


/*******************************************************************************
 * Batched / Strided GEMM Kernel name APIs
 ******************************************************************************/
rocblas_status rocblas_hgemm_kernel_name(rocblas_handle handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const rocblas_half *alpha,
                                         const rocblas_half *A,
                                         rocblas_int ld_a,
                                         rocblas_int stride_a,
                                         const rocblas_half *B,
                                         rocblas_int ld_b,
                                         rocblas_int stride_b,
                                         const rocblas_half *beta,
                                         rocblas_half *C,
                                         rocblas_int ld_c,
                                         rocblas_int stride_c,
                                         rocblas_int b_c)
{
    rocblas_status status = rocblas_gemm_kernel_name_impl<rocblas_half>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
    return status;
}

rocblas_status rocblas_sgemm_kernel_name(rocblas_handle handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const float *alpha,
                                         const float *A,
                                         rocblas_int ld_a,
                                         rocblas_int stride_a,
                                         const float *B,
                                         rocblas_int ld_b,
                                         rocblas_int stride_b,
                                         const float *beta,
                                         float *C,
                                         rocblas_int ld_c,
                                         rocblas_int stride_c,
                                         rocblas_int b_c)
{
    rocblas_status status = rocblas_gemm_kernel_name_impl<float>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
    return status;
}

rocblas_status rocblas_dgemm_kernel_name(rocblas_handle handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int m,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const double *alpha,
                                         const double *A,
                                         rocblas_int ld_a,
                                         rocblas_int stride_a,
                                         const double *B,
                                         rocblas_int ld_b,
                                         rocblas_int stride_b,
                                         const double *beta,
                                         double *C,
                                         rocblas_int ld_c,
                                         rocblas_int stride_c,
                                         rocblas_int b_c)
{
    rocblas_status status = rocblas_gemm_kernel_name_impl<double>(
        handle, trans_a, trans_b,
        m, n, k,
        alpha,
        A, ld_a, stride_a,
        B, ld_b, stride_b,
        beta,
        C, ld_c, stride_c, b_c);
    return status;
}
// clang-format on
