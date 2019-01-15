/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H
#include "status.h"

/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/

// half vectors
typedef _Float16 rocblas_half8 __attribute__((ext_vector_type(8)));
typedef _Float16 rocblas_half2 __attribute__((ext_vector_type(2)));

#ifndef GOOGLE_TEST // suppress warnings about __device__ when building tests
extern "C" __device__ rocblas_half2 llvm_fma_v2f16(rocblas_half2,
                                                   rocblas_half2,
                                                   rocblas_half2) __asm("llvm.fma.v2f16");

__device__ inline rocblas_half2
rocblas_fmadd_half2(rocblas_half2 multiplier, rocblas_half2 multiplicand, rocblas_half2 addend)
{
    return llvm_fma_v2f16(multiplier, multiplicand, addend);
}
#endif

#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                         \
    do                                                                      \
    {                                                                       \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;           \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                              \
        {                                                                   \
            return get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                   \
    } while(0)

#define RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)               \
    do                                                                \
    {                                                                 \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)            \
        {                                                             \
            return TMP_STATUS_FOR_CHECK;                              \
        }                                                             \
    } while(0)

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                         \
    do                                                                     \
    {                                                                      \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;          \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                             \
        {                                                                  \
            throw get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                  \
    } while(0)

#define THROW_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                \
    do                                                                \
    {                                                                 \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)            \
        {                                                             \
            throw TMP_STATUS_FOR_CHECK;                               \
        }                                                             \
    } while(0)

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    do                                                            \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    } while(0)

#define PRINT_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                \
    do                                                                \
    {                                                                 \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)            \
        {                                                             \
            fprintf(stderr,                                           \
                    "rocblas error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                             \
                    __FILE__,                                         \
                    __LINE__);                                        \
        }                                                             \
    } while(0)

#endif // DEFINITIONS_H
