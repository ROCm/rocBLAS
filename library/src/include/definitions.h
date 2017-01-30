/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#if BUILD_WITH_TENSILE
   #include "Tensile_status.h"
#endif

#include "status.h"

/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/


#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK) { \
    hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
    if (TMP_STATUS_FOR_CHECK != hipSuccess ) { \
        return get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
} }

#define RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK) { \
    rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
    if (TMP_STATUS_FOR_CHECK != rocblas_status_success) { \
        return TMP_STATUS_FOR_CHECK; \
} }

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK) { \
    hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
    if (TMP_STATUS_FOR_CHECK != hipSuccess ) { \
        throw get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
} }

#define THROW_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK) { \
    rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
    if (TMP_STATUS_FOR_CHECK != rocblas_status_success) { \
        throw TMP_STATUS_FOR_CHECK; \
} }

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK) {\
    hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
    if (TMP_STATUS_FOR_CHECK != hipSuccess) { \
        fprintf(stderr, "hip error code: %d at %s:%d\n",  TMP_STATUS_FOR_CHECK,__FILE__, __LINE__); \
} }


#define PRINT_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK) {\
    rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
    if (TMP_STATUS_FOR_CHECK != rocblas_status_success) { \
        fprintf(stderr, "rocblas error code: %d at %s:%d\n",  TMP_STATUS_FOR_CHECK,__FILE__, __LINE__); \
} }


#if BUILD_WITH_TENSILE

    #define RETURN_IF_TENSILE_ERROR(INPUT_STATUS_FOR_CHECK) { \
        TensileStatus TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if (TMP_STATUS_FOR_CHECK != tensileStatusSuccess) { \
            tensileStatusCheck( TMP_STATUS_FOR_CHECK ); \
            return get_rocblas_status_for_tensile_status(TMP_STATUS_FOR_CHECK); \
    } }

    #define THROW_IF_TENSILE_ERROR(INPUT_STATUS_FOR_CHECK) {\
        TensileStatus TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if (TMP_STATUS_FOR_CHECK != tensileStatusSuccess) { \
            tensileStatusCheck( TMP_STATUS_FOR_CHECK ); \
            throw get_rocblas_status_for_tensile_status(TMP_STATUS_FOR_CHECK); \
    } }

    #define PRINT_IF_TENSILE_ERROR(INPUT_STATUS_FOR_CHECK) {\
        TensileStatus TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;\
        if (TMP_STATUS_FOR_CHECK != tensileStatusSuccess) { \
            fprintf(stderr, "tensile error code: %d at %s:%d\n",  TMP_STATUS_FOR_CHECK,__FILE__, __LINE__); \
    } }\


#endif //BUILD_WITH_TENSILE


#endif //DEFINITIONS_H
