/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

/*******************************************************************************
 * Definitions
 ******************************************************************************/
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

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                                \
    do                                                                                            \
    {                                                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                                 \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                                    \
        {                                                                                         \
            rocblas_cerr << "hip error code: '" << hipGetErrorName(TMP_STATUS_FOR_CHECK)          \
                         << "':" << TMP_STATUS_FOR_CHECK << " at " << __FILE__ << ":" << __LINE__ \
                         << std::endl;                                                            \
        }                                                                                         \
    } while(0)

#define PRINT_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                                            \
    do                                                                                            \
    {                                                                                             \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                             \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)                                        \
        {                                                                                         \
            rocblas_cerr << "rocblas error: '" << rocblas_status_to_string(TMP_STATUS_FOR_CHECK)  \
                         << "':" << TMP_STATUS_FOR_CHECK << " at " << __FILE__ << ":" << __LINE__ \
                         << std::endl;                                                            \
        }                                                                                         \
    } while(0)

#define PRINT_AND_RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                                 \
    do                                                                                            \
    {                                                                                             \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                             \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)                                        \
        {                                                                                         \
            rocblas_cerr << "rocblas error: '" << rocblas_status_to_string(TMP_STATUS_FOR_CHECK)  \
                         << "':" << TMP_STATUS_FOR_CHECK << " at " << __FILE__ << ":" << __LINE__ \
                         << std::endl;                                                            \
            return TMP_STATUS_FOR_CHECK;                                                          \
        }                                                                                         \
    } while(0)

#endif // DEFINITIONS_H
