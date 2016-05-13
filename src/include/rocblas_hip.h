/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_HIP_H_
#define _ROCBLAS_HIP_H_

#include <hip_runtime.h>
#include <hip_vector_types.h>
#include "rocblas_types.h"

/*!\file
 * \brief ROCBLAS data types must inherit from HIP:
 */


    /*! \brief  HIP & CUDA both use float2/double2 to define complex number
     */

    typedef float2 rocblas_float_complex;
    typedef double2 rocblas_double_complex;
    // A Lot TODO about complex

    /* ============================================================================================ */
    /**
     *   @brief rocblas error codes definition, incorporating HIP error
     *   definitions.
     *
     *   This enumeration is a subset of the HIP error codes extended with some
     *   additional extra codes.  For example, hipErrorMemoryAllocation, which is
     *   defined in hip_runtime_api.h is aliased as rocblas_error_memory_allocation.
     */
#if 0
    typedef enum rocblas_status_ {

        rocblas_success                       =    hipSuccess ,                  ///< Successful completion.
        rocblas_error_memory_allocation       =    hipErrorMemoryAllocation,        ///< Memory allocation error.
        rocblas_error_memory_free             =    hipErrorMemoryFree,              ///< Memory free error.
        rocblas_error_unknown_symbol          =    hipErrorUnknownSymbol,           ///< Unknown symbol
        rocblas_error_outof_resources         =    hipErrorOutOfResources,          ///< Out of resources error
        rocblas_error_invalid_value           =    hipErrorInvalidValue,            ///< One or more of the paramters passed to the API call is NULL or not in an acceptable range.
        rocblas_error_invalid_resource_handle =    hipErrorInvalidResourceHandle,   ///< Resource handle (hipEvent_t or hipStream_t) invalid.
        rocblas_error_invalid_device          =    hipErrorInvalidDevice ,          ///< DeviceID must be in range 0...#compute-devices.
        rocblas_error_no_deive                =    hipErrorNoDevice ,               ///< Call to cudaGetDeviceCount returned 0 devices
        rocblas_error_not_ready               =    hipErrorNotReady ,               ///< indicates that asynchronous operations enqueued earlier are not ready.
                                                                                 /// This is not actually an error, but is used to distinguish from hipSuccess(which indicates completion).
                                                                                 /// APIs that return this error include hipEventQuery and hipStreamQuery.
        /* Extended error codes */
        rocblas_not_implemented         = -1024, /**< Functionality is not implemented */
        rocblas_not_initialized,                 /**< rocblas library is not initialized yet */
        rocblas_invalid_matA,                    /**< Matrix A is not a valid memory object */
        rocblas_invalid_matB,                    /**< Matrix B is not a valid memory object */
        rocblas_invalid_matC,                    /**< Matrix C is not a valid memory object */
        rocblas_invalid_vecX,                    /**< Vector X is not a valid memory object */
        rocblas_invalid_vecY,                    /**< Vector Y is not a valid memory object */
        rocblas_invalid_dim,                     /**< An input dimension (M,N,K) is invalid */
        rocblas_invalid_size,                     /**< An input data size is invalid, like <0 */
        rocblas_invalid_leadDimA,                /**< Leading dimension A must not be less than the size of the first dimension */
        rocblas_invalid_leadDimB,                /**< Leading dimension B must not be less than the size of the second dimension */
        rocblas_invalid_leadDimC,                /**< Leading dimension C must not be less than the size of the third dimension */
        rocblas_invalid_incx,                    /**< The increment for a vector X must not be 0 */
        rocblas_invalid_incy,                    /**< The increment for a vector Y must not be 0 */
    } rocblas_status;
#endif

#endif
