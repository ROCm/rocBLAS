/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ABLAS_HIP_H_
#define _ABLAS_HIP_H_

#include <hip_runtime.h> 

/*!\file
 * \brief ABLAS interface with HIP APIs: memory allocation, device management
 */


typedef hipStream_t ablas_queue;
typedef hipEvent_t  ablas_event;
typedef ablas_queue ablas_handle;


    /* ============================================================================================ */
    /**
     *   @brief ablas error codes definition, incorporating HIP error
     *   definitions.
     *
     *   This enumeration is a subset of the HIP error codes extended with some
     *   additional extra codes.  For example, hipErrorMemoryAllocation, which is
     *   defined in hip_runtime_api.h is aliased as ablas_error_memory_allocation.
     */
    typedef enum ablas_status_ {

        ablas_success                       =    hipSuccess = 0,                  ///< Successful completion.
        ablas_error_memory_allocation       =    hipErrorMemoryAllocation,        ///< Memory allocation error.
        ablas_error_memory_free             =    hipErrorMemoryFree,              ///< Memory free error.
        ablas_error_unknown_symbol          =    hipErrorUnknownSymbol,           ///< Unknown symbol
        ablas_error_outof_resources         =    hipErrorOutOfResources          ///< Out of resources error
        ablas_error_invalid_value           =    hipErrorInvalidValue            ///< One or more of the paramters passed to the API call is NULL or not in an acceptable range.
        ablas_error_invalid_resource_handle =    hipErrorInvalidResourceHandle   ///< Resource handle (hipEvent_t or hipStream_t) invalid.
        ablas_error_invalid_device          =    hipErrorInvalidDevice           ///< DeviceID must be in range 0...#compute-devices.
        ablas_error_no_deive                =    hipErrorNoDevice                ///< Call to cudaGetDeviceCount returned 0 devices
        ablas_error_not_ready               =    hipErrorNotReady                ///< indicates that asynchronous operations enqueued earlier are not ready.  
                                                                                 /// This is not actually an error, but is used to distinguish from hipSuccess(which indicates completion).  
                                                                                 /// APIs that return this error include hipEventQuery and hipStreamQuery.
        /* Extended error codes */
        ablas_not_implemented         = -1024, /**< Functionality is not implemented */
        ablas_not_initialized,                 /**< ablas library is not initialized yet */
        ablas_invalid_matA,                    /**< Matrix A is not a valid memory object */
        ablas_invalid_matB,                    /**< Matrix B is not a valid memory object */
        ablas_invalid_matC,                    /**< Matrix C is not a valid memory object */
        ablas_invalid_vecX,                    /**< Vector X is not a valid memory object */
        ablas_invalid_becY,                    /**< Vector Y is not a valid memory object */
        ablas_invalid_dim,                     /**< An input dimension (M,N,K) is invalid */
        ablas_invalid_leadDimA,                /**< Leading dimension A must not be less than the size of the first dimension */
        ablas_invalid_leadDimB,                /**< Leading dimension B must not be less than the size of the second dimension */
        ablas_invalid_leadDimC,                /**< Leading dimension C must not be less than the size of the third dimension */
        ablas_invalid_incx,                    /**< The increment for a vector X must not be 0 */
        ablas_invalid_incy,                    /**< The increment for a vector Y must not be 0 */
    } ablas_status;




    /* ============================================================================================ */
    /*! \brief   memory allocation on GPU devie memory */
    template<class T>
    ablas_status
    ablas_malloc_device(T** ptr, size_t bytes ){
        return hipMalloc(ptr, bytes);        
    };
    
    /*! \brief   memory allocation on GPU host pinned memmory */
    template<class T>
    ablas_status
    ablas_malloc_host(T** ptr, size_t bytes ){
        return hipMallocHost(ptr, bytes);        
    };


#endif

