/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ABLAS_RUNTIME_H_
#define _ABLAS_RUNTIME_H_

#include "ablas_types.h" 

/*!\file
 * \brief ABLAS Runtime APIs: error handling, memory allocation, device management, stream management.
 */

	/*
	 * ===========================================================================
	 *   READEME: ABLAS Wrapper of HIP data types and APIs
         *   HIP is still under development. Developers of aBLAS are encouraged to use ablas APIs
         *   in their code, in case HIP APIs would be changed in the future.
	 * ===========================================================================
	 */

    #define CHECK_ABLAS_ERROR(error) \
    if (error != ablas_success) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", ablas_get_error_string(error), error,__FILE__, __LINE__); \
    	exit(EXIT_FAILURE);\
    }

#ifdef __cplusplus
extern "C" {
#endif

    /* ============================================================================================ */

    /* Error Handling */

    //warning: the four error handling API only get and return recent HIP API error 
    //ablas_status is a superset of hip error, non-hip-runtime error won't be detected and reported by them 

    /*! \brief  Return last error returned by any HIP runtime API call and resets the stored error code to ablas_success. */
    ablas_status ablas_get_last_error( void );

    /*! \brief  Return last error returned by any HIP runtime API call. */
    ablas_status ablas_peek_at_last_error ( void );
	
 	/*! \brief Return name of the specified error code in text form. */
    const char* ablas_get_error_name(ablas_status ablas_error);

  	/*! \brief Return handy text string message to explain the error which occurred. On HCC, it is the same as ablas_get_error_name() */
    const char* ablas_get_error_string(ablas_status ablas_error);


    /* ============================================================================================ */
    /*! \brief   memory allocation on GPU devie memory in ablas_hip.h */

    /*! \brief   memory free on GPU devie memory */
    ablas_status
    ablas_free_device(void *ptr );

    /*! \brief   memory free on GPU host pinned memmory */
    ablas_status
    ablas_free_host( void *ptr );

    /*! \brief  host-synchronous, supports memory from host to device, device to host, device to device and host to host */
    ablas_status
    ablas_memcpy_host_to_host(void *          dst,
                              const void *    src,
                              size_t          sizeBytes);

    ablas_status
    ablas_memcpy_host_to_device(void *          dst,
                                const void *    src,
                                size_t          sizeBytes);
    ablas_status
    ablas_memcpy_device_to_host(void *          dst,
                                const void *    src,
                                size_t          sizeBytes);

    ablas_status
    ablas_memcpy_device_to_device(void *          dst,
                                  const void *    src,
                                  size_t          sizeBytes);

    /*! \brief   src to dst asynchronously, supports memory from host to device, device to host, device to device and host to host */
    ablas_status
    ablas_memcpy_async_host_to_host(void *           dst,
                                    const void *     src,
                                    size_t           sizeBytes,
                                    ablas_queue stream);

    ablas_status
    ablas_memcpy_async_host_to_device(void *           dst,
                                      const void *     src,
                                      size_t           sizeBytes,
                                      ablas_queue stream);

    ablas_status
    ablas_memcpy_async_device_to_host(void *           dst,
                                      const void *     src,
                                      size_t           sizeBytes,
                                      ablas_queue stream);

    ablas_status
    ablas_memcpy_async_device_to_device(void *           dst,
                                        const void *     src,
                                        size_t           sizeBytes,
                                        ablas_queue stream);

    /* ============================================================================================ */

    /* device management */

     /*! \brief  Blocks until the default device has completed all preceding requested tasks. */
    ablas_status ablas_device_synchronize(void);

    /*! \brief  Destroy all resources and reset all state on the default device in the current process. */
    ablas_status ablas_device_reset(void);
    }

    /*! \brief  Set default device to be used for subsequent hip API calls from this thread. */
     ablas_status ablas_set_device(ablas_int device);
     
    /*! \brief  Return the default device id for the calling host thread. */
    ablas_status ablas_get_device(ablas_int *device);

    /*! \brief  Return number of compute-capable devices. */
    ablas_status ablas_get_device_count(ablas_int *count);
 
    /* ============================================================================================ */
    /*  query device :*/
    void ablas_query_device();

    /* ============================================================================================ */

    /*   stream management */

    /*! \brief  Create an asynchronous stream. */
    ablas_status ablas_stream_create_withflags(ablas_queue *stream, unsigned ablas_int flags);
    /*! \brief  Make the specified compute stream wait for an event. */
    ablas_status ablas_stream_wait_event(ablas_queue stream, ablas_event event, unsigned ablas_int flags);
     
     /*! \brief  Wait for all commands in stream to complete. */
    ablas_status ablas_stream_synchronize(ablas_queue stream);
 
    /*! \brief  Destroys the specified stream. */
    ablas_status ablas_stream_destroy(ablas_queue stream);
      
     /*! \brief  Return flags associated with this stream. */
    ablas_status ablas_stream_get_flags(ablas_queue stream, unsigned ablas_int *flags);


#ifdef __cplusplus
}
#endif

#endif

