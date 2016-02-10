/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "stdio.h" 
#include "ablas_types.h" 
#include <hip_runtime.h> 


	/*
	 * ===========================================================================
	 *   READEME: ABLAS Wrapper of HIP data types and APIs
         *   HIP is still under development. Developers of aBLAS are encouraged to use ablas APIs
         *   in their code, in case HIP APIs would be changed in the future.
	 * ===========================================================================
	 */

    /* ============================================================================================ */

    /* Error Handling */

    //warning: the four error handling API only get and return recent HIP API error 
    //ablas_status is a superset of hip error, non-hip-runtime error won't be detected and reported by them 

    /*! \brief  Return last error returned by any HIP runtime API call and resets the stored error code to ablas_success. */
    extern "C"  ablas_status ablas_get_last_error( void ){
		return hipGetLastError();
	}

    /*! \brief  Return last error returned by any HIP runtime API call. */
    extern "C"  ablas_status ablas_peek_at_last_error ( void ){
		return hipPeekAtLastError();	
	} 	
	
 	/*! \brief Return name of the specified error code in text form. */
    extern "C"  const char* ablas_get_error_name(ablas_status ablas_error)
	{
		return hipGetErrorName (ablas_error);
	}

  	/*! \brief Return handy text string message to explain the error which occurred. On HCC, it is the same as ablas_get_error_name() */
    extern "C"  const char* ablas_get_error_string(ablas_status ablas_error)
	{
		return hipGetErrorString (ablas_error);
	}

    /* ============================================================================================ */
    /*! \brief   memory allocation on GPU devie memory is on header file */

    /*! \brief   memory free on GPU devie memory */
    extern "C"  ablas_status
    ablas_free_device(void *ptr ){
        return hipFree(ptr);    
    };

    /*! \brief   memory free on GPU host pinned memmory */
    extern "C"  ablas_status
    ablas_free_host( void *ptr ){
        return hipFreeHost(ptr);    
    };

    /*! \brief  host-synchronous, supports memory from host to device, device to host, device to device and host to host */
    extern "C"  ablas_status
    ablas_memcpy_host_to_host(void *          dst,
                              const void *    src,
                              size_t          sizeBytes){
    
        hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToHost);
    }     

    extern "C"  ablas_status
    ablas_memcpy_host_to_device(void *          dst,
                                const void *    src,
                                size_t          sizeBytes){
    
        hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToDevice);
    }  

    extern "C"  ablas_status
    ablas_memcpy_device_to_host(void *          dst,
                                const void *    src,
                                size_t          sizeBytes){
    
        hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToHost);
    }  

    extern "C"  ablas_status
    ablas_memcpy_device_to_device(void *          dst,
                                  const void *    src,
                                  size_t          sizeBytes){
    
        hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice);
    }  


    /*! \brief   src to dst asynchronously, supports memory from host to device, device to host, device to device and host to host */
    extern "C"  ablas_status
    ablas_memcpy_async_host_to_host(void *           dst,
                                    const void *     src,
                                    size_t           sizeBytes,
                                    ablas_queue stream){
    
        hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyHostToHost, stream);
    }     

    extern "C"  ablas_status
    ablas_memcpy_async_host_to_device(void *           dst,
                                      const void *     src,
                                      size_t           sizeBytes,
                                      ablas_queue stream){
    
        hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream);
    }  

    extern "C"  ablas_status
    ablas_memcpy_async_device_to_host(void *           dst,
                                      const void *     src,
                                      size_t           sizeBytes,
                                      ablas_queue stream){
    
        hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream);
    }  

    extern "C"  ablas_status
    ablas_memcpy_async_device_to_device(void *           dst,
                                        const void *     src,
                                        size_t           sizeBytes,
                                        ablas_queue stream){
    
        hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream);
    }  

    /* ============================================================================================ */

    /* device management */

     /*! \brief  Blocks until the default device has completed all preceding requested tasks. */
    extern "C"  ablas_status ablas_device_synchronize(void){
        return hipDeviceSynchronize();
    }

    /*! \brief  Destroy all resources and reset all state on the default device in the current process. */
    extern "C"  ablas_status ablas_device_reset(void){ 
        return hipDeviceReset();
    }

    /*! \brief  Set default device to be used for subsequent hip API calls from this thread. */
     extern "C"  ablas_status ablas_set_device(ablas_int device){
        return hipSetDevice (device);
    }
     
    /*! \brief  Return the default device id for the calling host thread. */
    extern "C"  ablas_status ablas_get_device(ablas_int *device){
        return hipGetDevice (device);
    }

    /*! \brief  Return number of compute-capable devices. */
    extern "C"  ablas_status ablas_get_device_count(ablas_int *count){
        return hipGetDeviceCount (count);
     }
 
    /*! \brief  Returns device properties. */
    extern "C"  ablas_status ablas_device_get_properties(hipDeviceProp_t *prop, ablas_int device){
        return hipDeviceGetProperties (prop, device);
     }
    
     /*! \brief  Set L1/Shared cache partition. */
     extern "C"  ablas_status ablas_device_set_cache_config(hipFuncCache cacheConfig){
        return hipDeviceSetCacheConfig(cacheConfig);
    }
 
    /*! \brief  Set Cache configuration for a specific function. */
    extern "C"  ablas_status ablas_device_get_cache_config(hipFuncCache *cacheConfig){
        return hipDeviceGetCacheConfig (cacheConfig);
    }
 
    /*! \brief  Set Cache configuration for a specific function. */
    extern "C"  ablas_status ablas_func_set_cache_config(hipFuncCache config){
        return hipFuncSetCacheConfig (config);
     }
    
     /*! \brief  Get Shared memory bank configuration. */
    extern "C"  ablas_status ablas_device_get_sharedMem_config (hipSharedMemConfig *pConfig){
        return hipDeviceGetSharedMemConfig(pConfig);
    }

    /*! \brief  Set Shared memory bank configuration. */
    extern "C"  ablas_status ablas_device_set_sharedMem_config(hipSharedMemConfig config){
        return     hipDeviceSetSharedMemConfig (config)
    }

    /* ============================================================================================ */
    /*  query device :*/
    extern "C"  void ablas_query_device()
    {
        int num_device, device_id=0;
        ablas_get_device_count(&num_device);
        ablas_set_device(device_id);

        printf("There are %d GPU devices; running on device ID %d \n", num_device, device_id);
    }

    /* ============================================================================================ */

    /*   stream management */

    /*! \brief  Create an asynchronous stream. */
    extern "C"  ablas_status ablas_stream_create_withflags(ablas_queue *stream, unsigned ablas_int flags){
        return hipStreamCreateWithFlags(stream, flags); 
    }
     
    /*! \brief  Make the specified compute stream wait for an event. */
    extern "C"  ablas_status ablas_stream_wait_event(ablas_queue stream, ablas_event event, unsigned ablas_int flags){
        return hipStreamWaitEvent(stream, event, flags);
    } 
     
     /*! \brief  Wait for all commands in stream to complete. */
    extern "C"  ablas_status ablas_stream_synchronize(ablas_queue stream){
        return     hipStreamSynchronize (stream);
     }
 
    /*! \brief  Destroys the specified stream. */
    extern "C"  ablas_status ablas_stream_destroy(ablas_queue stream){
        return hipStreamDestroy(stream);
    } 
      
     /*! \brief  Return flags associated with this stream. */
    extern "C"  ablas_status ablas_stream_get_flags(ablas_queue stream, unsigned ablas_int *flags){
        return hipStreamGetFlags(stream, flags);
    } 


