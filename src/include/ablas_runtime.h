/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#pragma once
#ifndef _ABLAS_RUNTIME_H_
#define _ABLAS_RUNTIME_H_

#include "ablas_types.h" 
#include <hip_runtime.h> 
#include <sys/time.h> 

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

    /* ============================================================================================ */

    /* Error Handling */

    //warning: the four error handling API only get and return recent HIP API error 
    //ablas_status is a superset of hip error, non-hip-runtime error won't be detected and reported by them 

    /*! \brief  Return last error returned by any HIP runtime API call and resets the stored error code to ablas_success. */
    ablas_status ablas_get_last_error( void ){
		return hipGetLastError();
	}

    /*! \brief  Return last error returned by any HIP runtime API call. */
    ablas_status ablas_peek_at_last_error ( void ){
		return hipPeekAtLastError();	
	} 	
	
 	/*! \brief Return name of the specified error code in text form. */
    const char* ablas_get_error_name(ablas_status ablas_error)
	{
		return hipGetErrorName (ablas_error);
	}

  	/*! \brief Return handy text string message to explain the error which occurred. On HCC, it is the same as ablas_get_error_name() */
    const char* ablas_get_error_string(ablas_status ablas_error)
	{
		return hipGetErrorString (ablas_error);
	}

    #define CHECK_ABLAS_ERROR(error) \
    if (error != ablas_success) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", ablas_get_error_string(error), error,__FILE__, __LINE__); \
    	exit(EXIT_FAILURE);\
    }

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

    /*! \brief   memory free on GPU devie memory */
    ablas_status
    ablas_free_device(void *ptr ){
        return hipFree(ptr);    
    };

    /*! \brief   memory free on GPU host pinned memmory */
    ablas_status
    ablas_free_host( void *ptr ){
        return hipFreeHost(ptr);    
    };

    /*! \brief  host-synchronous, supports memory from host to device, device to host, device to device and host to host */
    ablas_status
    ablas_memcpy_host_to_host(void *          dst,
                              const void *    src,
                              size_t          sizeBytes){
    
        hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToHost);
    }     

    ablas_status
    ablas_memcpy_host_to_device(void *          dst,
                                const void *    src,
                                size_t          sizeBytes){
    
        hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToDevice);
    }  

    ablas_status
    ablas_memcpy_device_to_host(void *          dst,
                                const void *    src,
                                size_t          sizeBytes){
    
        hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToHost);
    }  

    ablas_status
    ablas_memcpy_device_to_device(void *          dst,
                                  const void *    src,
                                  size_t          sizeBytes){
    
        hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice);
    }  


    /*! \brief   src to dst asynchronously, supports memory from host to device, device to host, device to device and host to host */
    ablas_status
    ablas_memcpy_async_host_to_host(void *           dst,
                                    const void *     src,
                                    size_t           sizeBytes,
                                    ablas_queue stream=0){
    
        hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyHostToHost, stream);
    }     

    ablas_status
    ablas_memcpy_async_host_to_device(void *           dst,
                                      const void *     src,
                                      size_t           sizeBytes,
                                      ablas_queue stream=0){
    
        hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream);
    }  

    ablas_status
    ablas_memcpy_async_device_to_host(void *           dst,
                                      const void *     src,
                                      size_t           sizeBytes,
                                      ablas_queue stream=0){
    
        hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream);
    }  

    ablas_status
    ablas_memcpy_async_device_to_device(void *           dst,
                                        const void *     src,
                                        size_t           sizeBytes,
                                        ablas_queue stream=0){
    
        hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream);
    }  

    /* ============================================================================================ */

    /* device management */

     /*! \brief  Blocks until the default device has completed all preceding requested tasks. */
    ablas_status ablas_device_synchronize(void){
        return hipDeviceSynchronize();
    }

    /*! \brief  Destroy all resources and reset all state on the default device in the current process. */
    ablas_status ablas_device_reset(void){ 
        return hipDeviceReset();
    }

    /*! \brief  Set default device to be used for subsequent hip API calls from this thread. */
     ablas_status ablas_set_device(int device){
        return hipSetDevice (device);
    }
     
    /*! \brief  Return the default device id for the calling host thread. */
    ablas_status ablas_get_device(int *device){
        return hipGetDevice (device);
    }

    /*! \brief  Return number of compute-capable devices. */
    ablas_status ablas_get_device_count(int *count){
        return hipGetDeviceCount (count);
     }
 
    /*! \brief  Returns device properties. */
    ablas_status ablas_device_get_properties(hipDeviceProp_t *prop, int device){
        return hipDeviceGetProperties (prop, device);
     }
    
     /*! \brief  Set L1/Shared cache partition. */
     ablas_status ablas_device_set_cache_config(hipFuncCache cacheConfig){
        return hipDeviceSetCacheConfig(cacheConfig);
    }
 
    /*! \brief  Set Cache configuration for a specific function. */
    ablas_status ablas_device_get_cache_config(hipFuncCache *cacheConfig){
        return hipDeviceGetCacheConfig (cacheConfig);
    }
 
    /*! \brief  Set Cache configuration for a specific function. */
    ablas_status ablas_func_set_cache_config(hipFuncCache config){
        return hipFuncSetCacheConfig (config);
     }
    
     /*! \brief  Get Shared memory bank configuration. */
     ablas_status ablas_device_get_sharedMem_config (hipSharedMemConfig *pConfig)
        return hipDeviceGetSharedMemConfig(pConfig);
    }

    /*! \brief  Set Shared memory bank configuration. */
    ablas_status ablas_device_set_sharedMem_config(hipSharedMemConfig config){
        return     hipDeviceSetSharedMemConfig (config)
    }

    /* ============================================================================================ */

    /*   stream management */

    /*! \brief  Create an asynchronous stream. */
    ablas_status ablas_stream_create_withflags(ablas_queue *stream, unsigned int flags){
        return hipStreamCreateWithFlags(stream, flags); 
    }
     
    /*! \brief  Make the specified compute stream wait for an event. */
    ablas_status ablas_stream_wait_event(ablas_queue stream, ablas_event event, unsigned int flags){
        return hipStreamWaitEvent(stream, event, flags);
    } 
     
     /*! \brief  Wait for all commands in stream to complete. */
    ablas_status ablas_stream_synchronize(ablas_queue stream){
        return     hipStreamSynchronize (stream);
     }
 
    /*! \brief  Destroys the specified stream. */
    ablas_status ablas_stream_destroy(ablas_queue stream){
        return hipStreamDestroy(stream);
    } 
      
     /*! \brief  Return flags associated with this stream. */
    ablas_status ablas_stream_get_flags(ablas_queue stream, unsigned int *flags){
        return hipStreamGetFlags(stream, flags);
    } 



#endif

