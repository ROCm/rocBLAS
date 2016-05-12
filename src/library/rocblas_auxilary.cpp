/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <hip_runtime.h>
#include "rocblas_types.h"
#include "rocblas_hip.h"

    /* ============================================================================================ */

    /*! \brief  indicates whether the pointer is on the host or device. currently HIP API can only recoginize the input ptr on deive or not
    can not recoginize it is on host or not */
    rocblas_pointer_type rocblas_get_pointer_type(void *ptr){
        hipPointerAttribute_t attribute;
        hipPointerGetAttributes(&attribute, ptr);
        //if( rocblas_success != (rocblas_status)hipPointerGetAttributes(&attribute, ptr) ){
        //    printf("failed to get the pointer type\n");
        //}
        if (ptr == attribute.devicePointer) {
            return DEVICE_POINTER;
        }
        else{
            return HOST_POINTER;
        }
    }


    /*! \brief   create rocblas handle called before any rocblas library routines*/
    extern "C"
    rocblas_status rocblas_create(rocblas_handle *handle){
        /* TODO */
        rocblas_int device;

        rocblas_status status;
        status = (rocblas_status)hipGetDevice(&device);// return the active device

        if (status != rocblas_success) {
            return status;
        }

        handle->device_id = device;
        return rocblas_success;
    }

    /*! \brief   release rocblas handle, will implicitly synchronize host and device */
    extern "C"
    rocblas_status rocblas_destroy(rocblas_handle handle){
        /* TODO */

        return rocblas_success;
    }


    /*! \brief   set rocblas stream used for all subsequent library function calls.
     *   If not set, all hip kernels will take the default NULL stream. stream_id must be created before this call */
    extern "C"
    rocblas_status
    rocblas_set_stream(rocblas_handle handle, hipStream_t stream_id){
        handle.stream = stream_id;
        return rocblas_success;
    }


    /*! \brief   get rocblas stream used for all subsequent library function calls.
     *   If not set, all hip kernels will take the default NULL stream. */
    extern "C"
    rocblas_status
    rocblas_get_stream(rocblas_handle handle, hipStream_t *stream_id){
        *stream_id = handle.stream;
        return rocblas_success;
    }
