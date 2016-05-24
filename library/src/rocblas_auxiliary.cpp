/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <hip_runtime.h>
#include "definitions.h"
#include "rocblas_types.h"
#include "handle.h"

    /* ============================================================================================ */

/*******************************************************************************
 * ! \brief  indicates whether the pointer is on the host or device.
 * currently HIP API can only recoginize the input ptr on deive or not
 *  can not recoginize it is on host or not
 ******************************************************************************/
rocblas_mem_location rocblas_get_pointer_location(void *ptr){
  hipPointerAttribute_t attribute;
  hipPointerGetAttributes(&attribute, ptr);
  if (ptr == attribute.devicePointer) {
    return rocblas_mem_location_device;
  } else {
    return rocblas_mem_location_host;
  }
}


/*******************************************************************************
 * ! \brief create rocblas handle called before any rocblas library routines
 ******************************************************************************/
extern "C"
rocblas_status rocblas_create_handle(rocblas_handle *handle){

  // if handle not valid
  if (handle == nullptr) {
    return rocblas_status_invalid_pointer;
  }

  // allocate on heap
  try {
    *handle = new _rocblas_handle();
  } catch (rocblas_status status) {
    return status;
  }

  return rocblas_status_success;
}


/*******************************************************************************
 *! \brief release rocblas handle, will implicitly synchronize host and device
 ******************************************************************************/
extern "C"
rocblas_status rocblas_destroy_handle(rocblas_handle handle){
  // call destructor
  try {
    delete handle;
  } catch (rocblas_status status) {
    return status;
  }
  return rocblas_status_success;
}


/*******************************************************************************
 *! \brief   set rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 *   stream_id must be created before this call
 ******************************************************************************/
extern "C"
rocblas_status
rocblas_set_stream(rocblas_handle handle, hipStream_t stream_id){
  return handle->set_stream( stream_id );
}


/*******************************************************************************
 *! \brief   get rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 ******************************************************************************/
extern "C"
rocblas_status
rocblas_get_stream(rocblas_handle handle, hipStream_t *stream_id){
  return handle->get_stream( stream_id );
}


/*******************************************************************************
 *! \brief  add stream to handle
 ******************************************************************************/
extern "C"
rocblas_status
rocblas_add_stream(rocblas_handle handle, hipStream_t stream_id ){
  return handle->add_stream( stream_id );
}

