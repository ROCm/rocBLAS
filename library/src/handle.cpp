
#include "definitions.h"
#include "status.h"
#include "handle.h"
#include <hip/hip_runtime_api.h>

#if BUILD_WITH_COBALT
    #include "Cobalt.h"
#endif

/*******************************************************************************
 * constructor 
 ******************************************************************************/
_rocblas_handle::_rocblas_handle() {

  // default device is active device
  THROW_IF_HIP_ERROR( hipGetDevice(&device) );
  THROW_IF_HIP_ERROR( hipGetDeviceProperties(&device_properties, device) );

  // rocblas by default take the system default stream 0 users cannot create

#if BUILD_WITH_COBALT
  // cobalt device profile
  cobalt_device_profile = cobaltCreateEmptyDeviceProfile();
  if ( strlen(device_properties.name) > cobalt_device_profile.devices[0].maxNameLength) {
    strncpy( cobalt_device_profile.devices[0].name,
        device_properties.name, cobalt_device_profile.devices[0].maxNameLength);
    cobalt_device_profile.devices[0].name[cobalt_device_profile.devices[0].maxNameLength-1] = '\0';
  } else {
    strcpy( cobalt_device_profile.devices[0].name, device_properties.name);
  }
  cobalt_device_profile.numDevices = 1;

  // cobalt control
  cobalt_control = cobaltCreateEmptyControl();
  cobalt_control.queues[0] = rocblas_stream;
  cobalt_control.numQueues = 1;

#endif

}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle() {
  //rocblas by default take the system default stream which user cannot destory
}

/*******************************************************************************
 * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
 ******************************************************************************/


/*******************************************************************************
 * set stream: 
   This API assumes user has already created a valid stream
   Associate the following rocblas API call with this user provided stream 
 ******************************************************************************/
rocblas_status _rocblas_handle::set_stream( hipStream_t user_stream ) {

  //TODO: check the user_stream valid or not
  rocblas_stream = user_stream;
#if BUILD_WITH_COBALT
  cobalt_control.queues[0] = user_stream;
  cobalt_control.numQueues = 1;
  // It is impossible to switch stream to another device in rocblas without destroying the handle
#endif
  return rocblas_status_success;
}


/*******************************************************************************
 * get stream
 ******************************************************************************/
rocblas_status _rocblas_handle::get_stream( hipStream_t *stream ) const {
  *stream = rocblas_stream;
  return rocblas_status_success;
}
