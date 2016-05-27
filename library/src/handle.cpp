
#include "definitions.h"
#include "status.h"
#include "handle.h"
#include "hip_runtime.h"

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocblas_handle::_rocblas_handle() {

  // default device is active device
  THROW_IF_HIP_ERROR( hipGetDevice(&device) );
  THROW_IF_HIP_ERROR( hipGetDeviceProperties(&device_properties, device) );

  // create a default stream for active device
  // TODO - how do we cast null stream as stream type?
  hipStream_t stream;
  THROW_IF_HIP_ERROR( hipStreamCreate(&stream) );
  streams.push_back(stream);

/*
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
  cobalt_control.queues[0] = streams[0];
  cobalt_control.numQueues = 1;
*/
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle() {
  // destroy streams
  /*
   * TODO put teardown back in; was having compiler errors
  while( !streams.empty() ) {
    hipStream_t stream = static_cast<hipStream_t>( streams.pop_back() );
    THROW_IF_HIP_ERROR( hipStreamDestroy(stream) );
  }
  */
}

/*******************************************************************************
 * add stream
 ******************************************************************************/
rocblas_status _rocblas_handle::add_stream( hipStream_t stream ) {
  streams.push_back(stream);
  cobalt_control.queues[cobalt_control.numQueues] = stream;
  cobalt_control.numQueues++;
  return rocblas_status_success;
}


/*******************************************************************************
 * set stream
 ******************************************************************************/
rocblas_status _rocblas_handle::set_stream( hipStream_t stream ) {
  // empty stream list
  /*
  // TODO add back in
  while( !streams.empty() ) {
    RETURN_IF_HIP_ERROR( hipStreamDestroy( streams.pop_back() ) );
  }
  */
  // add new stream
  streams.push_back(stream);
  cobalt_control.queues[0] = stream;
  cobalt_control.numQueues = 1;
  // TODO stream may point to new device
  // need to re-initialize device and cobalt info
  return rocblas_status_success;
}


/*******************************************************************************
 * get stream
 ******************************************************************************/
rocblas_status _rocblas_handle::get_stream( hipStream_t *stream ) const {
  *stream = streams[0];
  return rocblas_status_success;
}
