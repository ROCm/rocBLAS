
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

  // create a default stream for active device
  hipStream_t stream;
  THROW_IF_HIP_ERROR( hipStreamCreate(&stream) );
  streams.push_back(stream);

  // cobalt device profile
  // TODO - get device name via hip runtime
  cobaltDeviceProfile = cobaltCreateEmptyDeviceProfile();
  cobaltDeviceProfile.devices[0].name = "Fiji";
  cobaltDeviceProfile.numDevices = 1;


  // cobalt control
  cobaltControl = cobaltCreateEmptyControl();
  cobaltControl.queues[0] = streams[0];
  cobaltControl.numQueues = 1;

}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocblas_handle::~_rocblas_handle() {
  // destroy streams
  while( !streams.empty() ) {
    hipStream_t stream = static_cast<hipStream_t>( streams.pop_back() );
    THROW_IF_HIP_ERROR( hipStreamDestroy(stream) );
  }
}

/*******************************************************************************
 * add stream
 ******************************************************************************/
rocblas_status _rocblas_handle::add_stream( hipStream_t stream ) {
  streams.push_back(stream);
  cobaltControl.queues[cobaltControl.numQueues] = stream;
  cobaltControl.numQueues++;
  return rocblas_status_success;
}


/*******************************************************************************
 * set stream
 ******************************************************************************/
rocblas_status _rocblas_handle::set_stream( hipStream_t stream ) {
  // empty stream list
  while( !streams.empty() ) {
    RETURN_IF_HIP_ERROR( hipStreamDestroy( streams.pop_back() ) );
  }
  // add new stream
  streams.push_back(stream);
  cobaltControl.queues[0] = stream;
  cobaltControl.numQueues = 1;
  return rocblas_status_success;
}


/*******************************************************************************
 * get stream
 ******************************************************************************/
rocblas_status _rocblas_handle::get_stream( hipStream_t *stream ) const {
  *stream = streams[0];
  return rocblas_status_success;
}
