#ifndef HANDLE_H
#define HANDLE_H
#include <hip/hip_runtime_api.h>
#include <vector>

#include "rocblas.h"

#if BUILD_WITH_COBALT
    #include "Cobalt.h"
#endif

/*******************************************************************************
 * \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle() and the returned handle mus
 * It should be destroyed at the end using rocblas_destroy_handle().
 * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
******************************************************************************/
struct _rocblas_handle{

  _rocblas_handle();
  ~_rocblas_handle();

  rocblas_status set_stream( hipStream_t stream );
  rocblas_status get_stream( hipStream_t *stream ) const;

  rocblas_int device;
  hipDeviceProp_t device_properties;

  // rocblas by default take the system default stream 0 users cannot create
  hipStream_t rocblas_stream = 0; 

#if BUILD_WITH_COBALT
  /*****************************************************************************
   * \brief Cobalt Device Profile
   * describes device to which this control is assigned so
   * Cobalt can lookup optimal solution
   ****************************************************************************/
  CobaltDeviceProfile cobalt_device_profile;

  /*****************************************************************************
   * \brief Cobalt Control
   * for passing control state (stream) to Cobalt
   ****************************************************************************/
  CobaltControl cobalt_control;
#endif

};

#endif
