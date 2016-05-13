#ifndef HANDLE_H
#define HANDLE_H
#include "rocblas.h"
#include "Cobalt.h"
#include "hip_runtime.h"

/*******************************************************************************
 * \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create() and the returned handle mus
 * It should be destroyed at the end using rocblas_destroy().
******************************************************************************/
struct _rocblas_handle{

  _rocblas_handle();
  ~_rocblas_handle();
  rocblas_status add_stream( hipStream_t stream );
  rocblas_status set_stream( hipStream_t stream );
  rocblas_status get_stream() const; // stream 0

  rocblas_int device;
  std::vector<hipStream_t> streams;

  /*****************************************************************************
   * \brief Cobalt Device Profile
   * describes device to which this control is assigned so
   * Cobalt can lookup optimal solution
   ****************************************************************************/
  CobaltDeviceProfile cobaltDeviceProfile;

  /*****************************************************************************
   * \brief Cobalt Control
   * for passing control state (stream) to Cobalt
   ****************************************************************************/
  CobaltControl cobaltControl;


};

#endif
