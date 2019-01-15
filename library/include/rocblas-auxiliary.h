/* ************************************************************************
 * Copyright 2016-2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_AUXILIARY_H_
#define _ROCBLAS_AUXILIARY_H_

#include <hip/hip_runtime_api.h>
#include "rocblas-types.h"

/*!\file
 * \brief rocblas-auxiliary.h provides auxilary functions in rocblas
*/

/* ============================================================================================ */
/*! \brief  indicates whether the pointer is on the host or device. currently HIP API can only
recoginize the input ptr on deive or not
can not recoginize it is on host or not */
ROCBLAS_EXPORT rocblas_pointer_mode rocblas_pointer_to_mode(void* ptr);

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocblas_destroy_handle().
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_create_handle(rocblas_handle* handle);

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_destroy_handle(rocblas_handle handle);

/********************************************************************************
 * \brief add stream to handle
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_add_stream(rocblas_handle handle, hipStream_t stream);

/********************************************************************************
 * \brief remove any streams from handle, and add one
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream);

/********************************************************************************
 * \brief get stream [0] from handle
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t* stream);

/********************************************************************************
 * \brief set rocblas_pointer_mode
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_set_pointer_mode(rocblas_handle handle,
                                                       rocblas_pointer_mode pointer_mode);

/********************************************************************************
 * \brief get rocblas_pointer_mode
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_pointer_mode(rocblas_handle handle,
                                                       rocblas_pointer_mode* pointer_mode);

/********************************************************************************
 * \brief copy vector from host to device
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_set_vector(rocblas_int n,
                                                 rocblas_int elem_size,
                                                 const void* x,
                                                 rocblas_int incx,
                                                 void* y,
                                                 rocblas_int incy);

/********************************************************************************
 * \brief copy vector from device to host
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_vector(rocblas_int n,
                                                 rocblas_int elem_size,
                                                 const void* x,
                                                 rocblas_int incx,
                                                 void* y,
                                                 rocblas_int incy);

/********************************************************************************
 * \brief copy matrix from host to device
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_set_matrix(rocblas_int rows,
                                                 rocblas_int cols,
                                                 rocblas_int elem_size,
                                                 const void* a,
                                                 rocblas_int lda,
                                                 void* b,
                                                 rocblas_int ldb);

/********************************************************************************
 * \brief copy matrix from device to host
 *******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_matrix(rocblas_int rows,
                                                 rocblas_int cols,
                                                 rocblas_int elem_size,
                                                 const void* a,
                                                 rocblas_int lda,
                                                 void* b,
                                                 rocblas_int ldb);

#ifdef __cplusplus
}

namespace rocblas {
ROCBLAS_EXPORT void reinit_logs(); // For testing only
}
#endif

#endif /* _ROCBLAS_AUXILIARY_H_ */
