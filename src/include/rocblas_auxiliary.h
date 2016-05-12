/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_AUXILIARY_H_
#define _ROCBLAS_AUXILIARY_H_

#include <rocblas_types.h>

/*!\file
 * \brief rocblas_auxiliary.h provides auxilary functions in rocblas
*/



#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
      exit(EXIT_FAILURE);\
    }


    /* ============================================================================================ */
    /*! \brief  indicates whether the pointer is on the host or device. currently HIP API can only recoginize the input ptr on deive or not
    can not recoginize it is on host or not */
    rocblas_mem_location rocblas_get_pointer_location(void *ptr);

#ifdef __cplusplus
extern "C" {
#endif

    /*! \brief   create rocblas handle called before any rocblas library routines*/
    rocblas_status rocblas_create(rocblas_handle *handle);

    /*! \brief   release rocblas handle, will implicitly synchronize host and device */
    rocblas_status rocblas_destroy(rocblas_handle handle);


#endif  /* _ROCBLAS_AUXILIARY_H_ */
