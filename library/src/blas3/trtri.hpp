/* ************************************************************************
 *  * Copyright 2016 Advanced Micro Devices, Inc.
 *   *
 *    * ************************************************************************ */

#pragma once
#ifndef _TRTRI_HPP_
#define _TRTRI_HPP_

#include <hip/hip_runtime.h>
#include "trtri_device.h"
#include "rocblas_trtri_batched.hpp"

/* ============================================================================================ */

namespace trtri {

template <typename T>
rocblas_status rocblas_trtri_template(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      const T* A,
                                      rocblas_int lda,
                                      T* invA,
                                      rocblas_int ldinvA)
{
    return rocblas_trtri_batched_template<T>(
        handle, uplo, diag, n, A, lda, lda * n, invA, ldinvA, ldinvA * n, 1);
}
}

#endif // _TRTRI_HPP_
