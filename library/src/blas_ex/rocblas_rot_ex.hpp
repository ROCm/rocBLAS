/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_rot.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <rocblas_int NB, bool ISBATCHED = false>
rocblas_status rocblas_rot_ex_template(rocblas_handle   handle,
                                       rocblas_int      n,
                                       void*            x,
                                       rocblas_datatype x_type,
                                       rocblas_int      incx,
                                       rocblas_stride   stride_x,
                                       void*            y,
                                       rocblas_datatype y_type,
                                       rocblas_int      incy,
                                       rocblas_stride   stride_y,
                                       const void*      c,
                                       const void*      s,
                                       rocblas_datatype cs_type,
                                       rocblas_int      batch_count,
                                       rocblas_datatype execution_type);
