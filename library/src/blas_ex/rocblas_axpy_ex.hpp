/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_axpy.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <int NB, bool BATCHED = false>
rocblas_status rocblas_axpy_ex_template(const char*      name,
                                        rocblas_handle   handle,
                                        rocblas_int      n,
                                        const void*      alpha,
                                        rocblas_datatype alpha_type,
                                        rocblas_stride   stride_alpha,
                                        const void*      x,
                                        rocblas_datatype x_type,
                                        ptrdiff_t        offset_x,
                                        rocblas_int      incx,
                                        rocblas_stride   stride_x,
                                        void*            y,
                                        rocblas_datatype y_type,
                                        ptrdiff_t        offset_y,
                                        rocblas_int      incy,
                                        rocblas_stride   stride_y,
                                        rocblas_int      batch_count,
                                        rocblas_datatype execution_type);
