/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_scal.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <int NB, bool BATCHED = false>
rocblas_status rocblas_scal_ex_template(rocblas_handle   handle,
                                        rocblas_int      n,
                                        const void*      alpha,
                                        rocblas_datatype alpha_type,
                                        void*            x,
                                        rocblas_datatype x_type,
                                        rocblas_int      incx,
                                        rocblas_stride   stride_x,
                                        rocblas_int      batch_count,
                                        rocblas_datatype execution_type);
