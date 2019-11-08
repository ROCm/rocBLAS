/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "axpy_template.hpp"

template <int NB, typename T>
static rocblas_status rocblas_axpy_template(rocblas_handle handle,
                                            rocblas_int    n,
                                            const T*       alpha,
                                            const T*       x,
                                            rocblas_int    incx,
                                            T*             y,
                                            rocblas_int    incy)
{
    static constexpr rocblas_stride stride_0      = 0;
    static constexpr rocblas_int    batch_count_1 = 1;
    return axpy_template<NB>(handle, n, alpha, x, incx, stride_0, y, incy, stride_0, batch_count_1);
}
