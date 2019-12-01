/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "axpy_template.hpp"

template <int NB, typename T>
static rocblas_status rocblas_axpy_strided_batched_template(rocblas_handle handle,
                                                            rocblas_int    n,
                                                            const T*       alpha,
                                                            const T*       x,
                                                            rocblas_int    incx,
                                                            rocblas_stride stridex,
                                                            T*             y,
                                                            rocblas_int    incy,
                                                            rocblas_stride stridey,
                                                            rocblas_int    batch_count)
{
    return axpy_template<NB>(handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count);
}
