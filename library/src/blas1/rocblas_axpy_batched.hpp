/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "axpy_template.hpp"

template <int NB, typename T>
static rocblas_status rocblas_axpy_batched_template(rocblas_handle  handle,
                                                    rocblas_int     n,
                                                    const T*        alpha,
                                                    const T* const* x,
                                                    rocblas_int     incx,
                                                    T* const*       y,
                                                    rocblas_int     incy,
                                                    rocblas_int     batch_count)
{
    static constexpr rocblas_stride stride_0 = 0;
    return axpy_template<NB>(handle, n, alpha, x, incx, stride_0, y, incy, stride_0, batch_count);
}
