/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "tpmv_template.hpp"

template <typename A, typename X, typename W>
rocblas_status rocblas_tpmv_batched_template(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transa,
                                             rocblas_diagonal  diag,
                                             rocblas_int       m,
                                             A                 a,
                                             X                 x,
                                             rocblas_int       incx,
                                             W                 w,
                                             rocblas_stride    stridew,
                                             rocblas_int       batch_count)
{
    static constexpr rocblas_int    NB      = 512;
    static constexpr rocblas_stride stridex = 0;
    static constexpr rocblas_stride stridea = 0;
    static constexpr ptrdiff_t      offseta = 0;
    static constexpr ptrdiff_t      offsetx = 0;

    return tpmv_template<NB>(handle,
                             uplo,
                             transa,
                             diag,
                             m,
                             a,
                             offseta,
                             stridea,
                             x,
                             offsetx,
                             incx,
                             stridex,
                             w,
                             stridew,
                             batch_count);
}
