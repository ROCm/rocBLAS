/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "trmv_template.hpp"

template <typename A, typename X, typename W>
rocblas_status rocblas_trmv_strided_batched_template(rocblas_handle    handle,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation transa,
                                                     rocblas_diagonal  diag,
                                                     rocblas_int       m,
                                                     A                 a,
                                                     rocblas_int       lda,
                                                     rocblas_stride    stridea,
                                                     X                 x,
                                                     rocblas_int       incx,
                                                     rocblas_stride    stridex,
                                                     W                 w,
                                                     rocblas_stride    stridew,
                                                     rocblas_int       batch_count)
{
    static constexpr rocblas_int NB      = 512;
    static constexpr ptrdiff_t   offseta = 0;
    static constexpr ptrdiff_t   offsetx = 0;
    return rocblas_trmv_template<NB>(handle,
                                     uplo,
                                     transa,
                                     diag,
                                     m,
                                     a,
                                     offseta,
                                     lda,
                                     stridea,
                                     x,
                                     offsetx,
                                     incx,
                                     stridex,
                                     w,
                                     stridew,
                                     batch_count);
}
