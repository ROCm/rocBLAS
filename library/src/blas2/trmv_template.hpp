/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "../blas1/rocblas_copy.hpp"
#include "trmv_device.hpp"

template <rocblas_int NB, typename A, typename X, typename W>
ROCBLAS_EXPORT_NOINLINE rocblas_status rocblas_trmv_template(rocblas_handle    handle,
                                                             rocblas_fill      uplo,
                                                             rocblas_operation transa,
                                                             rocblas_diagonal  diag,
                                                             rocblas_int       m,
                                                             A                 a,
                                                             ptrdiff_t         offseta,
                                                             rocblas_int       lda,
                                                             rocblas_stride    stridea,
                                                             X                 x,
                                                             ptrdiff_t         offsetx,
                                                             rocblas_int       incx,
                                                             rocblas_stride    stridex,
                                                             W                 w,
                                                             rocblas_stride    stridew,
                                                             rocblas_int       batch_count)
{
    //
    // quick return
    //
    if(!m || !batch_count)
    {
        return rocblas_status_success;
    }

    hipStream_t rocblas_stream = handle->rocblas_stream;

    ptrdiff_t shiftx = incx < 0 ? offsetx + ptrdiff_t(incx) * (1 - m) : offsetx;

    dim3 trmv_grid((m - 1) / NB + 1, batch_count);
    dim3 trmv_threads(NB);
    switch(transa)
    {
    case rocblas_operation_none:
    {
        hipLaunchKernelGGL((trmvn_kernel<NB>),
                           trmv_grid,
                           trmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
                           lda,
                           stridea,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           w,
                           stridew);
        break;
    }

    case rocblas_operation_transpose:
    {
        hipLaunchKernelGGL((trmvt_kernel<NB>),
                           trmv_grid,
                           trmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
                           lda,
                           stridea,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           w,
                           stridew);
        break;
    }

    case rocblas_operation_conjugate_transpose:
    {
        hipLaunchKernelGGL((trmvc_kernel<NB>),
                           trmv_grid,
                           trmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
                           lda,
                           stridea,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           w,
                           stridew);

        break;
    }
    }

    //
    // Copy workspace to x.
    //
    {
        static constexpr rocblas_int offsetw = 0;
        static constexpr rocblas_int incw    = 1;
        return rocblas_copy_template<false, NB>(
            handle, m, w, offsetw, incw, stridew, x, offsetx, incx, stridex, batch_count);
    }
}
