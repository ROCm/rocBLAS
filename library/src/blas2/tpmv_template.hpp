/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"
#include "tpmv_device.hpp"

template <rocblas_int NB, typename A, typename X, typename W>
rocblas_status tpmv_template(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transa,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             A                 a,
                             ptrdiff_t         offseta,
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

    hipStream_t rocblas_stream = handle->get_stream();

    ptrdiff_t shiftx = incx < 0 ? offsetx + ptrdiff_t(incx) * (1 - m) : offsetx;

    dim3 tpmv_grid((m - 1) / NB + 1, batch_count);
    dim3 tpmv_threads(NB);

    switch(transa)
    {
    case rocblas_operation_none:
    {
        hipLaunchKernelGGL(tpmvn_kernel<NB>,
                           tpmv_grid,
                           tpmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
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
        hipLaunchKernelGGL(tpmvt_kernel<NB>,
                           tpmv_grid,
                           tpmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
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
        hipLaunchKernelGGL(tpmvc_kernel<NB>,
                           tpmv_grid,
                           tpmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
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

//TODO :-Add rocblas_check_numerics_tp_matrix_template for checking Matrix `A` which is a Triangular Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_tpmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_int    m,
                                           T              A,
                                           rocblas_int    offset_a,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_int    offset_x,
                                           rocblas_int    inc_x,
                                           rocblas_stride stride_x,
                                           rocblas_int    batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          m,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}
