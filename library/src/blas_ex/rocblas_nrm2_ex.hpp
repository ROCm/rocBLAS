/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "../blas1/rocblas_nrm2.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "utility.hpp"

template <int NB, bool ISBATCHED, typename Tx, typename Tr = Tx, typename Tex = Tr>
rocblas_status nrm2_ex_typecasting(rocblas_handle handle,
                                   rocblas_int    n,
                                   const void*    x,
                                   rocblas_stride shiftx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   rocblas_int    batch_count,
                                   void*          results,
                                   void*          workspace)
{
    if(ISBATCHED)
    {
        return rocblas_internal_nrm2_template<NB, ISBATCHED>(handle,
                                                             n,
                                                             (const Tx* const*)x,
                                                             shiftx,
                                                             incx,
                                                             stridex,
                                                             batch_count,
                                                             (Tr*)results,
                                                             (Tex*)workspace);
    }
    else
    {
        return rocblas_internal_nrm2_template<NB, ISBATCHED>(handle,
                                                             n,
                                                             (const Tx*)x,
                                                             shiftx,
                                                             incx,
                                                             stridex,
                                                             batch_count,
                                                             (Tr*)results,
                                                             (Tex*)workspace);
    }
}

template <rocblas_int NB, bool ISBATCHED>
rocblas_status rocblas_nrm2_ex_template(rocblas_handle   handle,
                                        rocblas_int      n,
                                        const void*      x,
                                        rocblas_datatype x_type,
                                        rocblas_stride   shiftx,
                                        rocblas_int      incx,
                                        rocblas_stride   stridex,
                                        rocblas_int      batch_count,
                                        void*            results,
                                        rocblas_datatype result_type,
                                        rocblas_datatype execution_type,
                                        void*            workspace)
{
#define NRM2_EX_TYPECASTING_PARAM \
    handle, n, x, shiftx, incx, stridex, batch_count, results, workspace

    if(x_type == rocblas_datatype_f16_r && result_type == rocblas_datatype_f16_r
       && execution_type == rocblas_datatype_f32_r)
    {
        return nrm2_ex_typecasting<NB, ISBATCHED, rocblas_half, rocblas_half, float>(
            NRM2_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_r && result_type == rocblas_datatype_f32_r
            && execution_type == rocblas_datatype_f32_r)
    {
        return nrm2_ex_typecasting<NB, ISBATCHED, float>(NRM2_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_r && result_type == rocblas_datatype_f64_r
            && execution_type == rocblas_datatype_f64_r)
    {
        return nrm2_ex_typecasting<NB, ISBATCHED, double>(NRM2_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_c && result_type == rocblas_datatype_f32_r
            && execution_type == rocblas_datatype_f32_r)
    {
        return nrm2_ex_typecasting<NB, ISBATCHED, rocblas_float_complex, float>(
            NRM2_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_c && result_type == rocblas_datatype_f64_r
            && execution_type == rocblas_datatype_f64_r)
    {
        return nrm2_ex_typecasting<NB, ISBATCHED, rocblas_double_complex, double>(
            NRM2_EX_TYPECASTING_PARAM);
    }

    return rocblas_status_not_implemented;
}
