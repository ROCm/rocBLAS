/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
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
    auto           check_numerics = handle->check_numerics;
    rocblas_status status         = rocblas_status_success;
    if(ISBATCHED)
    {
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status nrm2_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template("rocblas_nrm2_batched_ex",
                                                                  handle,
                                                                  n,
                                                                  (const Tx* const*)x,
                                                                  shiftx,
                                                                  incx,
                                                                  stridex,
                                                                  batch_count,
                                                                  check_numerics,
                                                                  is_input);
            if(nrm2_ex_check_numerics_status != rocblas_status_success)
                return nrm2_ex_check_numerics_status;
        }

        status = rocblas_internal_nrm2_template<NB, ISBATCHED>(handle,
                                                               n,
                                                               (const Tx* const*)x,
                                                               shiftx,
                                                               incx,
                                                               stridex,
                                                               batch_count,
                                                               (Tr*)results,
                                                               (Tex*)workspace);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status nrm2_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template("rocblas_nrm2_batched_ex",
                                                                  handle,
                                                                  n,
                                                                  (const Tx* const*)x,
                                                                  shiftx,
                                                                  incx,
                                                                  stridex,
                                                                  batch_count,
                                                                  check_numerics,
                                                                  is_input);
            if(nrm2_ex_check_numerics_status != rocblas_status_success)
                return nrm2_ex_check_numerics_status;
        }
    }
    else
    {
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status nrm2_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    stridex ? "rocblas_nrm2_strided_batched_ex" : "rocblas_nrm2_ex",
                    handle,
                    n,
                    (const Tx*)x,
                    shiftx,
                    incx,
                    stridex,
                    batch_count,
                    check_numerics,
                    is_input);
            if(nrm2_ex_check_numerics_status != rocblas_status_success)
                return nrm2_ex_check_numerics_status;
        }

        status = rocblas_internal_nrm2_template<NB, ISBATCHED>(handle,
                                                               n,
                                                               (const Tx*)x,
                                                               shiftx,
                                                               incx,
                                                               stridex,
                                                               batch_count,
                                                               (Tr*)results,
                                                               (Tex*)workspace);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status nrm2_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    stridex ? "rocblas_nrm2_strided_batched_ex" : "rocblas_nrm2_ex",
                    handle,
                    n,
                    (const Tx*)x,
                    shiftx,
                    incx,
                    stridex,
                    batch_count,
                    check_numerics,
                    is_input);
            if(nrm2_ex_check_numerics_status != rocblas_status_success)
                return nrm2_ex_check_numerics_status;
        }
    }
    return status;
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
