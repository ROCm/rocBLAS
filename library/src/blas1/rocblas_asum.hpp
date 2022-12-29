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

#include "fetch_template.hpp"
#include "rocblas_reduction_setup.hpp"

template <class To>
struct rocblas_fetch_asum
{
    template <typename Ti>
    __forceinline__ __device__ To operator()(Ti x) const
    {
        return {fetch_asum(x)};
    }
};

template <rocblas_int NB,
          typename FETCH,
          typename FINALIZE,
          typename TPtrX,
          typename To,
          typename Tr>
rocblas_status rocblas_reduction_template(rocblas_handle handle,
                                          rocblas_int    n,
                                          TPtrX          x,
                                          rocblas_stride shiftx,
                                          rocblas_int    incx,
                                          rocblas_stride stridex,
                                          rocblas_int    batch_count,
                                          To*            workspace,
                                          Tr*            result);

// allocate workspace inside this API
template <rocblas_int NB,
          bool        ISBATCHED,
          typename FETCH,
          typename REDUCE,
          typename FINALIZE,
          typename Tw,
          typename U,
          typename Tr>
rocblas_status rocblas_asum_template(rocblas_handle handle,
                                     rocblas_int    n,
                                     U              x,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     rocblas_int    batch_count,
                                     Tr*            results,
                                     const char*    name,
                                     const char*    name_bench)
{
    size_t         dev_bytes     = 0;
    rocblas_status checks_status = rocblas_reduction_setup<NB, ISBATCHED, Tw>(
        handle, n, x, incx, stridex, batch_count, results, name, name_bench, dev_bytes);

    if(checks_status != rocblas_status_continue)
    {
        return checks_status;
    }

    auto check_numerics = handle->check_numerics;

    if(check_numerics)
    {
        bool           is_input              = true;
        rocblas_status check_numerics_status = rocblas_internal_check_numerics_vector_template(
            name, handle, n, x, 0, incx, stridex, batch_count, check_numerics, is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    auto w_mem = handle->device_malloc(dev_bytes);
    if(!w_mem)
    {
        return rocblas_status_memory_error;
    }

    static constexpr rocblas_stride shiftx_0 = 0;
    rocblas_status                  status   = rocblas_reduction_template<NB, FETCH, FINALIZE>(
        handle, n, x, shiftx_0, incx, stridex, batch_count, (Tw*)w_mem, results);
    if(status != rocblas_status_success)
        return status;

    if(check_numerics)
    {
        bool           is_input              = false;
        rocblas_status check_numerics_status = rocblas_internal_check_numerics_vector_template(
            name, handle, n, x, 0, incx, stridex, batch_count, check_numerics, is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }
    return status;
}
