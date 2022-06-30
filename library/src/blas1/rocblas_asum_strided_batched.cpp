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

#include "rocblas_asum_strided_batched.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_reduction_impl.hpp"

namespace
{

    template <typename>
    constexpr char rocblas_asum_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_asum_strided_batched_name<float>[] = "rocblas_sasum_strided_batched";
    template <>
    constexpr char rocblas_asum_strided_batched_name<double>[] = "rocblas_dasum_strided_batched";
    template <>
    constexpr char rocblas_asum_strided_batched_name<rocblas_float_complex>[]
        = "rocblas_scasum_strided_batched";
    template <>
    constexpr char rocblas_asum_strided_batched_name<rocblas_double_complex>[]
        = "rocblas_dzasum_strided_batched";

    // allocate workspace inside this API
    template <rocblas_int NB, typename Ti, typename To>
    rocblas_status rocblas_asum_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     const Ti*      x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     rocblas_int    batch_count,
                                                     To*            results)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        static constexpr bool isbatched = true;

        return rocblas_reduction_impl<NB,
                                      isbatched,
                                      rocblas_fetch_asum<To>,
                                      rocblas_reduce_sum,
                                      rocblas_finalize_identity,
                                      To>(handle,
                                          n,
                                          x,
                                          incx,
                                          stridex,
                                          batch_count,
                                          results,
                                          rocblas_asum_strided_batched_name<Ti>,
                                          "asum_strided_batched");
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

#ifdef IMPL
#error IMPL IS ALREADY DEFINED
#endif

#define IMPL(name_, typei_, typeo_)                                \
    rocblas_status name_(rocblas_handle handle,                    \
                         rocblas_int    n,                         \
                         const typei_*  x,                         \
                         rocblas_int    incx,                      \
                         rocblas_stride stridex,                   \
                         rocblas_int    batch_count,               \
                         typeo_*        results)                   \
    try                                                            \
    {                                                              \
        return rocblas_asum_strided_batched_impl<ROCBLAS_ASUM_NB>( \
            handle, n, x, incx, stridex, batch_count, results);    \
    }                                                              \
    catch(...)                                                     \
    {                                                              \
        return exception_to_rocblas_status();                      \
    }

IMPL(rocblas_sasum_strided_batched, float, float);
IMPL(rocblas_dasum_strided_batched, double, double);
IMPL(rocblas_scasum_strided_batched, rocblas_float_complex, float);
IMPL(rocblas_dzasum_strided_batched, rocblas_double_complex, double);

#undef IMPL

} // extern "C"
