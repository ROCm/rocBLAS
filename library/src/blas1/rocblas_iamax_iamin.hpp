/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "handle.hpp"
#include "reduction.hpp"
#include "rocblas.h"
#include "utility.hpp"
#include <type_traits>
#include <utility>

//!
//! @brief Struct-operator a default_value of rocblas_index_value_t<T>
//!
template <typename T>
struct rocblas_default_value<rocblas_index_value_t<T>>
{
    __forceinline__ __host__ __device__ constexpr auto operator()() const
    {
        rocblas_index_value_t<T> x;
        x.index = -1;
        return x;
    }
};

//!
//! @brief Struct-operator to fetch absolute value
//!
template <typename To>
struct rocblas_fetch_amax_amin
{
    template <typename Ti>
    __forceinline__ __host__ __device__ rocblas_index_value_t<To>
                                        operator()(Ti x, rocblas_int index) const
    {
        return {index, fetch_asum(x)};
    }
};

//!
//! @brief Struct-operator to finalize the data.
//!
struct rocblas_finalize_amax_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ auto operator()(const rocblas_index_value_t<To>& x) const
    {
        return x.index + 1;
    }
};

// Replaces x with y if y.value > x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amax
{
    template <typename To>
    __forceinline__ __host__ __device__ void
        operator()(rocblas_index_value_t<To>& __restrict__ x,
                   const rocblas_index_value_t<To>& __restrict__ y) const
    {
        // If y.index == -1 then y.value is invalid and should not be compared
        if(y.index != -1)
        {
            if(x.index == -1 || y.value > x.value)
                x = y; // if larger or smaller, update max/min and index
            else if(y.index < x.index && x.value == y.value)
                x.index = y.index; // if equal, choose smaller index
        }
    }
};

// Replaces x with y if y.value < x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ void
        operator()(rocblas_index_value_t<To>& __restrict__ x,
                   const rocblas_index_value_t<To>& __restrict__ y) const
    {
        // If y.index == -1 then y.value is invalid and should not be compared
        if(y.index != -1)
        {
            if(x.index == -1 || y.value < x.value)
                x = y; // if larger or smaller, update max/min and index
            else if(y.index < x.index && x.value == y.value)
                x.index = y.index; // if equal, choose smaller index
        }
    }
};

template <rocblas_int NB,
          typename FETCH,
          typename REDUCE,
          typename FINALIZE,
          typename TPtrX,
          typename To,
          typename Tr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_iamin_template(rocblas_handle handle,
                                          rocblas_int    n,
                                          TPtrX          x,
                                          rocblas_stride shiftx,
                                          rocblas_int    incx,
                                          rocblas_stride stridex,
                                          rocblas_int    batch_count,
                                          To*            workspace,
                                          Tr*            result);

template <rocblas_int NB, typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_template(rocblas_handle            handle,
                                    rocblas_int               n,
                                    T                         x,
                                    rocblas_stride            shiftx,
                                    rocblas_int               incx,
                                    rocblas_stride            stridex,
                                    rocblas_int               batch_count,
                                    rocblas_int*              result,
                                    rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_template<NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amax,
                                                 rocblas_finalize_amax_amin>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}

template <rocblas_int NB, typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_template(rocblas_handle            handle,
                                    rocblas_int               n,
                                    T                         x,
                                    rocblas_stride            shiftx,
                                    rocblas_int               incx,
                                    rocblas_stride            stridex,
                                    rocblas_int               batch_count,
                                    rocblas_int*              result,
                                    rocblas_index_value_t<S>* workspace)
{
    return rocblas_internal_iamax_iamin_template<NB,
                                                 rocblas_fetch_amax_amin<S>,
                                                 rocblas_reduce_amin,
                                                 rocblas_finalize_amax_amin>(
        handle, n, x, shiftx, incx, stridex, batch_count, workspace, result);
}
