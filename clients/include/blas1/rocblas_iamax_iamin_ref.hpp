/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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

namespace rocblas_iamax_iamin_ref
{
    template <typename T>
    T asum(T x)
    {
        return x < 0 ? -x : x;
    }

    rocblas_half asum(rocblas_half x)
    {
        return rocblas_half(asum(float(x)));
    }

    template <typename T>
    bool lessthan(T x, T y)
    {
        return x < y;
    }

    bool lessthan(rocblas_half x, rocblas_half y)
    {
        return float(x) < float(y);
    }

    template <typename T>
    bool greatherthan(T x, T y)
    {
        return x > y;
    }

    bool greatherthan(rocblas_half x, rocblas_half y)
    {
        return float(x) > float(y);
    }

    template <typename T>
    void ref_iamin(int64_t N, const T* X, int64_t incx, int64_t* result)
    {
        int64_t minpos = -1;
        if(N > 0 && incx > 0)
        {
            auto min = asum(X[0]);
            minpos   = 0;
            for(size_t i = 1; i < N; ++i)
            {
                auto a = asum(X[i * incx]);
                if(lessthan(a, min))
                {
                    min    = a;
                    minpos = i;
                }
            }
        }
        *result = minpos;
    }

    template <typename T>
    void ref_iamax_ensure_minimum_index(int64_t N, const T* X, int64_t incx, int64_t* result)
    {
        int64_t maxpos = -1;
        if(N > 0 && incx > 0)
        {
            auto max = asum(X[0]);
            maxpos   = 0;
            for(size_t i = 1; i < N; ++i)
            {
                auto a = asum(X[i * incx]);
                if(greatherthan(a, max))
                {
                    max    = a;
                    maxpos = i;
                }
            }
        }
        *result = maxpos;
    }

    template <typename T>
    void iamin(int64_t N, const T* X, int64_t incx, int64_t* result)
    {
        ref_iamin(N, X, incx, result);
        *result += 1;
    }

    template <typename T>
    void iamax(int64_t N, const T* X, int64_t incx, int64_t* result)
    {
        ref_iamax_ensure_minimum_index(N, X, incx, result);
        *result += 1;
    }

}
