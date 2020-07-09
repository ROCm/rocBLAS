/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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
    void cblas_iamin(rocblas_int N, const T* X, rocblas_int incx, rocblas_int* result)
    {
        rocblas_int minpos = -1;
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
    void cblas_iamax_ensure_minimum_index(rocblas_int  N,
                                          const T*     X,
                                          rocblas_int  incx,
                                          rocblas_int* result)
    {
        rocblas_int maxpos = -1;
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
    void iamin(rocblas_int N, const T* X, rocblas_int incx, rocblas_int* result)
    {
        cblas_iamin(N, X, incx, result);
        *result += 1;
    }

    template <typename T>
    void iamax(rocblas_int N, const T* X, rocblas_int incx, rocblas_int* result)
    {
        cblas_iamax_ensure_minimum_index(N, X, incx, result);
        *result += 1;
    }

}
