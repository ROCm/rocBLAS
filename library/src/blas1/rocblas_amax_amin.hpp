/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "fetch_template.hpp"
#include "rocblas_reduction_template.hpp"

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
