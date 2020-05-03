/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
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
    __forceinline__ __host__ __device__ rocblas_index_value_t<To> operator()(Ti          x,
                                                                             rocblas_int index)
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
    __forceinline__ __host__ __device__ auto operator()(const rocblas_index_value_t<To>& x)
    {
        return x.index + 1;
    }
};
