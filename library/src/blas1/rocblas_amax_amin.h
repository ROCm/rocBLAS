/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_reduction_template.hpp"

//!
//! @brief Struct to define pair of value and index.
//!
template <typename T>
struct index_value_t
{
    //! @brief Important: index must come first, so that index_value_t* can be cast to rocblas_int*
    rocblas_int index;
    //! @brief The value.
    T value;
};

//!
//! @brief Struct-operator a default_value of index_value_t<T>
//!
template <typename T>
struct rocblas_default_value<index_value_t<T>>
{
    __forceinline__ __host__ __device__ constexpr auto operator()() const
    {
        index_value_t<T> x;
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
    __forceinline__ __host__ __device__ index_value_t<To> operator()(Ti x, rocblas_int index)
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
    __forceinline__ __host__ __device__ auto operator()(const index_value_t<To>& x)
    {
        return x.index + 1;
    }
};
