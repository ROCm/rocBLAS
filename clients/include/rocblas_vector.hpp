/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "d_vector.hpp"

#include "device_batch_vector.hpp"
#include "device_strided_batch_vector.hpp"
#include "device_vector.hpp"

#include "host_batch_vector.hpp"
#include "host_pinned_vector.hpp"
#include "host_strided_batch_vector.hpp"
#include "host_vector.hpp"

#include "rocblas_init.hpp"

//!
//! @brief Template for initializing a host (non_batched|batched|strided_batched)vector.
//! @param that That vector.
//! @param rand_gen The random number generator
//! @param seedReset Reset the seed if true, do not reset the seed otherwise.
//!
template <typename U, typename T>
void rocblas_init_template(U& that, T rand_gen(), bool seedReset)
{
    if(seedReset)
        rocblas_seedrand();

    for(rocblas_int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
    {
        auto*     batched_data = that[batch_index];
        ptrdiff_t inc          = that.inc();
        auto      n            = that.n();

        if(inc < 0)
            batched_data -= (n - 1) * inc;

        for(rocblas_int i = 0; i < n; ++i)
            batched_data[i * inc] = rand_gen();
    }
}

//!
//! @brief Initialize a host_strided_batch_vector with NaNs.
//! @param that The host_strided_batch_vector to be initialized.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init_nan(host_strided_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, random_nan_generator<T>, seedReset);
}

//!
//! @brief Initialize a host_batch_vector with NaNs.
//! @param that The host_batch_vector to be initialized.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init_nan(host_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, random_nan_generator<T>, seedReset);
}

//!
//! @brief Initialize a host_vector with NaNs.
//! @param that The host_vector to be initialized.
//! @param seedReset reset he seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init_nan(host_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, random_nan_generator<T>, seedReset);
}

//!
//! @brief Initialize a host_strided_batch_vector.
//! @param that The host_strided_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init_hpl(host_strided_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, random_hpl_generator<T>, seedReset);
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init_hpl(host_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, random_hpl_generator<T>, seedReset);
}

//!
//! @brief Initialize a host_strided_batch_vector.
//! @param that The host_strided_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init(host_strided_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, random_generator<T>, seedReset);
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init(host_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, random_generator<T>, seedReset);
}

//!
//! @brief Initialize a host_vector.
//! @param that The host_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init(host_vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        rocblas_seedrand();
    rocblas_init(that, that.size(), 1, 1);
}

//!
//! @brief trig Initialize of a host_strided_batch_vector.
//! @param that The host_strided_batch_vector.
//! @param init_cos cos initialize if true, else sin initialize.
//!
template <typename T>
inline void rocblas_init_trig(host_strided_batch_vector<T>& that, bool init_cos = false)
{
    if(init_cos)
    {
        for(rocblas_int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
        {
            auto*     batched_data = that[batch_index];
            ptrdiff_t inc          = that.inc();
            auto      n            = that.n();

            if(inc < 0)
                batched_data -= (n - 1) * inc;

            rocblas_init_cos(batched_data, 1, n, inc);
        }
    }
    else
    {
        for(rocblas_int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
        {
            auto*     batched_data = that[batch_index];
            ptrdiff_t inc          = that.inc();
            auto      n            = that.n();

            if(inc < 0)
                batched_data -= (n - 1) * inc;

            rocblas_init_sin(batched_data, 1, n, inc);
        }
    }
}

//!
//! @brief trig Initialize of a host_batch_vector.
//! @param that The host_batch_vector.
//! @param init_cos cos initialize if true, else sin initialize.
//!
template <typename T>
inline void rocblas_init_trig(host_batch_vector<T>& that, bool init_cos = false)
{
    if(init_cos)
    {
        for(rocblas_int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
        {
            auto*     batched_data = that[batch_index];
            ptrdiff_t inc          = that.inc();
            auto      n            = that.n();

            if(inc < 0)
                batched_data -= (n - 1) * inc;

            rocblas_init_cos(batched_data, 1, n, inc);
        }
    }
    else
    {
        for(rocblas_int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
        {
            auto*     batched_data = that[batch_index];
            ptrdiff_t inc          = that.inc();
            auto      n            = that.n();

            if(inc < 0)
                batched_data -= (n - 1) * inc;

            rocblas_init_sin(batched_data, 1, n, inc);
        }
    }
}

//!
//! @brief Initialize a host_strided_batch_vector.
//! @param hx The host_strided_batch_vector.
//! @param arg Specifies the argument class.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init_vector(host_strided_batch_vector<T>& hx,
                                const Arguments&              arg,
                                bool                          seedReset = false)
{
    if(rocblas_isnan(arg.alpha))
    {
        rocblas_init_nan(hx, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        rocblas_init_hpl(hx, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        rocblas_init(hx, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_trig(hx, seedReset);
    }
}

//!
//! @brief Initialize a host_batch_vector.
//! @param hx The host_batch_vector.
//! @param arg Specifies the argument class.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void
    rocblas_init_vector(host_batch_vector<T>& hx, const Arguments& arg, bool seedReset = false)
{
    if(rocblas_isnan(arg.alpha))
    {
        rocblas_init_nan(hx, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        rocblas_init_hpl(hx, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        rocblas_init(hx, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_trig(hx, seedReset);
    }
}

//!
//! @brief Initialize a host_vector.
//! @param hx The host_vector.
//! @param N Length of the host vector.
//! @param incx Increment for the host vector.
//! @param stride_x Incement between the host vector.
//! @param batch_count number of instances in the batch.
//! @param arg Specifies the argument class.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init_vector(host_vector<T>&  hx,
                                size_t           N,
                                size_t           incx,
                                rocblas_stride   stride_x,
                                rocblas_int      batch_count,
                                const Arguments& arg,
                                bool             seedReset = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(rocblas_isnan(arg.alpha))
    {
        rocblas_init_nan(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        rocblas_init_hpl(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        rocblas_init(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        if(seedReset)
            rocblas_init_cos(hx, 1, N, incx, stride_x, batch_count);
        else
            rocblas_init_sin(hx, 1, N, incx, stride_x, batch_count);
    }
}
