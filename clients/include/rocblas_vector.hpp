/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "d_vector.hpp"

#include "device_batch_vector.hpp"
#include "device_strided_batch_vector.hpp"
#include "device_vector.hpp"

#include "host_batch_vector.hpp"
#include "host_strided_batch_vector.hpp"
#include "host_vector.hpp"

//
// Get some specialized routine.
//

template <typename T>
void random_generator(T& n)
{
    n = random_generator<T>();
}

template <typename U>
void rocblas_init_template(U& that, bool seedReset = false)
{
    if(seedReset)
    {
        rocblas_seedrand();
    }

    for(rocblas_int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
    {
        auto data = that[batch_index];
        auto inc  = std::abs(that.inc());
        auto n    = that.n();
        if(inc >= 0)
        {
            for(rocblas_int i = 0; i < n; ++i)
            {
                random_generator(data[i * inc]);
            }
        }
        else
        {
            for(rocblas_int i = 0; i < n; ++i)
            {
                random_generator(data[(i + 1 - n) * inc]);
            }
        }
    }
}

template <typename T>
void rocblas_init(host_strided_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, seedReset);
}

template <typename T>
void rocblas_init(host_batch_vector<T>& that, bool seedReset = false)
{
    rocblas_init_template(that, seedReset);
}

template <typename T>
void rocblas_init(host_vector<T>& that, bool seedReset = false)
{
    if(seedReset)
    {
        rocblas_seedrand();
    }
    rocblas_init(that, 1, that.size(), 1);
}
