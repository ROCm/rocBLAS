/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
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
//! @brief enum to check for NaN initialization of the Input vector/matrix
//!
typedef enum rocblas_check_nan_init_
{
    // Alpha sets NaN
    rocblas_client_alpha_sets_nan,

    // Beta sets NaN
    rocblas_client_beta_sets_nan,

    //  Never set NaN
    rocblas_client_never_set_nan

} rocblas_check_nan_init;

//!
//! @brief Template for initializing a host (non_batched|batched|strided_batched)vector.
//! @param that That vector.
//! @param rand_gen The random number generator
//! @param seedReset Reset the seed if true, do not reset the seed otherwise.
//!
template <typename U, typename T>
void rocblas_init_template(U& that, T rand_gen(), bool seedReset, bool alternating_sign = false)
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

        if(alternating_sign)
        {
            for(rocblas_int i = 0; i < n; ++i)
            {
                auto value            = rand_gen();
                batched_data[i * inc] = (i ^ 0) & 1 ? value : negate(value);
            }
        }
        else
        {
            for(rocblas_int i = 0; i < n; ++i)
                batched_data[i * inc] = rand_gen();
        }
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
inline void rocblas_init_hpl(host_strided_batch_vector<T>& that,
                             bool                          seedReset        = false,
                             bool                          alternating_sign = false)
{
    rocblas_init_template(that, random_hpl_generator<T>, seedReset, alternating_sign);
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init_hpl(host_batch_vector<T>& that,
                             bool                  seedReset        = false,
                             bool                  alternating_sign = false)
{
    rocblas_init_template(that, random_hpl_generator<T>, seedReset, alternating_sign);
}

//!
//! @brief Initialize a host_strided_batch_vector.
//! @param that The host_strided_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void rocblas_init(host_strided_batch_vector<T>& that,
                         bool                          seedReset        = false,
                         bool                          alternating_sign = false)
{
    rocblas_init_template(that, random_generator<T>, seedReset, alternating_sign);
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void
    rocblas_init(host_batch_vector<T>& that, bool seedReset = false, bool alternating_sign = false)
{
    rocblas_init_template(that, random_generator<T>, seedReset, alternating_sign);
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
//! @param nan_init Initialize vector with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void rocblas_init_vector(host_strided_batch_vector<T>& hx,
                                const Arguments&              arg,
                                rocblas_check_nan_init        nan_init,
                                bool                          seedReset        = false,
                                bool                          alternating_sign = false)
{
    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_nan(hx, seedReset);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_nan(hx, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        rocblas_init_hpl(hx, seedReset, alternating_sign);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        rocblas_init(hx, seedReset, alternating_sign);
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
//! @param nan_init Initialize vector with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void rocblas_init_vector(host_batch_vector<T>&  hx,
                                const Arguments&       arg,
                                rocblas_check_nan_init nan_init,
                                bool                   seedReset        = false,
                                bool                   alternating_sign = false)
{
    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_nan(hx, seedReset);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_nan(hx, seedReset);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        rocblas_init_hpl(hx, seedReset, alternating_sign);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        rocblas_init(hx, seedReset, alternating_sign);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_trig(hx, seedReset);
    }
}

//!
//! @brief Initialize a host_vector.
//! @param hx The host_vector.
//! @param arg Specifies the argument class.
//! @param N Length of the host vector.
//! @param incx Increment for the host vector.
//! @param stride_x Incement between the host vector.
//! @param batch_count number of instances in the batch.
//! @param nan_init Initialize vector with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void rocblas_init_vector(host_vector<T>&        hx,
                                const Arguments&       arg,
                                size_t                 N,
                                size_t                 incx,
                                rocblas_stride         stride_x,
                                rocblas_int            batch_count,
                                rocblas_check_nan_init nan_init,
                                bool                   seedReset        = false,
                                bool                   alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_vector(random_nan_generator<T>, hx, N, incx, stride_x, batch_count);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_vector(random_nan_generator<T>, hx, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        if(alternating_sign)
            rocblas_init_vector_alternating_sign(
                random_hpl_generator<T>, hx, N, incx, stride_x, batch_count);
        else
            rocblas_init_vector(random_hpl_generator<T>, hx, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        if(alternating_sign)
            rocblas_init_vector_alternating_sign(
                random_generator<T>, hx, N, incx, stride_x, batch_count);
        else
            rocblas_init_vector(random_generator<T>, hx, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_vector_trig(hx, N, incx, stride_x, batch_count, seedReset);
    }
}

//!
//! @brief Initialize a host matrix.
//! @param hA The host matrix.
//! @param arg Specifies the argument class.
//! @param M Length of the host matrix.
//! @param N Length of the host matrix.
//! @param lda Leading dimension of the host matrix.
//! @param stride_A Incement between the host matrix.
//! @param batch_count number of instances in the batch.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T>
inline void rocblas_init_matrix(host_vector<T>&           hA,
                                const Arguments&          arg,
                                size_t                    M,
                                size_t                    N,
                                size_t                    lda,
                                rocblas_stride            stride_A,
                                rocblas_int               batch_count,
                                rocblas_check_nan_init    nan_init,
                                rocblas_check_matrix_type matrix_type,
                                bool                      seedReset        = false,
                                bool                      alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_matrix(
            matrix_type, arg.uplo, random_nan_generator<T>, hA, M, N, lda, stride_A, batch_count);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_matrix(
            matrix_type, arg.uplo, random_nan_generator<T>, hA, M, N, lda, stride_A, batch_count);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type,
                                                 arg.uplo,
                                                 random_hpl_generator<T>,
                                                 hA,
                                                 M,
                                                 N,
                                                 lda,
                                                 stride_A,
                                                 batch_count);
        else
            rocblas_init_matrix(matrix_type,
                                arg.uplo,
                                random_hpl_generator<T>,
                                hA,
                                M,
                                N,
                                lda,
                                stride_A,
                                batch_count);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_generator<T>, hA, M, N, lda, stride_A, batch_count);
        else
            rocblas_init_matrix(
                matrix_type, arg.uplo, random_generator<T>, hA, M, N, lda, stride_A, batch_count);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_matrix_trig(
            matrix_type, arg.uplo, hA, M, N, lda, stride_A, batch_count, seedReset);
    }
}
