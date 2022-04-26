/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "d_vector.hpp"

#include "device_batch_matrix.hpp"
#include "device_matrix.hpp"
#include "device_strided_batch_matrix.hpp"

#include "host_batch_matrix.hpp"
#include "host_matrix.hpp"
#include "host_strided_batch_matrix.hpp"
#include "rocblas_init.hpp"

//!
//! @brief Initialize a host_strided_batch_matrix.
//! @param hA The host_strided_batch_matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T>
inline void rocblas_init_matrix(host_strided_batch_matrix<T>& hA,
                                const Arguments&              arg,
                                rocblas_check_nan_init        nan_init,
                                rocblas_check_matrix_type     matrix_type,
                                bool                          seedReset        = false,
                                bool                          alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
}

//!
//! @brief Initialize a host_batch_matrix.
//! @param hA The host_batch_matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T>
inline void rocblas_init_matrix(host_batch_matrix<T>&     hA,
                                const Arguments&          arg,
                                rocblas_check_nan_init    nan_init,
                                rocblas_check_matrix_type matrix_type,
                                bool                      seedReset        = false,
                                bool                      alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
}

//!
//! @brief Initialize a host matrix.
//! @param hA The host matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T>
inline void rocblas_init_matrix(host_matrix<T>&           hA,
                                const Arguments&          arg,
                                rocblas_check_nan_init    nan_init,
                                rocblas_check_matrix_type matrix_type,
                                bool                      seedReset        = false,
                                bool                      alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        rocblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
}
