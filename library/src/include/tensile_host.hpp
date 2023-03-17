/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************/

/*********************************************************
 * Declaration of the rocBLAS<->Tensile interface layer. *
 *********************************************************/

#pragma once

/*****************************************************************************
 * WARNING: Tensile-specific data types, functions and macros should only be *
 * referenced from tensile_host.cpp. This header file defines the interface  *
 * that the rest of rocBLAS uses to access Tensile. If another Tensile       *
 * feature needs to be accessed, the API for accessing it should be defined  *
 * in this file, without referencing any Tensile-specific identifiers here.  *
 *****************************************************************************/

#include "handle.hpp"
#include "tuple_helper.hpp"
#include <atomic>

// Struct to represent tensile problem, algo, solution index.
typedef struct
{
    void*             problem;
    int32_t           solution_index;
    rocblas_gemm_algo algo;
} rocblas_tensile_problem_info;

/********************************************************************
 * RocblasContractionProblem captures the arguments for a GEMM-like *
 * contraction problem, to be passed to runContractionProblem.      *
 ********************************************************************/
template <typename Ti, typename To = Ti, typename Tc = To>
struct RocblasContractionProblem
{
    rocblas_handle     handle;
    rocblas_gemm_flags flags;
    rocblas_operation  trans_a;
    rocblas_operation  trans_b;

    // The RocblasContractionProblem data members should exactly match
    // Tensile's parameter types, even if rocBLAS uses differently
    // sized or signed types. The constructors should convert rocBLAS
    // types into the corresponding Tensile types stored in this class.
    size_t m;
    size_t n;
    size_t k;

    const Tc* alpha;

    const Ti*        A;
    const Ti* const* batch_A;
    size_t           row_stride_a;
    size_t           col_stride_a;
    size_t           batch_stride_a;
    size_t           buffer_offset_a;

    const Ti*        B;
    const Ti* const* batch_B;
    size_t           row_stride_b;
    size_t           col_stride_b;
    size_t           batch_stride_b;
    size_t           buffer_offset_b;

    const Tc* beta;

    const To*        C;
    const To* const* batch_C;
    size_t           row_stride_c;
    size_t           col_stride_c;
    size_t           batch_stride_c;
    size_t           buffer_offset_c;

    To*        D;
    To* const* batch_D;
    size_t     row_stride_d;
    size_t     col_stride_d;
    size_t     batch_stride_d;
    size_t     buffer_offset_d;

    size_t batch_count;
    bool   strided_batch;

    // gemm
    // gemm_strided_batched
    RocblasContractionProblem(rocblas_handle     handle,
                              rocblas_operation  trans_a,
                              rocblas_operation  trans_b,
                              rocblas_int        m,
                              rocblas_int        n,
                              rocblas_int        k,
                              const Tc*          alpha,
                              const Ti*          A,
                              const Ti* const*   batch_A,
                              rocblas_int        ld_a,
                              rocblas_stride     batch_stride_a,
                              rocblas_stride     offset_a,
                              const Ti*          B,
                              const Ti* const*   batch_B,
                              rocblas_int        ld_b,
                              rocblas_stride     batch_stride_b,
                              rocblas_stride     offset_b,
                              const Tc*          beta,
                              To*                C,
                              To* const*         batch_C,
                              rocblas_int        ld_c,
                              rocblas_stride     batch_stride_c,
                              rocblas_stride     offset_c,
                              rocblas_int        batch_count,
                              bool               strided_batch,
                              rocblas_gemm_flags flags)
        : handle(handle)
        , flags(flags)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , batch_A(batch_A)
        , row_stride_a(1)
        , col_stride_a(ld_a)
        , batch_stride_a(batch_stride_a)
        , buffer_offset_a(offset_a)
        , B(B)
        , batch_B(batch_B)
        , row_stride_b(1)
        , col_stride_b(ld_b)
        , batch_stride_b(batch_stride_b)
        , buffer_offset_b(offset_b)
        , beta(beta)
        , C(C)
        , batch_C(batch_C)
        , row_stride_c(1)
        , col_stride_c(ld_c)
        , batch_stride_c(batch_stride_c)
        , buffer_offset_c(offset_c)
        , D(C)
        , batch_D(batch_C)
        , row_stride_d(1)
        , col_stride_d(ld_c)
        , batch_stride_d(batch_stride_c)
        , buffer_offset_d(offset_c)
        , batch_count(batch_count)
        , strided_batch(strided_batch)
    {
    }

    // gemm_ex
    // gemm_strided_batched_ex
    RocblasContractionProblem(rocblas_handle     handle,
                              rocblas_operation  trans_a,
                              rocblas_operation  trans_b,
                              rocblas_int        m,
                              rocblas_int        n,
                              rocblas_int        k,
                              const Tc*          alpha,
                              const Ti*          A,
                              const Ti* const*   batch_A,
                              rocblas_int        ld_a,
                              rocblas_stride     batch_stride_a,
                              rocblas_stride     offset_a,
                              const Ti*          B,
                              const Ti* const*   batch_B,
                              rocblas_int        ld_b,
                              rocblas_stride     batch_stride_b,
                              rocblas_stride     offset_b,
                              const Tc*          beta,
                              const To*          C,
                              const To* const*   batch_C,
                              rocblas_int        ld_c,
                              rocblas_stride     batch_stride_c,
                              rocblas_stride     offset_c,
                              To*                D,
                              To* const*         batch_D,
                              rocblas_int        ld_d,
                              rocblas_stride     batch_stride_d,
                              rocblas_stride     offset_d,
                              rocblas_int        batch_count,
                              bool               strided_batch,
                              rocblas_gemm_flags flags)
        : handle(handle)
        , flags(flags)
        , trans_a(trans_a)
        , trans_b(trans_b)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , batch_A(batch_A)
        , row_stride_a(1)
        , col_stride_a(ld_a)
        , batch_stride_a(batch_stride_a)
        , buffer_offset_a(offset_a)
        , B(B)
        , batch_B(batch_B)
        , row_stride_b(1)
        , col_stride_b(ld_b)
        , batch_stride_b(batch_stride_b)
        , buffer_offset_b(offset_b)
        , beta(beta)
        , C(C)
        , batch_C(batch_C)
        , row_stride_c(1)
        , col_stride_c(ld_c)
        , batch_stride_c(batch_stride_c)
        , buffer_offset_c(offset_c)
        , D(D)
        , batch_D(batch_D)
        , row_stride_d(1)
        , col_stride_d(ld_d)
        , batch_stride_d(batch_stride_d)
        , buffer_offset_d(offset_d)
        , batch_count(batch_count)
        , strided_batch(strided_batch)
    {
    }

    // gemm_ext2
    // gemm_strided_batched_ext2
    RocblasContractionProblem(rocblas_handle   handle,
                              rocblas_int      m,
                              rocblas_int      n,
                              rocblas_int      k,
                              const Tc*        alpha,
                              const Ti*        A,
                              const Ti* const* batch_A,
                              rocblas_stride   row_stride_a,
                              rocblas_stride   col_stride_a,
                              rocblas_stride   batch_stride_a,
                              rocblas_stride   offset_a,
                              const Ti*        B,
                              const Ti* const* batch_B,
                              rocblas_stride   row_stride_b,
                              rocblas_stride   col_stride_b,
                              rocblas_stride   batch_stride_b,
                              rocblas_stride   offset_b,
                              const Tc*        beta,
                              const To*        C,
                              const To* const* batch_C,
                              rocblas_stride   row_stride_c,
                              rocblas_stride   col_stride_c,
                              rocblas_stride   batch_stride_c,
                              rocblas_stride   offset_c,
                              To*              D,
                              To* const*       batch_D,
                              rocblas_stride   row_stride_d,
                              rocblas_stride   col_stride_d,
                              rocblas_stride   batch_stride_d,
                              rocblas_stride   offset_d,
                              rocblas_int      batch_count,
                              bool             strided_batch)
        : handle(handle)
        , flags(rocblas_gemm_flags_none)
        , trans_a(rocblas_operation_none)
        , trans_b(rocblas_operation_none)
        , m(m)
        , n(n)
        , k(k)
        , alpha(alpha)
        , A(A)
        , batch_A(batch_A)
        , row_stride_a(row_stride_a)
        , col_stride_a(col_stride_a)
        , batch_stride_a(batch_stride_a)
        , buffer_offset_a(offset_a)
        , B(B)
        , batch_B(batch_B)
        , row_stride_b(row_stride_b)
        , col_stride_b(col_stride_b)
        , batch_stride_b(batch_stride_b)
        , buffer_offset_b(offset_b)
        , beta(beta)
        , C(C)
        , batch_C(batch_C)
        , row_stride_c(row_stride_c)
        , col_stride_c(col_stride_c)
        , batch_stride_c(batch_stride_c)
        , buffer_offset_c(offset_c)
        , D(D)
        , batch_D(batch_D)
        , row_stride_d(row_stride_d)
        , col_stride_d(col_stride_d)
        , batch_stride_d(batch_stride_d)
        , buffer_offset_d(offset_d)
        , batch_count(batch_count)
        , strided_batch(strided_batch)
    {
    }

    /***************************************************
     * Print a RocblasContractionProblem for debugging *
     ***************************************************/
    friend rocblas_internal_ostream& operator<<(rocblas_internal_ostream&        os,
                                                const RocblasContractionProblem& prob)
    {
        return tuple_helper::print_tuple_pairs(
            os,
            std::make_tuple("a_type",
                            rocblas_precision_string<Ti>,
                            "b_type",
                            rocblas_precision_string<Ti>,
                            "c_type",
                            rocblas_precision_string<To>,
                            "d_type",
                            rocblas_precision_string<To>,
                            "compute_type",
                            rocblas_precision_string<Tc>,
                            "transA",
                            rocblas_transpose_letter(prob.trans_a),
                            "transB",
                            rocblas_transpose_letter(prob.trans_b),
                            "M",
                            prob.m,
                            "N",
                            prob.n,
                            "K",
                            prob.k,
                            "alpha",
                            *prob.alpha,
                            "row_stride_a",
                            prob.row_stride_a,
                            "col_stride_a",
                            prob.col_stride_a,
                            "row_stride_b",
                            prob.row_stride_b,
                            "col_stride_b",
                            prob.col_stride_b,
                            "row_stride_c",
                            prob.row_stride_c,
                            "col_stride_c",
                            prob.col_stride_c,
                            "row_stride_d",
                            prob.row_stride_d,
                            "col_stride_d",
                            prob.col_stride_d,
                            "beta",
                            *prob.beta,
                            "batch_count",
                            prob.batch_count,
                            "strided_batch",
                            prob.strided_batch,
                            "stride_a",
                            prob.batch_stride_a,
                            "stride_b",
                            prob.batch_stride_b,
                            "stride_c",
                            prob.batch_stride_c,
                            "stride_d",
                            prob.batch_stride_d,
                            "atomics_mode",
                            prob.handle->atomics_mode));
    };
};

/*******************************************************************************
 * runContractionProblem() solves a RocblasContractionProblem                  *
 *******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblas_status runContractionProblem(RocblasContractionProblem<Ti, To, Tc> const& problem,
                                     rocblas_gemm_algo algo           = rocblas_gemm_algo_standard,
                                     int32_t           solution_index = 0);

template <typename Ti, typename To, typename Tc>
rocblas_status getAllSolutions(const RocblasContractionProblem<Ti, To, Tc>& prob,
                               rocblas_int*                                 list_array,
                               rocblas_int*                                 list_size);

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for testing) *
 ***********************************************************************************/
ROCBLAS_INTERNAL_EXPORT std::atomic_bool& rocblas_internal_tensile_is_initialized();

/***********************************************************************************
 * Whether rocblas_initialize() is invoked to load all tensile kernels at startup  *
 ***********************************************************************************/
std::atomic_bool& rocblas_initialize_called();

/**********************************************
 * Whether to suppress Tensile error messages *
 **********************************************/
inline bool& rocblas_suppress_tensile_error_messages()
{
    thread_local bool t_suppress = false;
    return t_suppress;
}
