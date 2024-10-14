/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

// The implementation of the rocBLAS<->HipBlasLT interface layer.

#include "rocblas.h"

extern "C" void rocblas_shutdown();

/*****************************************************************************
 * This is the only file in rocBLAS which should #include Hipblaslt headers    *
 * or reference Hipblaslt identifiers. hipblaslt_host.hpp defines the interface. *
 *****************************************************************************/

#include <functional>
#include <iostream>

#include "hipblaslt_host.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

namespace
{
    /********************************************************************
     * Variable template to map a rocBLAS type into a hipblasltDatatype_t *
     ********************************************************************/
    template <typename>
    constexpr auto hipblaslt_datatype = HIPBLASLT_DATATYPE_INVALID;

    template <>
    constexpr auto hipblaslt_datatype<int8_t> = HIP_R_8I;

    template <>
    constexpr auto hipblaslt_datatype<int32_t> = HIP_R_32I;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_f8> = HIP_R_8F_E4M3_FNUZ;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_bf8> = HIP_R_8F_E5M2_FNUZ;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_half> = HIP_R_16F;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_bfloat16> = HIP_R_16BF;

    template <>
    constexpr auto hipblaslt_datatype<float> = HIP_R_32F;

    template <>
    constexpr auto hipblaslt_datatype<double> = HIP_R_64F;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_float_complex> = HIP_C_32F;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_double_complex> = HIP_C_64F;

    /********************************************************************
     * Variable template to map a rocBLAS type into a hipblasLtComputeType_t *
     ********************************************************************/
    template <typename>
    constexpr auto hipblaslt_compute_type = HIPBLAS_COMPUTE_32F;

    template <>
    constexpr auto hipblaslt_compute_type<int32_t> = HIPBLAS_COMPUTE_32I;

    template <>
    constexpr auto hipblaslt_compute_type<rocblas_half> = HIPBLAS_COMPUTE_16F;

    template <>
    constexpr auto hipblaslt_compute_type<float> = HIPBLAS_COMPUTE_32F;

    template <>
    constexpr auto hipblaslt_compute_type<double> = HIPBLAS_COMPUTE_64F;

    template <typename T>
    auto convertScalarForHipblasLT(T num)
    {
        return static_cast<int8_t>(num);
    }

    template <>
    auto convertScalarForHipblasLT(rocblas_float_complex num)
    {
        return static_cast<int8_t>(std::real(num));
    }

    template <>
    auto convertScalarForHipblasLT(rocblas_double_complex num)
    {
        return static_cast<int8_t>(std::real(num));
    }

    /**************************************************************************
    * We normally print error messages only once, to avoid excessive logging *
    **************************************************************************/
    void print_once(const rocblas_internal_ostream& msg)
    {
        if(rocblas_suppress_tensile_error_messages())
            return;
        static constexpr char varname[] = "ROCBLAS_VERBOSE_HIPBLASLT_ERROR";
        static const char*    verbose   = getenv(varname);
        if(!verbose)
        {
            static auto& once = rocblas_cerr
                                << msg
                                << "\nThis message will be only be displayed once, unless the "
                                << varname << " environment variable is set." << std::endl;
        }
        else
            rocblas_cerr << msg << std::endl;
    }

    /****************************************************************
     * Construct a HipBlasLT GEMM from a RocblasContractionProblem *
     ****************************************************************/
    template <typename TiA,
              typename To,
              typename Tc,
              typename TiB = TiA,
              typename TcA = TiA,
              typename TcB = TiA>
    auto ConstructHipBlasLTGemm(const RocblasContractionProblem<TiA, To, Tc, TiB, TcA, TcB>& prob)
    {
        hipblasLtHandle_t& handle = *(prob.handle->getHipblasLtHandle());

        hipblaslt_ext::Gemm gemm(handle,
                                 (hipblasOperation_t)prob.trans_a,
                                 (hipblasOperation_t)prob.trans_b,
                                 hipblaslt_datatype<TiA>,
                                 hipblaslt_datatype<TiB>,
                                 hipblaslt_datatype<To>,
                                 hipblaslt_datatype<To>,
                                 hipblaslt_compute_type<Tc>);

        hipblaslt_ext::GemmProblemType problemType;
        problemType.op_a         = (hipblasOperation_t)prob.trans_a;
        problemType.op_b         = (hipblasOperation_t)prob.trans_b;
        problemType.type_a       = hipblaslt_datatype<TiA>;
        problemType.type_b       = hipblaslt_datatype<TiB>;
        problemType.type_c       = hipblaslt_datatype<To>;
        problemType.type_d       = hipblaslt_datatype<To>;
        problemType.type_compute = hipblaslt_compute_type<Tc>;

        hipblaslt_ext::GemmEpilogue epilogue;
        hipblaslt_ext::GemmInputs   inputs;
        inputs.a     = (void*)(prob.A + prob.buffer_offset_a);
        inputs.b     = (void*)(prob.B + prob.buffer_offset_b);
        inputs.c     = (void*)(prob.C + prob.buffer_offset_c);
        inputs.d     = (void*)(prob.D + prob.buffer_offset_d);
        inputs.alpha = (void*)prob.alpha;
        inputs.beta  = (void*)prob.beta;

        gemm.setProblem(prob.m,
                        prob.n,
                        prob.k,
                        prob.batch_count,
                        prob.col_stride_a,
                        prob.col_stride_b,
                        prob.col_stride_c,
                        prob.col_stride_d,
                        prob.batch_stride_a,
                        prob.batch_stride_b,
                        prob.batch_stride_c,
                        prob.batch_stride_d,
                        epilogue,
                        inputs,
                        problemType);
        return gemm;
    }

    /****************************************************************
     * Construct a HipBlasLT Groupped GEMM from a RocblasContractionProblem *
     ****************************************************************/
    template <typename TiA,
              typename To,
              typename Tc,
              typename TiB = TiA,
              typename TcA = TiA,
              typename TcB = TiA>
    auto ConstructHipBlasLTGroupedGemm(
        const RocblasContractionProblem<TiA, To, Tc, TiB, TcA, TcB>& prob)
    {
        hipblasLtHandle_t& handle = *(prob.handle->getHipblasLtHandle());

        hipblaslt_ext::GroupedGemm gemm(handle,
                                        (hipblasOperation_t)prob.trans_a,
                                        (hipblasOperation_t)prob.trans_b,
                                        hipblaslt_datatype<TiA>,
                                        hipblaslt_datatype<TiB>,
                                        hipblaslt_datatype<To>,
                                        hipblaslt_datatype<To>,
                                        hipblaslt_compute_type<Tc>);

        hipblaslt_ext::GemmProblemType problemType;
        problemType.op_a         = (hipblasOperation_t)prob.trans_a;
        problemType.op_b         = (hipblasOperation_t)prob.trans_b;
        problemType.type_a       = hipblaslt_datatype<TiA>;
        problemType.type_b       = hipblaslt_datatype<TiB>;
        problemType.type_c       = hipblaslt_datatype<To>;
        problemType.type_d       = hipblaslt_datatype<To>;
        problemType.type_compute = hipblaslt_compute_type<Tc>;

        std::vector<int64_t>                     Ms(prob.batch_count);
        std::vector<int64_t>                     Ns(prob.batch_count);
        std::vector<int64_t>                     Ks(prob.batch_count);
        std::vector<int64_t>                     ldas(prob.batch_count);
        std::vector<int64_t>                     ldbs(prob.batch_count);
        std::vector<int64_t>                     ldcs(prob.batch_count);
        std::vector<int64_t>                     ldds(prob.batch_count);
        std::vector<int64_t>                     strideas(prob.batch_count);
        std::vector<int64_t>                     stridebs(prob.batch_count);
        std::vector<int64_t>                     stridecs(prob.batch_count);
        std::vector<int64_t>                     strideds(prob.batch_count);
        std::vector<int64_t>                     batch_counts(prob.batch_count);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogues(prob.batch_count);
        std::vector<hipblaslt_ext::GemmInputs>   inputs(prob.batch_count);

        for(int batch = 0; batch < prob.batch_count; batch++)
        {
            Ms[batch]           = prob.m;
            Ns[batch]           = prob.n;
            Ks[batch]           = prob.k;
            ldas[batch]         = prob.col_stride_a;
            ldbs[batch]         = prob.col_stride_b;
            ldcs[batch]         = prob.col_stride_c;
            ldds[batch]         = prob.col_stride_d;
            strideas[batch]     = prob.batch_stride_a;
            stridebs[batch]     = prob.batch_stride_b;
            stridecs[batch]     = prob.batch_stride_c;
            strideds[batch]     = prob.batch_stride_d;
            batch_counts[batch] = 1;
            inputs[batch].a     = (void*)(prob.batch_A[batch] + prob.buffer_offset_a);
            inputs[batch].b     = (void*)(prob.batch_B[batch] + prob.buffer_offset_b);
            inputs[batch].c     = (void*)(prob.batch_C[batch] + prob.buffer_offset_c);
            inputs[batch].d     = (void*)(prob.batch_D[batch] + prob.buffer_offset_d);
            inputs[batch].alpha = (void*)prob.alpha;
            inputs[batch].beta  = (void*)prob.beta;
        }

        gemm.setProblem(Ms,
                        Ns,
                        Ks,
                        batch_counts,
                        ldas,
                        ldbs,
                        ldcs,
                        ldds,
                        strideas,
                        stridebs,
                        stridecs,
                        strideds,
                        epilogues,
                        inputs,
                        problemType);
        return gemm;
    }

    /*
     * Combine common initialization functionality for GEMM and Ggrouped GEMM.
    */
    auto hipBlasLTInit(hipblaslt_ext::GemmInstance&      gemm,
                       rocblas_gemm_algo                 algo,
                       int32_t                           solution_index,
                       const rocblas_handle              probHandle,
                       size_t&                           workspace_size,
                       hipblasLtMatmulHeuristicResult_t& heuristicResult,
                       size_t                            extra_malloc = 0)
    {
        hipblasLtHandle_t& handle = *(probHandle->getHipblasLtHandle());

        hipblaslt_ext::GemmPreference gemmPref;
        auto                          max_workspace_size = probHandle->get_available_workspace();
        gemmPref.setMaxWorkspaceBytes(max_workspace_size - extra_malloc);

        const int                                     request_solutions = 1;
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;

        if(algo == rocblas_gemm_algo_solution_index && solution_index > 0)
        {
            std::vector<int> solution_index_vec(1, solution_index - 1);
            if(hipblaslt_ext::getAlgosFromIndex(handle, solution_index_vec, heuristicResults)
               != HIPBLAS_STATUS_SUCCESS)
            {
                rocblas_internal_ostream msg;
                print_once(msg << "hipBLASLt error: Cannot find specified solution index!");
                return rocblas_status_invalid_value;
            }

            if(heuristicResults.empty())
            {
                rocblas_internal_ostream msg;
                print_once(msg << "rocBLAS error: No hipBLASLt solution found");
                return rocblas_status_invalid_value;
            }
        }
        else
        {
            if(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResults)
               != HIPBLAS_STATUS_SUCCESS)
            {
                rocblas_internal_ostream msg;
                print_once(msg << "hipBLASLt error: Heuristic Fetch Failed!");
                return rocblas_status_internal_error;
            }

            if(heuristicResults.empty())
            {
                rocblas_internal_ostream msg;
                print_once(msg << "rocBLAS error: No hipBLASLt solution found");
                return rocblas_status_not_implemented;
            }
        }

        heuristicResult = heuristicResults[0];
        workspace_size  = 0;

        if(algo == rocblas_gemm_algo_solution_index && solution_index > 0)
        {
            if(gemm.isAlgoSupported(heuristicResult.algo, workspace_size) != HIPBLAS_STATUS_SUCCESS)
            {
                rocblas_internal_ostream msg;
                print_once(msg << "hipBLASLt error: algo not supported.");
                return rocblas_status_invalid_value;
            }
        }
        else
        {
            workspace_size = heuristicResult.workspaceSize;
        }

        workspace_size += extra_malloc;

        if(workspace_size > max_workspace_size)
        {
            rocblas_internal_ostream msg;
            print_once(msg << "hipblaslt: algo not supported: insufficient workspace.");
            return rocblas_status_invalid_value;
        }
        return rocblas_status_success;
    }
}

/******************************************************************************
 * runContractionProblemHipBlasLT calls Hipblaslt to run a contraction problem described *
 * by RocblasContractionProblem                                               *
 ******************************************************************************/
template <typename TiA, typename To, typename Tc, typename TiB, typename TcA, typename TcB>
rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<TiA, To, Tc, TiB, TcA, TcB>& prob,
    rocblas_gemm_algo                                            algo,
    int32_t                                                      solution_index)
{
    if(prob.batch_A == 0)
    {
        auto gemm = ConstructHipBlasLTGemm(prob);

        size_t                           workspace_size = 0;
        hipblasLtMatmulHeuristicResult_t heuristicResult;

        auto init = hipBlasLTInit(
            gemm, algo, solution_index, prob.handle, workspace_size, heuristicResult);
        if(rocblas_status_success != init)
        {
            return init;
        }

        auto  gsu_malloc  = prob.handle->gsu_malloc_by_size(workspace_size);
        void* d_workspace = prob.handle->gsu_workspace;

        if(gemm.initialize(heuristicResult.algo, d_workspace, false) != HIPBLAS_STATUS_SUCCESS)
        {
            rocblas_internal_ostream msg;
            print_once(msg << "hipBLASLt error: hipBLASLt Initialization Failed!");
            return rocblas_status_internal_error;
        }
        if(gemm.run(prob.handle->get_stream()) != HIPBLAS_STATUS_SUCCESS)
        {
            rocblas_internal_ostream msg;
            print_once(msg << "hipBLASLt error: hipBLASLt Execution Failed!");
            return rocblas_status_internal_error;
        }
    }
    else
    {
        auto gemm         = ConstructHipBlasLTGroupedGemm(prob);
        auto userArgsSize = prob.batch_count * sizeof(hipblaslt_ext::UserArguments);

        size_t                           workspace_size = 0;
        hipblasLtMatmulHeuristicResult_t heuristicResult;

        auto init = hipBlasLTInit(
            gemm, algo, solution_index, prob.handle, workspace_size, heuristicResult, userArgsSize);
        if(rocblas_status_success != init)
        {
            return init;
        }

        auto  gsu_malloc  = prob.handle->gsu_malloc_by_size(workspace_size);
        void* d_workspace = prob.handle->gsu_workspace;

        if(gemm.initialize(heuristicResult.algo, d_workspace, false) != HIPBLAS_STATUS_SUCCESS)
        {
            rocblas_internal_ostream msg;
            print_once(msg << "hipBLASLt error: hipBLASLt Initialization Failed!");
            return rocblas_status_internal_error;
        }

        auto h_alpha = *(prob.alpha);
        auto h_beta  = *(prob.beta);

        hipblaslt_ext::UserArguments* userArgs;
        hipHostMalloc(&userArgs, userArgsSize);
        gemm.getDefaultValueForDeviceUserArguments(userArgs);
        for(int batch = 0; batch < prob.batch_count; batch++)
        {
            userArgs[batch].m        = prob.m;
            userArgs[batch].n        = prob.n;
            userArgs[batch].k        = prob.k;
            userArgs[batch].strideA1 = prob.col_stride_a;
            userArgs[batch].strideB1 = prob.col_stride_b;
            userArgs[batch].strideC1 = prob.col_stride_c;
            userArgs[batch].strideD1 = prob.col_stride_d;
            userArgs[batch].strideA2 = prob.batch_stride_a;
            userArgs[batch].strideB2 = prob.batch_stride_b;
            userArgs[batch].strideC2 = prob.batch_stride_c;
            userArgs[batch].strideD2 = prob.batch_stride_d;
            userArgs[batch].batch    = 1;
            userArgs[batch].a        = (void*)(prob.batch_A[batch] + prob.buffer_offset_a);
            userArgs[batch].b        = (void*)(prob.batch_B[batch] + prob.buffer_offset_b);
            userArgs[batch].c        = (void*)(prob.batch_C[batch] + prob.buffer_offset_c);
            userArgs[batch].d        = (void*)(prob.batch_D[batch] + prob.buffer_offset_d);

            userArgs[batch].alpha[0] = convertScalarForHipblasLT(h_alpha);
            userArgs[batch].beta[0]  = convertScalarForHipblasLT(h_beta);
        }

        // Copy them to device memory
        hipblaslt_ext::UserArguments* d_userArgs
            = (hipblaslt_ext::UserArguments*)((char*)(prob.handle->gsu_workspace)
                                              + (workspace_size - userArgsSize));
        hipMemcpy(d_userArgs, userArgs, userArgsSize, hipMemcpyHostToDevice);
        hipFree(userArgs);

        if(gemm.run(d_userArgs, prob.handle->get_stream()) != HIPBLAS_STATUS_SUCCESS)
        {
            rocblas_internal_ostream msg;
            print_once(msg << "hipBLASLt error: hipBLASLt Execution Failed!");
            return rocblas_status_internal_error;
        }
    }
    return rocblas_status_success;
}

template <typename TiA, typename To, typename Tc, typename TiB, typename TcA, typename TcB>
rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<TiA, To, Tc, TiB, TcA, TcB>& prob,
                             rocblas_tensile_get_solution_option                          option,
                             rocblas_int* list_array,
                             rocblas_int* list_size)
{

    if(list_size == nullptr)
    {
        return rocblas_status_invalid_pointer;
    }

    hipblasLtHandle_t& handle = *(prob.handle->getHipblasLtHandle());

    if(option == MATCHES_TYPE)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
        std::vector<hipblasOperation_t> ops = {HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_OP_C};

        for(auto op1 : ops)
        {
            for(auto op2 : ops)
            {
                std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults_temp;
                auto fetch = hipblaslt_ext::getAllAlgos(handle,
                                                        hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                        op1,
                                                        op2,
                                                        hipblaslt_datatype<TiA>,
                                                        hipblaslt_datatype<TiB>,
                                                        hipblaslt_datatype<To>,
                                                        hipblaslt_datatype<To>,
                                                        hipblaslt_compute_type<Tc>,
                                                        heuristicResults_temp);

                heuristicResults.insert(heuristicResults.end(),
                                        heuristicResults_temp.begin(),
                                        heuristicResults_temp.end());
            }
        }

        // Convert to indexes and remove duplicates.
        std::vector<rocblas_int> heuristicIndexes(heuristicResults.size());
        std::transform(heuristicResults.begin(),
                       heuristicResults.end(),
                       heuristicIndexes.begin(),
                       [](auto x) { return hipblaslt_ext::getIndexFromAlgo(x.algo) + 1; });
        std::sort(heuristicIndexes.begin(), heuristicIndexes.end());
        auto itr = std::unique(heuristicIndexes.begin(), heuristicIndexes.end());
        heuristicIndexes.resize(std::distance(heuristicIndexes.begin(), itr));

        if(list_array == nullptr)
        {
            *list_size = heuristicIndexes.size();
        }
        else
        {
            rocblas_int i  = 0;
            auto        it = heuristicIndexes.begin();
            while(i < *list_size && it != heuristicIndexes.end())
            {
                list_array[i] = *it;
                ++i;
                ++it;
            }
        }

        return rocblas_status_success;
    }
    else if(option == CAN_SOLVE)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
        auto                                          fetch = hipblaslt_ext::getAllAlgos(handle,
                                                hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                (hipblasOperation_t)prob.trans_a,
                                                (hipblasOperation_t)prob.trans_b,
                                                hipblaslt_datatype<TiA>,
                                                hipblaslt_datatype<TiB>,
                                                hipblaslt_datatype<To>,
                                                hipblaslt_datatype<To>,
                                                hipblaslt_compute_type<Tc>,
                                                heuristicResults);

        std::shared_ptr<hipblaslt_ext::GemmInstance> gemm;

        if(prob.batch_A == 0)
        {
            gemm = std::make_shared<hipblaslt_ext::GemmInstance>(ConstructHipBlasLTGemm(prob));
        }
        else
        {
            gemm = std::make_shared<hipblaslt_ext::GemmInstance>(
                ConstructHipBlasLTGroupedGemm(prob));
        }

        size_t retSize   = heuristicResults.size();
        size_t iter_size = list_array == nullptr ? retSize : *list_size;

        rocblas_int i  = 0;
        auto        it = heuristicResults.begin();
        size_t      tmpWorkspaceSize;
        while(i < iter_size && it != heuristicResults.end())
        {
            if(gemm->isAlgoSupported(it->algo, tmpWorkspaceSize) == HIPBLAS_STATUS_SUCCESS)
            {
                if(list_array != nullptr)
                {
                    list_array[i] = hipblaslt_ext::getIndexFromAlgo(it->algo) + 1;
                    ++i;
                }
            }
            else
            {
                --retSize;
            }
            ++it;
        }

        if(list_array == nullptr)
        {
            *list_size = retSize;
        }

        return rocblas_status_success;
    }
    else
    {
        return rocblas_status_invalid_value;
    }
}

/******************************************************************************
 * Intantiate the cases of runContractionProblemHipBlasLT which are needed to satisfy  *
 * rocBLAS dependencies. This file's template functions are not defined in a  *
 * header file, in order to keep hipBLASLt and rocBLAS separate.                *
 ******************************************************************************/

// Non-HPA/GEMM types
template rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<rocblas_half>&, rocblas_gemm_algo algo, int32_t solution_index);

template rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<float>&,
                                                       rocblas_gemm_algo algo,
                                                       int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<double>&,
                                                       rocblas_gemm_algo algo,
                                                       int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_float_complex>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_double_complex>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

// EX types

// f8 case0: Ti=f8 Tc=To=f32
template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_f8, float, float>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<rocblas_f8, rocblas_half, float>&,
    rocblas_gemm_algo algo,
    int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_f8, rocblas_f8, float>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_bf8, float, float>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<rocblas_bf8, rocblas_half, float>&,
    rocblas_gemm_algo algo,
    int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<rocblas_bf8, rocblas_bf8, float>&,
    rocblas_gemm_algo algo,
    int32_t           solution_index);

//hybrid // Change of f8 parameter convention in order to support existing usage
template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_f8,
                                                                   float,
                                                                   float,
                                                                   rocblas_bf8,
                                                                   rocblas_f8,
                                                                   rocblas_bf8>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_f8,
                                                                   rocblas_half,
                                                                   float,
                                                                   rocblas_bf8,
                                                                   rocblas_f8,
                                                                   rocblas_bf8>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_bf8,
                                                                                       float,
                                                                                       float,
                                                                                       rocblas_f8,
                                                                                       rocblas_bf8,
                                                                                       rocblas_f8>&,
                                                       rocblas_gemm_algo algo,
                                                       int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_bf8,
                                                                                       rocblas_half,
                                                                                       float,
                                                                                       rocblas_f8,
                                                                                       rocblas_bf8,
                                                                                       rocblas_f8>&,
                                                       rocblas_gemm_algo algo,
                                                       int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_f8,
                                                                   rocblas_f8,
                                                                   float,
                                                                   rocblas_bf8,
                                                                   rocblas_f8,
                                                                   rocblas_bf8>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_f8,
                                                                   rocblas_bf8,
                                                                   float,
                                                                   rocblas_bf8,
                                                                   rocblas_f8,
                                                                   rocblas_bf8>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_bf8,
                                                                                       rocblas_f8,
                                                                                       float,
                                                                                       rocblas_f8,
                                                                                       rocblas_bf8,
                                                                                       rocblas_f8>&,
                                                       rocblas_gemm_algo algo,
                                                       int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_bf8,
                                                                                       rocblas_bf8,
                                                                                       float,
                                                                                       rocblas_f8,
                                                                                       rocblas_bf8,
                                                                                       rocblas_f8>&,
                                                       rocblas_gemm_algo algo,
                                                       int32_t           solution_index);

// HPA types
template rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<rocblas_half, rocblas_half, float>&,
    rocblas_gemm_algo algo,
    int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_half, float, float>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<rocblas_bfloat16, rocblas_bfloat16, float>&,
    rocblas_gemm_algo algo,
    int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_bfloat16, float, float>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<int8_t, int32_t, int32_t>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

// ********** get all solutions explicits ********
// Non-HPA/GEMM types
template rocblas_status getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_half>&,
                                                 rocblas_tensile_get_solution_option option,
                                                 rocblas_int*                        list_array,
                                                 rocblas_int*                        list_size);

template rocblas_status getAllSolutionsHipBlasLT(const RocblasContractionProblem<float>&,
                                                 rocblas_tensile_get_solution_option option,
                                                 rocblas_int*                        list_array,
                                                 rocblas_int*                        list_size);

template rocblas_status getAllSolutionsHipBlasLT(const RocblasContractionProblem<double>&,
                                                 rocblas_tensile_get_solution_option option,
                                                 rocblas_int*                        list_array,
                                                 rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_float_complex>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_double_complex>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

// HPA types
template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_half, rocblas_half, float>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_half, float, float>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

template rocblas_status getAllSolutionsHipBlasLT(
    const RocblasContractionProblem<rocblas_bfloat16, rocblas_bfloat16, float>&,
    rocblas_tensile_get_solution_option option,
    rocblas_int*                        list_array,
    rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_bfloat16, float, float>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<int8_t, int32_t, int32_t>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);
