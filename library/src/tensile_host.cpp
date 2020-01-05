/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************/

// The implementation of the rocBLAS<->Tensile interface layer.

#ifdef USE_TENSILE_HOST

#include "tensile_host.hpp"
#include "rocblas.h"
//#include <Tensile/AMDGPU.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <glob.h>
#include <libgen.h>
#include <memory>
#include <string>
#include <type_traits>
#include <unistd.h>

namespace
{
    /******************************************************
     * Map a rocBLAS type to a corresponding Tensile type *
     ******************************************************/
    template <typename T>
    struct rocblas_to_tensile_type
    {
        using tensile_type = T;
    };

    template <>
    struct rocblas_to_tensile_type<rocblas_float_complex>
    {
        using tensile_type = std::complex<float>;
    };

    template <>
    struct rocblas_to_tensile_type<rocblas_double_complex>
    {
        using tensile_type = std::complex<double>;
    };

    template <>
    struct rocblas_to_tensile_type<rocblas_half>
    {
        using tensile_type = Tensile::Half;
    };

    template <>
    struct rocblas_to_tensile_type<rocblas_bfloat16>
    {
        using tensile_type = Tensile::BFloat16;
    };

    template <>
    struct rocblas_to_tensile_type<int8_t>
    {
        using tensile_type = Tensile::Int8x4;
    };

    /********************************************************************
     * Variable template to map a rocBLAS type into a Tensile::DataType *
     ********************************************************************/
    template <typename>
    constexpr auto tensile_datatype = nullptr;

    template <>
    constexpr auto tensile_datatype<int8_t> = Tensile::DataType::Int8x4;

    template <>
    constexpr auto tensile_datatype<int32_t> = Tensile::DataType::Int32;

    template <>
    constexpr auto tensile_datatype<rocblas_half> = Tensile::DataType::Half;

    template <>
    constexpr auto tensile_datatype<rocblas_bfloat16> = Tensile::DataType::BFloat16;

    template <>
    constexpr auto tensile_datatype<float> = Tensile::DataType::Float;

    template <>
    constexpr auto tensile_datatype<double> = Tensile::DataType::Double;

    template <>
    constexpr auto tensile_datatype<rocblas_float_complex> = Tensile::DataType::ComplexFloat;

    template <>
    constexpr auto tensile_datatype<rocblas_double_complex> = Tensile::DataType::ComplexDouble;

    /***************************************************************
     * Contruct a Tensile Problem from a RocblasContractionProblem *
     ***************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto ConstructTensileProblem(const RocblasContractionProblem<Ti, To, Tc>& prob)
    {
        // Tensile DataTypes corresponding to rocBLAS data types
        static constexpr Tensile::DataType Tensile_Ti = tensile_datatype<Ti>;
        static constexpr Tensile::DataType Tensile_To = tensile_datatype<To>;

        // Tensor descriptors for a, b
        Tensile::TensorDescriptor a, b;

        // Tensor ops for matrices, like complex conjugate
        Tensile::TensorOps aops, bops, cops, dops;

        // Tensile Indices for contraction problem
        Tensile::ContractionProblem::FreeIndices  freeIndex(2);
        Tensile::ContractionProblem::BoundIndices boundIndex(1);
        Tensile::ContractionProblem::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != rocblas_operation_none)
        {
            a = {Tensile_Ti, {prob.k, prob.m, prob.batch_count}, {1, prob.ld_a, prob.stride_a}};
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            a = {Tensile_Ti, {prob.m, prob.k, prob.batch_count}, {1, prob.ld_a, prob.stride_a}};
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If A is complex and conjugated, add a ComplexConjugate op to aops
        if(is_complex<Ti> && prob.trans_a == rocblas_operation_conjugate_transpose)
            aops.push_back(Tensile::TensorOp::Type::ComplexConjugate);

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != rocblas_operation_none)
        {
            b = {Tensile_Ti, {prob.n, prob.k, prob.batch_count}, {1, prob.ld_b, prob.stride_b}};
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            b = {Tensile_Ti, {prob.k, prob.n, prob.batch_count}, {1, prob.ld_b, prob.stride_b}};
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // If B is complex and conjugated, add a ComplexConjugate op to bops
        if(is_complex<Ti> && prob.trans_b == rocblas_operation_conjugate_transpose)
            bops.push_back(Tensile::TensorOp::Type::ComplexConjugate);

        // Descriptor for input matrix C
        Tensile::TensorDescriptor c{
            Tensile_To, {prob.m, prob.n, prob.batch_count}, {1, prob.ld_c, prob.stride_c}};

        // Descriptor for output matrix D
        Tensile::TensorDescriptor d{
            Tensile_To, {prob.m, prob.n, prob.batch_count}, {1, prob.ld_d, prob.stride_d}};

        // The ContractionProblem
        Tensile::ContractionProblem tensileProblem{a,
                                                   aops,
                                                   b,
                                                   bops,
                                                   c,
                                                   cops,
                                                   d,
                                                   dops,
                                                   freeIndex,
                                                   batchIndex,
                                                   boundIndex,
                                                   value_category(*prob.beta)};

        // If HPA is active, mark it as true
        if(sizeof(Tc) > sizeof(Ti))
            tensileProblem.setHighPrecisionAccumulate(true);

        return tensileProblem;
    }

    /*************************************************************************
     * Class for converting alpha and beta between rocBLAS and Tensile types *
     * By default, alpha and beta are the same type as Tc compute_type       *
     *************************************************************************/
    template <typename Ti, typename To = Ti, typename Tc = To>
    struct AlphaBeta
    {
        using tensile_type = typename rocblas_to_tensile_type<Tc>::tensile_type;
        static void copy(tensile_type* dst, const Tc* src)
        {
            static_assert(sizeof(*src) == sizeof(*dst),
                          "Tensile and rocBLAS types are not the same size");
            static_assert(std::is_standard_layout<tensile_type>{} && std::is_standard_layout<Tc>{},
                          "Tensile or rocBLAS types are not standard layout types");
            memcpy(dst, src, sizeof(*dst));
        }
    };

    /**************************************************************
     * Tensile does not support float alpha and beta for HPA half *
     * We must convert alpha and beta from float to half          *
     **************************************************************/
    template <>
    struct AlphaBeta<rocblas_half, rocblas_half, float>
    {
        using tensile_type = Tensile::Half;
        static void copy(tensile_type* dst, const float* float_src)
        {
            rocblas_half src(*float_src);
            AlphaBeta<rocblas_half>::copy(dst, &src);
        }
    };

    /***************************************************************
     * Construct the inputs to a Tensile ContractionProblem        *
     ***************************************************************/
    template <typename Ti, typename To, typename Tc>
    inline auto GetTensileInputs(const RocblasContractionProblem<Ti, To, Tc>& problem)
    {
        // Tensile types corresponding to Ti, To, Tc
        using Tensile_Ti          = typename rocblas_to_tensile_type<Ti>::tensile_type;
        using Tensile_To          = typename rocblas_to_tensile_type<To>::tensile_type;
        using Tensile_Talpha_beta = typename AlphaBeta<Ti, To, Tc>::tensile_type;

        // Make sure rocBLAS and Tensile types are compatible
        // For int8_t we allow the sizes to differ, assuming alignment
        static_assert((sizeof(Tensile_Ti) == sizeof(Ti) || std::is_same<Ti, int8_t>{})
                          && sizeof(Tensile_To) == sizeof(To),
                      "Tensile and rocBLAS types are not the same size");

        static_assert(std::is_standard_layout<Ti>{} && std::is_standard_layout<Tensile_Ti>{}
                          && std::is_standard_layout<To>{} && std::is_standard_layout<Tensile_To>{},
                      "Tensile or rocBLAS types are not standard layout types");

        // Structure describing the inputs (A, B, C, D, alpha, beta)
        Tensile::TypedContractionInputs<Tensile_Ti,
                                        Tensile_Ti,
                                        Tensile_To,
                                        Tensile_To,
                                        Tensile_Talpha_beta,
                                        Tensile_Talpha_beta>
            inputs;

        // Set the A, B, C, D matrices pointers in Tensile
        inputs.a = reinterpret_cast<const Tensile_Ti*>(problem.A);
        inputs.b = reinterpret_cast<const Tensile_Ti*>(problem.B);
        inputs.c = reinterpret_cast<const Tensile_To*>(problem.C);
        inputs.d = reinterpret_cast<Tensile_To*>(problem.D);

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        AlphaBeta<Ti, To, Tc>::copy(&inputs.alpha, problem.alpha);
        AlphaBeta<Ti, To, Tc>::copy(&inputs.beta, problem.beta);

        return inputs;
    }

    /*******************************************************************************
     * The TensileHostImpl class implements TensileHost as an opaque derived class *
     *******************************************************************************/
    class TensileHostImpl : public TensileHost
    {
        /************************************************************
         * Allow TensileHost to downcast and access TensileHostImpl *
         ************************************************************/
        friend class TensileHost;

        /*******************
         * Class variables *
         *******************/
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;
        std::shared_ptr<Tensile::Hardware>                                           hardware;
        Tensile::hip::SolutionAdapter                                                adapter;

        /******************************************************************************
         * GetProcessorName() returns the processor architecture name for directories *
         ******************************************************************************/
        static constexpr const char* GetProcessorName(Tensile::AMDGPU::Processor p)
        {
            switch(p)
            {
            case Tensile::AMDGPU::Processor::gfx803:
                return "gfx803";
            default: // Default falls to the most common hardware
            case Tensile::AMDGPU::Processor::gfx900:
                return "gfx900";
            case Tensile::AMDGPU::Processor::gfx906:
                return "gfx906";
            case Tensile::AMDGPU::Processor::gfx908:
                return "gfx908";
            }
        }

        /*******************************************************
         * Testpath() tests that a path exists and is readable *
         *******************************************************/
        static bool TestPath(const std::string& path)
        {
            return access(path.c_str(), R_OK) == 0;
        }

        /*************************************************************
         * Constructor loads host according to environment variables *
         * and default paths based on librocblas.so location and GPU *
         *************************************************************/
    public:
        TensileHostImpl()
            : hardware{Tensile::hip::GetCurrentDevice()}
        {
            std::string path;
            path.reserve(PATH_MAX);

            auto        pAMDGPU   = std::dynamic_pointer_cast<Tensile::AMDGPU>(hardware);
            std::string processor = GetProcessorName(pAMDGPU->processor);

            const char* env = getenv("ROCBLAS_TENSILE_LIBPATH");
            if(env)
            {
                path = env;
            }
            else
            {
                Dl_info info;

                // Find the location of librocblas.so
                if(dladdr((void*)createTensileHost, &info))
                {
                    path = info.dli_fname;
                    dirname(&path[0]);
                    path.resize(strlen(path.c_str()));
                }
                else
                {
                    path = "/opt/rocm/rocblas/lib";
                }

                // Find the location of the libraries
                if(TestPath(path + "/../../Tensile/library"))
                    path += "/../../Tensile/library";
                else
                    path += "/library";

                if(TestPath(path + "/" + processor))
                    path += "/" + processor;
            }

            auto dir = path + "/*co";

            glob_t glob_result;
            int    g = glob(dir.c_str(), GLOB_TILDE_CHECK | GLOB_NOSORT, nullptr, &glob_result);
            if(!g)
            {
                for(size_t i = 0; i < glob_result.gl_pathc; ++i)
                    adapter.loadCodeObjectFile(glob_result.gl_pathv[i]);
            }
            else
            {
                fprintf(stderr,
                        g == GLOB_NOMATCH ? "\nrocBLAS warning: No paths matched %s. Make sure "
                                            "ROCBLAS_TENSILE_LIBPATH is set correctly.\n"
                                          : "rocBLAS warning: glob(\"%s\", ...) returned %s.\n",
                        dir.c_str(),
                        g == GLOB_ABORTED
                            ? "GLOB_ABORTED"
                            : g == GLOB_NOSPACE ? "GLOB_NOSPACE" : "an unknown error");
            }
            globfree(&glob_result);

            path += "/TensileLibrary.yaml";
            if(!TestPath(path))
            {
                fprintf(stderr, "\nrocBLAS error: Cannot read %s: %m\n", path.c_str());
                abort();
            }

            library = std::dynamic_pointer_cast<
                Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>(
                Tensile::LoadLibraryFile<Tensile::ContractionProblem>(path));
        }
    };

} // namespace

/*****************************************************************************
 * createTensileHost returns an instance of TensileHostImpl as a TensileHost *
 *****************************************************************************/
TensileHost* createTensileHost()
{
    //    static int once = (tensileInitialize(), 0);
    return new TensileHostImpl;
}

/******************************************************************************
 * runContractionProblem calls Tensile to run a contraction problem described *
 * by RocblasContractionProblem                                               *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<Ti, To, Tc>& problem)
{
    rocblas_status                                status = rocblas_status_internal_error;
    std::shared_ptr<Tensile::ContractionSolution> solution;

    try
    {
        // We know that the TensileHost instance is a TensileHostImpl, so we can downcast to it
        auto* host            = static_cast<TensileHostImpl*>(this);
        auto  tensile_problem = ConstructTensileProblem(problem);
        solution              = host->library->findBestSolution(tensile_problem, *host->hardware);

        if(!solution)
        {
            // We print the error message only once, to avoid excessive logging
            static int once = (std::cerr << "Error: No Tensile solution found for " << problem, 0);
        }
        else
        {
            auto inputs = GetTensileInputs(problem);
            auto result = solution->solve(tensile_problem, inputs, *host->hardware);
            host->adapter.launchKernels(result);
            status = rocblas_status_success;
        }
    }
    catch(const std::exception& e)
    {
        static int once
            = (std::cerr << "Error: " << (solution ? "" : "No ") << "Tensile solution found, but "
                         << e.what() << " exception thown for " << problem,
               0);
    }
    catch(...)
    {
        static int once
            = (std::cerr << "Error: " << (solution ? "" : "No ")
                         << "Tensile solution found, but unknown exception thown for " << problem,
               0);
    }

    return status;
}

/******************************************************************************
 * Intantiate the cases of runContractionProblem which are needed to satisfy  *
 * rocBLAS dependencies. This file's template functions are not defined in a  *
 * header file, in order to keep Tensile and rocBLAS separate.                *
 ******************************************************************************/

// Non-EX types
template rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<rocblas_half>&);

template rocblas_status TensileHost::runContractionProblem(const RocblasContractionProblem<float>&);

template rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<double>&);

template rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<rocblas_float_complex>&);

template rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<rocblas_double_complex>&);

// EX types
template rocblas_status TensileHost::runContractionProblem(
    const RocblasContractionProblem<rocblas_half, rocblas_half, float>&);

template rocblas_status TensileHost::runContractionProblem(
    const RocblasContractionProblem<rocblas_bfloat16, rocblas_bfloat16, float>&);

template rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<int8_t, int32_t, int32_t>&);

#endif
