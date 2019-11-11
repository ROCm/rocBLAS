/* ************************************************************************
 * Copyright 2019 Advanced Micro Devices, Inc.
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
#include <dlfcn.h>
#include <glob.h>
#include <libgen.h>
#include <memory>
#include <string>
#include <unistd.h>
#include <sys/stat.h>

// Return the value category for a value as a double precision value, such as whether it's 0, 1,
// or some other value. Tensile uses a double precision value to express the category of beta.
// This function is required to convert complex or other types to a double representing the category.
template <typename T>
static constexpr double value_category(const T& beta)
{
    return beta == T(0) ? 0.0 : beta == T(1) ? 1.0 : -12345.0;
}

// Variable template to return Tensile type based on C++ rocBLAS type
template <typename>
static constexpr auto tensile_datatype = nullptr;

template <>
static constexpr auto tensile_datatype<rocblas_half> = Tensile::DataType::Half;

template <>
static constexpr auto tensile_datatype<float> = Tensile::DataType::Float;

template <>
static constexpr auto tensile_datatype<double> = Tensile::DataType::Double;

template <>
static constexpr auto tensile_datatype<rocblas_float_complex> = Tensile::DataType::ComplexFloat;

template <>
static constexpr auto tensile_datatype<rocblas_double_complex> = Tensile::DataType::ComplexDouble;

template <typename T>
static auto create_gemm_contraction_problem(rocblas_operation trans_a,
                                            rocblas_operation trans_b,
                                            size_t            m,
                                            size_t            n,
                                            size_t            k,
                                            T                 alpha,
                                            const T*          A,
                                            size_t            ld_a,
                                            const T*          B,
                                            size_t            ld_b,
                                            T                 beta,
                                            T*                C,
                                            size_t            ld_c,
                                            size_t            stride_a    = 0,
                                            size_t            stride_b    = 0,
                                            size_t            stride_c    = 0,
                                            size_t            batch_count = 1)
{
    auto dt = tensile_datatype<T>;

    Tensile::ContractionProblem::FreeIndices  freeIndex(2);
    Tensile::ContractionProblem::BoundIndices boundIndex(1);
    Tensile::ContractionProblem::BatchIndices batchIndex{{2, 2, 2, 2}};

    freeIndex[0].isA = true;
    freeIndex[1].isA = false;
    freeIndex[0].c = freeIndex[0].d = 0;
    freeIndex[1].c = freeIndex[1].d = 1;

    Tensile::TensorDescriptor a, b;

    if(trans_a != rocblas_operation_none)
    {
        a               = {dt, {k, m, batch_count}, {1, ld_a, stride_a}};
        freeIndex[0].i  = 1;
        boundIndex[0].a = 0;
    }
    else
    {
        a               = {dt, {m, k, batch_count}, {1, ld_a, stride_a}};
        freeIndex[0].i  = 0;
        boundIndex[0].a = 1;
    }

    if(trans_b != rocblas_operation_none)
    {
        b               = {dt, {n, k, batch_count}, {1, ld_b, stride_b}};
        freeIndex[1].i  = 0;
        boundIndex[0].b = 1;
    }
    else
    {
        b               = {dt, {k, n, batch_count}, {1, ld_b, stride_b}};
        freeIndex[1].i  = 1;
        boundIndex[0].b = 0;
    }

    Tensile::TensorOps aops;
    if(is_complex<T> && trans_a == rocblas_operation_conjugate_transpose)
        aops = {Tensile::TensorOp::Type::ComplexConjugate};

    Tensile::TensorOps bops;
    if(is_complex<T> && trans_b == rocblas_operation_conjugate_transpose)
        bops = {Tensile::TensorOp::Type::ComplexConjugate};

    Tensile::TensorDescriptor c{dt, {m, n, batch_count}, {1, ld_c, stride_c}};

    return Tensile::ContractionProblem{
        a, aops, b, bops, c, {}, c, {}, freeIndex, batchIndex, boundIndex, value_category(beta)};
}

template <typename PROBLEM>
static auto ConstructTensileProblem(const PROBLEM& problem)
{
    switch(problem.problem_type)
    {
    case ContractionProblemType::GEMM:
        return create_gemm_contraction_problem(problem.trans_a,
                                               problem.trans_b,
                                               problem.m,
                                               problem.n,
                                               problem.k,
                                               problem.alpha,
                                               problem.A,
                                               problem.ld_a,
                                               problem.B,
                                               problem.ld_b,
                                               problem.beta,
                                               problem.C,
                                               problem.ld_c);

    case ContractionProblemType::GEMMStridedBatched:
        return create_gemm_contraction_problem(problem.trans_a,
                                               problem.trans_b,
                                               problem.m,
                                               problem.n,
                                               problem.k,
                                               problem.alpha,
                                               problem.A,
                                               problem.ld_a,
                                               problem.B,
                                               problem.ld_b,
                                               problem.beta,
                                               problem.C,
                                               problem.ld_c,
                                               problem.stride_a,
                                               problem.stride_b,
                                               problem.stride_c,
                                               problem.batch_count);
    }
}

// Map a static C++ type into a corresponding Tensile type
template <typename T>
struct rocblas_to_tensile_type
{
    using type = T;
};

template <>
struct rocblas_to_tensile_type<rocblas_float_complex>
{
    using type = std::complex<float>;
};

template <>
struct rocblas_to_tensile_type<rocblas_double_complex>
{
    using type = std::complex<double>;
};

template <>
struct rocblas_to_tensile_type<rocblas_half>
{
    using type = Tensile::Half;
};

// Construct the inputs to a Tensile ContractionProblem
template <typename T, typename U, typename V>
static auto GetTensileInputs(const RocblasContractionProblem<T, U, V>& problem)
{
    using tensile_type = typename rocblas_to_tensile_type<T>::type;
    Tensile::TypedContractionInputs<tensile_type> inputs;
    switch(problem.problem_type)
    {
    case ContractionProblemType::GEMM:
    case ContractionProblemType::GEMMStridedBatched:
        inputs.a = reinterpret_cast<const tensile_type*>(problem.A);
        inputs.b = reinterpret_cast<const tensile_type*>(problem.B);
        inputs.c = reinterpret_cast<tensile_type*>(problem.C);
        inputs.d = reinterpret_cast<tensile_type*>(problem.C);
        memcpy(&inputs.alpha, &problem.alpha, sizeof(T));
        memcpy(&inputs.beta, &problem.beta, sizeof(T));
        break;
    }
    return inputs;
}

bool TestPath(std::string path)
{
  struct stat st;
  if (stat(path.c_str(), &st) == 0)
  {
    return true;
  }
  else
  {
    return false;
  }
}

std::string GetProcessorName(Tensile::AMDGPU::Processor p)
{
        switch(p)
        {
		case Tensile::AMDGPU::Processor::gfx803: return "gfx803";
		case Tensile::AMDGPU::Processor::gfx900: return "gfx900";
		case Tensile::AMDGPU::Processor::gfx906: return "gfx906";
		case Tensile::AMDGPU::Processor::gfx908: return "gfx908";
        }
}


// TensileHostImpl class implements TensileHost as an opaque derived class
struct TensileHostImpl : TensileHost
{
    // Constructor loads host according to environment variables and default paths based on librocblas.so location
    TensileHostImpl()
    {
        std::string path;
        path.reserve(PATH_MAX);

        hardware = Tensile::hip::GetCurrentDevice();
	std::shared_ptr<Tensile::AMDGPU> pAMDGPU = std::dynamic_pointer_cast<Tensile::AMDGPU>(hardware);
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
            if (TestPath(path + "/../../Tensile/library"))
                path += "/../../Tensile/library";
	    else
	        path += "/library";
	    if (TestPath(path + "/" + processor))
		path += "/" + processor;
        }

        auto dir = path + "/*co";

        glob_t glob_result;
        auto   g = glob(dir.c_str(), GLOB_TILDE_CHECK | GLOB_NOSORT, nullptr, &glob_result);
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
                    g == GLOB_ABORTED ? "GLOB_ABORTED"
                                      : g == GLOB_NOSPACE ? "GLOB_NOSPACE" : "an unknown error");
        }
        globfree(&glob_result);

        path += "/TensileLibrary.yaml";
        if(access(path.c_str(), R_OK))
        {
            fprintf(stderr, "\nrocBLAS error: Cannot read %s: %m\n", path.c_str());
            abort();
        }

        library = std::dynamic_pointer_cast<
            Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>(
            Tensile::LoadLibraryFile<Tensile::ContractionProblem>(path));
    }

private:
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;
    std::shared_ptr<Tensile::Hardware>                                           hardware;
    Tensile::hip::SolutionAdapter                                                adapter;
    friend class TensileHost;
};

// createTensileHost returns an instance of TensileHostImpl as a TensileHost
TensileHost* createTensileHost()
{
    return new TensileHostImpl;
}

// runContractionProblem calls Tensile to run a contraction problem described by RocblasContractionProblem
template <typename T, typename U, typename V>
rocblas_status TensileHost::runContractionProblem(const RocblasContractionProblem<T, U, V>& problem)
try
{
    auto* host            = static_cast<TensileHostImpl*>(this);
    auto  tensile_problem = ConstructTensileProblem(problem);
    auto  inputs          = GetTensileInputs(problem);
    auto  solution        = host->library->findBestSolution(tensile_problem, *host->hardware);
    auto  result          = solution->solve(tensile_problem, inputs, *host->hardware);
    host->adapter.launchKernels(result);
    return rocblas_status_success;
}
catch(...)
{
    return rocblas_status_internal_error;
}

// Intantiate the cases of runContractionProblem which are needed to satisfy rocBLAS dependencies
// This file's functions are not defined in a header file, in order to keep Tensile and rocBLAS separate
template rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<rocblas_half>&);

template rocblas_status TensileHost::runContractionProblem(const RocblasContractionProblem<float>&);

template rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<double>&);

template rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<rocblas_float_complex>&);

template rocblas_status
    TensileHost::runContractionProblem(const RocblasContractionProblem<rocblas_double_complex>&);

#endif
