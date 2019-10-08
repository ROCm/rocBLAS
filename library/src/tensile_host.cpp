#ifdef USE_TENSILE_HOST

#include "tensile_host.hpp"
#include "rocblas.h"
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <glob.h>
#include <iostream>
#include <string>

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

// return the value category for a value, such as whether it's 0 or 1
template <typename T>
constexpr auto value_category(const T* beta)
{
    return *beta == T(0) ? 0.0 : *beta == T(1) ? 1.0 : -12345.0;
}

template <typename T>
auto create_gemm_contraction_problem_strided(rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             unsigned long     m,
                                             unsigned long     n,
                                             unsigned long     k,
                                             const T*          alpha,
                                             const T*          A,
                                             unsigned long     ld_a,
                                             unsigned long     stride_a,
                                             const T*          B,
                                             unsigned long     ld_b,
                                             unsigned long     stride_b,
                                             const T*          beta,
                                             T*                C,
                                             unsigned long     ld_c,
                                             unsigned long     stride_c,
                                             unsigned long     batchSize)
{
    bool transposeA = trans_a != rocblas_operation_none;
    bool transposeB = trans_b != rocblas_operation_none;

    Tensile::DataType           dt = tensile_datatype<T>;
    Tensile::ContractionProblem problem
        = Tensile::ContractionProblem::GEMM_Strides(transposeA,
                                                    transposeB,
                                                    dt,
                                                    dt,
                                                    dt,
                                                    dt,
                                                    m,
                                                    n,
                                                    k,
                                                    batchSize,
                                                    ld_a,
                                                    stride_a,
                                                    ld_b,
                                                    stride_b,
                                                    ld_c,
                                                    stride_c,
                                                    ld_c,
                                                    stride_c,
                                                    value_category(beta));

    return problem;
}

// construct the gemm contraction problem
template <typename T>
auto create_gemm_contraction_problem(rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     unsigned long     m,
                                     unsigned long     n,
                                     unsigned long     k,
                                     const T*          alpha,
                                     const T*          A,
                                     unsigned long     ld_a,
                                     const T*          B,
                                     unsigned long     ld_b,
                                     const T*          beta,
                                     T*                C,
                                     unsigned long     ld_c)
{
    bool transposeA = trans_a != rocblas_operation_none;
    bool transposeB = trans_b != rocblas_operation_none;

    Tensile::ContractionProblem::FreeIndex  free;
    Tensile::ContractionProblem::BoundIndex bound;

    free.ca = free.da = 0;
    free.cb = free.db = 1;

    Tensile::TensorDescriptor a, b, c, d;

    Tensile::DataType dt = tensile_datatype<T>;
    if(transposeA)
    {
        a       = Tensile::TensorDescriptor(dt, {k, m}, {1, ld_a});
        free.a  = 1;
        bound.a = 0;
    }
    else
    {
        a       = Tensile::TensorDescriptor(dt, {m, k}, {1, ld_a});
        free.a  = 0;
        bound.a = 1;
    }

    if(transposeB)
    {
        b       = Tensile::TensorDescriptor(dt, {n, k}, {1, ld_b});
        free.b  = 0;
        bound.b = 1;
    }
    else
    {
        b       = Tensile::TensorDescriptor(dt, {k, n}, {1, ld_b});
        free.b  = 1;
        bound.b = 0;
    }

    Tensile::ContractionProblem::FreeIndices  freeIndices{free};
    Tensile::ContractionProblem::BatchIndices batchIndices;
    Tensile::ContractionProblem::BoundIndices boundIndices{bound};

    d = Tensile::TensorDescriptor(dt, {m, n}, {1, ld_c});

    unsigned int batchCount = 1;

    a.appendDim(batchCount);
    b.appendDim(batchCount);
    d.appendDim(batchCount);

    batchIndices.push_back({2, 2, 2, 2});

    if(value_category(beta) != 0)
        c = d;

    Tensile::TensorOps nop;

    return Tensile::ContractionProblem(a,
                                       nop,
                                       b,
                                       nop,
                                       c,
                                       nop,
                                       d,
                                       nop,
                                       freeIndices,
                                       batchIndices,
                                       boundIndices,
                                       value_category(beta));
}

template <typename T>
auto ConstructTensileProblem(RocblasContractionProblem<T>* problem)
{
    Tensile::ContractionProblem tensile_problem;
    switch(problem->problem_type)
    {
    case GEMM:
        tensile_problem = create_gemm_contraction_problem<T>(problem->trans_a,
                                                             problem->trans_b,
                                                             problem->m,
                                                             problem->n,
                                                             problem->k,
                                                             problem->alpha,
                                                             problem->A,
                                                             problem->ld_a,
                                                             problem->B,
                                                             problem->ld_b,
                                                             problem->beta,
                                                             problem->C,
                                                             problem->ld_c);
        break;
    case GEMMStridedBatch:
        tensile_problem = create_gemm_contraction_problem_strided(problem->trans_a,
                                                                  problem->trans_b,
                                                                  problem->m,
                                                                  problem->n,
                                                                  problem->k,
                                                                  problem->alpha,
                                                                  problem->A,
                                                                  problem->ld_a,
                                                                  problem->stride_a,
                                                                  problem->B,
                                                                  problem->ld_b,
                                                                  problem->stride_b,
                                                                  problem->beta,
                                                                  problem->C,
                                                                  problem->ld_c,
                                                                  problem->stride_c,
                                                                  problem->batch_size);
        break;
    }

    return tensile_problem;
}

template <typename T>
auto GetTensileInputs(RocblasContractionProblem<T>* problem)
{
    Tensile::TypedContractionInputs<T> inputs;
    switch(problem->problem_type)
    {
    case GEMM:
    case GEMMStridedBatch:
        inputs.a     = problem->A;
        inputs.b     = problem->B;
        inputs.c     = problem->C;
        inputs.d     = problem->C;
        inputs.alpha = *problem->alpha;
        inputs.beta  = *problem->beta;
        break;
    }

    return inputs;
}

struct TensileHostImpl : TensileHost
{
    void initializeHost(const char* lib_path)
    {
        std::string path(lib_path);
        std::string dir = path + "/*co";

        glob_t glob_result;
        glob(dir.c_str(), GLOB_TILDE, NULL, &glob_result);
        for(unsigned int i = 0; i < glob_result.gl_pathc; ++i)
            adapter.loadCodeObjectFile(glob_result.gl_pathv[i]);
        globfree(&glob_result);

        std::string filename = path + "/TensileLibrary.yaml";
        library              = std::dynamic_pointer_cast<
            Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>(
            Tensile::LoadLibraryFile<Tensile::ContractionProblem>(filename));

        hardware = Tensile::hip::GetCurrentDevice();
    }

    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;
    std::shared_ptr<Tensile::Hardware>                                           hardware;
    Tensile::hip::SolutionAdapter                                                adapter;

    ~TensileHostImpl() override = default; // Ensure virtual base class
};

template <typename T>
rocblas_status TensileHostCall<T>::runContractionProblem(RocblasContractionProblem<T>* problem,
                                                         TensileHost*                  host)
try
{
    auto tensile_problem = ConstructTensileProblem<T>(problem);
    auto inputs          = GetTensileInputs<T>(problem);
    auto hosti           = dynamic_cast<TensileHostImpl*>(host);
    if(!hosti)
        return rocblas_status_internal_error;

    auto solution = hosti->library->findBestSolution(tensile_problem, *hosti->hardware);
    auto result   = solution->solve(tensile_problem, inputs, *hosti->hardware);
    hosti->adapter.launchKernels(result);
    return rocblas_status_success;
}
catch(...)
{
    return rocblas_status_internal_error;
}

TensileHost* createTensileHost()
{
    return new TensileHostImpl();
}

template <typename T>
inline rocblas_status callTensileContraction(RocblasContractionProblem<T>* problem,
                                             TensileHost*                  host)
{
    TensileHostCall<T> hostCaller;
    return hostCaller.runContractionProblem(problem, host);
}

rocblas_status callTensileContraction_half(RocblasContractionProblem<rocblas_half>* problem,
                                           TensileHost*                             host)
{
    return callTensileContraction<rocblas_half>(problem, host);
}
rocblas_status callTensileContraction_float(RocblasContractionProblem<float>* problem,
                                            TensileHost*                      host)
{
    return callTensileContraction<float>(problem, host);
}
rocblas_status callTensileContraction_double(RocblasContractionProblem<double>* problem,
                                             TensileHost*                       host)
{
    return callTensileContraction<double>(problem, host);
}
rocblas_status
    callTensileContraction_float_complex(RocblasContractionProblem<rocblas_float_complex>* problem,
                                         TensileHost*                                      host)
{
    return callTensileContraction<rocblas_float_complex>(problem, host);
}
rocblas_status callTensileContraction_double_complex(
    RocblasContractionProblem<rocblas_double_complex>* problem, TensileHost* host)
{
    return callTensileContraction<rocblas_double_complex>(problem, host);
}
#endif
