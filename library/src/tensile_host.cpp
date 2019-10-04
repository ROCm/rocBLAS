#ifdef USE_TENSILE_HOST

#include "rocblas.h"
#include "tensile_host.hpp"


#include <Tensile/Tensile.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <Tensile/Utils.hpp>
#include <Tensile/TensorDescriptor.hpp>

#include <string>
#include <iostream>
//#include <filesystem>

#include <glob.h>

template <typename>
Tensile::DataType tensile_datatype()
{
    throw std::runtime_error("undefined datatype");
    return Tensile::DataType::Float;
}

template <>
Tensile::DataType tensile_datatype<rocblas_half>()
{
    return Tensile::DataType::Half;
}

template <>
Tensile::DataType tensile_datatype<float>()
{
    return Tensile::DataType::Float;
}
 
template <>
Tensile::DataType tensile_datatype<double>()
{
    return Tensile::DataType::Double;
}

template <>
Tensile::DataType tensile_datatype<rocblas_float_complex>()
{
    return Tensile::DataType::ComplexFloat;
}

template <>
Tensile::DataType tensile_datatype<rocblas_double_complex>()
{
    return Tensile::DataType::ComplexDouble;
}

template <typename T>
Tensile::ContractionProblem create_gemm_contraction_problem_strided (
                                     rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     unsigned long       m,
                                     unsigned long       n,
                                     unsigned long       k,
                                     const T*          alpha,
                                     const T*          A,
                                     unsigned long       ld_a,
                                     unsigned long     stride_a,
                                     const T*          B,
                                     unsigned long       ld_b,
                                     unsigned long     stride_b,
                                     const T*          beta,
                                     T*                C,
                                     unsigned long       ld_c,
                                     unsigned long     stride_c,
                                     unsigned long     batchSize)
{

    bool transposeA = false;
    if (trans_a == rocblas_operation_conjugate_transpose)
        transposeA = true;

    bool transposeB = false;
    if(trans_b == rocblas_operation_conjugate_transpose)
        transposeB = true;

    Tensile::DataType dt = tensile_datatype<T>();
    Tensile::ContractionProblem problem = Tensile::ContractionProblem::GEMM_Strides(
                                                        transposeA, transposeB,
                                                        dt, dt, dt, dt,
                                                        m, n, k, batchSize,
                                                        ld_a, stride_a,
                                                        ld_b, stride_b,
                                                        ld_c, stride_c,
                                                        ld_c, stride_c,
                                                        *beta);

    return problem;
}


// construct the gemm contraction problem
template <typename T>
Tensile::ContractionProblem create_gemm_contraction_problem (
                                     rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     unsigned long       m,
                                     unsigned long       n,
                                     unsigned long       k,
                                     const T*          alpha,
                                     const T*          A,
                                     unsigned long       ld_a,
                                     const T*          B,
                                     unsigned long       ld_b,
                                     const T*          beta,
                                     T*                C,
                                     unsigned long       ld_c)
{

    bool transposeA = false;
    if (trans_a == rocblas_operation_conjugate_transpose)
        transposeA = true;

    bool transposeB = false;
    if(trans_b == rocblas_operation_conjugate_transpose)
        transposeB = true;    


    Tensile::ContractionProblem::FreeIndex free;
    Tensile::ContractionProblem::BoundIndex bound;

    free.ca = free.da = 0;
    free.cb = free.db = 1;

    Tensile::TensorDescriptor a, b, c, d;

    Tensile::DataType dt = tensile_datatype<T>();
    if(transposeA)
    {
        a = Tensile::TensorDescriptor(dt, {k, m}, {1, ld_a});
        free.a = 1;
        bound.a = 0;
    }
    else
    {
        a = Tensile::TensorDescriptor(dt, {m, k}, {1, ld_a});
        free.a = 0;
        bound.a = 1;
    }

    if(transposeB)
    {
        b = Tensile::TensorDescriptor(dt, {n, k}, {1, ld_b});
        free.b = 0;
        bound.b = 1;
    }
    else
    {
        b = Tensile::TensorDescriptor(dt, {k, n}, {1, ld_b});
        free.b = 1;
        bound.b = 0;
    }

    Tensile::ContractionProblem::FreeIndices freeIndices{free};
    Tensile::ContractionProblem::BatchIndices batchIndices;
    Tensile::ContractionProblem::BoundIndices boundIndices{bound};

    d = Tensile::TensorDescriptor(dt, {m, n}, {1, ld_c});

    unsigned int batchCount = 1;

    a.appendDim(batchCount);
    b.appendDim(batchCount);
    d.appendDim(batchCount);

    batchIndices.push_back({2,2,2,2});

    if(*beta != 0.0)
        c = d;

    Tensile::TensorOps nop;

    return Tensile::ContractionProblem(a, nop, b, nop, c, nop, d, nop, freeIndices, batchIndices, boundIndices, *beta);
}


template <typename T>                               
Tensile::ContractionProblem ConstructTensileProblem(RocblasContractionProblem<T> *problem)
{
    Tensile::ContractionProblem tensile_problem;
    switch(problem->problem_type)
    {
        case GEMM: 
            tensile_problem = create_gemm_contraction_problem<T> (
                                     problem->trans_a,problem->trans_b,
                                     problem->m,problem->n,problem->k,
                                     problem->alpha,
                                     problem->A,problem->ld_a,
                                     problem->B,problem->ld_b,
                                     problem->beta,
                                     problem->C,problem->ld_c);
            break;
        case GEMMStridedBatch:
            tensile_problem = create_gemm_contraction_problem_strided (
                                     problem->trans_a,problem->trans_b,
                                     problem->m,problem->n,problem->k,
                                     problem->alpha,
                                     problem->A,problem->ld_a,problem->stride_a,
                                     problem->B,problem->ld_b,problem->stride_b,
                                     problem->beta,
                                     problem->C,problem->ld_c,problem->stride_c,
                                     problem->batch_size);
            break;      
    }

    return tensile_problem;
}


template <typename T>
Tensile::TypedContractionInputs<T> GetTensileInputs(RocblasContractionProblem<T> *problem)
{
    Tensile::TypedContractionInputs<T> inputs;
    switch(problem->problem_type)
    {
        case GEMM:
        case GEMMStridedBatch:
            inputs.a = problem->A;
            inputs.b = problem->B;
            inputs.c = problem->C;
            inputs.d = problem->C;
            inputs.alpha = *(problem->alpha);
            inputs.beta = *(problem->beta);
            break;
    }

    return inputs;
}

class TensileHostImpl : public TensileHost
{
public:
  void initializeHost(const char* lib_path)
  {
    //std::string dir ("/home/wgilmart/dev/wbgilmartin/tasks/new_client_integration/iteration3/Tensile/build/1_BenchmarkProblems/Cijk_Ailk_Bljk_SB_00/00_Final/source/library/*hsaco");
    
    std::string path (lib_path);
    std::string dir = path + "/*co";


    glob_t glob_result;
    glob(dir.c_str(),GLOB_TILDE,NULL,&glob_result);
    //vector<string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        //files.push_back(string(glob_result.gl_pathv[i]));
      //std::cout << glob_result.gl_pathv[i] << std::endl;
      std::string cofilename = std::string(glob_result.gl_pathv[i]);
      adapter.loadCodeObjectFile(cofilename);
    }
    globfree(&glob_result);

    std::string filename = path + "/TensileLibrary.yaml";
    library = std::dynamic_pointer_cast<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>(
                        Tensile::LoadLibraryFile<Tensile::ContractionProblem>(filename));

 
   // namespace fs = std::filesystem;

   // for (const auto & entry : fs::directory_iterator(dir))
   //     std::cout << entry.path() << std::endl;    
 

     //std::DIR *dp;
    //if((dp  = opendir(dir.c_str())) == NULL) {
    //    std::cout << "Error(" << errno << ") opening " << dir << std::endl;
    //    return errno;
    //}

    //while ((dirp = readdir(dp)) != NULL) {
    //    std::cout << dirp->d_name << std::endl;
        //files.push_back(string(dirp->d_name));
    //}
    //closedir(dp);
 
    //std::string filename ("/home/wgilmart/dev/wbgilmartin/tasks/new_client_integration/iteration3/Tensile/build/1_BenchmarkProblems/Cijk_Ailk_Bljk_SB_00/00_Final/source/library/TensileLibrary.yaml");
    //library = std::dynamic_pointer_cast<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>(
    //                    Tensile::LoadLibraryFile<Tensile::ContractionProblem>(filename));

    //std::string cofilename ("/home/wgilmart/dev/wbgilmartin/tasks/new_client_integration/iteration3/Tensile/build/1_BenchmarkProblems/Cijk_Ailk_Bljk_SB_00/00_Final/source/library/TensileLibrary_gfx906.co");
    //adapter.loadCodeObjectFile(cofilename);

    //std::string k906filename ("/home/wgilmart/dev/wbgilmartin/tasks/new_client_integration/iteration3/Tensile/build/1_BenchmarkProblems/Cijk_Ailk_Bljk_SB_00/00_Final/source/library/Kernels.so-000-gfx906.hsaco");
    //adapter.loadCodeObjectFile(k906filename);

    //std::string k900filename ("/home/wgilmart/dev/wbgilmartin/tasks/new_client_integration/iteration3/Tensile/build/1_BenchmarkProblems/Cijk_Ailk_Bljk_SB_00/00_Final/source/library/Kernels.so-000-gfx900.hsaco");
    //adapter.loadCodeObjectFile(k900filename);

    //std::string k803filename ("/home/wgilmart/dev/wbgilmartin/tasks/new_client_integration/iteration3/Tensile/build/1_BenchmarkProblems/Cijk_Ailk_Bljk_SB_00/00_Final/source/library/Kernels.so-000-gfx803.hsaco");
    //adapter.loadCodeObjectFile(k803filename);

    hardware = Tensile::hip::GetCurrentDevice();
  }

//private:
  std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;
  std::shared_ptr<Tensile::Hardware> hardware;
  Tensile::hip::SolutionAdapter adapter;
};

template <typename T>
rocblas_status TensileHostCall<T>::runContractionProblem(RocblasContractionProblem<T> *problem, TensileHost *host)
{
    Tensile::ContractionProblem tensile_problem;
    try
    {
        tensile_problem = ConstructTensileProblem<T>(problem);
    }
    catch(...)
    {
        return rocblas_status_internal_error;
    }
    Tensile::TypedContractionInputs<T> inputs;
    try
    {
        inputs = GetTensileInputs<T>(problem);
    }
    catch(...)
    {
        return rocblas_status_internal_error;
    }

    TensileHostImpl * hosti = dynamic_cast<TensileHostImpl *>(host);
    if (hosti == nullptr)
    {
        return rocblas_status_internal_error;
    }

    std::vector<Tensile::KernelInvocation> result;
    try
    {
        auto solution = hosti->library->findBestSolution(tensile_problem, *(hosti->hardware));
        result = solution->solve(tensile_problem, inputs, *(hosti->hardware));
    }
    catch(...)
    {
        return rocblas_status_internal_error;
    }

    try
    {
        hosti->adapter.launchKernels(result);
    }
    catch(...)
    {
        return rocblas_status_internal_error;
    }

    return rocblas_status_success; 
}

TensileHost *createTensileHost()
{ 

  TensileHostImpl *host = new TensileHostImpl();
  return host;
}

template <typename T>
rocblas_status callTensileContraction( RocblasContractionProblem<T> *problem, TensileHost *host)
{
    TensileHostCall<T> hostCaller;
    return hostCaller.runContractionProblem(problem, host);
}

rocblas_status callTensileContraction_half(RocblasContractionProblem<rocblas_half> *problem, TensileHost *host)
{
    return callTensileContraction<rocblas_half>(problem, host);
}
rocblas_status callTensileContraction_float(RocblasContractionProblem<float> *problem, TensileHost *host)
{
    return callTensileContraction<float>(problem, host);
}
rocblas_status callTensileContraction_double(RocblasContractionProblem<double> *problem, TensileHost *host)
{
    return callTensileContraction<double>(problem, host);
}
rocblas_status callTensileContraction_float_complex(RocblasContractionProblem<rocblas_float_complex> *problem, TensileHost *host)
{
    return callTensileContraction<rocblas_float_complex>(problem, host);
}
rocblas_status callTensileContraction_double_complex(RocblasContractionProblem<rocblas_double_complex> *problem, TensileHost *host)
{
    return callTensileContraction<rocblas_double_complex>(problem, host);
}
#endif
