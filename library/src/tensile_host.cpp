/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************/

// The implementation of the rocBLAS<->Tensile interface layer.

#include "rocblas.h"

#ifndef USE_TENSILE_HOST

// In the old Tensile client, rocblas_initialize() is a no-op
extern "C" void rocblas_initialize() {}

#else

/*****************************************************************************
 * This is the only file in rocBLAS which should #include Tensile headers    *
 * or reference Tensile identifiers. tensile_host.hpp defines the interface. *
 *****************************************************************************/

#include "tensile_host.hpp"
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
#include <atomic>
#include <complex>
#include <dlfcn.h>
#include <exception>
#include <glob.h>
#include <iomanip>
#include <libgen.h>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

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

        // Tensile does not short-circuit alpha==0. As a workaround, we set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the inputs.
        // It optimizes all problems with alpha==0 into K=0 and alpha=(don't care)
        auto k = prob.k && *prob.alpha ? prob.k : 0;

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != rocblas_operation_none)
        {
            a = {
                    Tensile_Ti,
                    {k, prob.m, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            a = {
                    Tensile_Ti,
                    {prob.m, k, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If A is complex and conjugated, add a ComplexConjugate op to aops
        if(is_complex<Ti> && prob.trans_a == rocblas_operation_conjugate_transpose)
            aops.push_back(Tensile::TensorOp::Type::ComplexConjugate);

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != rocblas_operation_none)
        {
            b = {
                    Tensile_Ti,
                    {prob.n, k, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            b = {
                    Tensile_Ti,
                    {k, prob.n, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // If B is complex and conjugated, add a ComplexConjugate op to bops
        if(is_complex<Ti> && prob.trans_b == rocblas_operation_conjugate_transpose)
            bops.push_back(Tensile::TensorOp::Type::ComplexConjugate);

        // Descriptor for input matrix C
        Tensile::TensorDescriptor c{Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c}};

        // Descriptor for output matrix D
        Tensile::TensorDescriptor d{Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_c, prob.col_stride_d, prob.batch_stride_d}};

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

        // Pass atomics mode to Tensile interface
        if(prob.handle->atomics_mode == rocblas_atomics_not_allowed)
            tensileProblem.setDeterministicMode(true);

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
    inline auto GetTensileInputs(const RocblasContractionProblem<Ti, To, Tc>& prob)
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
        inputs.a = reinterpret_cast<const Tensile_Ti*>(prob.A);
        inputs.b = reinterpret_cast<const Tensile_Ti*>(prob.B);
        inputs.c = reinterpret_cast<const Tensile_To*>(prob.C);
        inputs.d = reinterpret_cast<Tensile_To*>(prob.D);

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set inputs.alpha=0
        if(prob.k)
            AlphaBeta<Ti, To, Tc>::copy(&inputs.alpha, prob.alpha);
        else
            memset(&inputs.alpha, 0, sizeof(inputs.alpha));
        AlphaBeta<Ti, To, Tc>::copy(&inputs.beta, prob.beta);

        return inputs;
    }

    /*************************************************
     * The TensileHost struct interfaces with Tensile *
     *************************************************/
    struct TensileHost
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;

        struct adapter_s
        {
            mutable std::atomic<Tensile::hip::SolutionAdapter*> adapter{nullptr};
            mutable std::mutex                                  mutex;
        };

        std::vector<adapter_s> const adapters;

        TensileHost()
            : adapters(GetDeviceCount())
        {
        }

        TensileHost(const TensileHost&) = delete;
        TensileHost& operator=(const TensileHost&) = delete;

        static int GetDeviceCount()
        {
            int count;
            if(hipGetDeviceCount(&count) != hipSuccess)
            {
                rocblas_cerr
                    << "\nrocBLAS error: Could not initialize Tensile host: No devices found"
                    << std::endl;
                rocblas_abort();
            }
            return count;
        }

        ~TensileHost()
        {
            for(auto& a : adapters)
                delete a.adapter;
        }

        /*******************************************************
         * Testpath() tests that a path exists and is readable *
         *******************************************************/
        static bool TestPath(const std::string& path)
        {
            return access(path.c_str(), R_OK) == 0;
        }

        /*********************************************************************
         * Initialize adapter and library according to environment variables *
         * and default paths based on librocblas.so location and GPU         *
         *********************************************************************/
        void initialize(Tensile::hip::SolutionAdapter& adapter)
        {
            std::string path;
            path.reserve(PATH_MAX);

            // The name of the current GPU platform
            std::string processor = rocblas_get_arch_name();

            const char* env = getenv("ROCBLAS_TENSILE_LIBPATH");
            if(env)
            {
                path = env;
            }
            else
            {
#ifndef ROCBLAS_STATIC_LIB
                Dl_info info;

                // Find the location of librocblas.so
                // Fall back on hard-coded path if static library or not found

                if(dladdr((void*)rocblas_initialize, &info))
                {
                    path = info.dli_fname;
                    path = std::string{dirname(&path[0])};
                }
                else
#endif
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

            // only load modules for the current architecture
            auto dir = path + "/*" + processor + "*co";

            glob_t glob_result{};
            int    g = glob(dir.c_str(), GLOB_NOSORT, nullptr, &glob_result);
            if(!g)
            {
                for(size_t i = 0; i < glob_result.gl_pathc; ++i)
                    adapter.loadCodeObjectFile(glob_result.gl_pathv[i]);
            }
            else if(g == GLOB_NOMATCH)
            {
                static auto& once = rocblas_cerr
                                    << "\nrocBLAS warning: No paths matched " << dir
                                    << ". Make sure that ROCBLAS_TENSILE_LIBPATH is set correctly."
                                    << std::endl;
            }
            else
            {
                static auto& once = rocblas_cerr
                                    << "\nrocBLAS warning: glob(\"" << dir << "\", ...) returned "
                                    << (g == GLOB_ABORTED ? "GLOB_ABORTED"
                                                          : g == GLOB_NOSPACE ? "GLOB_NOSPACE"
                                                                              : "an unknown error")
                                    << "." << std::endl;
            }
            globfree(&glob_result);

            // We initialize a local static variable with a lambda function call to avoid
            // race conditions when multiple threads with different device IDs try to
            // initialize library. This ensures that only one thread initializes library,
            // and other threads trying to initialize library wait for it to complete.
            static int once = [&] {
#ifdef TENSILE_YAML
                path += "/TensileLibrary.yaml";
#else
                path += "/TensileLibrary.dat";
#endif
                if(!TestPath(path))
                {
                    rocblas_cerr << "\nrocBLAS error: Cannot read " << path << ": "
                                 << strerror(errno) << std::endl;
                    rocblas_abort();
                }

                library = std::dynamic_pointer_cast<
                    Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>(
                    Tensile::LoadLibraryFile<Tensile::ContractionProblem>(path));

                return 0;
            }();

            if(!library)
            {
                rocblas_cerr << "\nrocBLAS error: Could not initialize Tensile library"
                             << std::endl;
                rocblas_abort();
            }
        }
    };

    // Return the library and adapter for the current HIP device
    Tensile::hip::SolutionAdapter& get_library_and_adapter(
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>* library
        = nullptr)
    try
    {
        // TensileHost is initialized on the first call
        static TensileHost host;

        int device;
        hipGetDevice(&device);

        // Adapter entry for the current HIP device ID
        auto& a       = host.adapters.at(device);
        auto* adapter = a.adapter.load(std::memory_order_acquire);

        // Once set, a.adapter contains the adapter for the current HIP device ID
        if(!adapter)
        {
            // Lock so that only one thread performs initialization of the adapter
            std::lock_guard<std::mutex> lock(a.mutex);

            adapter = a.adapter.load(std::memory_order_relaxed);
            if(!adapter)
            {
                // Allocate a new adapter using the current HIP device
                adapter = new Tensile::hip::SolutionAdapter;

                // Initialize the adapter and possibly the library
                host.initialize(*adapter);

                // Atomically change the adapter stored for this device ID
                a.adapter.store(adapter, std::memory_order_release);
            }
        }

        // If an adapter is found, it is assumed that the library is initialized
        if(library)
            *library = host.library;
        return *adapter;
    }
    catch(const std::exception& e)
    {
        rocblas_cerr << "\nrocBLAS error: Could not initialize Tensile host:\n"
                     << e.what() << std::endl;
        rocblas_abort();
    }
    catch(...)
    {
        rocblas_cerr
            << "\nrocBLAS error: Could not initialize Tensile host:\nUnknown exception thrown"
            << std::endl;
        rocblas_abort();
    }
} // namespace

/******************************************************************************
 * runContractionProblem calls Tensile to run a contraction problem described *
 * by RocblasContractionProblem                                               *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblas_status runContractionProblem(const RocblasContractionProblem<Ti, To, Tc>& prob)
{
    rocblas_status                                status = rocblas_status_internal_error;
    std::shared_ptr<Tensile::ContractionSolution> solution;

    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;
        auto& adapter      = get_library_and_adapter(&library);
        auto  hardware     = Tensile::hip::GetCurrentDevice();
        auto  tensile_prob = ConstructTensileProblem(prob);
        solution           = library->findBestSolution(tensile_prob, *hardware);

        if(!solution)
        {
            // We print the error message only once, to avoid excessive logging
            static auto& once = rocblas_cerr << "\nrocBLAS error: No Tensile solution found for "
                                             << prob;
            status = rocblas_status_not_implemented;
        }
        else
        {
            auto handle = prob.handle;
            adapter.launchKernels(solution->solve(tensile_prob, GetTensileInputs(prob), *hardware),
                                  handle->rocblas_stream,
                                  handle->startEvent,
                                  handle->stopEvent);
            status = rocblas_status_success;
        }
    }
    catch(const std::exception& e)
    {
        static auto& once = rocblas_cerr << "\nrocBLAS error: " << (solution ? "" : "No ")
                                         << "Tensile solution found, but exception thown for "
                                         << prob << e.what() << std::endl;
    }
    catch(...)
    {
        static auto& once = rocblas_cerr
                            << "\nrocBLAS error: " << (solution ? "" : "No ")
                            << "Tensile solution found, but unknown exception thown for " << prob
                            << std::endl;
    }

    return status;
}

/***************************************************************
 * ! \brief  Initialize rocBLAS for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
extern "C" void rocblas_initialize()
{
    get_library_and_adapter();
}

/******************************************************************************
 * Intantiate the cases of runContractionProblem which are needed to satisfy  *
 * rocBLAS dependencies. This file's template functions are not defined in a  *
 * header file, in order to keep Tensile and rocBLAS separate.                *
 ******************************************************************************/

// Non-EX types
template rocblas_status runContractionProblem(const RocblasContractionProblem<rocblas_half>&);

template rocblas_status runContractionProblem(const RocblasContractionProblem<float>&);

template rocblas_status runContractionProblem(const RocblasContractionProblem<double>&);

template rocblas_status
    runContractionProblem(const RocblasContractionProblem<rocblas_float_complex>&);

template rocblas_status
    runContractionProblem(const RocblasContractionProblem<rocblas_double_complex>&);

// EX types
template rocblas_status
    runContractionProblem(const RocblasContractionProblem<rocblas_half, rocblas_half, float>&);

template rocblas_status runContractionProblem(
    const RocblasContractionProblem<rocblas_bfloat16, rocblas_bfloat16, float>&);

template rocblas_status
    runContractionProblem(const RocblasContractionProblem<int8_t, int32_t, int32_t>&);

#endif
