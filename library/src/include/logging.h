/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCBLAS_LOGGING_H_
#define _ROCBLAS_LOGGING_H_
#include "handle.h"
#include "rocblas_ostream.hpp"
#include "tuple_helper.hpp"
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

/************************************************************************************
 * Profile kernel arguments
 ************************************************************************************/
template <typename TUP>
class argument_profile
{
    // Output stream
    rocblas_ostream& os;

    // Mutex for multithreaded access to table
    std::shared_timed_mutex mutex;

    // Table mapping argument tuples into atomic counts
    std::unordered_map<TUP,
                       std::atomic_size_t*,
                       typename tuple_helper::hash_t<TUP>,
                       typename tuple_helper::equal_t<TUP>>
        map;

public:
    // A tuple of arguments is looked up in an unordered map.
    // A count of the number of calls with these arguments is kept.
    // arg is assumed to be an rvalue for efficiency
    void operator()(TUP&& arg)
    {
        decltype(map.end()) p;
        {
            // Acquire a shared lock for reading map
            std::shared_lock<std::shared_timed_mutex> lock(mutex);

            // Look up the tuple in the map
            p = map.find(arg);

            // If tuple already exists, atomically increment count and return
            if(p != map.end())
            {
                ++*p->second;
                return;
            }
        } // Release shared lock

        // Acquire an exclusive lock for modifying map
        std::lock_guard<std::shared_timed_mutex> lock(mutex);

        // If doesn't already exist, insert tuple by moving
        bool inserted;
        std::tie(p, inserted) = map.emplace(std::move(arg), nullptr);

        // If new entry inserted, replace nullptr with new value
        // If tuple already exists, atomically increment count
        if(inserted)
            p->second = new std::atomic_size_t{1};
        else
            ++*p->second;
    }

    // Constructor
    explicit argument_profile(rocblas_ostream& os)
        : os(os)
    {
    }

    // Cleanup handler which dumps profile at destruction
    ~argument_profile()
    try
    {
        // Print all of the tuples in the map
        for(auto& p : map)
        {
            os << "- ";
            tuple_helper::print_tuple_pairs(
                os, std::tuple_cat(p.first, std::make_tuple("call_count", p.second->load())));
            os << "\n";
            delete p.second;
        }
        os.flush();
    }
    catch(...)
    {
    }
};

// if profile logging is turned on with
// (handle->layer_mode & rocblas_layer_mode_log_profile) != 0
// log_profile will call argument_profile to profile actual arguments,
// keeping count of the number of times each set of arguments is used
template <typename... Ts>
inline void log_profile(rocblas_handle handle, const char* func, Ts&&... xs)
{
    // Make a tuple with the arguments
    auto tup = std::make_tuple("rocblas_function", func, std::forward<Ts>(xs)...);

    // Set up profile
    static argument_profile<decltype(tup)> profile(*handle->log_profile_os);

    // Add at_quick_exit handler is added in case the program terminates early
    static int aqe = at_quick_exit([] { profile.~argument_profile(); });

    // Profile the tuple
    profile(std::move(tup));
}

/********************************************
 * Log values (for log_trace and log_bench) *
 ********************************************/
template <typename H, typename... Ts>
static inline void log_arguments(rocblas_ostream& os, const char* sep, H head, Ts&&... xs)
{
    os << head;
    int x[] = {(os << sep << std::forward<Ts>(xs), 0)...};
    os << "\n";
    os.flush();
}

// if trace logging is turned on with
// (handle->layer_mode & rocblas_layer_mode_log_trace) != 0
// log_function will call log_arguments to log arguments with a comma separator
template <typename... Ts>
inline void log_trace(rocblas_handle handle, Ts&&... xs)
{
    log_arguments(*handle->log_trace_os, ",", std::forward<Ts>(xs)...);
}

// if bench logging is turned on with
// (handle->layer_mode & rocblas_layer_mode_log_bench) != 0
// log_bench will call log_arguments to log a string that
// can be input to the executable rocblas-bench.
template <typename... Ts>
inline void log_bench(rocblas_handle handle, Ts&&... xs)
{
    log_arguments(*handle->log_bench_os, " ", std::forward<Ts>(xs)...);
}

/*************************************************
 * Trace log scalar values pointed to by pointer *
 *************************************************/
inline float log_trace_scalar_value(const rocblas_half* value)
{
    return value ? float(*value) : std::numeric_limits<float>::quiet_NaN();
}

template <typename T, typename std::enable_if<!is_complex<T>, int>::type = 0>
inline T log_trace_scalar_value(const T* value)
{
    return value ? *value : std::numeric_limits<T>::quiet_NaN();
}

template <typename T, typename std::enable_if<+is_complex<T>, int>::type = 0>
inline T log_trace_scalar_value(const T* value)
{
    return value ? *value
                 : T{std::numeric_limits<typename T::value_type>::quiet_NaN(),
                     std::numeric_limits<typename T::value_type>::quiet_NaN()};
}

/*************************************************
 * Bench log scalar values pointed to by pointer *
 *************************************************/
inline std::string log_bench_scalar_value(const char* name, const rocblas_half* value)
{
    std::stringstream ss;
    rocblas_ostream   os(ss);
    os << "--" << name << " " << (value ? float(*value) : std::numeric_limits<float>::quiet_NaN());
    return ss.str();
}

template <typename T, typename std::enable_if<!is_complex<T>, int>::type = 0>
inline std::string log_bench_scalar_value(const char* name, const T* value)
{
    std::stringstream ss;
    rocblas_ostream   os(ss);
    os << "--" << name << " " << (value ? *value : std::numeric_limits<T>::quiet_NaN());
    return ss.str();
}

template <typename T, typename std::enable_if<+is_complex<T>, int>::type = 0>
inline std::string log_bench_scalar_value(const char* name, const T* value)
{
    std::stringstream ss;
    rocblas_ostream   os(ss);
    os << "--" << name << " "
       << (value ? std::real(*value) : std::numeric_limits<typename T::value_type>::quiet_NaN());
    if(value && std::imag(*value))
        os << " --" << name << "i " << std::imag(*value);
    return ss.str();
}

#define LOG_BENCH_SCALAR_VALUE(name) log_bench_scalar_value(#name, name)

/******************************************************************
 * Log alpha and beta with dynamic compute_type in *_ex functions *
 ******************************************************************/
inline rocblas_status log_trace_alpha_beta_ex(rocblas_datatype   compute_type,
                                              const void*        alpha,
                                              const void*        beta,
                                              std::stringstream& alphass,
                                              std::stringstream& betass)
{
    rocblas_ostream alphaos(alphass);
    rocblas_ostream betaos(betass);
    switch(compute_type)
    {
    case rocblas_datatype_f16_r:
        alphaos << log_trace_scalar_value(reinterpret_cast<const rocblas_half*>(alpha));
        betaos << log_trace_scalar_value(reinterpret_cast<const rocblas_half*>(beta));
        break;
    case rocblas_datatype_f32_r:
        alphaos << log_trace_scalar_value(reinterpret_cast<const float*>(alpha));
        betaos << log_trace_scalar_value(reinterpret_cast<const float*>(beta));
        break;
    case rocblas_datatype_f64_r:
        alphaos << log_trace_scalar_value(reinterpret_cast<const double*>(alpha));
        betaos << log_trace_scalar_value(reinterpret_cast<const double*>(beta));
        break;
    case rocblas_datatype_i32_r:
        alphaos << log_trace_scalar_value(reinterpret_cast<const int32_t*>(alpha));
        betaos << log_trace_scalar_value(reinterpret_cast<const int32_t*>(beta));
        break;
    case rocblas_datatype_f32_c:
        alphaos << log_trace_scalar_value(reinterpret_cast<const rocblas_float_complex*>(alpha));
        betaos << log_trace_scalar_value(reinterpret_cast<const rocblas_float_complex*>(beta));
        break;
    case rocblas_datatype_f64_c:
        alphaos << log_trace_scalar_value(reinterpret_cast<const rocblas_double_complex*>(alpha));
        alphaos << log_trace_scalar_value(reinterpret_cast<const rocblas_double_complex*>(beta));
        break;
    default:
        return rocblas_status_not_implemented;
    }
    return rocblas_status_success;
}

inline rocblas_status log_bench_alpha_beta_ex(rocblas_datatype compute_type,
                                              const void*      alpha,
                                              const void*      beta,
                                              std::string&     alphas,
                                              std::string&     betas)
{
    switch(compute_type)
    {
    case rocblas_datatype_f16_r:
        alphas = log_bench_scalar_value("alpha", reinterpret_cast<const rocblas_half*>(alpha));
        betas  = log_bench_scalar_value("beta", reinterpret_cast<const rocblas_half*>(beta));
        break;
    case rocblas_datatype_f32_r:
        alphas = log_bench_scalar_value("alpha", reinterpret_cast<const float*>(alpha));
        betas  = log_bench_scalar_value("beta", reinterpret_cast<const float*>(beta));
        break;
    case rocblas_datatype_f64_r:
        alphas = log_bench_scalar_value("alpha", reinterpret_cast<const double*>(alpha));
        betas  = log_bench_scalar_value("beta", reinterpret_cast<const double*>(beta));
        break;
    case rocblas_datatype_i32_r:
        alphas = log_bench_scalar_value("alpha", reinterpret_cast<const int32_t*>(alpha));
        betas  = log_bench_scalar_value("beta", reinterpret_cast<const int32_t*>(beta));
        break;
    case rocblas_datatype_f32_c:
        alphas = log_bench_scalar_value("alpha",
                                        reinterpret_cast<const rocblas_float_complex*>(alpha));
        betas
            = log_bench_scalar_value("beta", reinterpret_cast<const rocblas_float_complex*>(beta));
        break;
    case rocblas_datatype_f64_c:
        alphas = log_bench_scalar_value("alpha",
                                        reinterpret_cast<const rocblas_double_complex*>(alpha));
        betas
            = log_bench_scalar_value("beta", reinterpret_cast<const rocblas_double_complex*>(beta));
        break;
    default:
        return rocblas_status_not_implemented;
    }
    return rocblas_status_success;
}

#endif
