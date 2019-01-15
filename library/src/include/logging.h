/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef LOGGING_H
#define LOGGING_H
#include "handle.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <memory>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <string>

class tuple_helper
{
    protected:
    /************************************************************************************
     * Print values
     ************************************************************************************/
    // Default output
    template <typename T>
    static void print_value(std::ostream& os, const T& x)
    {
        os << x;
    }

    // Floating-point output
    static void print_value(std::ostream& os, double x)
    {
        if(std::isnan(x))
            os << ".nan";
        else if(std::isinf(x))
            os << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            os << s;
        }
    }

    // Character output
    static void print_value(std::ostream& os, char c)
    {
        char s[]{c, 0};
        os << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& os, bool b) { os << (b ? "true" : "false"); }

    // string output
    static void print_value(std::ostream& os, const char* s) { os << std::quoted(s); }
    static void print_value(std::ostream& os, const std::string& s) { print_value(os, s.c_str()); }

    /************************************************************************************
     * Print tuples
     ************************************************************************************/
    template <typename T, size_t idx = std::tuple_size<T>()>
    struct print_tuple_recurse
    {
        template <typename F>
        __attribute__((always_inline)) void operator()(F& print_argument, const T& tuple)
        {
            print_tuple_recurse<T, idx - 2>{}(print_argument, tuple);
            print_argument(std::get<idx - 2>(tuple), std::get<idx - 1>(tuple));
        }
    };

    template <typename T>
    struct print_tuple_recurse<T, 0>
    {
        template <typename F>
        __attribute__((always_inline)) void operator()(F& print_argument, const T& tuple)
        {
        }
    };

    // Print a tuple which is expected to be (name1, value1, name2, value2, ...)
    template <typename TUP>
    static void print_tuple(std::ostream& os, const TUP& tuple)
    {
        static_assert(std::tuple_size<TUP>() % 2 == 0, "Tuple size must be even");

        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print_argument = [&, delim = '{' ](auto name, auto value) mutable
        {
            os << delim << " " << name << ": ";
            print_value(os, value);
            delim = ',';
        };
        print_tuple_recurse<TUP>{}(print_argument, tuple);
        os << " }" << std::endl;
    }

    /************************************************************************************
     * Compute value hashes for (key1, value1, key2, value2, ...) tuples
     ************************************************************************************/
    // Workaround for compilers which don't implement C++14 enum hash (LWG 2148)
    template <typename T>
    static typename std::enable_if<std::is_enum<T>{}, size_t>::type hash(const T& x)
    {
        return std::hash<typename std::underlying_type<T>::type>{}(x);
    }

    // Default hash for non-enum types
    template <typename T>
    static typename std::enable_if<!std::is_enum<T>{}, size_t>::type hash(const T& x)
    {
        return std::hash<T>{}(x);
    }

    // C-style string hash since std::hash does not hash them
    static constexpr size_t hash(const char* s)
    {
        size_t seed = 0xcbf29ce484222325;
        for(const char* p = s; *p; ++p)
            seed = (seed ^ *p) * 0x100000001b3;
        return seed;
    }

    // For consistency with above
    static size_t hash(const std::string& s) { return hash(s.c_str()); }

    // Combine tuple value hashes, computing hash of all tuple values
    template <typename TUP, size_t idx = std::tuple_size<TUP>{}>
    struct tuple_hash_recurse
    {
        __attribute__((always_inline)) size_t operator()(const TUP& tup)
        {
            size_t seed = tuple_hash_recurse<TUP, idx - 2>{}(tup);
            return seed ^ (hash(std::get<idx - 1>(tup)) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
        }
    };

    // Leaf node
    template <typename TUP>
    struct tuple_hash_recurse<TUP, 0>
    {
        __attribute__((always_inline)) size_t operator()(const TUP&) { return 0; }
    };

    // Hash function class compatible with STL containers
    template <typename TUP>
    struct hash_t
    {
        static_assert(std::tuple_size<TUP>() % 2 == 0, "Tuple size must be even");
        size_t operator()(const TUP& x) const { return tuple_hash_recurse<TUP>{}(x); }
    };

    /************************************************************************************
     * Test (key1, value1, key2, value2, ...) tuples for equality of values
     ************************************************************************************/
    template <typename T>
    static bool equal(const T& x1, const T& x2)
    {
        return std::equal_to<T>{}(x1, x2);
    }

    static bool equal(const char* s1, const char* s2) { return !strcmp(s1, s2); }
    static bool equal(const std::string& s1, const char* s2) { return !strcmp(s1.c_str(), s2); }
    static bool equal(const char* s1, const std::string& s2) { return !strcmp(s1, s2.c_str()); }

    // Recursively compare tuple values, short-circuiting
    template <typename TUP, size_t idx = std::tuple_size<TUP>()>
    struct tuple_equal_recurse
    {
        bool operator()(const TUP& t1, const TUP& t2)
        {
            return equal(std::get<idx - 1>(t1), std::get<idx - 1>(t2)) &&
                   tuple_equal_recurse<TUP, idx - 2>{}(t1, t2);
        }
    };

    // Leaf node
    template <typename TUP>
    struct tuple_equal_recurse<TUP, 0>
    {
        bool operator()(const TUP&, const TUP&) { return true; }
    };

    // Equality test class compatible with STL containers
    template <typename TUP>
    struct equal_t
    {
        static_assert(std::tuple_size<TUP>() % 2 == 0, "Tuple size must be even");
        __attribute__((flatten)) bool operator()(const TUP& x, const TUP& y) const
        {
            return tuple_equal_recurse<TUP>{}(x, y);
        }
    };

    /************************************************************************************
     * Log values (for log_trace and log_bench)
     ************************************************************************************/
    public:
    template <typename H, typename... Ts>
    static void log_arguments(std::ostream& os, const char* sep, H head, Ts&&... xs)
    {
        os << head;
        int x[] = {(os << sep << std::forward<Ts>(xs), 0)...};
        os << "\n";
    }
};

/************************************************************************************
 * Profile kernel arguments
 ************************************************************************************/
template <typename TUP>
class argument_profile : tuple_helper
{
    // Output stream
    std::ostream& os;

    // Mutex for multithreaded access to table
    std::shared_timed_mutex mutex;

    // Table mapping argument tuples into atomic counts
    std::unordered_map<TUP, std::atomic_size_t*, hash_t<TUP>, equal_t<TUP>> map;

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
    explicit argument_profile(std::ostream& os) : os(os) {}

    // Cleanup handler which dumps profile at destruction
    ~argument_profile()
    {
        // Print all of the tuples in the map
        for(auto& p : map)
        {
            print_tuple(os,
                        std::tuple_cat(p.first, std::make_tuple("call_count", p.second->load())));
            delete p.second;
        }
    }
};

// if profile logging is turned on with
// (handle->layer_mode & rocblas_layer_mode_log_profile) != 0
// log_profile will call argument_profile to profile actual arguments,
// keeping count of the number of times each set of arguments is used
template <typename... Ts>
inline void log_profile(rocblas_handle handle, const char* func, Ts&&... xs)
{
    auto tup = std::make_tuple("rocblas_function", func, std::forward<Ts>(xs)...);
    static argument_profile<decltype(tup)> profile(*handle->log_profile_os);
    profile(std::move(tup));
}

// if trace logging is turned on with
// (handle->layer_mode & rocblas_layer_mode_log_trace) != 0
// log_function will call log_arguments to log arguments with a comma separator
template <typename... Ts>
inline void log_trace(rocblas_handle handle, Ts&&... xs)
{
    tuple_helper::log_arguments(*handle->log_trace_os, ",", std::forward<Ts>(xs)...);
}

// if bench logging is turned on with
// (handle->layer_mode & rocblas_layer_mode_log_bench) != 0
// log_bench will call log_arguments to log a string that
// can be input to the executable rocblas-bench.
template <typename... Ts>
inline void log_bench(rocblas_handle handle, Ts&&... xs)
{
    tuple_helper::log_arguments(*handle->log_bench_os, " ", std::forward<Ts>(xs)...);
}

#endif
