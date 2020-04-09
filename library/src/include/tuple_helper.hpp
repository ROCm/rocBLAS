/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCBLAS_TUPLE_HPP_
#define _ROCBLAS_TUPLE_HPP_

#include "handle.h"
#include "rocblas_ostream.hpp"
#include <cstddef>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

/*****************************************************
 * Tuple helper class provides operations on tuples  *
 *****************************************************/
class tuple_helper
{
    /********************************************************************
     * Traverse (key, value) pairs, applying functions or printing YAML *
     ********************************************************************/
    template <typename FUNC, typename TUP, size_t... I>
    static void apply_pairs_impl(FUNC&& func, const TUP& tuple, std::index_sequence<I...>)
    {
        // TODO: Replace with C++17 fold expression
        // (func(std::get<I * 2>(tuple), std::get<I * 2 + 1>(tuple)), ...);
        (void)(int[]){(func(std::get<I * 2>(tuple), std::get<I * 2 + 1>(tuple)), 0)...};
    }

public:
    // Apply a function to pairs in a tuple (name1, value1, name2, value2, ...)
    template <typename FUNC, typename TUP>
    static void apply_pairs(FUNC&& func, const TUP& tuple)
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");
        apply_pairs_impl(std::forward<FUNC>(func),
                         tuple,
                         std::make_index_sequence<std::tuple_size<TUP>{} / 2>{});
    }

    // Print a tuple which is expected to be (name1, value1, name2, value2, ...)
    template <typename TUP>
    static rocblas_ostream& print_tuple_pairs(rocblas_ostream& os, const TUP& tuple)
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");

        // delim starts as "{ " and becomes ", " afterwards
        auto print_pair = [&, delim = "{ "](const char* name, const auto& value) mutable {
            os << delim << std::make_pair(name, value);
            delim = ", ";
        };

        // Call print_argument for each (name, value) tuple pair
        apply_pairs(print_pair, tuple);

        // Closing brace
        return os << " }\n";
    }

    /*********************************************************************
     * Compute value hashes for (key1, value1, key2, value2, ...) tuples *
     *********************************************************************/
    // Default hash for non-enum types
    template <typename T, std::enable_if_t<!std::is_enum<T>{}, int> = 0>
    static size_t hash(const T& x)
    {
        return std::hash<T>{}(x);
    }

    // Workaround for compilers which don't implement C++14 enum hash
    template <typename T, std::enable_if_t<std::is_enum<T>{}, int> = 0>
    static size_t hash(const T& x)
    {
        return std::hash<typename std::underlying_type<T>::type>{}(x);
    }

    // C-style string hash since std::hash does not hash them
    static size_t hash(const char* s)
    {
        size_t seed = 0xcbf29ce484222325;
        for(auto p = reinterpret_cast<const unsigned char*>(s); *p; ++p)
            seed = (seed ^ *p) * 0x100000001b3; // FNV-1a
        return seed;
    }

    // For std::string consistency with above
    static size_t hash(const std::string& s)
    {
        return hash(s.c_str());
    }

    // Iterate over pairs, combining hash values
    template <typename TUP, size_t... I>
    static size_t hash(const TUP& tuple, std::index_sequence<I...>)
    {
        size_t seed = 0;
        for(size_t h : {hash(std::get<I * 2 + 1>(tuple))...})
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    // Hash function class compatible with STL containers
    template <typename TUP>
    struct hash_t
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");
        size_t operator()(const TUP& tuple) const
        {
            return hash(tuple, std::make_index_sequence<std::tuple_size<TUP>{} / 2>{});
        }
    };

    /************************************************************************
     * Test (key1, value1, key2, value2, ...) tuples for equality of values *
     ************************************************************************/
private:
    // Default comparison
    template <typename T>
    static bool equal(const T& x1, const T& x2)
    {
        return x1 == x2;
    }

    // C-string == C-string
    static bool equal(const char* s1, const char* s2)
    {
        return !strcmp(s1, s2);
    }

    // Compute equality of values in tuple (name, value) pairs
    template <typename TUP, size_t... I>
    static bool equal(const TUP& t1, const TUP& t2, std::index_sequence<I...>)
    {
        // TODO: Replace with C++17 fold expression
        // return (equal(std::get<I * 2 + 1>(t1), std::get<I * 2 + 1>(t2)) && ...);
        bool ret = true;
        (void)(bool[]){(ret = ret && equal(std::get<I * 2 + 1>(t1), std::get<I * 2 + 1>(t2)))...};
        return ret;
    }

public:
    // Tuple (name, value) equality test class is compatible with STL associative containers
    template <typename TUP>
    struct equal_t
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");
        bool operator()(const TUP& t1, const TUP& t2) const
        {
            return equal(t1, t2, std::make_index_sequence<std::tuple_size<TUP>{} / 2>{});
        }
    };
};

#endif
