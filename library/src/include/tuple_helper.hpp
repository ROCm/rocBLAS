/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ROCBLAS_TUPLE_HPP_
#define _ROCBLAS_TUPLE_HPP_

#include "handle.h"
#include "rocblas_ostream.hpp"
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

/***********************************************************************
 * Tuple helper class provides operations on std::tuple argument lists *
 ***********************************************************************/
class tuple_helper
{
    // Recursion to traverse the tuple
    template <typename TUP, size_t idx = std::tuple_size<TUP>{}>
    struct apply_pairs_recurse
    {
        template <typename FUNC>
        void operator()(FUNC&& action, const TUP& tuple)
        {
            static constexpr size_t i = std::tuple_size<TUP>{} - idx;
            action(std::get<i>(tuple), std::get<i + 1>(tuple));
            apply_pairs_recurse<TUP, idx - 2>{}(std::forward<FUNC>(action), tuple);
        }
    };

    template <typename TUP>
    struct apply_pairs_recurse<TUP, 0>
    {
        template <typename FUNC>
        void operator()(FUNC&& action, const TUP& tuple)
        {
        }
    };

public:
    // Apply a function to pairs in a tuple which is expected to be (name1, value1, name2, value2, ...)
    template <typename FUNC, typename TUP>
    __attribute__((flatten)) static void apply_pairs(FUNC&& action, const TUP& tuple)
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");
        apply_pairs_recurse<TUP>{}(std::forward<FUNC>(action), tuple);
    }

    // Print a tuple which is expected to be (name1, value1, name2, value2, ...)
    template <typename TUP>
    static rocblas_ostream& print_tuple_pairs(rocblas_ostream& str, const TUP& tuple)
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");

        // delim starts as '{' and becomes ',' afterwards
        auto print_argument = [&, delim = '{'](const char* name, auto&& value) mutable {
            str << delim << ' ' << name << ": " << value;
            delim = ',';
        };
        apply_pairs(print_argument, tuple);
        return str << " }";
    }

    /************************************************************************************
     * Compute value hashes for (key1, value1, key2, value2, ...) tuples
     ************************************************************************************/
private:
    // Workaround for compilers which don't implement C++14 enum hash (LWG 2148)
    template <typename T, typename std::enable_if<std::is_enum<T>{}, int>::type = 0>
    static size_t hash(const T& x)
    {
        return std::hash<typename std::underlying_type<T>::type>{}(x);
    }

    // Default hash for non-enum types
    template <typename T, typename std::enable_if<!std::is_enum<T>{}, int>::type = 0>
    static size_t hash(const T& x)
    {
        return std::hash<T>{}(x);
    }

    // C-style string hash since std::hash does not hash them
    static size_t hash(const char* s)
    {
        size_t seed = 0xcbf29ce484222325;
        for(auto p = reinterpret_cast<const unsigned char*>(s); *p; ++p)
            seed = (seed ^ *p) * 0x100000001b3; // FNV-1a
        return seed;
    }

    // For consistency with above
    static size_t hash(const std::string& s)
    {
        return hash(s.c_str());
    }

    // Combine tuple value hashes, computing hash of all tuple values
    template <typename TUP, size_t idx = std::tuple_size<TUP>{}>
    struct tuple_hash_recurse
    {
        size_t operator()(const TUP& tup)
        {
            static constexpr size_t i = std::tuple_size<TUP>{} - idx;
            size_t seed = hash(std::get<i + 1>(tup)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed ^ tuple_hash_recurse<TUP, idx - 2>{}(tup);
        }
    };

    // Leaf node
    template <typename TUP>
    struct tuple_hash_recurse<TUP, 0>
    {
        size_t operator()(const TUP&)
        {
            return 0;
        }
    };

public:
    // Hash function class compatible with STL containers
    template <typename TUP>
    struct hash_t
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");
        __attribute__((flatten)) size_t operator()(const TUP& x) const
        {
            return tuple_hash_recurse<TUP>{}(x);
        }
    };

    /************************************************************************************
     * Test (key1, value1, key2, value2, ...) tuples for equality of values
     ************************************************************************************/
private:
    template <typename T>
    static bool equal(const T& x1, const T& x2)
    {
        return x1 == x2;
    }

    static bool equal(const char* s1, const char* s2)
    {
        return !strcmp(s1, s2);
    }

    static bool equal(const std::string& s1, const char* s2)
    {
        return !strcmp(s1.c_str(), s2);
    }

    static bool equal(const char* s1, const std::string& s2)
    {
        return !strcmp(s1, s2.c_str());
    }

    // Recursively compare tuple values, short-circuiting
    template <typename TUP, size_t idx = std::tuple_size<TUP>{}>
    struct tuple_equal_recurse
    {
        bool operator()(const TUP& t1, const TUP& t2) const
        {
            static constexpr size_t i = std::tuple_size<TUP>{} - idx;
            return equal(std::get<i + 1>(t1), std::get<i + 1>(t2))
                   && tuple_equal_recurse<TUP, idx - 2>{}(t1, t2);
        }
    };

    // Leaf node
    template <typename TUP>
    struct tuple_equal_recurse<TUP, 0>
    {
        bool operator()(const TUP&, const TUP&) const
        {
            return true;
        }
    };

public:
    // Equality test class compatible with STL containers
    template <typename TUP>
    struct equal_t
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");
        __attribute__((flatten)) bool operator()(const TUP& x, const TUP& y) const
        {
            return tuple_equal_recurse<TUP>{}(x, y);
        }
    };
};

#endif
