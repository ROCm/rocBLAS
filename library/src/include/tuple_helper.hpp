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
#include <utility>

/*****************************************************
 * Tuple helper class provides operations on tuples  *
 *****************************************************/
class tuple_helper
{
    /********************************************************************
     * Traverse (key, value) pairs, applying functions or printing YAML *
     ********************************************************************/

    // Recursion to traverse the tuple
    template <typename TUP, size_t size = std::tuple_size<TUP>{}>
    struct apply_pairs_recurse
    {
        template <typename FUNC>
        void operator()(FUNC&& action, const TUP& tuple)
        {
            // Current pair is at (i, i+1)
            constexpr size_t i = std::tuple_size<TUP>{} - size;

            //Perform the action, passing the 2 elements of the pair
            action(std::get<i>(tuple), std::get<i + 1>(tuple));

            // Recurse to the next pair, forwarding the action
            apply_pairs_recurse<TUP, size - 2>{}(std::forward<FUNC>(action), tuple);
        }
    };

    // Leaf node
    template <typename TUP>
    struct apply_pairs_recurse<TUP, 0>
    {
        template <typename FUNC>
        void operator()(FUNC&& action, const TUP& tuple)
        {
        }
    };

public:
    // Apply a function to pairs in a tuple (name1, value1, name2, value2, ...)
    template <typename FUNC, typename TUP>
    __attribute__((flatten)) static void apply_pairs(FUNC&& action, const TUP& tuple)
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");
        apply_pairs_recurse<TUP>{}(std::forward<FUNC>(action), tuple);
    }

    // Print a tuple which is expected to be (name1, value1, name2, value2, ...)
    template <typename TUP>
    static rocblas_ostream& print_tuple_pairs(rocblas_ostream& os, const TUP& tuple)
    {
        static_assert(std::tuple_size<TUP>{} % 2 == 0, "Tuple size must be even");

        // Turn YAML formatting on
        os << rocblas_ostream::yaml_on;

        // delim starts as "{ " and becomes ", " afterwards
        auto print_pair = [&os, delim = "{ "](const char* name, const auto& value) mutable {
            os << delim << name << ": " << value;
            delim = ", ";
        };

        // Call print_argument for each (name, value) tuple pair
        apply_pairs(std::move(print_pair), tuple);

        // Closing brace and turn YAML formatting off
        return os << " }\n" << rocblas_ostream::yaml_off;
    }

    /*********************************************************************
     * Compute value hashes for (key1, value1, key2, value2, ...) tuples *
     *********************************************************************/
private:
    // Default hash
    template <typename T>
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

    // For std::string consistency with above
    static size_t hash(const std::string& s)
    {
        return hash(s.c_str());
    }

    // Combine tuple value hashes, computing hash of all tuple values
    template <typename TUP, size_t size = std::tuple_size<TUP>{}>
    struct tuple_hash_recurse
    {
        size_t operator()(const TUP& tup)
        {
            // Current pair is at (i, i+1)
            constexpr size_t i = std::tuple_size<TUP>{} - size;

            // Compute the hash of the remaining pairs
            size_t seed = tuple_hash_recurse<TUP, size - 2>{}(tup);

            // Combine the hash of the remaining pairs with the hash of the current pair
            return seed ^ (hash(std::get<i + 1>(tup)) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
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

    // Recursively compare tuple values for equality, short-circuiting on false
    template <typename TUP, size_t size = std::tuple_size<TUP>{}>
    struct tuple_equal_recurse
    {
        bool operator()(const TUP& t1, const TUP& t2) const
        {
            // Current pair is at (i, i+1)
            constexpr size_t i = std::tuple_size<TUP>{} - size;

            // Compare the values of the current pair
            // Continue with the later pairs, short-circuiting on false
            return equal(std::get<i + 1>(t1), std::get<i + 1>(t2))
                   && tuple_equal_recurse<TUP, size - 2>{}(t1, t2);
        }
    };

    // Leaf node returns true when there are no more values to compare
    template <typename TUP>
    struct tuple_equal_recurse<TUP, 0>
    {
        bool operator()(const TUP&, const TUP&) const
        {
            return true;
        }
    };

public:
    // Tuple key,value equality test class compatible with STL associative containers
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
