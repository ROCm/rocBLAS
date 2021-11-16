/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_arguments.hpp"
#include "tuple_helper.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <istream>
#include <ostream>
#include <utility>

#ifdef WIN32
// Clang specific code
template <typename T>
rocblas_internal_ostream& operator<<(rocblas_internal_ostream& os, std::pair<char const*, T> p)
{
    os << p.first << ":";
    os << p.second;
    return os;
}

rocblas_internal_ostream& operator<<(rocblas_internal_ostream&                os,
                                     std::pair<char const*, rocblas_datatype> p)
{
    os << p.first << ":";
    os << rocblas_datatype_string(p.second);
    return os;
}

rocblas_internal_ostream& operator<<(rocblas_internal_ostream&                      os,
                                     std::pair<char const*, rocblas_initialization> p)
{
    os << p.first << ":";
#define CASE(x) \
    case x:     \
        return os << #x
    switch(p.second)
    {
        CASE(rocblas_initialization::rand_int);
        CASE(rocblas_initialization::trig_float);
        CASE(rocblas_initialization::hpl);
    }
    return os << "unknown";
}
#undef CASE

rocblas_internal_ostream& operator<<(rocblas_internal_ostream& os, std::pair<char const*, bool> p)
{
    os << p.first << ":";
    os << (p.second ? "true" : "false");
    return os;
}
// End of Clang specific code
#endif

// Function to print Arguments out to stream in YAML format
rocblas_internal_ostream& operator<<(rocblas_internal_ostream& os, const Arguments& arg)
{
    // delim starts as "{ " and becomes ", " afterwards
    auto print_pair = [&, delim = "{ "](const char* name, const auto& value) mutable {
        os << delim << std::make_pair(name, value);
        delim = ", ";
    };

    // Print each (name, value) tuple pair
#define NAME_VALUE_PAIR(NAME) print_pair(#NAME, arg.NAME)
    FOR_EACH_ARGUMENT(NAME_VALUE_PAIR, ;);

    // Closing brace
    return os << " }\n";
}

// Google Tests uses this automatically with std::ostream to dump parameters
std::ostream& operator<<(std::ostream& os, const Arguments& arg)
{
    rocblas_internal_ostream oss;
    // Print to rocblas_internal_ostream, then transfer to std::ostream
    return os << (oss << arg);
}

// Function to read Structures data from stream
std::istream& operator>>(std::istream& is, Arguments& arg)
{
    is.read(reinterpret_cast<char*>(&arg), sizeof(arg));
    return is;
}

// Error message about incompatible binary file format
static void validation_error [[noreturn]] (const char* name)
{
    rocblas_cerr << "Arguments field \"" << name
                 << "\" does not match format.\n\n"
                    "Fatal error: Binary test data does match input format.\n"
                    "Ensure that rocblas_arguments.hpp and rocblas_common.yaml\n"
                    "define exactly the same Arguments, that rocblas_gentest.py\n"
                    "generates the data correctly, and that endianness is the same."
                 << std::endl;
    rocblas_abort();
}

// rocblas_gentest.py is expected to conform to this format.
// rocblas_gentest.py uses rocblas_common.yaml to generate this format.
void Arguments::validate(std::istream& ifs)
{
    char      header[8]{}, trailer[8]{};
    Arguments arg{};

    ifs.read(header, sizeof(header));
    ifs >> arg;
    ifs.read(trailer, sizeof(trailer));

    if(strcmp(header, "rocBLAS"))
        validation_error("header");

    if(strcmp(trailer, "ROCblas"))
        validation_error("trailer");

    auto check_func = [sig = 0u](const char* name, const auto& value) mutable {
        static_assert(sizeof(value) <= 256,
                      "Fatal error: Arguments field is too large (greater than 256 bytes).");
        for(size_t i = 0; i < sizeof(value); ++i)
        {
            if(reinterpret_cast<const unsigned char*>(&value)[i] ^ sig ^ i)
                validation_error(name);
        }
        sig = (sig + 89) % 256;
    };

    // Apply check_func to each pair (name, value) of Arguments as a tuple
#define CHECK_FUNC(NAME) check_func(#NAME, arg.NAME)
    FOR_EACH_ARGUMENT(CHECK_FUNC, ;);
}
