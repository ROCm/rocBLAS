/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_arguments.hpp"
#include "../../library/src/include/tuple_helper.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>

// Function to print Arguments out to stream in YAML format
rocblas_ostream& operator<<(rocblas_ostream& os, const Arguments& arg)
{
    return tuple_helper::print_tuple_pairs(os, arg.as_tuple());
}

// Google Tests uses this automatically with std::ostream to dump parameters
std::ostream& operator<<(std::ostream& os, const Arguments& arg)
{
    rocblas_ostream oss;
    // Print to rocblas_ostream, then transfer to std::ostream
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
        if(sizeof(value) > 256)
        {
            rocblas_cerr << "Fatal error: Arguments field \"" << name
                         << "\" is too large (greater than 256 bytes)." << std::endl;
            rocblas_abort();
        }
        for(size_t i = 0; i < sizeof(value); ++i)
        {
            if(reinterpret_cast<const unsigned char*>(&value)[i] ^ sig ^ i)
                validation_error(name);
        }
        sig = (sig + 89) % 256;
    };

    // Apply check_func to each pair (name, value) of Arguments as a tuple
    tuple_helper::apply_pairs(check_func, arg.as_tuple());
}
