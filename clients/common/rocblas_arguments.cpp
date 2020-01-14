/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_arguments.hpp"
#include "../../library/src/include/rocblas_ostream.hpp"
#include "../../library/src/include/tuple_helper.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

// Function to print Arguments out to stream in YAML format
// Google Tests uses this automatically to dump parameters
std::ostream& operator<<(std::ostream& os, const Arguments& arg)
{
    rocblas_ostringstream str;
    tuple_helper::print_tuple_pairs(str, arg.as_tuple());
    return os << str.str();
}

// Function to read Structures data from stream
std::istream& operator>>(std::istream& str, Arguments& arg)
{
    str.read(reinterpret_cast<char*>(&arg), sizeof(arg));
    return str;
}

// rocblas_gentest.py is expected to conform to this format.
// rocblas_gentest.py uses rocblas_common.yaml to generate this format.
void Arguments::validate(std::istream& ifs)
{
    auto error = [](const char* name) {
        std::cerr << "Arguments field " << name
                  << " does not match format.\n\n"
                     "Fatal error: Binary test data does match input format.\n"
                     "Ensure that rocblas_arguments.hpp and rocblas_common.yaml\n"
                     "define exactly the same Arguments, that rocblas_gentest.py\n"
                     "generates the data correctly, and that endianness is the same."
                  << std::endl;
        abort();
    };

    char      header[8]{}, trailer[8]{};
    Arguments arg{};
    ifs.read(header, sizeof(header));
    ifs >> arg;
    ifs.read(trailer, sizeof(trailer));

    if(strcmp(header, "rocBLAS"))
        error("header");
    if(strcmp(trailer, "ROCblas"))
        error("trailer");

    auto check_func = [&, sig = (unsigned char)0](const char* name, auto&& value) mutable {
        static_assert(sizeof(value) <= 255,
                      "One of the fields of Arguments is too large (> 255 bytes)");
        for(unsigned char i = 0; i < sizeof(value); ++i)
            if(reinterpret_cast<const unsigned char*>(&value)[i] ^ sig ^ i)
                error(name);
        sig += 89;
    };

    // Apply check_func to each member of Arguments as a tuple
    tuple_helper::apply_pairs(check_func, arg.as_tuple());
}
