/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_test.hpp"
#include "testing_ostream_threadsafety.hpp"
#include "type_dispatch.hpp"

namespace
{
    template <typename...>
    struct ostream_threadsafety_testing : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "ostream_threadsafety"))
                testing_ostream_threadsafety(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct ostream_threadsafety : RocBLAS_Test<ostream_threadsafety, ostream_threadsafety_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "ostream_threadsafety");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<ostream_threadsafety>(arg.name);
        }
    };

    TEST_P(ostream_threadsafety, auxilliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<ostream_threadsafety_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(ostream_threadsafety);

} // namespace
