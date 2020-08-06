/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "utility.hpp"
#include <string>

namespace
{
    template <typename...>
    struct testing_set_get_atomics_mode : rocblas_test_valid
    {
        void operator()(const Arguments&)
        {
            rocblas_atomics_mode mode = rocblas_atomics_mode(-1);
            rocblas_handle       handle;
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

            // Make sure the default atomics_mode is rocblas_atomics_allowed
            CHECK_ROCBLAS_ERROR(rocblas_get_atomics_mode(handle, &mode));
            EXPECT_EQ(rocblas_atomics_allowed, mode);

            // Make sure set()/get() functions work
            CHECK_ROCBLAS_ERROR(rocblas_set_atomics_mode(handle, rocblas_atomics_not_allowed));
            CHECK_ROCBLAS_ERROR(rocblas_get_atomics_mode(handle, &mode));
            EXPECT_EQ(rocblas_atomics_not_allowed, mode);

            CHECK_ROCBLAS_ERROR(rocblas_set_atomics_mode(handle, rocblas_atomics_allowed));
            CHECK_ROCBLAS_ERROR(rocblas_get_atomics_mode(handle, &mode));
            EXPECT_EQ(rocblas_atomics_allowed, mode);

            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
        }
    };

    struct set_get_atomics_mode : RocBLAS_Test<set_get_atomics_mode, testing_set_get_atomics_mode>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments&)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "set_get_atomics_mode");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<set_get_atomics_mode>(arg.name);
        }
    };

    TEST_P(set_get_atomics_mode, auxilliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(testing_set_get_atomics_mode<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_atomics_mode)

} // namespace
