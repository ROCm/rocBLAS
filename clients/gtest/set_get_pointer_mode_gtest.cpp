/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "utility.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{
    template <typename...>
    struct testing_set_get_pointer_mode : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            rocblas_pointer_mode mode = rocblas_pointer_mode_device;
            rocblas_local_handle handle(arg.atomics_mode);
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR(rocblas_get_pointer_mode(handle, &mode));
            EXPECT_EQ(rocblas_pointer_mode_device, mode);
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR(rocblas_get_pointer_mode(handle, &mode));
            EXPECT_EQ(rocblas_pointer_mode_host, mode);
        }
    };

    struct set_get_pointer_mode : RocBLAS_Test<set_get_pointer_mode, testing_set_get_pointer_mode>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments&)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "set_get_pointer_mode");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<set_get_pointer_mode>(arg.name);
        }
    };

    TEST_P(set_get_pointer_mode, auxilliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(testing_set_get_pointer_mode<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_pointer_mode)

} // namespace
