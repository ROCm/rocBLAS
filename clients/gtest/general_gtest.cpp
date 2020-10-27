/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{

    template <typename T>
    void expect_decimals_eq(T a, T b, int decimals)
    {
        EXPECT_EQ(std::round(a * pow(10, decimals)), std::round(b * pow(10, decimals)));
    }

    // half floats

    template <typename T>
    void testing_half_operators(const Arguments& arg)
    {

        T c(0.5);
        T s(2.0);

        T result = -(c + c) * s;
        result /= s;
        EXPECT_EQ(result, T(-1.0));

        c      = T(0.5);
        s      = T(2.0);
        result = c * s + s / c;
        EXPECT_EQ((float)result, 5.0f);

        // unique harmonic convergence
        // search half-precision-arithmetic-fp16-versus-bfloat16 harmonic
        result = T(0);
        if(std::is_same<T, rocblas_half>{})
        {
            for(int i = 1; i <= 513; i++)
                result += T(1.0) / T(i);
            expect_decimals_eq((float)result, 7.08594f, 5);
        }
        else if(std::is_same<T, rocblas_bfloat16>{})
        {
            for(int i = 1; i <= 65; i++)
                result += T(1.0) / T(i);
            expect_decimals_eq((float)result, 5.0625f, 4);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct half_operators_testing : rocblas_test_invalid
    {
    };

    template <typename T>
    struct half_operators_testing<
        T,
        std::enable_if_t<std::is_same<T, rocblas_half>{} || std::is_same<T, rocblas_bfloat16>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "half_operators"))
                testing_half_operators<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct half_operators : RocBLAS_Test<half_operators, half_operators_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "half_operators");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<half_operators> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);
            return std::move(name);
        }
    };

    TEST_P(half_operators, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<half_operators_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(half_operators);

    //
    // complex

    template <typename T>
    void testing_complex_operators(const Arguments& arg)
    {
        using R = real_t<T>;

        T c(0.5, 0.25);
        R s(2.0);

        T result = c * s;
        EXPECT_EQ(result, T(1.0, 0.5));

        result /= s;
        EXPECT_EQ(result, c);

        T val(1.0, -2.0);
        result = (s - val) / s;
        EXPECT_EQ(result, T(0.5, 1.0));

        result = T(20.0, -4.0) / T(3.0, 2.0);
        EXPECT_EQ(result, T(4.0, -4.0));

        result = 1.0 / T(1.0, 0.0);
        EXPECT_EQ(result, T(1.0, 0.0));
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct complex_operators_testing : rocblas_test_invalid
    {
    };

    template <typename T>
    struct complex_operators_testing<T,
                                     std::enable_if_t<std::is_same<T, rocblas_float_complex>{}
                                                      || std::is_same<T, rocblas_double_complex>{}>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "complex_operators"))
                testing_complex_operators<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct complex_operators : RocBLAS_Test<complex_operators, complex_operators_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "complex_operators");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<complex_operators> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);
            return std::move(name);
        }
    };

    TEST_P(complex_operators, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<complex_operators_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(complex_operators);

} // namespace
