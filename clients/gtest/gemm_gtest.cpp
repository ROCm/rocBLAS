/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include "testing_gemm.hpp"

using namespace std;

/* =====================================================================
     BLAS-3 GEMM:
   =================================================================== */

namespace {

// GEMM testing class
struct gemm : ::testing::TestWithParam<Arguments>
{
    // Filter for which tests get into gemm right now
    static function<bool(const Arguments&)> filter()
    {
        return [](const Arguments &arg)
        {
            return (arg.a_type == rocblas_datatype_f64_r ||
                    arg.a_type == rocblas_datatype_f32_r ||
                    arg.a_type == rocblas_datatype_f16_r) &&
                   (!strcmp(arg.function, "testing_gemm") ||
                    !strcmp(arg.function, "testing_gemm_NaN") ||
                    !strcmp(arg.function, "testing_gemm_bad_arg"));
        };
    }
};

template<class ...T>
void testit(const Arguments &arg)
{
    if (!strcmp(arg.function, "testing_gemm")) {
        rocblas_status status = testing_gemm<T...>(arg);

        /* if not success, then the input argument is problematic,
           so detect the error message */
        if (status != rocblas_status_success)
        {
            if (arg.M < 0 || arg.N < 0 || arg.K < 0)
            {
                EXPECT_EQ(rocblas_status_invalid_size, status);
            }
            else if (arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
            {
                EXPECT_EQ(rocblas_status_invalid_size, status);
            }
            else if (arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N)
            {
                EXPECT_EQ(rocblas_status_invalid_size, status);
            }
            else if (arg.ldc < arg.M)
            {
                EXPECT_EQ(rocblas_status_invalid_size, status);
            }
        } else if (!strcmp(arg.function, "testing_gemm_NaN")) {
            testing_gemm_NaN<T...>(arg);
        } else if (!strcmp(arg.function, "testing_gemm_bad_arg")) {
            testing_gemm_bad_arg<T...>();
        }
    }
}

// The testit function is instantiated with zero or more types
// depending on the *_type Arguments.
TEST_P(gemm, gemm)
{
    const Arguments &arg = GetParam();
    switch (arg.a_type)
    {
    default: return; // Skip unrecognized types
    case rocblas_datatype_f64_r: return testit<double>(arg);
    case rocblas_datatype_f32_r: return testit<float>(arg);
    case rocblas_datatype_f16_r: return testit<half>(arg);
 // case rocblas_datatype_f64_c: return testit<rocblas_double_complex>(arg);
 // case rocblas_datatype_f32_c: return testit<rocblas_float_complex>(arg);
 // case rocblas_datatype_f16_c: return testit<rocblas_half_complex>(arg);
 //
 // TODO:^   NOTE: testit<...> is ready for multiple data types, if
 // a_type, c_type, compute_type, etc. need to be passed separately.
    }
}

// The tests are instantiated by filtering through the RocBLAS_Data stream
INSTANTIATE_TEST_CASE_P(prefix, gemm, \
 ::testing::ValuesIn(RocBLAS_Data::begin(gemm::filter()), RocBLAS_Data::end()));

} // namespace
