/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
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
    struct testing_set_get_events : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            hipEvent_t e, startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

            {
                rocblas_local_handle handle;

                EXPECT_ROCBLAS_STATUS(rocblas_set_start_event(nullptr, startEvent),
                                      rocblas_status_invalid_handle);
                CHECK_ROCBLAS_ERROR(rocblas_set_start_event(handle, startEvent));
                EXPECT_ROCBLAS_STATUS(rocblas_get_start_event(nullptr, &e),
                                      rocblas_status_invalid_handle);
                EXPECT_ROCBLAS_STATUS(rocblas_get_start_event(handle, nullptr),
                                      rocblas_status_invalid_pointer);
                CHECK_ROCBLAS_ERROR(rocblas_get_start_event(handle, &e));
                EXPECT_EQ(e, startEvent);

                EXPECT_ROCBLAS_STATUS(rocblas_set_stop_event(nullptr, stopEvent),
                                      rocblas_status_invalid_handle);
                CHECK_ROCBLAS_ERROR(rocblas_set_stop_event(handle, stopEvent));
                EXPECT_ROCBLAS_STATUS(rocblas_get_stop_event(nullptr, &e),
                                      rocblas_status_invalid_handle);
                EXPECT_ROCBLAS_STATUS(rocblas_get_stop_event(handle, nullptr),
                                      rocblas_status_invalid_pointer);
                CHECK_ROCBLAS_ERROR(rocblas_get_stop_event(handle, &e));
                EXPECT_EQ(e, stopEvent);
            }

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));
        }
    };

    struct set_get_events : RocBLAS_Test<set_get_events, testing_set_get_events>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments&)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "set_get_events");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments&)
        {
            return RocBLAS_TestName<set_get_events>{};
        }
    };

    TEST_P(set_get_events, auxilliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(testing_set_get_events<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_events)

} // namespace
