/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_iamin.hpp"
#include "rocblas_reduction_impl.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_iamin_name[] = "unknown";
    template <>
    constexpr char rocblas_iamin_name<float>[] = "rocblas_isamin";
    template <>
    constexpr char rocblas_iamin_name<double>[] = "rocblas_idamin";
    template <>
    constexpr char rocblas_iamin_name<rocblas_float_complex>[] = "rocblas_icamin";
    template <>
    constexpr char rocblas_iamin_name<rocblas_double_complex>[] = "rocblas_izamin";

    // allocate workspace inside this API
    template <typename S, typename T>
    rocblas_status rocblas_iamin_impl(
        rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result)
    {
        static constexpr bool           isbatched     = false;
        static constexpr rocblas_stride stridex_0     = 0;
        static constexpr rocblas_int    batch_count_1 = 1;
        static constexpr int            NB            = 1024;

        return rocblas_reduction_impl<NB,
                                      isbatched,
                                      rocblas_fetch_amax_amin<S>,
                                      rocblas_reduce_amin,
                                      rocblas_finalize_amax_amin,
                                      rocblas_index_value_t<S>>(
            handle, n, x, incx, stridex_0, batch_count_1, result, rocblas_iamin_name<T>, "iamin");
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

#ifdef IMPL
#error IMPL IS ALREADY DEFINED
#endif

#define IMPL(name_, typei_, typew_)                                     \
    rocblas_status name_(rocblas_handle handle,                         \
                         rocblas_int    n,                              \
                         const typei_*  x,                              \
                         rocblas_int    incx,                           \
                         rocblas_int*   results)                        \
    try                                                                 \
    {                                                                   \
        return rocblas_iamin_impl<typew_>(handle, n, x, incx, results); \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocblas_status();                           \
    }

IMPL(rocblas_isamin, float, float);
IMPL(rocblas_idamin, double, double);
IMPL(rocblas_icamin, rocblas_float_complex, float);
IMPL(rocblas_izamin, rocblas_double_complex, double);

#undef IMPL

} // extern "C"
